import copy
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def snip(model, keep_ratio, train_dataloader, loss, device="cpu"):
    inputs, targets = next(iter(train_dataloader))
    inputs, targets = inputs.to(device), targets.to(device)
    network = copy.deepcopy(model)

    for layer in network.modules():
        conv2 = isinstance(layer, nn.Conv2d)
        lin = isinstance(layer, nn.Linear)
        if conv2 or lin:
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            if conv2:
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
            if lin:
                layer.forward = types.MethodType(snip_forward_linear, layer)

    network.to(device)
    network.zero_grad()
    outputs = network(inputs)
    total_loss = loss(outputs, targets)
    total_loss.backward()

    grads_abs = []
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight_mask.grad is not None:
                grads_abs.append(torch.abs(layer.weight_mask.grad))
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    return (keep_masks)


def apply_prune_mask(model, masks):
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
        model.modules()
    )

    for layer, mask in zip(prunable_layers, masks):
        assert (layer.weight.shape == mask.shape)

        def hook_factory(mask):
            def hook(grads):
                return grads * mask

            return hook

        layer.weight.data[mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(mask))


def snip_skip_layers(model, keep_ratio, loader, loss, count_last_pruned, device='cpu', reinit=True):
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    _model = copy.deepcopy(model)
    count_pruned = 0

    for layer in _model.modules():
        conv2 = isinstance(layer, nn.Conv2d)
        lin = isinstance(layer, nn.Linear)
        if conv2 or lin:
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            if reinit and count_pruned > count_last_pruned:
                nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            count_pruned += 1
            if conv2:
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
            if lin:
                layer.forward = types.MethodType(snip_forward_linear, layer)

    #compte le nombre de layers qui ont été pruned dans le modèle
    print("count_pruned in snip: {}".format(count_pruned))

    _model.to(device)
    _model.zero_grad()
    print("shape of input: {}".format(inputs.shape))
    outputs = _model(inputs)
    total_loss = loss(outputs, targets)
    total_loss.backward()

    grads_abs = []
    for layer in _model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if layer.weight_mask.grad is not None:
                grads_abs.append(torch.abs(layer.weight_mask.grad))
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())


    return (keep_masks, count_pruned-1)


def apply_prune_mask_skip_layers(model, masks, count_pruned):
    prunable_layers = filter(
        lambda layer: isinstance(layer, (nn.Linear, nn.Conv2d)),
        model.modules()
    )
    count = 0
    for layer, mask in zip(prunable_layers, masks):
        assert (layer.weight.shape == mask.shape)

        def hook_factory(mask):
            def hook(grads):
                return grads * mask

            return hook

        if count >= count_pruned:
            # l'id des couches (linear and conv2d not blocks) that are pruned
            print("pruned: {}".format(count))
            layer.weight.data[mask == 0.] = 0.
            layer.weight.register_hook(hook_factory(mask))
        count += 1

def snip_bloc_iterative(model, keep_ratio, mini_ratio, steps, loader, loss, device='cpu', reinit=True):
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    _model = copy.deepcopy(model)

    # détection et répartition des blocs
    blocks = get_blocs(_model)
    # le tableau blocks contient les différentes layers comprisent entre les IC
    for layer in _model.modules():
        conv2 = isinstance(layer, nn.Conv2d)
        lin = isinstance(layer, nn.Linear)
        if conv2 or lin:
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            layer.weight.requires_grad = False
            if conv2:
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
            if lin:
                layer.forward = types.MethodType(snip_forward_linear, layer)

    _model.to(device)
    _model.zero_grad()
    outputs = _model(inputs)
    total_loss = loss(outputs, targets)
    total_loss.backward()

    # ranger les gradients par blocs
    masks = []
    for id, bloc in enumerate(blocks):
        grads_abs = []
        for layer in bloc.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) and layer.weight_mask.grad is not None:
                grads_abs.append(torch.abs(layer.weight_mask.grad))
        # print("grad_abs: {}".format(grads_abs))
        if len(grads_abs) == 0:
            masks.append([])
            continue
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        tmp = keep_ratio ** (steps[id] + 1)
        #intern_keep_ratio = tmp if tmp > mini_ratio else mini_ratio
        intern_keep_ratio = tmp
        # print("keep ratio {}: {}".format(id, intern_keep_ratio))

        num_params_to_keep = int(len(all_scores) * intern_keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g/norm_factor) >= acceptable_score).float())
        masks.append(keep_masks)
    return masks

def apply_prune_mask_bloc_iterative(model, masks):
    blocks = get_blocs(model)

    for id, bloc in enumerate(blocks):
        prunable_layers = filter(
            lambda layer: isinstance(layer, (nn.Conv2d, nn.Linear)),
            bloc.modules()
        )
        mask = masks[id]
        for layer, mask in zip(prunable_layers, mask):
            assert(layer.weight.shape == mask.shape)

            def hook_factory(mask):
                def hook(grads):
                    return grads * mask
                return hook

            layer.weight.data[mask == 0.] = 0.
            layer.weight.register_hook(hook_factory(mask))


def get_blocs(_model):
    indexes = [0]
    for i, v in enumerate(_model.ics):
        if v == 1:
            indexes.append(i)
    indexes.append(len(_model.ics) - 1)
    blocks = []
    for i in range(len(indexes) - 1):
        first, second = indexes[i], indexes[i + 1] + 1
        if first != 0:
            first += 1
        blocks.append(_model.layers[first:second])
    blocks[0].insert(0, _model.init_conv)
    if sum(_model.ics) == _model.num_output:
        blocks[-1].append(_model.end_layers)
    return blocks
