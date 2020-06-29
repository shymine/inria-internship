import copy
import sys
import types
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


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


def snip_skip_layers(model, keep_ratio, loader, loss, index_to_prune, previous_masks, device='cpu', reinit=True):
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    _model = copy.deepcopy(model)

    blocks = get_blocs(_model)

    if index_to_prune >= len(blocks):
        print("index out of bloc range: index {}, number of blocks {}".format(index_to_prune, len(blocks)))
        return None
    
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

    masks = []
    for idx, bloc in enumerate(blocks):
        if idx != index_to_prune:
            masks.append([])
            continue
        grads_abs = []
        for layer in bloc.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) and layer.weight_mask.grad is not None:
                grads_abs.append(torch.abs(layer.weight_mask.grad))
        if len(grads_abs) == 0:
            masks.append([])
            continue
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
        num_params_to_keep = int(len(all_scores) * keep_ratio[idx])
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())
        masks.append(keep_masks)
    
    if previous_masks is not None:
        for i in range(index_to_prune):
            masks[i] = previous_masks[i]

    return masks


def apply_prune_mask_skip_layers(model, masks, index_to_prune):
    if masks is None:
        return
    blocks = get_blocs(model)

    for idx, bloc in enumerate(blocks):
        if idx != index_to_prune:
            continue
        prunable_layers = filter(lambda layer: isinstance(layer, (nn.Linear, nn.Conv2d)),
            bloc.modules())
        mask = masks[idx]

        for layer, msk in zip(prunable_layers, mask):
            assert (layer.weight.shape == msk.shape)

            layer.weight.data[msk == 0.] = 0.
            
            layer.weight_mask = nn.Parameter(msk)
            layer.weight_mask.requires_grad = False
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer) 
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)             

def snip_bloc_iterative(model, keep_ratio, mini_ratio, steps, loader, loss, device='cpu', reinit=True):
    # mini_ratio is now an array for every bloc
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    _model = copy.deepcopy(model)

    blocks = get_blocs(_model)
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
    for idx, bloc in enumerate(blocks):
        grads_abs = []
        for layer in bloc.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) and layer.weight_mask.grad is not None:
                grads_abs.append(torch.abs(layer.weight_mask.grad))
        if len(grads_abs) == 0:
            masks.append([])
            continue
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)

        intern_keep_ratio = keep_ratio[idx] ** (steps[idx] + 1)
        if mini_ratio is not None:
            intern_keep_ratio = intern_keep_ratio if intern_keep_ratio > mini_ratio[idx] else mini_ratio[idx]
        print("keep ratio {}: {}".format(idx, intern_keep_ratio))

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
        for layer, msk in zip(prunable_layers, mask):
            assert(layer.weight.shape == msk.shape)

            layer.weight.data[msk == 0.] = 0.
            # fuse the weight masks
            if hasattr(layer, 'weight_mask'):
                msk = msk*layer.weight_mask
            layer.weight_mask = nn.Parameter(msk)
            layer.weight_mask.requires_grad = False
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer) 
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)

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
    if sum(_model.ics)+1 == _model.num_output:
        blocks[-1].append(_model.end_layers)
    return blocks
