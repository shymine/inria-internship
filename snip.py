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
        for idy, layer in enumerate(bloc.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) and layer.weight_mask.grad is not None:
                grads_abs.append(torch.abs(layer.weight_mask.grad))
            # if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            #     print("{} layer: {}, grad: {}".format(idy, layer, layer.weight_mask.grad is not None))
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
        sums = [x.flatten().sum() for x in keep_masks]
        sums = sum(sums)
        numelem = sum([x.numel() for x in keep_masks])
        print("{} kept params: {}".format(idx, sums/numelem))
    
    if previous_masks is not None:
        for i in range(index_to_prune):
            masks[i] = previous_masks[i]
    
    # for mask in masks:
    #     print("len mask: {}".format(len(mask)))

    return masks


def apply_prune_mask_skip_layers(model, masks, index_to_prune):
    if masks is None:
        return
    blocks = get_blocs(model)

    for idx, bloc in enumerate(blocks):
        if idx != index_to_prune:
            continue
        prunable_layers = filter(
            lambda layer: isinstance(layer, (nn.Linear, nn.Conv2d)),
            bloc.modules()
        )
        mask = masks[idx]

        for layer, msk in zip(prunable_layers, mask):
            assert (layer.weight.shape == msk.shape)

            layer.weight.data[msk == 0.] = 0.
            
            layer.weight_mask = nn.Parameter(msk)
            layer.weight_mask.requires_grad = False
            # print("layer id: {}, {}".format(layer, id(layer)))
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer) 
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)             

    for i, b in enumerate(blocks):
        if i>index_to_prune:
            break
        p_z = sum([torch.sum(layer.weight != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())]) 
        #sum(p.numel() for p in b.parameters())
        t_p = sum([layer.weight.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())])
        #sum(len(list(filter(lambda x: x != 0., p.flatten()))) for p in b.parameters() if p.requires_grad)
        print("{} total: {}, non zero: {}, ratio: {:.2f}".format(i, t_p, p_z, float(p_z)/float(t_p)))
        # for layer in b.modules():
        #     if isinstance(layer, (nn.Linear, nn.Conv2d)):
        #         print("weight: {}".format(layer.weight))

    total_param = sum([layer.weight.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), model.modules())])
    param_z = sum([torch.sum(layer.weight != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), model.modules())]) 
    total_param = total_param-25610 if index_to_prune<3 else total_param
    param_z = param_z-25610 if index_to_prune<3 else param_z
    print("total param: {}\nparam_z: {}".format(total_param, param_z))
    print("keep ratio: {}".format(float(param_z)/float(total_param)))

def snip_bloc_iterative(model, keep_ratio, mini_ratio, steps, loader, loss, device='cpu', reinit=True):
    # mini_ratio is now an array for every bloc
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

        intern_keep_ratio = keep_ratio[id] ** (steps[id] + 1)
        if mini_ratio is not None:
            intern_keep_ratio = intern_keep_ratio if intern_keep_ratio > mini_ratio[id] else mini_ratio[id]
        print("keep ratio {}: {}".format(id, intern_keep_ratio))

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
    #print("num ics: {}, num outputs: {}".format(sum(_model.ics), _model.num_output))
    if sum(_model.ics)+1 == _model.num_output:
        blocks[-1].append(_model.end_layers)
    return blocks
