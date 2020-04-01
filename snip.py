import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

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

    network.zero_grad()
    outputs = network(inputs)
    total_loss = loss(outputs, targets)
    total_loss.backward()

    grads_abs = []
    for layer in network.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
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
    print("masks: {}".format(keep_masks))
    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

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
                return grads*mask
            return hook
        layer.weight.data[mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(mask))
