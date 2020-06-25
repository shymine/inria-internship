import sys

from torch import nn

import aux_funcs as af
import network_architectures as arcs
import snip
import data
import model_funcs
import torch
import numpy as np
from snip import get_blocs


def main():
    print("test")
    device = af.get_pytorch_device()
    model, param = arcs.create_resnet_iterative(
        "networks/", "dense", "0", (True, [0.1, 0.1, 0.1, 0.1], 128), False
    )

    dataset = data.CIFAR10()
    optimizer, scheduler = af.get_full_optimizer(
        model, (param['learning_rate'], param['weight_decay'], param['momentum'], -1), ([4], [0.1])
    )
    train_params = dict(
        epochs=10,
        epoch_growth=[2, 4, 6],
        epoch_prune=[1, 3, 5, 7, 8, 9],
        prune_batch_size=128,
        prune_type="0",
        reinit=False,
        min_ratio=None#[0.5, 0.4, 0.3, 0.2],
    )
    params, best_model = model_funcs.iter_training_0(
        model, dataset, train_params, optimizer, scheduler, device
    )

    print("\nblocs:\n")
    for i,b in enumerate(snip.get_blocs(best_model)):
        p_z = sum([torch.sum(layer.weight != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())])
        t_p = sum([layer.weight.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())])
        print("{} total: {}, non zero: {}, ratio: {:.2f}".format(i, t_p, p_z, float(p_z)/float(t_p)))

    # calculer la somme pour chaque parameters
    p_z = sum([torch.sum(layer.weight != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), best_model.modules())])
    t_p = sum([layer.weight.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), best_model.modules())])
    print("\nfinal\n{} total: {}, non zero: {}, ratio: {:.2f}".format(i, t_p, p_z, float(p_z)/float(t_p)))


if __name__ == "__main__":
    main()
