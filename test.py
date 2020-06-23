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
    # metrics = [
    #     dict(
    #         name='truc1',
    #         valid_top1_acc=[
    #             [25.3],
    #             [26.2],
    #             [28.1, 20.0],
    #             [29.0, 26.0],
    #             [29.0, 30.0],
    #             [30.0, 34.0, 30.0],
    #             [30.0, 36.0, 35.0],
    #             [30.0, 37.0, 40.0],
    #             [30.0, 37.0, 41.0, 45.0]
    #         ],
    #         test_top1_acc=[30.0, 38.0, 40.0, 44.0]
    #     ),
    #     dict(
    #         name='truc2',
    #         valid_top1_acc=[
    #             [20.0],
    #             [27.0],
    #             [30.0, 25.0],
    #             [31.0, 29.0],
    #             [30.0, 32.0],
    #             [31.0, 34.0, 30.0],
    #             [31.0, 36.0, 37.0],
    #             [31.0, 39.0, 41.0],
    #             [31.0, 39.0, 42.0, 45.0]
    #         ],
    #         test_top1_acc=[31.0, 40.0, 41.0, 44.0]
    #     )
    # ]
    # af.print_acc([(None, i) for i in metrics], True)
    print("test")
    device = af.get_pytorch_device()
    model, param = arcs.create_resnet_iterative("networks/", "dense", "0", (True, [0.1, 0.1, 0.1, 0.1], 128), False)

    dataset = data.CIFAR10()
    optimizer, scheduler = af.get_full_optimizer(model,
                                                 (0.01, 0.0001, 0.9, -1),
                                                 ([4], [0.1]))
    train_params = dict(
        epochs=10,
        epoch_growth=[2, 3, 4],
        epoch_prune=[1, 3, 5, 7],
        prune_batch_size=128,
        prune_type="0",
        reinit=False,
        min_ratio=[0.8, 0.7, 0.5, 0.2]
    )
    params, best_model = model_funcs.iter_training_0(model,
                                                     dataset,
                                                     train_params,
                                                     optimizer,
                                                     scheduler,
                                                     device)
    # arr = [(best_model, params)]
    # total_param = [sum(p.numel() for p in model.parameters(True)) for model in [m[0] for m in arr]]
    # param_z = [sum(len(list(filter(lambda x: x != 0., p.flatten()))) for p in model.parameters(True) if p.requires_grad) for model in
    #      [m[0] for m in arr]]
    # print("number of parameters: {}".format(total_param))
    # print("number of trainable parameters: {}".format(param_z))
    # print("ratio calc = {}".format([x/y for x,y in zip(param_z, total_param)]))
    # print("parameters: {}".format(list(model.parameters())[0]))
    blocs = get_blocs(best_model)
    parameters = []
    for b, bloc in enumerate(blocs):
        param_bloc = []
        print("bloc: {}".format(b))
        for id, layer in enumerate(bloc.modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                param_bloc.append(layer.weight.data)
                print("layer: {}".format(layer))
                print(layer.weight.data)
        parameters.append(param_bloc)

    # calculer la somme pour chaque parameters
    total_sum = sum([x.numel() for y in model.parameters() for x in y])
    print("total sum: {}".format(total_sum))
    zero_param = sum([len(list(filter(lambda x: x != 0., y.flatten()))) for a in model.parameters() for y in a])
    print("zero param: {}".format(zero_param))
    print("ratio: {}".format(zero_param/total_sum))

if __name__ == '__main__':
    main()
