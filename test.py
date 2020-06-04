import sys

import aux_funcs as af
import network_architectures as arcs
import snip
import data
import model_funcs
import torch
import numpy as np


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
    model, param = arcs.create_resnet_iterative("results/", "iterative", "0", (True, 0.8, 128), False)
    # loader = model_funcs.get_loader(data.CIFAR10(), False)

    # masks1 = snip.snip_bloc_iterative(model, 0.8, 0.5, [0, 0, 0, 0], loader, model_funcs.sdn_loss, reinit=False)
    # flat = torch.tensor([y for x in masks1[0] for y in torch.flatten(x)])
    # # print("flat: {}".format(flat))
    # kept = torch.sum(flat)/len(flat)
    # print("kept: {}".format(kept))
    # snip.apply_prune_mask_bloc_iterative(model, masks1)
    # masks2 = snip.snip_bloc_iterative(model, 0.8, 0.5, [1, 0, 0, 0], loader, model_funcs.sdn_loss, reinit=False)
    # flat2 = torch.tensor([y for x in masks2[0] for y in torch.flatten(x)])
    # kept = torch.sum(flat2) / len(flat2)
    # print("kept2: {}".format(kept))
    # print("len1: {}, len2: {}".format(len(flat), len(flat2)))
    # a = sum([1 if a == 0 and b != 0 else 0 for a, b in zip(flat, flat2)])
    # print("num of elem in 1 that are not in 2: {}".format(a))
    # b = sum([1 if a != 0 and b == 0 else 0 for a, b in zip(flat, flat2)])
    # print("num of elem in 2 that are not in 1: {}".format(b))

    dataset = data.CIFAR10()
    optimizer, scheduler = af.get_full_optimizer(model,
                                                 (0.01, 0.0001, 0.9, -1),
                                                 ([4], [0.1]))
    train_params = dict(
        epochs=5,
        epoch_growth=[2, 3, 4],
        epoch_prune=[1, 3, 5, 7],
        prune_batch_size=128,
        prune_type="2",
        reinit=False,
        min_ratio=0.5
    )
    model_funcs.iter_training_0(model,
                                dataset,
                                train_params,
                                optimizer,
                                scheduler,
                                device)


if __name__ == '__main__':
    main()
