import getopt
import sys

import aux_funcs as af
import network_architectures as arcs

"""
Tests to do:
- train every layer for 100 epochs
- train for 100 epochs after a new layer is added and freeze the previous layers
- train for 50 epochs between growth and freeze the previous layers, 100 epochs of full training
- grow every 50, freeze the 25 first and learn the 25 others for beneath layers
"""


def train_model(models_path, params, device):
    type, mode, pruning = params
    model, params = arcs.create_resnet_iterative(models_path, type, mode, pruning, False)
    dataset = af.get_dataset('cifar10')
    params['name'] = params['base_model'] + '_{}_{}'.format(type, mode)
    opti_param = (params['learning_rate'], params['weight_decay'], params['momentum'], -1)
    lr_schedule_params = (params['milestones'], params['gammas'])

    model.to(device)

    optimizer, scheduler = af.get_full_optimizer(model, opti_param, lr_schedule_params)
    metrics = model.train_func(model, dataset, params['epochs'], optimizer, scheduler, device)
    _link_metrics(params, metrics)

    arcs.save_model(model, params, models_path, params['name'], epoch=-1)
    return (model, params)


def multi_experiments(models_path, params, device):
    for create, bool in params:
        if bool:
            yield train_model(models_path, create, device)


def _link_metrics(params, metrics):
    params['train_top1_acc'] = metrics['train_top1_acc']
    params['train_top3_acc'] = metrics['train_top3_acc']
    params['test_top1_acc'] = metrics['test_top1_acc']
    params['test_top3_acc'] = metrics['test_top3_acc']
    params['epoch_times'] = metrics['epoch_times']
    params['lrs'] = metrics['lrs']


def main(mode):
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    # iter, full_ic, full = train_model(models_path, device, mode)
    # print("accuracies:\niter: {}, full_ic: {}, full: {}".format(iter[1]['test_top1_acc'][-1], full_ic[1]['test_top1_acc'][-1], full[1]['test_top1_acc'][-1]))
    create_params = [
        ('iterative', '0', (False, None)),
        ('iterative', '1', (False, None)),
        ('iterative', '2', (False, None)),
        ('full', None, (False, None)),
        ('full_ic', None, (False, None))
    ]
    create_bool = [
        1,
        0,
        0,
        0,
        0
    ]

    arr = multi_experiments(models_path, device)
    af.print_acc(arr)


if __name__ == '__main__':
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'm:')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    mode = 0
    for opt, arg in optlist:
        if opt == "-m":
            mode = arg

    main(mode)
