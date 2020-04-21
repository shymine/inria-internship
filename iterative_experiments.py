import copy
import getopt
import sys
import matplotlib.pyplot as plt

import aux_funcs as af
import network_architectures as arcs


def train_model(models_path, cr_params, device):
    type, mode, pruning = cr_params
    model, params = arcs.create_resnet_iterative(models_path, type, mode, pruning, False)
    dataset = af.get_dataset('cifar10')
    params['name'] = params['base_model'] + '_{}_{}'.format(type, mode)
    if model.prune:
        params['name'] += "_prune_{}".format(model.keep_ratio * 100)
    if mode == "0":
        params['epochs'] = 300
    if type == "full":
        params['learning_rate'] = 0.1
    print("lr: {}".format(params['learning_rate']))

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


def main(mode, load):
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    create_params = [
        ('iterative', '0', (False, None)),
        ('iterative', '1', (False, None)),
        ('iterative', '2', (False, None)),
        ('iterative', '3', (False, None)),
        ('iterative', '4', (False, None)),
        # ('iterative', '0', (True, 0.5)),
        # ('iterative', '1', (True, 0.5)),
        # ('iterative', '2', (True, 0.5)),

        ('full', None, (False, None)),
        ('full_ic', None, (False, None))
    ]
    # create_params = [
    #     ('iterative', '0', (False, None)),
    #     ('iterative', '0', (True, 0.8)),
    #     ('iterative', '0', (True, 0.6)),
    #     ('iterative', '0', (True, 0.5)),
    #     ('iterative', '0', (True, 0.4)),
    #     ('iterative', '0', (True, 0.3)),
    #     ('iterative', '0', (True, 0.2)),
    #     ('iterative', '0', (True, 0.1)),
    #     ('iterative', '0', (True, 0.05)),
    #
    #     ('full', None, (False, None)),
    #     ('full_ic', None, (False, None))
    # ]
    create_bool = [
        1 if i == 0 or i == 6
        else 0 for i in range(len(create_params))
    ]
    if load is not None:
        model, param = arcs.load_model(models_path, load, -1)
        arr = [(model, param)]
    else:
        arr = list(multi_experiments(models_path, zip(create_params, create_bool), device))
    af.print_acc(arr)
    # print("parameters")
    # print("arr: {}".format([i for i in arr]))
    # for m in arr[:0]:
    #     print("m[0]: {}".format(m[0]))
    #     params = m[0].parameters(True)
    #     for p in params:
    #         print("{}\n".format(p))
    af.plot_acc([m[1] for m in arr])

if __name__ == '__main__':
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'm:l:')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    mode = 0
    load = None
    for opt, arg in optlist:
        if opt == "-m":
            mode = arg
        if opt == "-l":
            load = arg

    main(mode, load)
