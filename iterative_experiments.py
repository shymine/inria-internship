import copy
import getopt
import sys

import aux_funcs as af
import network_architectures as arcs



def train_model(models_path, cr_params, device, num=0):
    type, mode, pruning = cr_params
    model, params = arcs.create_resnet_iterative(models_path, type, mode, pruning, False)
    dataset = af.get_dataset('cifar10')
    params['name'] = params['base_model'] + '_{}_{}'.format(type, mode)
    if model.prune:
        params['name'] += "_prune_{}".format(model.keep_ratio * 100)
        print("prune: {}".format(model.keep_ratio))
    if mode == "0":
        params['epochs'] = 250
        params['milestones'] = [120, 160, 180]
        params['gammas'] = [0.1, 0.01, 0.01]

    if mode == "4":
        params['epochs'] = 300
        params['milestones'] = [100, 150, 200]
        params['gammas'] = [0.1, 0.1, 0.1]

    if "full" in type:
        params['learning_rate'] = 0.1
    print("lr: {}".format(params['learning_rate']))

    opti_param = (params['learning_rate'], params['weight_decay'], params['momentum'], -1)
    lr_schedule_params = (params['milestones'], params['gammas'])

    model.to(device)
    train_params = dict(
        epochs=params['epochs'],
        epoch_growth=[25, 50, 75],
        epoch_prune=[10, 35, 60, 85],  # [10, 35, 60, 85, 95, 105, 125, 130],
        prune_batch_size=pruning[2],
        prune_type='0',  # 0 skip layer, 1 normal full, 2 iterative
        reinit=False,
        min_ratio=[0.8, 0.7, 0.6, 0.5]  # not needed if skip layers, minimum for the iterative pruning
    )
    optimizer, scheduler = af.get_full_optimizer(model, opti_param, lr_schedule_params)
    metrics, best_model = model.train_func(model, dataset,
                                           train_params,
                                           optimizer, scheduler, device)
    _link_metrics(params, metrics)

    arcs.save_model(best_model, params, models_path, params['name'], epoch=-1)
    print("test acc: {}, last val: {}".format(params['test_top1_acc'], params['valid_top1_acc'][-1]))
    return best_model, params


def multi_experiments(models_path, params, device):
    count = 0
    last_mode = None
    for create, bool in params:
        if bool:
            if last_mode is None:
                last_mode = create[1]
            elif create[1] != last_mode:
                last_mode = create[1]
                count = 0
            yield train_model(models_path, create, device, num=count)
            count += 1

def _link_metrics(params, metrics):
    params['train_top1_acc'] = metrics['train_top1_acc']
    params['train_top3_acc'] = metrics['train_top3_acc']
    params['test_top1_acc'] = metrics['test_top1_acc']
    params['test_top3_acc'] = metrics['test_top3_acc']
    params['valid_top1_acc'] = metrics['valid_top1_acc']
    params['valid_top3_acc'] = metrics['valid_top3_acc']
    params['epoch_times'] = metrics['epoch_times']
    params['lrs'] = metrics['lrs']
    params['best_model_epoch'] = metrics['best_model_epoch']

def main(mode, load):
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    create_params = [
        # type, training, (prune?, keep_ratio for ics, batch size)
        ('dense', '0', (True, [0.8, 0.8, 0.8, 0.8], 128)),
        ('dense', '0', (True, [0.8, 0.8, 0.8, 0.8], 128)),
        ('dense', '0', (True, [0.8, 0.8, 0.8, 0.8], 128)),
        ('dense', '0', (True, [0.8, 0.8, 0.8, 0.8], 128)),
        ('dense', '0', (True, [0.8, 0.8, 0.8, 0.8], 128))
    ]
    create_bool = [
        1 if True
        else 0 for i in range(len(create_params))
    ]
    if load is not None:
        model, param = arcs.load_model(models_path, load, -1)
        arr = [(model, param)]
    else:
        arr = list(multi_experiments(models_path, zip(create_params, create_bool), device))
    #af.print_acc(arr, groups=[5], extend=True)
    af.print_acc(arr, extend=True)
    #af.print_acc(arr, extend=False)
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
