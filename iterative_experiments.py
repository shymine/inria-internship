import sys
import getopt

import aux_funcs as af
import network_architectures as arcs
import model_funcs as mf

"""
Tests to do:
- train every layer for 100 epochs
- train for 100 epochs after a new layer is added and freeze the previous layers
- train for 50 epochs between growth and freeze the previous layers, 100 epochs of full training
- grow every 50, freeze the 25 first and learn the 25 others for beneath layers
"""

def train_model(models_path, device, mode):
    iter_model, iter_params = arcs.create_resnet_iterative(models_path, 'iterative', mode=mode, return_name=False)
    full_ic_model, full_ic_params = arcs.create_resnet_iterative(models_path, 'full_ic', return_name=False)
    full_model, full_params = arcs.create_resnet_iterative(models_path, 'full', return_name=False)

    print("Training...")
    dataset = af.get_dataset('cifar10', 128)
    lr = iter_params['learning_rate']/10 #0.01
    momentum = iter_params['momentum']
    weight_decay = iter_params['weight_decay']
    milestones = iter_params['milestones']
    gammas = iter_params['gammas']
    num_epochs = iter_params['epochs'] #100
    iter_params['optimizer'] = 'SGD'

    iter_name = iter_params['base_model']
    full_ic_name = full_ic_params['base_model']
    full_name = full_params['base_model']

    opti_param = (lr, weight_decay, momentum, -1)
    lr_schedule_params = (milestones, gammas)

    iter_model.to(device)
    full_ic_model.to(device)
    full_model.to(device)

    iter_name = iter_name + '_training'
    full_ic_name = full_ic_name + '_training'
    full_name = full_name + '_training'

    iter_optimizer, iter_scheduler = af.get_full_optimizer(iter_model, opti_param, lr_schedule_params)
    full_ic_optimizer, full_ic_scheduler = af.get_full_optimizer(full_ic_model, opti_param, lr_schedule_params)
    full_optimizer, full_scheduler = af.get_full_optimizer(full_model, opti_param, lr_schedule_params)

    iter_metrics = iter_model.train_func(iter_model, dataset, num_epochs, iter_optimizer, iter_scheduler, device)
    full_ic_metrics = full_ic_model.train_func(full_ic_model, dataset, num_epochs, full_ic_optimizer, full_ic_scheduler, device)
    full_metrics = full_model.train_func(full_model, dataset, num_epochs, full_optimizer, full_scheduler, device, True)

    _link_metrics(iter_params, iter_metrics)
    _link_metrics(full_ic_params, full_ic_metrics)
    _link_metrics(full_params, full_metrics)

    arcs.save_model(iter_model, iter_params, models_path, iter_name, epoch=-1)
    arcs.save_model(full_ic_model, full_ic_params, models_path, full_ic_name, epoch=-1)
    arcs.save_model(full_model, full_params, models_path, full_name, epoch=-1)

    return (iter_model, iter_params), (full_ic_model, full_ic_params), (full_model, full_params)

def multi_experiments(models_path, device):
    train0_model, train0_params = arcs.create_resnet_iterative(models_path, 'iterative', mode='0', return_name=False)
    train1_model, train1_params = arcs.create_resnet_iterative(models_path, 'iterative', mode='1', return_name=False)
    train2_model, train2_params = arcs.create_resnet_iterative(models_path, 'iterative', mode='2', return_name=False)
    full_ic_model, full_ic_params = arcs.create_resnet_iterative(models_path, 'full_ic', return_name=False)
    full_model, full_params = arcs.create_resnet_iterative(models_path, 'full', return_name=False)

    print("Training...")
    dataset = af.get_dataset('cifar10', 128)
    lr = train0_params['learning_rate'] / 10  # 0.01
    momentum = train0_params['momentum']
    weight_decay = train0_params['weight_decay']
    milestones = train0_params['milestones']
    gammas = train0_params['gammas']
    num_epochs = train0_params['epochs']  # 100
    train0_params['optimizer'] = 'SGD'

    train0_name = train0_params['base_model'] + 'train0_training'
    train1_name = train1_params['base_model'] + 'train1_training'
    train2_name = train2_params['base_model'] + 'train2_training'
    full_ic_name = full_ic_params['base_model'] + '_training'
    full_name = full_params['base_model'] + 'training'

    train0_params['name'] = train0_name
    train1_params['name'] = train1_name
    train2_params['name'] = train2_name
    full_ic_params['name'] = full_ic_name
    full_params['name'] = full_name

    train0_model.to(device)
    train1_model.to(device)
    train2_model.to(device)
    full_ic_model.to(device)
    full_model.to(device)

    opti_param = (lr, weight_decay, momentum, -1)
    lr_schedule_params = (milestones, gammas)

    train0_optimizer, train0_scheduler = af.get_full_optimizer(train0_model, opti_param, lr_schedule_params)
    train1_optimizer, train1_scheduler = af.get_full_optimizer(train1_model, opti_param, lr_schedule_params)
    train2_optimizer, train2_scheduler = af.get_full_optimizer(train2_model, opti_param, lr_schedule_params)
    full_ic_optimizer, full_ic_scheduler = af.get_full_optimizer(full_ic_model, opti_param, lr_schedule_params)
    full_optimizer, full_scheduler = af.get_full_optimizer(full_model, opti_param, lr_schedule_params)

    train0_metrics = train0_model.train_func(train0_model, dataset, num_epochs, train0_optimizer, train0_scheduler, device)
    train1_metrics = train1_model.train_func(train1_model, dataset, num_epochs, train1_optimizer, train1_scheduler, device)
    train2_metrics = train2_model.train_func(train2_model, dataset, num_epochs, train2_optimizer, train2_scheduler, device)
    full_ic_metrics = full_ic_model.train_func(full_ic_model, dataset, num_epochs, full_ic_optimizer, full_ic_scheduler, device)
    full_metrics = full_model.train_func(full_model, dataset, num_epochs, full_optimizer, full_scheduler, device, True)

    _link_metrics(train0_params, train0_metrics)
    _link_metrics(train1_params, train1_metrics)
    _link_metrics(train2_params, train2_metrics)
    _link_metrics(full_ic_params, full_ic_metrics)
    _link_metrics(full_params, full_metrics)

    arcs.save_model(train0_model, train0_params, models_path, train0_name, epoch=-1)
    arcs.save_model(train1_model, train1_params, models_path, train1_name, epoch=-1)
    arcs.save_model(train2_model, train2_params, models_path, train2_name, epoch=-1)

    arcs.save_model(full_ic_model, full_ic_params, models_path, full_ic_name, epoch=-1)
    arcs.save_model(full_model, full_params, models_path, full_name, epoch=-1)

    return [(train0_model, train0_params), (train1_model, train1_params), (train2_model, train2_params), (full_ic_model, full_ic_params), (full_model, full_params)]


def _link_metrics(params, metrics):
    params['train_top1_acc'] = metrics['train_top1_acc']
    params['train_top3_acc'] = metrics['train_top3_acc']
    params['test_top1_acc'] = metrics['test_top1_acc']
    params['test_top3_acc'] = metrics['test_top3_acc']
    params['epoch_times'] = metrics['epoch_times']
    params['lrs'] = metrics['lrs']

def main(mode):
    def print_acc(arr):
        str = "accuracies:\n"
        for i in arr:
            str += "{}: {}, ".format(i[1]['name'], i[1]['test_top1_acc'][-1])
        print(str)

    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    #iter, full_ic, full = train_model(models_path, device, mode)
    #print("accuracies:\niter: {}, full_ic: {}, full: {}".format(iter[1]['test_top1_acc'][-1], full_ic[1]['test_top1_acc'][-1], full[1]['test_top1_acc'][-1]))
    arr = multi_experiments(models_path, device)
    print_acc(arr)



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
