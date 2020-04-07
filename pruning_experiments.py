import aux_funcs as af
import network_architectures as arcs


def train(models_path, device):
    model, model_params = arcs.create_resnet_iterative(models_path, 'iterative', mode=3, return_name=False)
    full, full_params = arcs.create_resnet_iterative(models_path, 'full', mode=0, return_name=False)

    print("Training")
    dataset = af.get_dataset('cifar10', 128)
    lr = model_params['learning_rate'] / 10  # 0.01
    momentum = model_params['momentum']
    weight_decay = model_params['weight_decay']
    milestones = model_params['milestones']
    gammas = model_params['gammas']
    num_epochs = model_params['epochs']  # 100
    model_params['optimizer'] = 'SGD'

    model_name = model_params['base_model'] + "iter_snip"
    full_name = full_params['base_model'] + "_snip"

    opti_param = (lr, weight_decay, momentum, -1)
    lr_schedule_params = (milestones, gammas)

    model.to(device)
    full.to(device)

    model_name = model_name + 'prune_training'
    model_params['name'] = model_name
    full_params['name'] = full_name

    optimizer, scheduler = af.get_full_optimizer(model, opti_param, lr_schedule_params)
    full_opti, full_sch = af.get_full_optimizer(full, opti_param, lr_schedule_params)

    metrics = model.train_func(model, dataset, num_epochs, optimizer, scheduler, device)
    full_metrics = full.train_func(full, dataset, num_epochs, optimizer, scheduler, device)

    model_params['train_top1_acc'] = metrics['train_top1_acc']
    model_params['train_top3_acc'] = metrics['train_top3_acc']
    model_params['test_top1_acc'] = metrics['test_top1_acc']
    model_params['test_top3_acc'] = metrics['test_top3_acc']
    model_params['epoch_times'] = metrics['epoch_times']
    model_params['lrs'] = metrics['lrs']

    full_params['train_top1_acc'] = full_metrics['train_top1_acc']
    full_params['train_top3_acc'] = full_metrics['train_top3_acc']
    full_params['test_top1_acc'] = full_metrics['test_top1_acc']
    full_params['test_top3_acc'] = full_metrics['test_top3_acc']
    full_params['epoch_times'] = full_metrics['epoch_times']
    full_params['lrs'] = full_metrics['lrs']

    return (model, model_params), (full, full_params)


def main():
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    arr = train(models_path, device)
    af.print_acc(arr)


if __name__ == "__main__":
    main()
