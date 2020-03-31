import aux_funcs as af
import network_architectures as arcs
import snip

def train(models_path, device):
    model, model_params = arcs.create_resnet_iterative(models_path, 'full', return_name=False)
    print("Training")
    dataset = af.get_dataset('cifar10', 128)
    lr = model_params['learning_rate'] / 10  # 0.01
    momentum = model_params['momentum']
    weight_decay = model_params['weight_decay']
    milestones = model_params['milestones']
    gammas = model_params['gammas']
    num_epochs = model_params['epochs']  # 100
    model_params['optimizer'] = 'SGD'

    model_name = model_params['base_model'] + "_snip"

    opti_param = (lr, weight_decay, momentum, -1)
    lr_schedule_params = (milestones, gammas)

    model.to(device)

    model_name = model_name + '_training'

    mask = snip.snip(model, 0.1, dataset, device)
    snip.apply_prune_mask(model, mask)

    optimizer, scheduler = af.get_full_optimizer(model, opti_param, lr_schedule_params)

    metrics = model.train_func(model, dataset, num_epochs, optimizer, scheduler, device)

    model_params['train_top1_acc'] = metrics['train_top1_acc']
    model_params['train_top3_acc'] = metrics['train_top3_acc']
    model_params['test_top1_acc'] = metrics['test_top1_acc']
    model_params['test_top3_acc'] = metrics['test_top3_acc']
    model_params['epoch_times'] = metrics['epoch_times']
    model_params['lrs'] = metrics['lrs']

    return model, model_params

def main():
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    train(models_path, device)