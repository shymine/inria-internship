import aux_funcs as af
import network_architectures as arcs

def train_model(models_path, device):
    sdn = arcs.create_resnet56(models_path, 'cifar10', save_type='d')
    print('snd name: {}'.format(sdn))
    # train_sdn(models_path, sdn, device)
    print("Training model...")
    trained_model, model_params = arcs.load_model(models_path, sdn, 0)
    dataset = af.get_dataset(model_params['task'])
    lr = model_params['learning_rate']
    momentum = model_params['momentum']
    weight_decay = model_params['weight_decay']
    milestones = model_params['milestones']
    gammas = model_params['gammas']
    num_epochs = model_params['epochs']

    model_params['optimizer'] = 'SGD'

    opti_param = (lr, weight_decay, momentum)
    lr_schedule_params = (milestones, gammas)

    optimizer, scheduler = af.get_full_optimizer(trained_model, opti_param, lr_schedule_params)
    trained_model_name = sdn+'_training'

    print('Training: {}...'.format(trained_model_name))
    trained_model.to(device)
    metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)
    model_params['train_top1_acc'] = metrics['train_top1_acc']
    model_params['test_top1_acc'] = metrics['test_top1_acc']
    model_params['train_top3_acc'] = metrics['train_top3_acc']
    model_params['test_top3_acc'] = metrics['test_top3_acc']
    model_params['epoch_times'] = metrics['epoch_times']
    model_params['lrs'] = metrics['lrs']
    total_training_time = sum(model_params['epoch_times'])
    model_params['total_time'] = total_training_time
    print('Training took {} seconds...'.format(total_training_time))
    arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)


def main():
    random_seed = af.get_random_seed()
    print('Random seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}'.format(random_seed)
    af.create_path(models_path)
    af.set_logger('outputs/train_models')

    train_model(models_path, device)

if __name__ == '__main__':
    main()