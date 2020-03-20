import aux_funcs as af
import network_architectures as arcs
import model_funcs as mf

def train_model(models_path, device):
    _, model_name = arcs.create_resnet_iterative(models_path, return_name=True)
    print("Training iteratively...")
    trained_model, model_params = arcs.load_model(models_path, model_name, 0)
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
    print('model: {}'.format(model_name))
    optimizer, scheduler = af.get_full_optimizer(trained_model, opti_param, lr_schedule_params)
    trained_model_name = model_name + '_training'

    trained_model.to(device)
    model_name.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device)

    arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)
    return trained_model

def main():
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    model = train_model(models_path, device)

if __name__ == '__main__':
    main()