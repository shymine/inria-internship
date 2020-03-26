import aux_funcs as af
import network_architectures as arcs
import model_funcs as mf

def train_model(models_path, device):
    iter_model, iter_params = arcs.create_resnet_iterative(models_path, 'iterative', return_name=False)
    print("Training...")
    dataset = af.get_dataset(iter_params['task'], 128)
    lr = iter_params['learning_rate']/10 #0.01
    momentum = iter_params['momentum']
    weight_decay = iter_params['weight_decay']
    milestones = iter_params['milestones']
    gammas = iter_params['gammas']
    num_epochs = iter_params['epochs'] #100
    model_name = iter_params['base_model']
    iter_params['optimizer'] = 'SGD'


    opti_param = (lr, weight_decay, momentum, -1)
    lr_schedule_params = (milestones, gammas)

    iter_model.to(device)
    #optimizer, scheduler = af.get_full_optimizer(trained_model, opti_param, lr_schedule_params)
    trained_model_name = model_name + '_training'

    iter_model.train_func(iter_model, dataset, num_epochs, opti_param, lr_schedule_params, device)

    arcs.save_model(iter_model, iter_params, models_path, trained_model_name, epoch=-1)
    return iter_model

def main():
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    model = train_model(models_path, device)

if __name__ == '__main__':
    main()