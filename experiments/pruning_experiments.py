import aux_funcs as af
import network_architectures as arcs

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
    

def main():
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    train(models_path, device)