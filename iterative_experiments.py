import aux_funcs as af
import network_architectures as arcs

def train_model(models_path, device):
    model = arcs.create_resnet_iterative(models_path, return_name=False)
    print("Training iteratively...")
    print('model: {}'.format(model))
    return model

def main():
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    model = train_model(models_path, device)

if __name__ == '__main__':
    main()