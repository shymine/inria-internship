import aux_funcs as af
import network_architectures as arcs

def train_model(models_path, params, device):
    res56_model, res56_params = arcs.create_resnet56(models_path, 'cifar10', 'd', return_model=True)
    dense_model, dense_params = arcs.create_dense_iterative(models_path, params)


if __name__=="__main__":
    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    create_params = [
        # keep_ratio, min_ratio, pruning mode
        ([0.46, 0.46, 0.46, 0.46], [0.1, 0.1, 0.1, 0.1], "2")
    ]
    arr = [train_model(model_path, param, device) for param in create_params]