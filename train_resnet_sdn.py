import aux_funcs as af
import network_architectures as arcs
import pandas as pd

import sys
import getopt

def train_model(models_path, device):
    _, sdn = arcs.create_resnet56(models_path, 'cifar10', save_type='d')
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

    opti_param = (lr, weight_decay, momentum, -1)
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
    return trained_model, dataset


def main(confusion, model_name):
    random_seed = af.get_random_seed()
    model = None
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()

    if model_name == "":
        # af.create_path(models_path)
        # af.set_logger('outputs/train_models')
    
        model, dataset = train_model(models_path, device)
    else :
        model, model_params = arcs.load_model(models_path, model_name,epoch=-1)

    if confusion:
        confusion, confusion_correct = af.calculate_confusion(model, 'cifar10', device)
        confusion_df = pd.DataFrame(confusion)
        correct_df = pd.DataFrame(confusion_correct)
        confusion_df.to_csv("confusion.csv")
        correct_df.to_csv("confusion_correct.csv")

if __name__ == '__main__':
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'hcm:')
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    confusion = False
    model_name = ""
    print("optlist: {}".format(optlist))
    print("args: {}".format(args))
    for opt, arg in optlist:
        if opt == "-c":
            confusion = True
        if opt == "-m":
            model_name = arg
    print(model_name)
    
    main(confusion, model_name)
