import getopt
import sys

import aux_funcs as af
import network_architectures as arcs


def test(models_path, device):
    model, params = arcs.create_resnet_iterative(models_path, 'iterative', mode='0', return_name=False)
    params['name'] = "test_name"
    params['test_top1_acc'] = [85.0]
    dataset = af.get_dataset('cifar10')
    opti_param = (params['learning_rate'] / 10, params['weight_decay'], params['momentum'], -1)
    scheduler_param = (params['milestones'], params['gammas'])

    model.to(device)

    # optim, scheduler = af.get_full_optimizer(model, opti_param, scheduler_param)

    """
    """

    print("modules: \n{}".format([x for x in model.modules()]))

    return (model, params),


def main(mode):
    def print_acc(arr):
        str = "accuracies:\n"
        for i in arr:
            str += "{}: {}, ".format(i[1]['name'], i[1]['test_top1_acc'][-1])
        return str

    random_seed = af.get_random_seed()
    models_path = 'networks/{}'.format(random_seed)
    device = af.get_pytorch_device()
    # iter, full_ic, full = train_model(models_path, device, mode)
    # print("accuracies:\niter: {}, full_ic: {}, full: {}".format(iter[1]['test_top1_acc'][-1], full_ic[1]['test_top1_acc'][-1], full[1]['test_top1_acc'][-1]))
    arr = test(models_path, device)
    p = print_acc(arr)
    print(p)


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
