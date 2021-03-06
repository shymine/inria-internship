# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and SDNs and also plotting

import copy
import itertools as it
import math
import os
import os.path
import pickle
import random
import sys
import time
import statistics
from functools import reduce

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 13})

from bisect import bisect_right
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR
from torch.nn import CrossEntropyLoss

import network_architectures as arcs
import snip
from profiler import profile

from data import CIFAR10, CIFAR100, TinyImagenet


# to log the output of the experiments to a file
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")
        print(self.log)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()


def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')
    # sys.stderr = Logger(log_file, 'err')


# the learning rate scheduler
class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr, 8)
            lrs.append(new_lr)
        print("base_lrs: {}".format(self.base_lrs))
        print("af scheduler: {}".format(lrs))
        return lrs

class SGDForPruning(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDForPruning, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDForPruning, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                d_p[p==0.] = 0.
                p.add_(d_p, alpha=-group['lr'])
        return loss

# flatten the output of conv layers for fully connected layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size / 4)
    else:
        return -1


# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        # red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size)  # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels * input_size * input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size / red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels * red_input_size * red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))


def get_random_seed():
    return 1221  # 121 and 1221


def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))


def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())


def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])


def overlay_two_histograms(save_path, save_name, hist_first_values, hist_second_values, first_label, second_label,
                           title):
    plt.hist([hist_first_values, hist_second_values], bins=25, label=[first_label, second_label])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()


def get_confusion_scores(outputs, normalize=None, device='cpu'):
    p = 1
    confusion_scores = torch.zeros(outputs[0].size(0))
    confusion_scores = confusion_scores.to(device)

    for output in outputs:
        cur_disagreement = nn.functional.pairwise_distance(outputs[-1], output, p=p)
        cur_disagreement = cur_disagreement.to(device)
        for instance_id in range(outputs[0].size(0)):
            confusion_scores[instance_id] += cur_disagreement[instance_id]

    if normalize is not None:
        for instance_id in range(outputs[0].size(0)):
            cur_confusion_score = confusion_scores[instance_id]
            cur_confusion_score = cur_confusion_score - normalize[0]  # subtract mean
            cur_confusion_score = cur_confusion_score / normalize[1]  # divide by the standard deviation
            confusion_scores[instance_id] = cur_confusion_score

    return confusion_scores


def get_dataset(dataset, batch_size=128, add_trigger=False):
    if dataset == 'cifar10':
        return load_cifar10(batch_size, add_trigger)
    elif dataset == 'cifar100':
        return load_cifar100(batch_size)
    elif dataset == 'tinyimagenet':
        return load_tinyimagenet(batch_size)


def load_cifar10(batch_size, add_trigger=False):
    cifar10_data = CIFAR10(batch_size=batch_size, add_trigger=add_trigger)
    return cifar10_data


def load_cifar100(batch_size):
    cifar100_data = CIFAR100(batch_size=batch_size)
    return cifar100_data


def load_tinyimagenet(batch_size):
    tiny_imagenet = TinyImagenet(batch_size=batch_size)
    return tiny_imagenet


def get_output_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth

    # output_depths.append(total_depth)

    return np.array(output_depths) / total_depth, total_depth


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def model_exists(models_path, model_name):
    return os.path.isdir(models_path + '/' + model_name)


def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]


def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']


def get_full_optimizer(model, lr_params, stepsize_params):
    lr = lr_params[0]
    weight_decay = lr_params[1]
    momentum = lr_params[2]
    epoch = lr_params[3]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    # optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = SGDForPruning(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas, last_epoch=epoch)

    return optimizer, scheduler


def get_sdn_ic_only_optimizer(model, lr_params, stepsize_params):
    freeze_except_outputs(model)

    lr = lr_params[0]
    weight_decay = lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    param_list = []
    for layer in model.layers:
        if layer.no_output == False:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})

    optimizer = Adam(param_list, lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device


def get_loss_criterion():
    return CrossEntropyLoss()


def get_all_trained_models_info(models_path, use_profiler=False, device='gpu'):
    print('Testing all models in: {}'.format(models_path))

    for model_name in sorted(os.listdir(models_path)):
        try:
            model_params = arcs.load_params(models_path, model_name, -1)
            train_time = model_params['total_time']
            num_epochs = model_params['epochs']
            architecture = model_params['architecture']
            print(model_name)
            task = model_params['task']
            print(task)
            net_type = model_params['network_type']
            print(net_type)

            top1_test = model_params['test_top1_acc']
            top1_train = model_params['train_top1_acc']
            top5_test = model_params['test_top5_acc']
            top5_train = model_params['train_top5_acc']

            print('Top1 Test accuracy: {}'.format(top1_test[-1]))
            print('Top5 Test accuracy: {}'.format(top5_test[-1]))
            print('\nTop1 Train accuracy: {}'.format(top1_train[-1]))
            print('Top5 Train accuracy: {}'.format(top5_train[-1]))

            print('Training time: {}, in {} epochs'.format(train_time, num_epochs))

            if use_profiler:
                model, _ = arcs.load_model(models_path, model_name, epoch=-1)
                model.to(device)
                input_size = model_params['input_size']

                if architecture == 'dsn':
                    total_ops, total_params = profile_dsn(model, input_size, device)
                    print("#Ops (GOps): {}".format(total_ops))
                    print("#Params (mil): {}".format(total_params))

                else:
                    total_ops, total_params = profile(model, input_size, device)
                    print("#Ops: %f GOps" % (total_ops / 1e9))
                    print("#Parameters: %f M" % (total_params / 1e6))

            print('------------------------')
        except:
            print('FAIL: {}'.format(model_name))
            continue


def sdn_prune(sdn_path, sdn_name, prune_after_output, epoch=-1, preloaded=None):
    print('Pruning an SDN...')

    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    output_layer = get_nth_occurance_index(sdn_model.add_output, prune_after_output)

    pruned_model = copy.deepcopy(sdn_model)
    pruned_params = copy.deepcopy(sdn_params)

    new_layers = nn.ModuleList()
    prune_add_output = []

    for layer_id, layer in enumerate(sdn_model.layers):
        if layer_id == output_layer:
            break
        new_layers.append(layer)
        prune_add_output.append(sdn_model.add_output[layer_id])

    last_conv_layer = sdn_model.layers[output_layer]
    end_layer = copy.deepcopy(last_conv_layer.output)

    last_conv_layer.output = nn.Sequential()
    last_conv_layer.forward = last_conv_layer.only_forward
    last_conv_layer.no_output = True
    new_layers.append(last_conv_layer)

    pruned_model.layers = new_layers
    pruned_model.end_layers = end_layer

    pruned_model.add_output = prune_add_output
    pruned_model.num_output = prune_after_output + 1

    pruned_params['pruned_after'] = prune_after_output
    pruned_params['pruned_from'] = sdn_name

    return pruned_model, pruned_params


# convert a cnn to a sdn by adding output layers to internal layers
def cnn_to_sdn(cnn_path, cnn_name, sdn_params, epoch=-1, preloaded=None):
    print('Converting a CNN to a SDN...')
    if preloaded is None:
        cnn_model, _ = arcs.load_model(cnn_path, cnn_name, epoch=epoch)
    else:
        cnn_model = preloaded

    sdn_params['architecture'] = 'sdn'
    sdn_params['converted_from'] = cnn_name
    sdn_model = arcs.get_sdn(cnn_model)(sdn_params)

    sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, cnn_layer in enumerate(cnn_model.layers):
        sdn_layer = sdn_model.layers[layer_id]
        sdn_layer.layers = cnn_layer.layers
        layers.append(sdn_layer)

    sdn_model.layers = layers

    sdn_model.end_layers = cnn_model.end_layers

    return sdn_model, sdn_params


def sdn_to_cnn(sdn_path, sdn_name, epoch=-1, preloaded=None):
    print('Converting a SDN to a CNN...')
    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    cnn_params = copy.deepcopy(sdn_params)
    cnn_params['architecture'] = 'cnn'
    cnn_params['converted_from'] = sdn_name
    cnn_model = arcs.get_cnn(sdn_model)(cnn_params)

    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)

    cnn_model.layers = layers

    cnn_model.end_layers = sdn_model.end_layers

    return cnn_model, cnn_params


def freeze_except_outputs(model):
    model.frozen = True
    for param in model.init_conv.parameters():
        param.requires_grad = False

    for layer in model.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.end_layers.parameters():
        param.requires_grad = False


def save_tinyimagenet_classname():
    filename = 'tinyimagenet_classes'
    dataset = get_dataset('tinyimagenet')
    tinyimagenet_classes = {}

    for index, name in enumerate(dataset.testset_paths.classes):
        tinyimagenet_classes[index] = name

    with open(filename, 'wb') as f:
        pickle.dump(tinyimagenet_classes, f, pickle.HIGHEST_PROTOCOL)


def get_tinyimagenet_classes(prediction=None):
    filename = 'tinyimagenet_classes'
    with open(filename, 'rb') as f:
        tinyimagenet_classes = pickle.load(f)

    if prediction is not None:
        return tinyimagenet_classes[prediction]

    return tinyimagenet_classes


def calculate_confusion(model, dataset, device='cpu'):
    print("calculating confusion")
    loader = get_dataset(dataset).train_loader  # test_loader
    confusion = []
    confusion_correct = []
    correct_found = []
    print("batch size {}".format(loader.batch_size))

    for l in range(model.num_output):
        confusion.append([0 for _ in range(model.num_output)])
        confusion_correct.append([0 for _ in range(model.num_output)])
        correct_found.append(0)
    # confusion[x][y] example predicted in x that are correct and are the same in y
    model.eval()
    model.cuda()
    with torch.no_grad():
        example_num = 0
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            outputs = model(b_x)
            pred = format_outputs(outputs)
            for example in range(b_y.size(0)):
                for i in range(model.num_output):
                    pred_i = pred[example][i]
                    count = False
                    if pred_i == b_y[i]:
                        count = True
                        correct_found[i] += 1
                    for j in range(i, model.num_output):
                        pred_j = pred[example][j]
                        if pred_i == pred_j:
                            confusion[i][j] += 1
                            if count:
                                confusion_correct[i][j] += 1
            example_num += b_y.size(0)
        confusion = (np.array(confusion) / example_num).tolist()
        for i, arr in enumerate(confusion_correct):
            confusion_correct[i] = [elm / correct_found[i] for elm in arr]

    print("confusion: {}".format(confusion))
    print("confusion_correct: {}".format(confusion_correct))
    return confusion, confusion_correct


def format_outputs(outputs):
    res = []
    for example in range(len(outputs[0])):
        out = []
        for output in range(len(outputs)):
            out.append(outputs[output][example].argmax())
        res.append(out)
    return res


def print_acc(arr, groups=None, extend=False):
    str = "accuracies:\n"
    for i in arr:
        str += "{}: {}, {},\n".format(i[1]['name'], i[1]['test_top1_acc'], i[1]['best_model_epoch'])
    print(str)
    def mean_(arr):
        return math.fsum(arr)/len(arr)
    if extend:
        acc = [i[1]['test_top1_acc'] for i in arr]
        if groups is None:
            tr = reverse(acc)
            means = [statistics.mean([float(a) for a in i]) for i in tr]
            stds = [statistics.stdev([float(a) for a in i]) for i in tr]
            print("means: {}".format(means))
            print("stds: {}".format(stds))
            print("std%: {}".format([100*std/mean for std, mean in zip(stds, means)]))
        else: # groups=[2,2,2]
            groups_cum = [0] + [sum(groups[:i+1]) for i in range(len(groups))]
            for i in range(len(groups)):
                group_acc = acc[groups_cum[i]:groups_cum[i+1]]
                tr = reverse(group_acc)
                means = [statistics.mean([float(j) for j in i]) for i in tr]
                stds = [statistics.stdev([float(j) for j in i]) for i in tr]
                print("{} means: {}".format(i, means))
                print("{} stds: {}".format(i, stds))
                print("{} std /100: {}".format(i, [100 * std / mean for std, mean in zip(stds, means)]))

def reverse(test_acc):
    max_len = len(test_acc[-1])
    tmp = []
    for i in test_acc:
        if len(i) < max_len:
            a = [i[ind] if ind < len(i) else 0 for ind in range(max_len)]
        else:
            a = copy.deepcopy(i)
        tmp.append(a)
    res = [[] for _ in tmp[0]]
    for i in tmp:
        for id, j in enumerate(i):
            res[id].append(j)
    return res

def plot_acc(arr):
    figs = []
    for i, m in enumerate(arr):
        acc = m['valid_top1_acc']
        tr = reverse(acc)
        fig, ax = plt.subplots()
        ax.text(m['epochs']-80, 20, "best model epoch: \n{}".format(m['best_model_epoch']))
        ax.set_xlabel('epochs')
        ax.set_ylabel('accuracy')
        name = "_".join(m['name'].split('_')[3:])
        ax.set_title(name + '\naccuracy: {}'.format(m['test_top1_acc']))

        ax.plot([i for i in range(len(acc))],
                tr[0],
                label="IC 1")
        ax.plot([i for i in range(len(acc))],
                tr[1],
                label="IC 2")
        ax.plot([i for i in range(len(acc))],
                tr[2],
                label="IC 3")
        ax.plot([i for i in range(len(acc))],
                tr[3],
                label="final output")
        
        for epoch_prune in m['epoch_prune']:
            ax.axvline(x=epoch_prune)

        figs.append(fig)
    name = time.asctime(time.localtime(time.time())).replace(" ", "_")
    if not os.path.exists("results/{}".format(name)):
        os.makedirs("results/{}".format(name))
    for i, fig in enumerate(figs):
        fig.savefig("results/{}/{}".format(name, i))


def print_sparsity(model, mask=False):
    blocks = snip.get_blocs(model)
    #grown_index = -1
    print("blocks:")
    for i, b in enumerate(blocks):
        if len(b)==0:
            break
        #grown_index += 1
        if mask:
            p_z = sum([torch.sum(layer.weight_mask != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())]) 
            t_p = sum([layer.weight_mask.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())])
        else:
            p_z = sum([torch.sum(layer.weight != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())]) 
            t_p = sum([layer.weight.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), b.modules())])
        print("    {} total: {}, non zero: {}, ratio: {:.2f}".format(i, t_p, p_z, float(p_z)/float(t_p)))
    
    if mask:
        total_param = sum([layer.weight_mask.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), model.modules())])
        param_z = sum([torch.sum(layer.weight_mask != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), model.modules())]) 
    else:
        total_param = sum([layer.weight.nelement() for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), model.modules())])
        param_z = sum([torch.sum(layer.weight != 0) for layer in filter(lambda l: isinstance(l, (nn.Linear, nn.Conv2d)), model.modules())]) 

    final_layers_param = sum(layer.weight.nelement() for layer in filter(lambda l: isinstance(l, (nn.Conv2d, nn.Linear)), model.end_layers.modules()))
    total_param = total_param-final_layers_param if len(blocks[-1]) == 0 else total_param
    param_z = param_z-final_layers_param if len(blocks[-1]) == 0 else param_z
    print("total: {}, non_zero: {}, ratio: {}".format(total_param, param_z, float(param_z)/float(total_param)))

def connection_importance(model):
    print("truc")
    distances = [[] for _ in range(len(model.layers)-1)]
    for cell in model.layers:
        nb_input = cell.layers[0]
    # TODO: connection importance

def calculate_flops(model, input_shape):
    def flop_linear(layer):
        if layer.bias is not None:
            flops = 2*layer.in_features*layer.out_features
        else:
            flops = layer.out_features*(2*layer.in_features - 1)
        if hasattr(layer, 'weight_mask'):
            flops -= 2*sum(layer.weight_mask.flatten()==0.).item() # if one value is zero then there is 1mul and 1add that are removed
            flops += sum([1 if sum(line)==0. else 0 for line in layer.weight_mask]) # if one line is empty, then we have to count back the addition that has been removed that is too much
        return flops
    def flop_conv2d(layer, **kwargs):
        in_shape = kwargs['in_shape']
        if isinstance(layer.kernel_size, int):
            n = in_shape[0] * layer.kernel_size * layer.kernel_size
        else: 
            n = layer.in_channels*layer.kernel_size[0]*layer.kernel_size[1]
        flops_per_filter = []
        empty_filter = 0
        if isinstance(layer.kernel_size, int):
            num_instances_per_filter = math.floor(size_func(in_shape[1], layer.padding, layer.dilation if hasattr(layer, 'dilation') else 1, layer.kernel_size, layer.stride))
            num_instances_per_filter *= math.floor(size_func(in_shape[2], layer.padding, layer.dilation if hasattr(layer, 'dilation') else 1, layer.kernel_size, layer.stride))
        else:
            num_instances_per_filter = math.floor(size_func(in_shape[1], layer.padding[0], layer.dilation[0], layer.kernel_size[0], layer.stride[0]))
            num_instances_per_filter *= math.floor(size_func(in_shape[2], layer.padding[1], layer.dilation[1], layer.kernel_size[1], layer.stride[1]))

        if hasattr(layer, 'weight_mask'):
            for filt in layer.weight_mask:
                f_n = n - sum(filt.flatten()==0.).item()
                if f_n < 0:
                    print("f_n: {}, n: {}, mask: {}, layer: {}, in_shape: {}".format(f_n, n, filt.shape, layer, in_shape))
                flops_per_instance = f_n + 1
                if sum(filt.flatten()) == 0.:
                    empty_filter += 1
                flops_per_filter.append(num_instances_per_filter * flops_per_instance)
        else:
            flops_per_instance = n + 1
            ite = layer.out_channels if isinstance(layer, nn.Conv2d) else in_shape[0]
            flops_per_filter.extend([num_instances_per_filter * flops_per_instance for _ in range(ite)])
        flops = sum(flops_per_filter)
        if hasattr(layer, 'bias') and layer.bias is not None:
            flops += layer.out_channels

        flops -= empty_filter
        return flops
    def flop_relu(input_shape):
        return reduce(lambda x,y: x*y, input_shape)
    
    def size_func(in_shape, padding, dilation, kernel_size, stride):
        return ((in_shape + 2*padding - dilation*(kernel_size-1)-1)/stride)+1
    
    in_shape = input_shape
    flops = 0
    ic_buffer = []
    ic = None
    for layer in model.modules():
        if isinstance(layer, InternalClassifier):
            ic_buffer.append([layer, in_shape])
            ic = 0
            continue
        if ic is not None:
            ic += 1
            if ic == 3:
                ic = None
            continue
        linear = isinstance(layer, nn.Linear)
        conv2d = isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d))
        relu = isinstance(layer, nn.ReLU)
        if linear or conv2d or relu:
            flops += flop_linear(layer) if linear else flop_conv2d(layer, in_shape=in_shape) if conv2d else flop_relu(in_shape)
            if conv2d:
                if isinstance(layer, nn.Conv2d) and layer.padding[0] != 0.:
                    in_shape = (
                        layer.out_channels,
                        math.floor(size_func(in_shape[1], layer.padding[0], layer.dilation[0], layer.kernel_size[0], layer.stride[0])),
                        math.floor(size_func(in_shape[2], layer.padding[1], layer.dilation[1], layer.kernel_size[1], layer.stride[1]))
                    )
                elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
                        in_shape = (
                            in_shape[0],
                            math.floor(size_func(in_shape[1], layer.padding, layer.dilation if hasattr(layer, 'dilation') else 1, layer.kernel_size, layer.stride)),
                            math.floor(size_func(in_shape[2], layer.padding, layer.dilation if hasattr(layer, 'dilation') else 1, layer.kernel_size, layer.stride))
                        )
                    
    for ic, in_shape in ic_buffer:
        for layer in ic.modules():
            linear = isinstance(layer, nn.Linear)
            conv2d = isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d))
            relu = isinstance(layer, nn.ReLU)
            if linear or conv2d or relu:
                flops += flop_linear(layer) if linear else flop_conv2d(layer, in_shape=in_shape) if conv2d else flop_relu(in_shape)
                if conv2d:
                    if isinstance(layer, nn.Conv2d):
                        in_shape = (
                            layer.out_channels,
                            math.floor(size_func(in_shape[1], layer.padding[0], layer.dilation[0], layer.kernel_size[0], layer.stride[0])),
                            math.floor(size_func(in_shape[2], layer.padding[1], layer.dilation[1], layer.kernel_size[1], layer.stride[1]))
                        )
                    elif isinstance(layer, nn.AvgPool2d): # Avg as it is the last pooling method used and that it should take the same input as MaxPool
                        in_shape = (
                            in_shape[0],
                            math.floor(size_func(in_shape[1], layer.padding, layer.dilation if hasattr(layer, 'dilation') else 1, layer.kernel_size, layer.stride)),
                            math.floor(size_func(in_shape[2], layer.padding, layer.dilation if hasattr(layer, 'dilation') else 1, layer.kernel_size, layer.stride))
                        )
    return flops

