# model_funcs.py
# implements the functions for training, testing SDNs and CNNs
# also implements the functions for computing confusion and confidence
import copy
import time

import numpy as np
import torch
import torch.nn as nn

import aux_funcs as af
import data
import snip
#from GPUtil import showUtilization as gpu_usage

def sdn_training_step(optimizer, model, coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  # clear gradients for this training step
    # total_loss = 0.0
    #
    # for ic_id in range(model.num_output - 1):
    #     cur_output = output[ic_id]
    #     cur_loss = float(coeffs[ic_id])*af.get_loss_criterion()(cur_output, b_y)
    #     total_loss += cur_loss
    #
    # total_loss += af.get_loss_criterion()(output[-1], b_y)
    total_loss = sdn_loss(output, b_y, coeffs)
    total_loss.backward()
    optimizer.step()  # apply gradients

    return total_loss


def sdn_ic_only_step(optimizer, model, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  # clear gradients for this training step
    total_loss = 0.0

    for output_id, cur_output in enumerate(output):
        if output_id == model.num_output - 1:  # last output
            break

        cur_loss = af.get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss.backward()
    optimizer.step()  # apply gradients

    return total_loss


def get_loader(data, augment):
    if augment:
        train_loader = data.aug_train_loader
    else:
        train_loader = data.train_loader

    return train_loader


def sdn_train(model, data, params, optimizer, scheduler, device='cpu'):
    augment = model.augment_training
    print("sdn training")
    metrics = {'epoch_times': [],
               'valid_top1_acc': [],
               'valid_top3_acc': [],
               'test_top1_acc': [],
               'test_top3_acc': [],
               'train_top1_acc': [],
               'train_top3_acc': [],
               'lrs': []}
    max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])  # max tau_i --- C_i values
    epochs = params['epochs']
    if model.ic_only:
        print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    else:
        print('sdn will be trained from scratch...(The SDN training)')

    if model.prune:
        loader = get_loader(data, False)
        prune2(model, model.keep_ratio, loader, sdn_loss, device)
    best_model, accuracies, best_epoch = None, None, 0
    for epoch in range(1, epochs + 1):
        epoch_routine(model, data, optimizer, scheduler, epoch, epochs, augment, metrics, device)

        print("best model evaluation: {}/{}".format(metrics['valid_top1_acc'][-1], accuracies))
        if best_model is None:
            best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
            best_epoch = epoch
            print("Begin best_model: {}".format(accuracies))
        elif sum(metrics['valid_top1_acc'][-1]) > sum(accuracies):
            best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
            best_epoch = epoch
            print("New best model: {}".format(accuracies))
    metrics['test_top1_acc'], metrics['test_top3_acc'] = sdn_test(best_model, data.test_loader, device)
    test_top1, test_top3 = sdn_test(model, data.test_loader, device)
    metrics['best_model_epoch'] = best_epoch
    print("best epoch: {}".format(best_epoch))
    print("comparison best and latest: {}/{}".format(metrics['test_top1_acc'], test_top1))
    return metrics, best_model


def sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top3 = []
    for output_id in range(model.num_output):
        t1 = data.AverageMeter()
        t3 = data.AverageMeter()
        top1.append(t1)
        top3.append(t3)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec3 = data.accuracy(cur_output, b_y, topk=(1, 3))
                top1[output_id].update(prec1[0], b_x.size(0))
                top3[output_id].update(prec3[0], b_x.size(0))

    top1_accs = []
    top3_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(top1[output_id].avg.data.cpu().numpy()[()])
        top3_accs.append(top3[output_id].avg.data.cpu().numpy()[()])

    return top1_accs, top3_accs


def sdn_get_detailed_results(model, loader, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id * loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, layer_predictions, layer_confidence


def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, confusion_stats, device)

            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion


# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return (mean_con, std_con)


def sdn_test_early_exits(model, loader, device='cpu'):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = data.AverageMeter()
    top3 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1, prec3 = data.accuracy(output, b_y, topk=(1, 3))
            top1.update(prec1[0], b_x.size(0))
            top3.update(prec3[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top3_acc = top3.avg.data.cpu().numpy()[()]

    return top1_acc, top3_acc, early_output_counts, non_conf_output_counts, total_time


def cnn_training_step(model, optimizer, data, labels, device='cpu', islist=False):
    b_x = data.to(device)  # batch x
    b_y = labels.to(device)  # batch y
    output = model(b_x)
    if isinstance(output, list):  # cnn final output
        output = output[0]
    criterion = af.get_loss_criterion()
    loss = criterion(output, b_y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients


def cnn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
    metrics = {'epoch_times': [], 'test_top1_acc': [], 'test_top3_acc': [], 'train_top1_acc': [], 'train_top3_acc': [],
               'lrs': []}
    print("cnn training")
    for epoch in range(1, epochs + 1):
        scheduler.step()

        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y in train_loader:
            cnn_training_step(model, optimizer, x, y, device)

        end_time = time.time()

        top1_test, top3_test = cnn_test(model, data.test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top3 Test accuracy: {}'.format(top3_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top3_acc'].append(top3_test)

        top1_train, top3_train = cnn_test(model, train_loader, device)
        print('Top1 Train accuracy: {}'.format(top1_train))
        print('top3 Train accuracy: {}'.format(top3_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top3_acc'].append(top3_train)
        epoch_time = int(end_time - start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        metrics['epoch_times'].append(epoch_time)

        metrics['lrs'].append(cur_lr)

    return metrics


def cnn_test_time(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top3 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            prec1, prec3 = data.accuracy(output, b_y, topk=(1, 3))
            top1.update(prec1[0], b_x.size(0))
            top3.update(prec3[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top3_acc = top3.avg.data.cpu().numpy()[()]

    return top1_acc, top3_acc, total_time


def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top3 = data.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)

            if isinstance(output, list):
                output = output[0]

            prec1, prec3 = data.accuracy(output, b_y, topk=(1, 3))
            top1.update(prec1[0], b_x.size(0))
            top3.update(prec3[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top3_acc = top3.avg.data.cpu().numpy()[()]

    return top1_acc, top3_acc


def cnn_get_confidence(model, loader, device='cpu'):
    model.eval()
    correct = set()
    wrong = set()
    instance_confidence = {}
    correct_cnt = 0

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = nn.functional.softmax(output, dim=1)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            pred_prob = model_pred[0].to(device)

            is_correct = pred.eq(b_y.view_as(pred))
            correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id * loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)

    return correct, wrong, instance_confidence


# default training
def iter_training_0(model, data, params, optimizer, scheduler, device='cpu'):
    print("iter training 0")
    augment = model.augment_training
    metrics = {
        'epoch_times': [],
        'valid_top1_acc': [],
        'valid_top3_acc': [],
        'train_top1_acc': [],
        'train_top3_acc': [],
        'test_top1_acc': [],
        'test_top3_acc': [],
        'lrs': []
    }
    epochs, epoch_growth, epoch_prune = params['epochs'], params['epoch_growth'], params['epoch_prune']
    pruning_batch_size, pruning_type, reinit = params['prune_batch_size'], params['prune_type'], params['reinit']

    #epoch_growth = [25, 50, 75]  # [(i + 1) * epochs / (model.num_ics + 1) for i in range(model.num_ics)]
    print("array params: num_ics {}, epochs {}".format(model.num_ics, epochs))
    print("epochs growth: {}".format(epoch_growth))

    max_coeffs = calc_coeff(model)
    print('max_coeffs: {}'.format(max_coeffs))
    model.to(device)
    model.to_train()

    if model.prune:
        prune_dataset = af.get_dataset('cifar10', batch_size=pruning_batch_size)
        print("pruning_batch_size: {}, prune_type: {}, reinit: {}".format(pruning_batch_size, pruning_type, reinit))

    best_model, accuracies, best_epoch = None, None, 0
    count_pruned = 0
    for epoch in range(1, epochs + 1):

        if epoch in epoch_growth:
            grown_layers = model.grow()
            model.to(device)
            optimizer.add_param_group({'params': grown_layers})
            print("model grow")

        if epoch in epoch_prune and model.prune:
            loader = get_loader(prune_dataset, False)
            if pruning_type == '0':
                count_pruned = prune(model, model.keep_ratio, loader, sdn_loss, count_pruned, device, reinit)
            elif pruning_type == '1':
                prune2(model, model.keep_ratio, loader, sdn_loss, device)

        epoch_routine(model, data, optimizer, scheduler, epoch, epochs, augment, metrics, device)

        if model.num_output == model.num_ics + 1:
            print("best model evaluation: {}/{}".format(metrics['valid_top1_acc'][-1], accuracies))
            if best_model is None:
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                best_epoch = epoch
                print("Begin best_model: {}".format(accuracies))
            elif sum(metrics['valid_top1_acc'][-1]) > sum(accuracies):
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                best_epoch = epoch
                print("New best model: {}".format(accuracies))

        #print(gpu_usage())

    metrics['test_top1_acc'], metrics['test_top3_acc'] = sdn_test(best_model, data.test_loader, device)
    test_top1, test_top3 = sdn_test(model, data.test_loader, device)
    metrics['best_model_epoch'] = best_epoch
    print("best epoch: {}".format(best_epoch))
    print("comparison best and latest: {}/{}".format(metrics['test_top1_acc'], test_top1))
    return metrics, best_model


# training with freezing the previous layers
def iter_training_1(model, data, epochs, optimizer, scheduler, device='cpu'):
    print("iter training 1")
    augment = model.augment_training
    metrics = dict(epoch_times=[],
                   valid_top1_acc=[],
                   valid_top3_acc=[],
                   train_top1_acc=[],
                   train_top3_acc=[],
                   test_top1_acc=[],
                   test_top3_acc=[],
                   lrs=[])
    epoch_growth = [(i + 1) * epochs / (model.num_ics + 1) for i in range(model.num_ics)]
    print("epoch growth: {}".format(epoch_growth))
    freeze_epochs = (np.array([0, 25, 50]) + epochs).tolist()
    print("freeze epochs: {}".format(freeze_epochs))
    max_coeffs = calc_coeff(model)

    model.to(device)
    model.to_train()

    max_epoch = int(epoch_growth[-1]) + epochs
    print("max_epoch: {}".format(max_epoch))

    if model.prune:
        loader = get_loader(data, False)
        count_pruned = prune(model, model.keep_ratio, loader, sdn_loss, 0, device)
    best_model, accuracies = None, None
    for epoch in range(max_epoch):
        epoch_routine(model, data, optimizer, scheduler, epoch, epochs, augment, metrics, device)

        if epoch in epoch_growth:
            grown_layers = model.grow()
            model.to(device)
            optimizer.add_param_group({'params': grown_layers})
            print("model grow")
            print("layers: {}".format(model.layers))
            if model.prune:
                loader = get_loader(data, False)
                count_pruned = prune(model, model.keep_ratio, loader, sdn_loss, count_pruned, device)

        if epoch in freeze_epochs:
            index = freeze_epochs.index(epoch)
            index_to_freeze = 0  # the index of the layer until which we freeze the network
            print("index of freeze_epoch: {}".format(index))
            nb = 0
            for ic in model.ics:
                nb += ic
                index_to_freeze += 1
                if nb > index:
                    break
            print("index_to_freeze: {}".format(index_to_freeze))
            for bloc in model.layers[:index_to_freeze]:
                for param in bloc.parameters(True):
                    param.require_grad = False
        if model.num_output == model.num_ics + 1:
            print("best model evaluation")
            if best_model is None:
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("Begin best_model: {}".format(accuracies))
            elif sum(metrics['valid_top1_acc'][-1]) > sum(accuracies):
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("New best model: {}".format(accuracies))
    metrics['test_top1_acc'], metrics['test_top3_acc'] = sdn_test(best_model, data.test_loader, device)
    return metrics, best_model


def sdn_loss(output, label, coeffs=None):
    total_loss = 0.0
    if coeffs is None:
        coeffs = [1 for _ in range(len(output) - 1)]
    for ic_id in range(len(output) - 1):
        total_loss += float(coeffs[ic_id]) * af.get_loss_criterion()(output[ic_id], label)
    total_loss += af.get_loss_criterion()(output[-1], label)
    return total_loss


# training with freezing previous layers and defrezing
def iter_training_2(model, data, epochs, optimizer, scheduler, device='cpu'):
    print("iter training 2")
    augment = model.augment_training
    metrics = {
        'epoch_times': [],
        'valid_top1_acc': [],
        'valid_top3_acc': [],
        'train_top1_acc': [],
        'train_top3_acc': [],
        'test_top1_acc': [],
        'test_top3_acc': [],
        'lrs': []
    }
    epoch_growth = [(i + 1) * epochs / (model.num_ics + 1) for i in range(model.num_ics)]
    print("epoch growth: {}".format(epoch_growth))

    def calc_inter_growth(array, last_epoch):
        res = []
        last = None
        arr = array + [last_epoch]
        for i in arr:
            if last:
                res.append(int((last + i) / 2))
            last = i
        return res

    unfreeze_epochs = calc_inter_growth(epoch_growth, epochs)
    print("unfreeze_epochs: {}".format(unfreeze_epochs))
    max_coeffs = calc_coeff(model)

    model.to(device)
    model.to_train()

    if model.prune:
        loader = get_loader(data, False)
        count_pruned = prune(model, model.keep_ratio, loader, sdn_loss, 0, device)
    best_model, accuracies = None, None
    for epoch in range(epochs):

        epoch_routine(model, data, optimizer, scheduler, epoch, epochs, augment, metrics, device)

        if epoch in epoch_growth:
            for params in model.parameters(True):
                params.require_grad = False
            grown_layers = model.grow()
            model.to(device)
            optimizer.add_param_group({'params': grown_layers})
            print("model grow")
            if model.prune:
                loader = get_loader(data, False)
                count_pruned = prune(model, model.keep_ratio, loader, sdn_loss, count_pruned, device)

        if epoch in unfreeze_epochs:
            for params in model.parameters(True):
                params.require_grad = True
        if model.num_output == model.num_ics + 1:
            print("best model evaluation")
            if best_model is None:
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("Begin best_model: {}".format(accuracies))
            elif sum(metrics['valid_top1_acc'][-1]) > sum(accuracies):
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("New best model: {}".format(accuracies))
    metrics['test_top1_acc'], metrics['test_top3_acc'] = sdn_test(best_model, data.test_loader, device)
    return metrics, best_model


# same as 2 but with more and more epochs between the growings
def iter_training_3(model, data, epochs, optimizer, scheduler, device='cpu'):
    print("iter training 3")
    augment = model.augment_training
    metrics = {
        'epoch_times': [],
        'valid_top1_acc': [],
        'valid_top3_acc': [],
        'train_top1_acc': [],
        'train_top3_acc': [],
        'test_top1_acc': [],
        'test_top3_acc': [],
        'lrs': []
    }
    increase_value = epochs / (model.num_ics + 1)
    print("increase value: {}".format(increase_value))
    epoch_growth = [(i + 1) * increase_value for i in range(model.num_ics + 1)]
    print("epoch increasing: {}".format(epoch_growth))
    tmp = [0] + epoch_growth
    epoch_growth = [25, 50, 100, 200]  # [sum(tmp[:i+2]) for i in range(len(tmp)-1)]
    print("epoch growth: {}".format(epoch_growth))

    def calc_inter_growth(array):
        res = []
        last = None
        arr = array
        for i in arr:
            if last:
                res.append(int((last + i) / 2))
            last = i
        return res

    unfreeze_epochs = [33, 75, 125]  # calc_inter_growth(epoch_growth)
    print("unfreeze_epochs: {}".format(unfreeze_epochs))
    # max_coeffs = calc_coeff(model)

    model.to(device)
    model.to_train()

    if model.prune:
        loader = get_loader(data, False)
        count_pruned = prune(model, model.keep_ratio, loader, sdn_loss, 0, device)
    best_model, accuracies = None, None
    for epoch in range(int(epoch_growth[-1])):
        epoch_routine(model, data, optimizer, scheduler, epoch, int(epoch_growth[-1]), augment, metrics, device)

        if epoch in epoch_growth[:-1]:
            for params in model.parameters(True):
                params.require_grad = False
            grown_layers = model.grow()
            model.to(device)
            optimizer.add_param_group({'params': grown_layers})
            print("model grow")
            if model.prune:
                loader = get_loader(data, False)
                count_pruned = prune(model, model.keep_ratio, loader, sdn_loss, count_pruned, device)

        if epoch in unfreeze_epochs:
            for params in model.parameters(True):
                params.require_grad = True
        if model.num_output == model.num_ics + 1:
            print("best model evaluation")
            if best_model is None:
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("Begin best_model: {}".format(accuracies))
            elif sum(metrics['valid_top1_acc'][-1]) > sum(accuracies):
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("New best model: {}".format(accuracies))
    metrics['test_top1_acc'], metrics['test_top3_acc'] = sdn_test(best_model, data.test_loader, device)
    return metrics, best_model


# grow when loss stagnate
def iter_training_4(model, data, epochs, optimizer, scheduler, device='cpu'):
    print("iter training 4")
    augment = model.augment_training
    losses = []
    metrics = {
        'epoch_times': [],
        'valid_top1_acc': [],
        'valid_top3_acc': [],
        'train_top1_acc': [],
        'train_top3_acc': [],
        'test_top1_acc': [],
        'test_top3_acc': [],
        'lrs': []
    }
    # def grow(model, previous_loss, new_loss):

    model.to(device)
    model.to_train()

    def to_grow(acc_s):
        grow = False
        if len(acc_s[0])>=2:
            for a,b in zip(acc_s[0], acc_s[1]):
                grow = a-b<0.8 and a-b>0
                if not grow:
                    break
        return grow
    best_model, accuracies = None, None
    for epoch in range(epochs):
        epoch_routine(model, data, optimizer, scheduler, epoch, epochs, augment, metrics, device)

        if len(metrics['valid_top1_acc']) >=2 and to_grow([metrics['valid_top1_acc'][-2], metrics['valid_top1_acc'][-1]]):
            print("num_output, ic_num: {}, {}".format(model.num_output, model.num_ics))
            if model.num_output == model.num_ics + 1:
                break
            grown_layers = model.grow()
            model.to(device)
            optimizer.add_param_group({'params': grown_layers})
            print("model grow: {}".format(model.num_output))
        if model.num_output == model.num_ics + 1:
            print("best model evaluation")
            if best_model is None:
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("Begin best_model: {}".format(accuracies))
            elif sum(metrics['valid_top1_acc'][-1]) > sum(accuracies):
                best_model, accuracies = copy.deepcopy(model), metrics['valid_top1_acc'][-1]
                print("New best model: {}".format(accuracies))
    metrics['test_top1_acc'], metrics['test_top3_acc'] = sdn_test(best_model, data.test_loader, device)
    return metrics, best_model


def epoch_routine(model, datas, optimizer, scheduler, epoch, epochs, augment, metrics, device):
    scheduler.step()
    cur_lr = af.get_lr(optimizer)
    print('\nEpoch: {}/{}'.format(epoch, epochs))
    print('cur_lr: {}'.format(cur_lr))
    print("scheduler state dict: {}".format(scheduler.state_dict()))
    max_coeffs = calc_coeff(model)
    cur_coeffs = 0.01 + epoch * (np.array(max_coeffs) / epochs)
    cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
    print("current coeffs: {}".format(cur_coeffs))

    start_time = time.time()
    model.train()
    loader = get_loader(datas, augment)
    losses = []
    for i, batch in enumerate(loader):
        total_loss = sdn_training_step(optimizer, model, cur_coeffs, batch, device)
        losses.append(total_loss)
        if i % 100 == 0:
            print("Loss: {}".format(total_loss))

    top1_test, top3_test = sdn_test(model, datas.aug_valid_loader if augment else datas.valid_loader, device)
    end_time = time.time()

    print('Top1 Valid accuracies: {}'.format(top1_test))
    print('Top3 Valid accuracies: {}'.format(top3_test))
    top1_train, top3_train = sdn_test(model, get_loader(datas, augment), device)
    print('Top1 Train accuracies: {}'.format(top1_train))
    print('Top3 Train accuracies: {}'.format(top3_train))

    epoch_time = int(end_time - start_time)
    print('Epoch took {} seconds.'.format(epoch_time))

    metrics['valid_top1_acc'].append(top1_test)
    metrics['valid_top3_acc'].append(top3_test)
    metrics['train_top1_acc'].append(top1_train)
    metrics['train_top3_acc'].append(top3_train)
    metrics['epoch_times'].append(epoch_time)
    metrics['lrs'].append(cur_lr)

    loss_moy = sum(losses) / len(losses)
    print("mean loss: {}".format(loss_moy))
    return loss_moy


def calc_coeff(model):
    return [0.01 + (1 / model.num_output) * (i + 1) for i in range(model.num_output - 1)]


# max tau: % of the network for the IC -> if 3 outputs: 0.33, 0.66, 1

def prune(model, keep_ratio, loader, loss, count_pruned, device, reinit):
    masks, cur_pruned = snip.snip_skip_layers(model, keep_ratio, loader, loss, device, reinit)
    snip.apply_prune_mask_skip_layers(model, masks, count_pruned)
    return cur_pruned

def prune_whole():
    print("prune")

def prune2(layers, keep_ratio, loader, loss, device):
    masks = snip.snip(layers, keep_ratio, loader, loss, device)
    snip.apply_prune_mask(layers, masks)