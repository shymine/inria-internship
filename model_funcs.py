# model_funcs.py
# implements the functions for training, testing SDNs and CNNs
# also implements the functions for computing confusion and confidence
import sys

import torch
import math
import copy
import time
import random

import torch.nn as nn
import numpy as np

from torch.optim import SGD
from random import choice, shuffle
from collections import Counter

import aux_funcs as af
import data


def sdn_training_step(optimizer, model, coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
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
    optimizer.step()                # apply gradients

    return total_loss

def sdn_ic_only_step(optimizer, model, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    for output_id, cur_output in enumerate(output):
        if output_id == model.num_output - 1: # last output
            break
        
        cur_loss = af.get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss

def get_loader(data, augment):
    if augment:
        train_loader = data.aug_train_loader
    else:
        train_loader = data.train_loader

    return train_loader


def sdn_train(model, data, epochs, optimizer, scheduler, device='cpu'):
    augment = model.augment_training
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top3_acc':[], 'train_top1_acc':[], 'train_top3_acc':[], 'lrs':[]}
    max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]) # max tau_i --- C_i values

    if model.ic_only:
        print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    else:
        print('sdn will be trained from scratch...(The SDN training)')

    for epoch in range(1, epochs+1):
        scheduler.step()
        cur_lr = af.get_lr(optimizer)
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        if model.ic_only is False:
            # calculate the IC coeffs for this epoch for the weighted objective function
            cur_coeffs = 0.01 + epoch*(max_coeffs/epochs) # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            print('Cur coeffs: {}'.format(cur_coeffs))

        start_time = time.time()
        model.train()
        loader = get_loader(data, augment)
        for i, batch in enumerate(loader):
            if model.ic_only is False:
                total_loss = sdn_training_step(optimizer, model, cur_coeffs, batch, device)
            else:
                total_loss = sdn_ic_only_step(optimizer, model, batch, device)

            if i % 100 == 0:
                print('Loss: {}: '.format(total_loss))

        top1_test, top3_test = sdn_test(model, data.test_loader, device)

        print('Top1 Test accuracies: {}'.format(top1_test))
        print('Top3 Test accuracies: {}'.format(top3_test))
        end_time = time.time()

        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top3_acc'].append(top3_test)

        top1_train, top3_train = sdn_test(model, get_loader(data, augment), device)
        print('Top1 Train accuracies: {}'.format(top1_train))
        print('Top3 Train accuracies: {}'.format(top3_train))
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top3_acc'].append(top3_train)

        epoch_time = int(end_time-start_time)
        metrics['epoch_times'].append(epoch_time)
        print('Epoch took {} seconds.'.format(epoch_time))

        metrics['lrs'].append(cur_lr)

    return metrics

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
                    cur_instance_id = test_id + cur_batch_id*loader.batch_size
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
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
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
            total_time+= (end_time - start_time)
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

def cnn_training_step(model, optimizer, data, labels, device='cpu', list=False):
    b_x = data.to(device)   # batch x
    b_y = labels.to(device)   # batch y
    output = model(b_x) if not list else model(b_x)[0]          # cnn final output
    criterion = af.get_loss_criterion()
    loss = criterion(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients


def cnn_train(model, data, epochs, optimizer, scheduler, device='cpu', list=False):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top3_acc':[], 'train_top1_acc':[], 'train_top3_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
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
            cnn_training_step(model, optimizer, x, y, device, list)
        
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
        epoch_time = int(end_time-start_time)
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
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)

   
    return correct, wrong, instance_confidence

def iter_training(model, data, epochs, optimizer, scheduler, device='cpu'):
    augment = model.augment_training
    metrics = {
        'epoch_times': [],
        'test_top1_acc': [],
        'test_top3_acc': [],
        'train_top1_acc': [],
        'train_top3_acc': [],
        'lrs': []
    }
    def calc_coeff(model):
        return [0.01+(1/model.num_output)*(i+1) for i in range(model.num_output-1)]
    # max tau: % of the network for the IC -> if 3 outputs: 0.33, 0.66, 1
    max_coeffs = calc_coeff(model)
    print('max_coeffs: {}'.format(max_coeffs))
    print('layers: {}'.format(model.layers))
    model.to(device)
    model.to_train()
    for epoch in range(1, epochs+1):
        scheduler.step()
        cur_lr = af.get_lr(optimizer)
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('cur_lr: {}'.format(cur_lr))
        max_coeffs = calc_coeff(model)
        cur_coeffs = 0.01 + epoch*(np.array(max_coeffs)/epochs)
        cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
        print("current coeffs: {}".format(cur_coeffs))

        start_time = time.time()
        model.train()
        loader = get_loader(data, augment)
        for i, batch in enumerate(loader):
            total_loss = sdn_training_step(optimizer, model, cur_coeffs, batch, device) #iter_training_step()
            if i%100 == 0:
                print("Loss: {}".format(total_loss))

        top1_test, top3_test = sdn_test(model, data.test_loader, device)
        end_time = time.time()

        print('Top1 Test accuracies: {}'.format(top1_test))
        print('Top3 Test accuracies: {}'.format(top3_test))
        top1_train, top3_train = sdn_test(model, get_loader(data, augment), device)
        print('Top1 Train accuracies: {}'.format(top1_train))
        print('Top3 Train accuracies: {}'.format(top3_train))

        epoch_time = int(end_time - start_time)
        print('Epoch took {} seconds.'.format(epoch_time))

        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top3_acc'].append(top3_test)
        metrics['train_top1_acc'].append(top1_train)
        metrics['train_top3_acc'].append(top3_train)
        metrics['epoch_times'].append(epoch_time)
        metrics['lrs'].append(cur_lr)

        if epoch in [25,50,75]: #,100,125,150]:
            grown_layers = model.grow()
            model.to(device)
            optimizer.add_param_group({'params':grown_layers}) #= af.get_full_optimizer(model, optim_param2, scheduler_params)
            print("model grow")
            print("layers: {}".format(model.layers))
    return metrics

def sdn_loss(output, label, coeffs=None):
    total_loss = 0.0
    if coeffs == None:
        coeffs = [1 for _ in len(output)-1]
    for ic_id in range(len(output)-1):
        total_loss += float(coeffs[ic_id])*af.get_loss_criterion()(output[ic_id], label)
    total_loss += af.get_loss_criterion()(output[-1], label)
    return total_loss

