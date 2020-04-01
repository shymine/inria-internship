import torch
import torch.nn as nn

import aux_funcs as af
import model_funcs as mf
import architectures.SDNs.ResNet_SDN as resNet
from torch.optim import SGD, Adam

class ResNet_Baseline(nn.Module):
    def __init__(self, params):
        super(ResNet_Baseline, self).__init__()
        """ Create the baseline ResNet for the iterative training
        Is concentrated on cifar10 dataset
        :param params: {    'network_type': 'resnet_iterative',
                            'augment_training': True,
                            'init_weights': True,
                            'block_type': 'basic',
                            'weight_decay': 0.0001,
                            'learning_rate': 0.1,
                            'epochs': 100,
                            'milestones': [35, 60, 85],
                            'gammas': [0.1, 0.1, 0.1],
                            'init_type': 'iterative',
                            'size': 9,
                            'ics': [0,1,0,1,0,1,0,1,0]
                        }
        :type params:
        """
        self.ic_only = False

        self.augment_training = params['augment_training']
        self.init_weights = params['init_weights']
        self.block_type = params['block_type']
        self.init_type = params['init_type']
        self.total_size = params['size'] #the size to reach (number of units)
        self.ics = params['ics'] if 'ics' in params else []
        self.num_ics = sum(self.ics)

        if 'mode' in params:
            self.mode = params['mode']

        if self.init_type != 'full' and len(self.ics) != self.total_size:
            raise ValueError("final size of network does not match the length of ics array: {}; {}".format(self.total_size, self.ics))

        self.num_class = 10

        self.num_output = 0

        if self.block_type == 'basic':
            self.block = resNet.BasicBlockWOutput

        self.input_size = 32 # cifar10
        self.in_channels = 16

        init_conv = []
        init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU())

        end_layers = []
        end_layers.append(nn.AvgPool2d(kernel_size=8))
        end_layers.append(af.Flatten())
        end_layers.append(nn.Linear(256*self.block.expansion, self.num_class))

        self.init_conv = nn.Sequential(*init_conv)
        self.layers = nn.ModuleList()
        self.end_layers = nn.Sequential(*end_layers)

        train_funcs = {
            0: mf.iter_training_0,
            1: mf.iter_training_1,
            2: mf.iter_training_2,
            3: mf.iter_training_3,
            4: mf.iter_training_4
        }

        if self.init_type == "full":
            self.train_func = mf.cnn_train
            self.test_func = mf.cnn_test
            layers = [self.block(self.in_channels,
                        16, (False, self.num_class, 32, 1)) for _ in range(self.total_size)]
            self.layers.extend(layers)
            self.num_output = 1
        elif self.init_type == "full_ic":
            self.train_func = mf.sdn_train
            self.test_func = mf.sdn_test
            layers = [self.block(self.in_channels,
                        16, (self.ics[i], self.num_class, 32, 1)) for i in range(self.total_size)]
            self.layers.extend(layers)
            self.num_output = sum(self.ics) + 1
        elif self.init_type == "iterative":
            self.train_func = train_funcs[self.mode]
            self.test_func = mf.sdn_test
            self.grow()
        else:
            raise KeyError("the init_type should be either 'full', 'full_ic' or 'iterative' and it is: {}".format(self.init_type))

        self.to_eval()

        if self.init_weights:
            self._init_weights(self.modules())


    def _init_weights(self, iter):
        for m in iter:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # the final output is considered as the final output
    def forward_eval(self, input):
        outputs = []
        fwd = self.init_conv(input)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)
        return outputs

    # the final output is not taken in account
    def forward_train(self, input):
        outputs = []
        fwd = self.init_conv(input)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        return outputs

    def to_train(self):
        self.forward = self.forward_train

    def to_eval(self):
        self.forward = self.forward_eval

    def grow(self): #grow to the next ic or, if no ic till the end is found, grow the needed layers
        nb_grow = 0
        add_ic = False
        ics_index = 0
        tmp = 0
        for ind, ic in enumerate(self.ics):
            tmp += ic
            # print("loop ({}), tmp:{}, ic:{}".format(ind, tmp, ic))
            if tmp >= self.num_output:
                ics_index = ind
                break
        # print("tmp: {}, num_ics: {}".format(tmp, self.num_ics))
        if tmp == self.num_ics: # no more ICs are to be grown
            print("Eval mode")
            self.to_eval()
        # print("ics_index: {}".format(ics_index))
        # print("iter on: {}".format(self.ics[ics_index:]))
        pos = 1
        if ics_index == 0:
            pos = 0
        for ic in self.ics[ics_index+pos:]:
            nb_grow += 1
            if ic:
                add_ic = True
                break
        # print("nb_grow: {}".format(nb_grow))
        layers = [
            self.block(self.in_channels,
                       16, (add_ic if i == nb_grow-1 else False,
                           self.num_class, 32, 1))
            for i in range(nb_grow)
        ]
        # print("grown layers: {}".format(layers))
        self._init_weights(layers)
        self.layers.extend(layers)
        self.num_output += 1
        return filter(lambda p: p.requires_grad, [p for l in layers for p in l.parameters(True)])

    # Not needed anymore as grow itself work well
    # def grow_copy(self, add_ic=True):
    #     model = ResNet_Baseline({
    #         'augment_training':self.augment_training,
    #         'init_weights': False,
    #         'block_type': 'basic'#,
    #         # 'size': self.num_output,
    #         # 'init_type': self.init_type
    #     })
    #     for i in range(self.num_output):
    #         model.grow(add_ic=True if hasattr(self.layers[3*i+2], 'output') else False)
    #     with torch.no_grad():
    #         for id, layer in enumerate(self.init_conv):
    #             if hasattr(layer, 'weight'):
    #                 model.init_conv[id].weight.copy_(layer.weigh)
    #         for id, unit in enumerate(self.layers):
    #             for id2, layer in enumerate(unit):
    #                 if hasattr(layer, 'weight'):
    #                     model.layers[id][id2].weight.copy_(layer.weight)
    #     self.init_conv = model.init_conv
    #     self.layers = model.layers
    #     return filter(lambda p: p.requires_grad, model.parameters())
