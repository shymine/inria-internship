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
                            'gammas': [0.1, 0.1, 0.1]
                        }
        :type params:
        """
        self.augment_training = params['augment_training']
        self.init_weights = params['init_weights']
        self.block_type = params['block_type']
        self.num_class = 10
        self.train_func = mf.iter_training
        self.test_func = None
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
        end_layers.append(nn.Linear(16, self.num_class))

        self.init_conv = nn.Sequential(*init_conv)
        self.layers = nn.ModuleList()
        self.end_layers = nn.Sequential(*end_layers)

        self.grow()
        print("model: {}".format(self))

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

    def grow(self):
        layers = [self.block(self.in_channels, 16,
                        (True if i == 2 else False,
                         self.num_class,
                         32,
                         1))
             for i in range(3)]
        self._init_weights(layers)
        self.layers.extend(layers)
        self.num_output += 1
        return filter(lambda p: p.requires_grad, layers)

    def grow_copy(self):
        model = ResNet_Baseline({
            'augment_training':self.augment_training,
            'init_weights': False,
            'block_type': 'basic'
        })
        for _ in range(self.num_output):
            model.grow()
        with torch.no_grad():
            for id, layer in enumerate(self.init_conv):
                if hasattr(layer, 'weight'):
                    model.init_conv[id].weight.copy_(layer.weigh)
            for id, unit in enumerate(self.layers):
                for id2, layer in enumerate(unit):
                    if hasattr(layer, 'weight'):
                        model.layers[id][id2].weight.copy_(layer.weight)
        self.init_conv = model.init_conv
        self.layers = model.layers
        return filter(lambda p: p.requires_grad, model.parameters())
