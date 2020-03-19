import torch
import torch.nn as nn

import aux_funcs as af
import model_funcs as mf
import architectures.SDNs.ResNet_SDN as resNet

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
        self.layers.extend([self.block(self.in_channels, 16, (True if i == 2 else False, self.num_class, 32, 1)) for i in range(3)])

        self.end_layers = nn.Sequential(*end_layers)
        print("model: {}".format(self))

    def forward(self, input):
        outputs = []
        fwd = self.init_conv(input)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)
        return outputs
