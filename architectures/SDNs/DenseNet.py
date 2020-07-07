import torch
import torch.nn as nn

from itertools import chain

import aux_funcs as af
import model_funcs as mf

class DenseNet(nn.Module):

    def __init__(self, total_size, ics, init_weights=True):
        super(DenseNet, self).__init__()
        self.total_size = total_size
        self.init_weights = init_weights
        self.ics = ics
        self.num_ics = sum(self.ics)
        self.num_class = 10
        self.num_output = 0

        self.train_func = mf.iter_training_0
        self.test_func = mf.sdn_test

        self.input_size = 32
        self.in_channels = 16
        self.cum_in_channels = self.in_channels

        self.init_conv = nn.Sequential(*[
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLu()
        ])

        self.end_layers = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=8),
            af.Flatten(),
            nn.linear(2560, self.num_class)
        ])
        self.grow()

        if self.init_weights:
            self._init_weights(self.modules())
    
    def _init_weights(self, it):
        for m in it:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        added_input = fwd
        for layer in self.layers:
            fwd, is_output, output = layer(added_input)
            added_input = torch.cat([added_input, fwd], 1)
            if is_output:
                outputs.append(output)
        if self.num_output == self.num_ics + 1:
            outputs.append(self.end_layers(added_input))
        return outputs

    def grow(self):
        nb_grow = 0
        add_ic = False
        ics_index = 0
        tmp = 0
        for ind, ic in enumerate(self.ics):
            tmp += ic
            if tmp >= self.num_output:
                ics_index = ind
                break
        pos = 1
        if ics_index == 0:
            pos = 0
        for ic in self.ics[ics_index + pos:]:
            nb_grow += 1
            if ic:
                add_ic = True
                break
        layers = []
        for i in range(nb_grow):
            layers.append(ConvBNRLUnit(
                self.in_channels, 16*self.num_output, 
                add_ic = add_ic if i == nb_grow-1 else False
            ))
            self.in_channels += layers[-1].out_channels
        if self.init_weights:
            self._init_weights(layers)
        self.layers.extend(layers)
        self.num_output += 1

class ConvBNRLUnit(nn.Module):

    def __init__(self, in_channels, out_channels, add_ic=False, num_classes=10, input_size=32):
        super(ConvBNRLUnit, self)
        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLu()
        ])
        
        if add_ic:
            self.ic = af.InternalClassifier(input_size, out_channels, num_classes)
        else:
            self.ic = None
        
    def forward(self, x):
        fwd = self.layers(x)
        return fwd, self.ic is not None, None if self.ic is None else self.ic(fwd)
    
    def only_output(self, x):
        return self.ic(self.layers(x))

    def get_parameters(self):
        if hasattr(self, 'output'):
            return chain(self.layers.parameters(True), self.output.parameter)
        else:
            return self.layers.parameters(True)
    
