from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx

import torch
import torch.nn as nn

from torch.nn import init

def weights_init_kaiming(m):
    """Torch Weights initializer"""
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(
            m.weight.data, a=0, mode="fan_in"
        )  # For old pytorch, you may use kaiming_normal.
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    """Torch Weights initializer"""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    """A basic block of Linear, BatchNorm, activation function and Dropout in torch, 
    followed by a classification layer.

    Arguments:
        input_dim: Integer, the size of the input
        class_num: Integer, the size of the output
        droprate: Float in [0,1], rate of the Dropout
        relu: Boolean, to add a LeakyRelu layer
        bnorm: Boolean, to add a BatchNorm layer
        num_bottleneck: Integer, output size of the linear layer
        linear: Boolean, to add a Linear layer
        return_f: Boolean, to return the intermediate output before the classifier or not
    
    """

    def __init__(
        self,
        input_dim,
        class_num,
        droprate,
        relu=False,
        bnorm=True,
        num_bottleneck=512,
        linear=True,
        return_f=False,
    ):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x
