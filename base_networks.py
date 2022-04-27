import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from math import sqrt
import random

class ConvBlock(torch.nn.Module):
    def __init__(self,
                 input_size, # input number of channels???
                 # output_size
                 output_size, # output number of channels???
                 # kernel size is 3x3
                 kernel_size=3, # kernel size 3x3 by default???
                 # stride is 1x1
                 stride=1, # step aside by just one step, when sliding the window
                 padding=1, # the correct padding for 3x3 kernel so image will stay the same size
                 bias=True, # use bias
                 activation='prelu', # activation function
                 norm=None): # and do not use the normalization inside
        super(ConvBlock, self).__init__() # So is is a module, so we inherit from it
        # we just proxy the parameters to the convolutional layer, and then
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        # norm normalization
        self.norm = norm
        if self.norm =='batch': # if we use batch-normalization
            self.bn = torch.nn.BatchNorm2d(output_size) # torch.nn.BatchNorm2d
        elif self.norm == 'instance': # or better: instance normalization
            self.bn = torch.nn.InstanceNorm2d(output_size) # # Use Instance Normalization as anybody

        self.activation = activation
        if self.activation == 'relu': # relu activation function
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu': # prelu and all the other stuff
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True) # torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh() # torch.nn.Tanh
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            # out = self.bn(self.conv(x))
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    # Is exactly symmetric to the convolutional variant
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        # but uses transposed convolution
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      # To upsample we just use ConvTranspose
      super(UpsampleConvLayer, self).__init__()
      # conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    # This is a residual block
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # ConvLayer does not have activation, this is a weird construction
        # conv1
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1) # first layer
        # conv2
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1) # second layer
        # relu
        self.relu = nn.ReLU() # ReLU: Rectified linear unit

    def forward(self, x):
        residual = x # This tensor is used to form the residual
        out = self.relu(self.conv1(x)) # Apply convolution, then relu, then
        out = self.conv2(out) * 0.1 # apply convolution second time, it is not clear why it is scaled like this
        # why this magic numbers???
        out = torch.add(out, residual) # final output
        return out

def init_linear(linear): # what?? it takes some tensor, or some layer and modified the weight inside???
    init.xavier_normal(linear.weight) # use xavier_normal
    linear.bias.data.zero_() # zero out the bias inside of this thing


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight) # special initialization for
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name    # input of what??? It seeams that it performs some kind of initialization

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module