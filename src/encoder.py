import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_modules import *


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dense_depth=4):
        super(EncoderBlock, self).__init__()
        self.conv1 = BasicConv(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.dense = DenseBlock(depth=dense_depth, in_channels=out_channels)
        self.conv2 = BasicConv(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 3),
                               stride=(1, 2),
                               padding=(0,1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense(out)
        out = self.conv2(out)

        return out


class EncoderBlock_depth(nn.Module):

    def __init__(self, in_channels, out_channels, dense_depth=4):
        super(EncoderBlock_depth, self).__init__()
        self.conv1 = BasicConv(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.dense = DenseBlock_depth(in_channels=out_channels,
                                      groups=out_channels,
                                      depth=dense_depth)
        self.conv2 = BasicConv(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(1, 3),
                               stride=(1, 2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense(out)
        out = self.conv2(out)

        return out


if __name__ == '__main__':
    a = torch.rand(4, 3, 122, 257)
    model = EncoderBlock(3, 64)
    convnum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    b = model(a)
    print(b.size())
    print('conv:',convnum)
    model2 = EncoderBlock_depth(3, 64)
    depnum = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    c = model2(a)
    print('dep dense',c.size())
    print('dep dense:',depnum)
