import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from .attention import *


class BasicConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1),
                 padding=(0, 0),
                 dilation=1,
                 groups=1,
                 relu=True,
                 inorm=True,
                 bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.inorm = nn.InstanceNorm2d(out_channels,
                                       affine=True) if inorm else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):

        x = self.conv(x)
        if self.inorm is not None:
            x = self.inorm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class DepthwiseConv(nn.Module):
    """
    添加上扩张选项，使得可分离卷积在时间维度上扩大感受野
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups,
                 stride,
                 dilation,
                 padding=0,
                 bias=False):
        super(DepthwiseConv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              groups=groups,
                              stride=stride,
                              dilation=dilation,
                              padding=padding,
                              bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class PointwiseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=0,
                 bias=True):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=stride,
                              padding=padding,
                              bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


# 基于point和dw实现dense所需要的卷积层
class DepthdenseConv(nn.Module):
    """
    bias分开设置，point需要bias，dw的bias设置为false
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size,
                 groups,
                 stride,
                 dilation,
                 padding=0):
        super(DepthdenseConv, self).__init__()
        self.point1 = PointwiseConv(in_channels,
                                    hidden_channels,
                                    stride=stride,
                                    padding=padding,
                                    bias=True)
        self.dw = DepthwiseConv(hidden_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                groups=groups,
                                stride=stride,
                                dilation=dilation,
                                padding=padding,
                                bias=False)

    def forward(self, x):
        x = self.point1(x)
        x = self.dw(x)

        return x


# denseblock use the depthwise
class DenseBlock_depth(nn.Module):

    def __init__(self, in_channels, groups, depth=4):
        super(DenseBlock_depth, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1),
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(
                self, 'conv{}'.format(i + 1),
                DepthdenseConv(self.in_channels * (i + 1),
                               self.in_channels * (i + 1),
                               self.in_channels,
                               kernel_size=self.kernel_size,
                               groups=groups,
                               stride=1,
                               dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1),
                    nn.InstanceNorm2d(self.in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out  # type:ignore


# 正常的dense block
class DenseBlock(nn.Module):

    def __init__(self, depth=4, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1),
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(
                self, 'conv{}'.format(i + 1),
                nn.Conv2d(self.in_channels * (i + 1),
                          self.in_channels,
                          kernel_size=self.kernel_size,
                          dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1),
                    nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out  # type:ignore


# 添加la的dense block
class DenseBlock_La(nn.Module):

    def __init__(self, depth=4, in_channels=64):
        super(DenseBlock_La, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1),
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(
                self, 'conv{}'.format(i + 1),
                nn.Conv2d(self.in_channels * (i + 1),
                          self.in_channels,
                          kernel_size=self.kernel_size,
                          dilation=(dil, 1)))
            setattr(self, 'la{}'.format(i + 1),
                    LocalAttention(inchannels=self.in_channels))
            setattr(self, 'norm{}'.format(i + 1),
                    nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'la{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out  # type:ignore


# 添加了lpa的dense block
class DenseBlock_Lpa(nn.Module):

    def __init__(self,
                 depth=4,
                 in_channels=64,
                 t_length=126,
                 num_features=257):
        super(DenseBlock_Lpa, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1),
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(
                self, 'conv{}'.format(i + 1),
                nn.Conv2d(self.in_channels * (i + 1),
                          self.in_channels,
                          kernel_size=self.kernel_size,
                          dilation=(dil, 1)))
            setattr(
                self, 'flpa{}'.format(i + 1),
                LocalpatchAttention(in_channels=self.in_channels,
                                    reduction=2,
                                    pool_t=t_length,
                                    pool_f=1,
                                    add_input=True))
            setattr(
                self, 'tlpa{}'.format(i + 1),
                LocalpatchAttention(in_channels=self.in_channels,
                                    reduction=2,
                                    pool_t=1,
                                    pool_f=num_features,
                                    add_input=True))
            setattr(self, 'norm{}'.format(i + 1),
                    nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'tlpa{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out  # type:ignore


# class sptransconv
class SPConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels,
                              out_channels * r,
                              kernel_size=kernel_size,
                              stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


if __name__ == '__main__':
    a = torch.rand(4, 8, 16, 17)
    deconv = nn.Conv2d(8, 8, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
    dea = deconv(a)
    print('deconv:', dea.size())
    upconv = SPConvTranspose2d(8, 8, (1, 3), r=2)
    b = upconv(dea)
    print(b.size())
    conv = nn.Conv2d(8, 8, (1, 2))
    c = conv(b)
    print('整形', c.size())
    # dense = DenseBlock(in_channels=8)
    # b = dense(a)
    # print('dense:',b.size())
    # conv = BasicConv(8,4,1)
    # c = conv(a)
    # print('conv',c.size())
