import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ca
class ChannelAttention(nn.Module):
    """
    channels:denseblock的输出的通道数，应该和前面的卷积块的输出通道数是一样的
    """

    def __init__(self, channels):
        super(ChannelAttention, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1, bias=False))

    def forward(self, x):
        """
        input:[b,n,t,f]
        output:[b,n,t,f]
        """
        attn_max = F.adaptive_max_pool2d(x, 1)
        attn_avg = F.adaptive_avg_pool2d(x, 1)

        attn_max = self.fc(attn_max)
        attn_avg = self.fc(attn_avg)

        attn = attn_max + attn_avg

        attn = F.sigmoid(attn)

        x = x * attn

        return x


# eca channel attention
class EcaAttention(nn.Module):

    def __init__(self, channel):
        super(EcaAttention, self).__init__()
        self.k_size = math.floor(math.log2(channel) // 2 + 0.5)  # 向下取整
        if self.k_size % 2 == 0:
            self.k_size = self.k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,
                              1,
                              kernel_size=self.k_size,
                              padding=(self.k_size - 1) // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,
                                              -2)).transpose(-1,
                                                             -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class LocalAttention(nn.Module):
    """
    inchannels:输入的通道数
    """

    def __init__(self, inchannels):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, 1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=3,
                               out_channels=1,
                               kernel_size=7,
                               stride=1,
                               padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        convout = self.conv1(x)
        out = torch.cat([avgout, maxout, convout], dim=1)
        out = self.sigmoid(self.conv2(out))
        # out = out * x
        return out


# 基于lanet的local patch attention
class LocalpatchAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 reduction=8,
                 pool_t=8,
                 pool_f=8,
                 add_input=False):
        """
        in_channels:输入的通道数
        reduction:通道缩放的大小
        pool_t:t维度上分割的大小
        pool_f:f维度上分割的大小
        """
        super(LocalpatchAttention, self).__init__()
        self.pool_t = pool_t
        self.pool_f = pool_f
        self.add_input = add_input
        self.SA = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.GroupNorm(in_channels // reduction,
                         in_channels // reduction),  # layernorm
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid())

    def forward(self, x):
        b, c, f, t = x.size()  # 这里可能需要改动
        t_patch = t // self.pool_t
        f_patch = f // self.pool_f

        a1 = F.adaptive_avg_pool2d(x, (f_patch, t_patch))
        a2 = F.adaptive_max_pool2d(x, (f_patch, t_patch))

        A = a1 + a2  # 这里仔细考虑一下如何将两个提取出来的特征融合
        A = self.SA(A)

        A = F.interpolate(A, (f, t), mode='bilinear')
        output = x * A

        if self.add_input:
            output += x

        return output


# 单个trans
class SingalTrans(nn.Module):

    def __init__(self,
                 in_channels,
                 nhead=2,
                 bidirectional=True,
                 dropout=0.0,
                 activation="relu"):
        super(SingalTrans, self).__init__()
        self.d_model = in_channels // 2
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=self.d_model,
                               kernel_size=1,
                               bias=False)
        self.conv_norm1 = nn.BatchNorm1d(self.d_model)
        self.ffn1 = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(self.d_model, self.d_model))
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.self_attn = nn.MultiheadAttention(self.d_model,
                                               nhead,
                                               dropout=dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.rnn2 = nn.GRU(self.d_model,
                           self.d_model * 2,
                           1,
                           bidirectional=bidirectional)
        self.ffn2 = nn.Sequential(
            nn.ReLU(), nn.Linear(self.d_model * 2 * 2, self.d_model))
        self.drop3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.conv2 = nn.Conv1d(in_channels=self.d_model,
                               out_channels=self.d_model * 2,
                               kernel_size=1,
                               bias=False)
        self.conv_norm2 = nn.BatchNorm1d(self.d_model * 2)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(SingalTrans, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # conv1
        src_conv = self.conv1(src)
        src_conv = self.conv_norm1(src_conv)  #[b*t,c,f]
        src_conv = src_conv.permute(0, 2, 1).contiguous()  #[b*t,f,c]
        # ffn1
        src1 = self.ffn1(src_conv)
        src1 = self.norm1(1 / 2 * self.drop1(src1) + src_conv)
        # mha
        src2 = self.self_attn(src1,
                              src1,
                              src1,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.norm2(src1 + self.drop2(src2))
        # ffn2
        self.rnn2.flatten_parameters()
        src3, h_n = self.rnn2(src2)
        del h_n
        src3 = self.ffn2(src3)
        src3 = self.norm3(src2 + 1 / 2 * self.drop3(src3))

        # conv2
        src3 = src3.permute(0, 2, 1).contiguous()  #[b*t,c,f]
        src_out = self.conv_norm2(self.conv2(src3))

        return src_out


# global attention base the trans其实就是trans
class GlobalAttention(nn.Module):

    def __init__(self, in_channels, nhead, dropout=0.0, num_layers=1):
        super(GlobalAttention, self).__init__()
        self.trans = nn.ModuleList([])
        self.norm = nn.ModuleList([])

        for i in range(num_layers):
            self.trans.append(
                SingalTrans(in_channels, nhead=nhead, dropout=dropout))
            self.norm.append(nn.GroupNorm(1, in_channels, eps=1e-8))

    def forward(self, input):
        b, c, t, f = input.shape  #[b,c,t,f]
        output = input

        for i in range(len(self.trans)):
            input = output.permute(0, 2, 1,
                                   3).contiguous().view(b * t, -1,
                                                        f)  # [b*t,c,f]
            trans_output = self.trans[i](input)  #[b*t,c,f]
            trans_output = trans_output.view(b, t, c, f).permute(
                0, 2, 1, 3).contiguous()  #[b,c,t,f]
            trans_output = self.norm[i](trans_output)
            output = output + trans_output

        return output


if __name__ == '__main__':
    a = torch.rand(4, 32, 122, 257)
    lpa = LocalpatchAttention(32)
    b = lpa(a)
    print('lpa output:', b.size())
    # model = GlobalAttention(32, 2, 0.1, 2)
    # c_model = EcaAttention(32)
    # # b = model(a)
    # d = c_model(a)
    # print(d.size())
    # print(b.size())
