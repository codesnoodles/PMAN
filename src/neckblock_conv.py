import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .conformer_conv import * 
from .utils import AFF
from .attention import LocalAttention, ChannelAttention
from .conv_modules import BasicConv

class ConnectConformer(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers=4):
        super(ConnectConformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels // 2

        self.input = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.mid_channels,
                      kernel_size=1), nn.PReLU())
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])

        for i in range(num_layers):
            self.row_trans.append(
                ConformerBlock(dim=self.mid_channels,
                               dim_head=self.mid_channels // 4,
                               conv_kernel_size=self.mid_channels // 2 - 1,
                               attn_dropout=0.2,
                               ff_dropout=0.2))
            self.col_trans.append(
                ConformerBlock(dim=self.mid_channels,
                               dim_head=self.mid_channels // 4,
                               conv_kernel_size=self.mid_channels // 2 - 1,
                               attn_dropout=0.2,
                               ff_dropout=0.2))
            self.row_norm.append(nn.GroupNorm(1, self.mid_channels, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, self.mid_channels, eps=1e-8))

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(self.mid_channels, out_channels, 1))

    def forward(self, input):
        b, c, t, f = input.shape
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(0, 3, 2,
                                       1).contiguous().view(b * f, t, -1)
            row_output = self.row_trans[i](row_input)
            row_output = row_output.view(b, f, t, -1).permute(0, 3, 2,
                                                              1).contiguous()
            row_output = self.row_norm[i](row_output)

            output = output + row_output
            col_input = output.permute(0, 2, 3,
                                       1).contiguous().view(b * t, f, -1)
            col_output = self.col_trans[i](col_input)
            col_output = col_output.view(b, t, f, -1).permute(0, 3, 1,
                                                              2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = output + col_output
        del row_input,row_output,col_input,col_output # type:ignore
        output = self.output(output)
        return output # type:ignore

class CrossConformer_ori(nn.Module):
    """
    just like the dual branch's paraller trans
    """

    def __init__(self, in_channels, out_channels, num_layers=1):
        super(CrossConformer_ori, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels // 2
        self.k1 = Parameter(torch.ones(1))
        self.k2 = Parameter(torch.ones(1))

        self.input = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.mid_channels,
                      kernel_size=1), nn.PReLU())

        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])

        for i in range(num_layers):
            self.row_trans.append(
                ConformerBlock(dim=self.mid_channels,
                               dim_head=self.mid_channels // 4,
                               conv_kernel_size=self.mid_channels // 2 - 1,
                               attn_dropout=0.2,
                               ff_dropout=0.2))
            self.col_trans.append(
                ConformerBlock(dim=self.mid_channels,
                               dim_head=self.mid_channels // 4,
                               conv_kernel_size=self.mid_channels // 2 - 1,
                               attn_dropout=0.2,
                               ff_dropout=0.2))
            self.row_norm.append(nn.GroupNorm(1, self.mid_channels, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, self.mid_channels, eps=1e-8))

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(self.mid_channels, out_channels, 1))

    def forward(self, input):
        b, c, t, f = input.shape
        output_list = []
        output = self.input(input)
        for i in range(len(self.row_trans)):
            row_input = output.permute(0, 3, 2,
                                       1).contiguous().view(b * f, t, -1)
            row_output = self.row_trans[i](row_input)
            row_output = row_output.view(b, f, t, -1).permute(0, 3, 2,
                                                              1).contiguous()
            row_output = self.row_norm[i](row_output)

            col_input = output.permute(0, 2, 3,
                                       1).contiguous().view(b * t, f, -1)
            col_output = self.col_trans[i](col_input)
            col_output = col_output.view(b, t, f, -1).permute(0, 3, 1,
                                                              2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = output + self.k1 * row_output + self.k2 * col_output
            output_i = self.output(output)
            output_list.append(output_i)
        del row_input, row_output, col_input, col_output  # type:ignore
        return output_i, output_list  # type:ignore


class CrossConformer_attn(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers=1):
        super(CrossConformer_attn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels // 2

        self.input = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.mid_channels,
                      kernel_size=1), nn.PReLU())

        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        self.la = nn.ModuleList([])
        self.ca = nn.ModuleList([])
        self.in_aff = nn.ModuleList([])
        self.out_aff = nn.ModuleList([])

        for i in range(num_layers):
            self.row_trans.append(
                ConformerBlock(dim=self.mid_channels,
                               dim_head=self.mid_channels // 4,
                               conv_kernel_size=self.mid_channels // 2 - 1,
                               attn_dropout=0.2,
                               ff_dropout=0.2))
            self.col_trans.append(
                ConformerBlock(dim=self.mid_channels,
                               dim_head=self.mid_channels // 4,
                               conv_kernel_size=self.mid_channels // 2 - 1,
                               attn_dropout=0.2,
                               ff_dropout=0.2))
            self.row_norm.append(nn.GroupNorm(1, self.mid_channels, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, self.mid_channels, eps=1e-8))
            self.la.append(LocalAttention(inchannels=self.mid_channels))
            self.ca.append(ChannelAttention(channels=self.mid_channels))
            self.in_aff.append(AFF(channels=self.mid_channels, r=1))
            self.out_aff.append(AFF(channels=self.mid_channels, r=1))

            self.output = nn.Sequential(
                nn.PReLU(), nn.Conv2d(self.mid_channels, out_channels, 1))

    def forward(self, input):
        b, c, t, f = input.shape
        output_list = []
        output = self.input(input)
        for i in range(len(self.row_trans)):
            # 计算ca和la生成权重，用于fa和ta的加和
            ca = self.ca[i](output)
            la = self.la[i](output)
            cla = ca + la
            w_cla = self.in_aff[i](cla)
            cla_w = ca * w_cla + (1 - w_cla) * la
            w_fta = self.out_aff[i](cla_w)
            # 计算fa和ta
            row_input = output.permute(0, 3, 2,
                                       1).contiguous().view(b * f, t, -1)
            row_output = self.row_trans[i](row_input)
            row_output = row_output.view(b, f, t, -1).permute(0, 3, 2,
                                                              1).contiguous()
            row_output = self.row_norm[i](row_output)

            col_input = output.permute(0, 2, 3,
                                       1).contiguous().view(b * t, f, -1)
            col_output = self.col_trans[i](col_input)
            col_output = col_output.view(b, t, f, -1).permute(0, 3, 1,
                                                              2).contiguous()
            col_output = self.col_norm[i](col_output)
            output = cla_w + w_fta * row_output + (1 - w_fta) * col_output
            output_i = self.output(output)
            output_list.append(output_i)
        del row_input, row_output, col_input, col_output, ca, la, cla, cla_w, w_cla, w_fta  # type:ignore
        return output_i, output_list  # type:ignore


class CrossConformer_big(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CrossConformer_big, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels // 2

        self.input = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=self.mid_channels,
                      kernel_size=1,
                      stride=1), nn.PReLU())

        self.tc1 = ConformerBlock(dim=self.mid_channels,
                                  dim_head=self.mid_channels // 4,
                                  conv_kernel_size=self.mid_channels // 2 - 1,
                                  attn_dropout=0.2,
                                  ff_dropout=0.2)
        self.tc2 = ConformerBlock(dim=self.mid_channels,
                                  dim_head=self.mid_channels // 4,
                                  conv_kernel_size=self.mid_channels // 2 - 1,
                                  attn_dropout=0.2,
                                  ff_dropout=0.2)
        self.fc1 = ConformerBlock(dim=self.mid_channels,
                                  dim_head=self.mid_channels // 4,
                                  conv_kernel_size=self.mid_channels // 2 - 1,
                                  attn_dropout=0.2,
                                  ff_dropout=0.2)
        self.fc2 = ConformerBlock(dim=self.mid_channels,
                                  dim_head=self.mid_channels // 4,
                                  conv_kernel_size=self.mid_channels // 2 - 1,
                                  attn_dropout=0.2,
                                  ff_dropout=0.2)
        self.ca1 = ChannelAttention(channels=self.mid_channels)
        self.ca2 = ChannelAttention(channels=self.mid_channels)
        self.la1 = LocalAttention(inchannels=self.mid_channels)
        self.la2 = LocalAttention(inchannels=self.mid_channels)
        self.conv1 = BasicConv(in_channels=self.mid_channels * 2,
                               out_channels=self.mid_channels,
                               kernel_size=1)
        self.conv2 = BasicConv(in_channels=self.mid_channels * 2,
                               out_channels=self.mid_channels,
                               kernel_size=1)
        self.aff1 = AFF(channels=self.mid_channels)
        self.aff2 = AFF(channels=self.mid_channels)
        self.aff3 = AFF(channels=self.mid_channels)
        self.aff4 = AFF(channels=self.mid_channels)
        self.norm1 = nn.GroupNorm(1, self.mid_channels, eps=1e-8)
        self.norm2 = nn.GroupNorm(1, self.mid_channels, eps=1e-8)
        self.norm3 = nn.GroupNorm(1, self.mid_channels, eps=1e-8)
        self.norm4 = nn.GroupNorm(1, self.mid_channels, eps=1e-8)
        self.output = nn.Sequential(nn.PReLU(),nn.Conv2d(self.mid_channels,out_channels,1))

    def forward(self, x):
        b, c, t, f = x.size()
        x = self.input(x)
        # ca和la计算加权系数
        x_ca = self.ca1(x)
        x_la = self.la1(x)
        x_aff1 = x_ca + x_la
        x_cla_w = self.aff1(x_aff1)
        x_cla = x_ca * x_cla_w + (1 - x_cla_w) * x_la
        x_ft_w = self.aff2(x_cla)
        # 计算第一阶段fa和ta
        x_t = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, -1)
        x_t = self.tc1(x_t)
        x_t = x_t.view(b, f, t, -1).permute(0, 3, 2, 1).contiguous()
        x_t = self.norm1(x_t)
        x_f = x.permute(0, 2, 3, 1).contiguous().view(b * t, f, -1)
        x_f = self.fc1(x_f)
        x_f = x_f.view(b, t, f, -1).permute(0, 3, 1, 2).contiguous()
        x_f = self.norm2(x_f)

        mid_out = x_cla + x_ft_w * x_f + (1 - x_ft_w) * x_t
        
        mid_ca = self.ca2(mid_out)
        mid_la = self.la2(mid_out)
        mid_aff2 = mid_ca+mid_la
        mid_cla_w = self.aff3(mid_aff2)
        mid_cla = mid_ca*mid_cla_w+(1-mid_cla_w)*mid_la
        y_ft_w = self.aff4(mid_cla)

        # 中间变量
        y_t = torch.cat((x_f, mid_out), dim=1)
        y_t = self.conv1(y_t)
        y_f = torch.cat((x_t, mid_out), dim=1)
        y_f = self.conv2(y_f)

        # 第二阶段
        y_t = y_t.permute(0,3,2,1).contiguous().view(b*f,t,-1)
        y_t = self.tc2(y_t)
        y_t = y_t.view(b, f, t, -1).permute(0, 3, 2, 1).contiguous()
        y_t = self.norm3(y_t)
        y_f = y_f.permute(0,2,3,1).contiguous().view(b*t,f,-1)
        y_f = self.fc2(y_f)
        y_f = y_f.view(b,t,f,-1).permute(0,3,1,2).contiguous()
        y_f = self.norm4(y_f)

        output = mid_cla+y_ft_w*y_f+(1-y_ft_w)*y_t
        output = self.output(output)

        return output 


if __name__ == '__main__':
    a = torch.rand(2, 64, 257, 128)
    # model = CrossConformer_ori(64, 64)
    # model = CrossConformer_attn(64, 64, num_layers=2)
    model = ConnectConformer(64,64) 
    b = model(a)
    print(b.size())

