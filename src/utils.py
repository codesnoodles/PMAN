import torch
import torch.nn as nn
import logging
import os
import numpy as np
import random
from pystoi import stoi
from pesq import pesq
from joblib import Parallel, delayed
from .attention import ChannelAttention, LocalpatchAttention, LocalAttention


# 语音频谱的幂律压缩,输入stft的结果，直接获取mag
def power_compress(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


# 这个的输入理论上是增强后的mag结合相位所生成的real和imag
def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**(1. / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)

    return torch.complex(real_compress, imag_compress)


# 特征压缩模块，输入三个张量，打算的是一个是at的输出，一个是af的输出，一个是经过ca和la之后的输出
class FeatureFuse(nn.Module):

    def __init__(self):
        super(FeatureFuse, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, at, af, weight):
        w = self.sigmoid(weight)
        d_w = 1 - w
        output = w * at + d_w * af
        return output


class AFF(nn.Module):
    """
    权重生成模块
    """

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        mid_channels = int(channels // r)
        self.local_attn = nn.Sequential(
            nn.Conv2d(channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      channels,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.BatchNorm2d(channels))
        self.global_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels,
                      mid_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      channels,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.BatchNorm2d(channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_attn(x)
        xg = self.global_attn(x)
        xlg = xl + xg
        out = self.sigmoid(xlg)

        return out


def count_paramerters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# log
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def Normalization(clean, noisy):
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
    noisy, clean = torch.transpose(noisy * c, 0,
                                   1), torch.transpose(clean * c, 0, 1)

    return clean, noisy


def seed_init(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type:ignore
    torch.backends.cudnn.benchmark = False  # type:ignore
    os.environ['PYTHONHASHSEED'] = str(seed)


# get scores
def get_scores(clean,est,sr=16000):
    est = est.numpy()
    clean = clean.numpy()

    pesq_i = get_pesq(clean,est,sr=sr)
    stoi_i = get_stoi(clean,est,sr=sr)

    return pesq_i, stoi_i

def get_pesq(ref_sig,out_sig,sr):
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr,ref_sig[i],out_sig[i],'wb')
    return pesq_val

def get_stoi(ref_sig,out_sig,sr):
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    return stoi_val

    




if __name__ == '__main__':
    a = torch.rand(2, 16, 122, 257)
    b = torch.rand(2, 16, 122, 257)
    # c = torch.rand(2, 16, 122, 257)
    # clean = torch.rand(2, 16000)
    # noisy = torch.rand(2, 16000)
    # clean, noisy = Normalization(clean, noisy)
    # print(clean.size())
    # print(noisy.size())
    ca_layer = ChannelAttention(16)
    la_layer = LocalAttention(16)
    ca = ca_layer(a)
    la = la_layer(a)
    aff_in = ca + la
    aff_layer = AFF(16, r=2)
    aff_out = aff_layer(aff_in)
    print(aff_out.size())
    ca = ca * aff_out
    la = la * (1 - aff_out)
    out = ca + la
    print('out', out.size())
