import torch
import torch.nn as nn
from .conv_modules import *


# mask deocder
class MaskDecoder(nn.Module):
    def __init__(self,num_features,num_channel=64,out_channel=1):
        """
        num_features: 频点数
        """
        super(MaskDecoder,self).__init__()
        self.dense = DenseBlock(in_channels=num_channel)
        self.sub = SPConvTranspose2d(num_channel,num_channel,(1,3),2)
        self.conv1 = BasicConv(num_channel,out_channel,(1,2))
        self.conv2 = nn.Conv2d(out_channel,out_channel,(1,1))
        self.prelu = nn.PReLU(num_features,init=-0.25)

    def forward(self,x):
        x = self.dense(x)
        x = self.sub(x)
        x = self.conv1(x)
        x = self.conv2(x).permute(0,3,2,1).squeeze(-1)
        x = self.prelu(x)
        return x.permute(0,2,1).unsqueeze(1)

class MaskDecoder_dep(nn.Module):
    def __init__(self,num_features,num_channel=64,out_channel=1):
        super(MaskDecoder_dep,self).__init__()
        self.dense = DenseBlock_depth(in_channels=num_channel,groups=num_channel)
        self.sub = SPConvTranspose2d(num_channel,num_channel,(1,3),2)
        self.conv1 = BasicConv(num_channel,out_channel,(1,2))
        self.conv2 = nn.Conv2d(out_channel,out_channel,(1,1))
        self.prelu = nn.PReLU(num_features,init=-0.25)
    def forward(self,x):
        x = self.dense(x)
        x = self.sub(x)
        x = self.conv1(x)
        x = self.conv2(x).permute(0,3,2,1).squeeze(-1)
        x = self.prelu(x)
        return x.permute(0,2,1).unsqueeze(1)




# complex decoder
class ComplexDecoder(nn.Module):
    def __init__(self,num_channel=64):
        super(ComplexDecoder,self).__init__()
        self.dense = DenseBlock(in_channels=num_channel)
        self.sub = SPConvTranspose2d(num_channel,num_channel,(1,3),2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel,affine=True)
        self.conv = nn.Conv2d(num_channel,2,(1,2))

    def forward(self,x):
        x = self.dense(x)
        x = self.sub(x)
        x = self.prelu(x)
        x = self.conv(x)

        return x

class ComplexDecoder_dep(nn.Module):
    def __init__(self,num_channel=64):
        super(ComplexDecoder_dep,self).__init__()
        self.dense = DenseBlock_depth(in_channels=num_channel,groups=num_channel)
        self.sub = SPConvTranspose2d(num_channel,num_channel,(1,3),2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel,affine=True)
        self.conv = nn.Conv2d(num_channel,2,(1,2))

    def forward(self,x):
        x = self.dense(x)
        x = self.sub(x)
        x = self.prelu(x)
        x = self.conv(x)

        return x



if __name__ == '__main__':
    a = torch.rand(4,64,122,129)
    # model = MaskDecoder_dep(257)
    # model = ComplexDecoder()
    model = ComplexDecoder_dep()
    b = model(a)
    print(b.size())

