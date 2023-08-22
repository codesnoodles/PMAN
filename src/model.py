import torch 
from .encoder import *
from .decoder import *
from .neckblock import *
from .utils import *
from torchinfo import summary


class Model_connect(nn.Module):
    def __init__(self,num_channel=64,num_features=257):
        super(Model_connect,self).__init__()
        self.encoder = EncoderBlock(in_channels=3,out_channels=num_channel)
        self.neck = ConnectConformer(num_channel,num_channel,num_layers=4)
        self.decoder = ComplexDecoder()
        self.mask_decoder = MaskDecoder(num_features=num_features)

    def forward(self,x):
        mag = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        phase = torch.angle(torch.complex(x[:,0,:,:,],x[:,1,:,:])).unsqueeze(1)
        x = torch.cat([mag,x],dim=1)
        en_out = self.encoder(x)
        neck_out = self.neck(en_out)
        complex = self.decoder(neck_out)
        mag_mask = self.mask_decoder(neck_out)
        # mag = x[:,0,:,:]*mag_mask
        mag = mag*mag_mask

        # output 
        mag_real = mag*torch.cos(phase)
        mag_imag = mag*torch.sin(phase)
        final_real = mag_real+complex[:,0,:,:].unsqueeze(1)
        final_imag = mag_imag+complex[:,1,:,:].unsqueeze(1)

        return final_real,final_imag

        
# modle
class Model_ori(nn.Module):
    def __init__(self,num_channel=64,num_features=257):
        super(Model_ori,self).__init__()
        self.encoder = EncoderBlock(in_channels=3,out_channels=num_channel)
        self.neck = CrossConformer_ori(num_channel,num_channel)
        self.decoder = ComplexDecoder()
        self.mask_decoder = MaskDecoder(num_features=num_features)

    def forward(self,x):
        # mag calculate
        mag = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        phase = torch.angle(torch.complex(x[:,0,:,:,],x[:,1,:,:])).unsqueeze(1)
        x = torch.cat([mag,x],dim=1)
        # model input
        en_out = self.encoder(x)
        neck_out,neck_list = self.neck(en_out)
        complex = self.decoder(neck_out)
        mag_mask = self.mask_decoder(neck_out)
        mag = x[:,0,:,:]*mag_mask

        # output 
        mag_real = mag*torch.cos(phase)
        mag_imag = mag*torch.sin(phase)
        final_real = mag_real+complex[:,0,:,:].unsqueeze(1)
        final_imag = mag_imag+complex[:,1,:,:].unsqueeze(1)

        return final_real,final_imag 

class Model_attn(nn.Module):
    def __init__(self,num_channel=64,num_features=257):
        super(Model_attn,self).__init__()
        self.encoder = EncoderBlock(in_channels=3,out_channels=num_channel)
        self.neck = CrossConformer_attn(num_channel,num_channel)
        self.decoder = ComplexDecoder()
        self.mask_decoder = MaskDecoder(num_features=num_features)

    def forward(self,x):
        # mag calculate
        mag = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        phase = torch.angle(torch.complex(x[:,0,:,:,],x[:,1,:,:])).unsqueeze(1)
        x = torch.cat([mag,x],dim=1)
        # model input
        en_out = self.encoder(x)
        neck_out,neck_list = self.neck(en_out)
        complex = self.decoder(neck_out)
        mag_mask = self.mask_decoder(neck_out)
        mag = x[:,0,:,:]*mag_mask

        # output 
        mag_real = mag*torch.cos(phase)
        mag_imag = mag*torch.sin(phase)
        final_real = mag_real+complex[:,0,:,:].unsqueeze(1)
        final_imag = mag_imag+complex[:,1,:,:].unsqueeze(1)

        return final_real,final_imag 

class Model_big(nn.Module):
    def __init__(self,num_channel=64,num_features=257):
        super(Model_big,self).__init__()
        self.encoder = EncoderBlock(in_channels=3,out_channels=num_channel)
        self.neck = CrossConformer_big(num_channel,num_channel)
        self.decoder = ComplexDecoder()
        self.mask_decoder = MaskDecoder(num_features=num_features)

    def forward(self,x):
        # mag calculate
        mag = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        phase = torch.angle(torch.complex(x[:,0,:,:,],x[:,1,:,:])).unsqueeze(1)
        x = torch.cat([mag,x],dim=1)
        # model input
        en_out = self.encoder(x)
        neck_out = self.neck(en_out)
        complex = self.decoder(neck_out)
        mag_mask = self.mask_decoder(neck_out)
        # mag = x[:,0,:,:]*mag_mask
        mag = mag*mag_mask

        # output 
        mag_real = mag*torch.cos(phase)
        mag_imag = mag*torch.sin(phase)
        final_real = mag_real+complex[:,0,:,:].unsqueeze(1)
        final_imag = mag_imag+complex[:,1,:,:].unsqueeze(1)

        return final_real,final_imag 


# 并联的方式
class Model_pbigger(nn.Module):
    def __init__(self,num_channel=64,num_features=257):
        super(Model_pbigger,self).__init__()
        self.encoder = EncoderBlock(in_channels=3,out_channels=num_channel)
        self.neck1 = CrossConformer_big(num_channel,num_channel)
        self.neck2 = CrossConformer_big(num_channel,num_channel)
        self.decoder = ComplexDecoder()
        self.mask_decoder = MaskDecoder(num_features=num_features)

    def forward(self,x):
        # mag calculate
        mag = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        phase = torch.angle(torch.complex(x[:,0,:,:,],x[:,1,:,:])).unsqueeze(1)
        x = torch.cat([mag,x],dim=1)
        # model input
        en_out = self.encoder(x)
        mag_neck = self.neck1(en_out)
        complex_neck = self.neck2(en_out)
        complex = self.decoder(complex_neck)
        mag_mask = self.mask_decoder(mag_neck)
        # mag = x[:,0,:,:]*mag_mask
        mag = mag*mag_mask

        # output 
        mag_real = mag*torch.cos(phase)
        mag_imag = mag*torch.sin(phase)
        final_real = mag_real+complex[:,0,:,:].unsqueeze(1)
        final_imag = mag_imag+complex[:,1,:,:].unsqueeze(1)

        return final_real,final_imag 
# 并联方式，输出的特征进行交互
class Model_pbiggerc(nn.Module):
    def __init__(self,num_channel=64,num_features=257):
        super(Model_pbiggerc,self).__init__()
        self.encoder = EncoderBlock(in_channels=3,out_channels=num_channel)
        self.neck1 = CrossConformer_big(num_channel,num_channel)
        self.neck2 = CrossConformer_big(num_channel,num_channel)
        self.featur_fuse = AFF(channels=num_channel)
        self.decoder = ComplexDecoder()
        self.mask_decoder = MaskDecoder(num_features=num_features)

    def forward(self,x):
        # mag calculate
        mag = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        phase = torch.angle(torch.complex(x[:,0,:,:,],x[:,1,:,:])).unsqueeze(1)
        x = torch.cat([mag,x],dim=1)
        # model input
        en_out = self.encoder(x)
        mag_neck = self.neck1(en_out)
        complex_neck = self.neck2(en_out)
        # feature fuse
        aff_input = complex_neck+mag_neck
        decoder_w = self.featur_fuse(aff_input)
        mag_w = decoder_w*mag_neck
        complex_w = complex_neck*decoder_w
        decoder_input = mag_w+complex_w

        complex = self.decoder(decoder_input)
        mag_mask = self.mask_decoder(decoder_input)
        mag = mag*mag_mask

        # output 
        mag_real = mag*torch.cos(phase)
        mag_imag = mag*torch.sin(phase)
        final_real = mag_real+complex[:,0,:,:].unsqueeze(1)
        final_imag = mag_imag+complex[:,1,:,:].unsqueeze(1)

        return final_real,final_imag 
        
# 串联的方式
class Model_cbigger(nn.Module):
    def __init__(self,num_channel=64,num_features=257):
        super(Model_cbigger,self).__init__()
        self.encoder = EncoderBlock(in_channels=3,out_channels=num_channel)
        self.neck1 = CrossConformer_big(num_channel,num_channel)
        self.neck2 = CrossConformer_big(num_channel,num_channel)
        self.decoder = ComplexDecoder()
        self.mask_decoder = MaskDecoder(num_features=num_features)

    def forward(self,x):
        # mag calculate
        mag = torch.sqrt(x[:,0,:,:]**2+x[:,1,:,:]**2).unsqueeze(1)
        phase = torch.angle(torch.complex(x[:,0,:,:,],x[:,1,:,:])).unsqueeze(1)
        x = torch.cat([mag,x],dim=1)
        # model input
        en_out = self.encoder(x)
        neck_out = self.neck1(en_out)
        neck_out = self.neck2(neck_out)
        complex = self.decoder(neck_out)
        mag_mask = self.mask_decoder(neck_out)
        # mag = x[:,0,:,:]*mag_mask
        mag = mag*mag_mask

        # output 
        mag_real = mag*torch.cos(phase)
        mag_imag = mag*torch.sin(phase)
        final_real = mag_real+complex[:,0,:,:].unsqueeze(1)
        final_imag = mag_imag+complex[:,1,:,:].unsqueeze(1)

        return final_real,final_imag 





if __name__ == '__main__':
    a = torch.rand(3,2,251,257)
    # model = Model_ori()
    model = Model_big()
    summary(model,[(3,2,251,257)])
    # b = model(a)
    # print(b[1].size())

