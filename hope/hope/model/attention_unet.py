from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)   
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttentionUNet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, in_channels=3, out_channels=1, init_features = 32):
        super(AttentionUNet, self).__init__()

        features = init_features

        self.encoder1 = AttentionUNet._convblock(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = AttentionUNet._convblock(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = AttentionUNet._convblock(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = AttentionUNet._convblock(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = AttentionUNet._convblock(features * 8, features * 16, name="bottleneck")

        self.upconv4 = AttentionUNet._upconv(features * 16,features * 8, name="upconv3")
        self.decoder4 = AttentionUNet._convblock((features * 8) * 2, features * 8, name="dec4")
        self.att4 = Attention_block(features * 8, features * 8, features*4)

        self.upconv3 = AttentionUNet._upconv(features * 8, features * 4, name="upconv2")
        self.decoder3 = AttentionUNet._convblock(features * 8, features * 4, name="dec3")
        self.att3 = Attention_block(features * 4, features * 4, features*2)

        self.decoder2 = AttentionUNet._convblock(features*4, features * 2, name="dec2")
        self.upconv2 = AttentionUNet._upconv(features * 4, features*2, name = "upconv2")
        self.att2 = Attention_block(features * 2, features * 2, features)

        self.decoder1 = AttentionUNet._convblock(features * 2, features, name="dec1")
        self.upconv1 = AttentionUNet._upconv(features * 2, features, name = "upconv1")
        self.att1 = Attention_block(features, features, out_channels)


        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        
    @staticmethod
    def _convblock(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [   
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(inplace=True)),
                ]
            )
        )
    @staticmethod
    def _upconv(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [   
                    (
                    name,
                    nn.ConvTranspose2d(
                        in_channels = in_channels, 
                        out_channels = features, 
                        kernel_size=2, 
                        stride=2
                        )
                    )
                ]
            )
        )


    def forward(self, x):

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
    
        dec4 = self.upconv4(bottleneck)
        att4 = self.att4(g = enc4, x = dec4)
        dec4 = torch.cat((att4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        att3 = self.att3(g = enc3, x = dec3)
        dec3 = torch.cat((att3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        att2 = self.att2(g = enc2, x = dec2)
        dec2 = torch.cat((att2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        att1 = self.att1(g = dec1, x = enc1)
        dec1 = torch.cat((att1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = torch.sigmoid(self.conv(dec1))

        return out

if __name__ == "__main__":
    """
    (name + "norm", nn.BatchNorm2d(num_features=features)),
    (name + "relu", nn.LeakyReLU(inplace=True))

    """
    """x_prev = torch.rand(8,256,24,24)
    X = torch.rand(8,512,12,12)
    transpose = nn.ConvTranspose2d(512,256,kernel_size=2, stride = 2)
    #up = nn.Upsample(scale_factor = 2)
    #c = nn.Conv2d(in_channels = 512,out_channels = 256, kernel_size = 3,padding = 1, stride = 1)
    
    
    #c_x = c(up(X))
    x_up = transpose(X)
    #print(c_x.shape)
    attenion_block = Attention_block(256, 256,128)
    att = attenion_block(x_up, x_prev)"""
    X = torch.rand(8,3,192,192)
    model = AttentionUNet(3,1,32)
    y = model(X)
    print(y.shape)
