from collections import OrderedDict

import torch
import torch.nn as nn

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        y = x * self.tanh(self.softplus(x))
        return y

class MiniUnet(nn.Module):

    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 1, 
        init_features: int = 32, 
        ) -> None:

        super(MiniUnet, self).__init__()
        
        features = init_features
    
        self.encoder1 = MiniUnet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = MiniUnet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = MiniUnet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = MiniUnet._block(features * 4, features * 8, name="enc4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        
        self.decoder3 = MiniUnet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        
        self.decoder2 = MiniUnet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        
        self.decoder1 = MiniUnet._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(
        self, x: torch.tensor
        ) -> torch.tensor:

        enc1 = nn.Dropout(0.05)(self.encoder1(x))
        enc2 = nn.Dropout(0.05)(self.encoder2(self.pool1(enc1)))
        enc3 = nn.Dropout(0.05)(self.encoder3(self.pool2(enc2)))
        
        bottleneck = nn.Dropout(0.05)(self.bottleneck(self.pool3(enc3))) #check if this works, remove if not
 
        dec3 = nn.Dropout(0.05)(self.upconv3(bottleneck))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = nn.Dropout(0.05)(self.upconv2(dec3))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = nn.Dropout(0.05)(self.upconv1(dec2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
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
                    (name + "Mish2", Mish()),
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
                    (name + "Mish2", Mish()),
                ]
            )
        )
