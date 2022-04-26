from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 1, 
        init_features: int = 32, 
        ) -> None:

        super(UNet, self).__init__()
        
        features = init_features
        """self.num_layer = num_layer

        if dropout_mode == "normal":
            if dropout_probability == None or dropout_probability > 1.0 or dropout_probability <= 0.0:
                raise RuntimeError(f"The probability is {dropout_probability}, it has to be in [0,1] interval")
            else:
                self.dropout = nn.Dropout(dropout_probability)
                print(f"Using dropout with probability {dropout_probability}")
        elif dropout_mode == "monte_carlo":
            "future implementation"
            pass 
        else:
            self.dropout = lambda x : x"""
    
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(
        self, x: torch.tensor
        ) -> torch.tensor:

        enc1 = nn.Dropout(0.05)(self.encoder1(x))
        enc2 = nn.Dropout(0.05)(self.encoder2(self.pool1(enc1)))
        enc3 = nn.Dropout(0.05)(self.encoder3(self.pool2(enc2)))
        enc4 = nn.Dropout(0.05)(self.encoder4(self.pool3(enc3)))
        
        bottleneck = nn.Dropout(0.05)(self.bottleneck(self.pool4(enc4))) #check if this works, remove if not

        dec4 = nn.Dropout(0.05)(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
 
        dec3 = nn.Dropout(0.05)(self.upconv3(dec4))
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
                    (name + "relu1", nn.ReLU(inplace=True)),
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
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def Unet(
    in_channels=3, 
    out_channels=1, 
    init_features=32, 
    pretrained_model_torch_model = False) -> UNet:
    
    if pretrained_model_torch_model == False:
        print("Not using pretrained unet")
        return UNet(in_channels, out_channels, init_features)
    else:
        print("Using pretrained unet")
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
        return model


if __name__ == "__main__":
    unet = UNet(3,1,32)

    X = torch.rand(8,3,192,192)
    unet = UNet(3,1,32)
    y = unet(X)
    print(y.shape)
    

    """
        encoder_layer = []
        encoder_layer.append(UNet._block(in_channels, features, name="enc1"))
        encoder_layer.append(dropout)
        encoder_layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        for layer_number_encoder in range(1, self.num_layer - 1, 1):
            encoder_layer.append(
                    UNet._block(features*2**(layer_number_encoder), 
                    features*2**(layer_number_encoder + 1), 
                    name = "enc" + str(layer_number_encoder)
                )
            )
            encoder_layer.append(dropout)
            encoder_layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.encoder_layer = encoder_layer
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        
        decoder_layer = []
        for layer_number_decoder in range(self.num_layer, 0, -1):
            decoder_layer.append(
                nn.ConvTranspose2d(
                    features*2**(layer_number_decoder), 
                    features*2**(layer_number_decoder - 1), 
                    kernel_size=2, 
                    stride=2
                    )
                )
            decoder_layer.append(dropout)
            decoder_layer.append(
                UNet._block(
                    features*2**(layer_number_decoder), 
                    features*2**(layer_number_decoder - 1),
                    name = "dec" + str(layer_number_decoder)
                ),
            )
        self.decoder_layer = decoder_layer
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )"""
