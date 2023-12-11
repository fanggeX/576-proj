import torch
import torch.nn as nn
import torch.nn.functional as F

# unet
class Generator(nn.Module):
    """
    Generator for pixel2pixel GAN. It is a U-net.
    """
    def __init__(self, n_channels=1):
        super().__init__()
        self.down_conv1 = self.double_conv(n_channels, 32, 32) # 128x128
        self.down_conv2 = self.double_conv(32, 64, 64) # 64x64
        self.down_conv3 = self.double_conv(64, 128, 128) # 32x32
        self.down_conv4 = self.double_conv(128, 256, 256) # 16x16
        self.down_conv5 = self.double_conv(256, 256, 512) # 8x8
        self.bottle_neck_conv = self.double_conv(512, 1024, 1024) # 4x4

        self.up_trans5 = self.conv_trans(1024, 512) # 8x8
        self.up_conv5 = self.double_conv(1024, 512, 256)
        self.up_trans4 = self.conv_trans(256) # 16x16
        self.up_conv4 = self.double_conv(512, 256, 128)
        self.up_trans3 = self.conv_trans(128) # 32x32
        self.up_conv3 = self.double_conv(256, 128, 64)
        self.up_trans2 = self.conv_trans(64) # 64x64
        self.up_conv2 = self.double_conv(128, 64, 32)
        self.up_trans1 = self.conv_trans(32)
        self.up_conv1 = self.double_conv(64, 32, 32)

        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)


    def double_conv(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def conv_trans(self, in_channels, out_channels=None):
        if out_channels is None:
            out_channels = in_channels
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        x1 = self.down_conv1(x) #128x128
        x2 = self.down_conv2(F.max_pool2d(x1, 2)) # 64x64
        x3 = self.down_conv3(F.max_pool2d(x2, 2)) # 32x32
        x4 = self.down_conv4(F.max_pool2d(x3, 2)) # 16x16
        x5 = self.down_conv5(F.max_pool2d(x4, 2)) # 8x8
        x_bottle_neck = self.bottle_neck_conv(F.max_pool2d(x5, 2)) # 4x4

        up_x = self.up_trans5(x_bottle_neck) #8x8\
        up_x = self.up_conv5(torch.cat([up_x, x5], dim=1))
        up_x = self.up_trans4(up_x) # 16x16
        up_x = self.up_conv4(torch.cat([up_x, x4], dim=1))
        up_x = self.up_trans3(up_x) # 32x32
        up_x = self.up_conv3(torch.cat([up_x, x3], dim=1))
        up_x = self.up_trans2(up_x) # 64x64
        up_x = self.up_conv2(torch.cat([up_x, x2], dim=1))
        up_x = self.up_trans1(up_x) # 128x128
        up_x = self.up_conv1(torch.cat([up_x, x1], dim=1))
        return self.output_layer(up_x)
    
# This descriminator is based on the descriminator in this web page https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Discriminator(nn.Module):
    """
    Discriminator for pixel2pixel GAN. It is a conv net.
    """
    def __init__(self, n_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            # 128x128
            nn.Conv2d(n_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            # 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            # 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            # 4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator2(nn.Module):
    """
    Discriminator for pixel2pixel GAN. It is a conv net.
    """
    def __init__(self, n_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            # 128x128
            nn.Conv2d(n_channels, 32, 3, 1, 1, bias=False),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            # 32x32
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            # 16x16
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            # 8x8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.LeakyReLU(inplace=True),
            # 4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

