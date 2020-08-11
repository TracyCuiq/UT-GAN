#https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, input):
        output = torch.abs(input)
        return output

class LRelu(nn.Module):
    def __init__(self, alpha=0.1):
        super(LRelu, self).__init__()
        self.alpha=alpha

    def forward(self, input):
        output = F.relu_(input) - self.alpha*F.relu_(-input)
        return output

class DoubleConv(nn.Module):
    """docstring for DoubleConv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, relu=False, lrelu=False):
        super(DoubleConv, self).__init__()
        if relu:
            self.double_conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        elif lrelu:
            self.double_conv = nn.Sequential(
                LRelu(),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                LRelu(),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3, relu=False, lrelu=False):
        super(DownConv, self).__init__()

        if relu:
            self.down_conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        elif lrelu:
            self.down_conv = nn.Sequential(
                LRelu(),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.down_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, lrelu=True, inc=False):
        super(Down, self).__init__()

        if inc:
            self.conv_block = nn.Sequential(
                DownConv(in_channels, out_channels, lrelu=False, relu=False),
            )
        elif lrelu:
            self.conv_block = nn.Sequential(
                DownConv(in_channels, out_channels, lrelu=True),
            )
        else:
            self.conv_block = nn.Sequential(
                #nn.MaxPool2d(2),
                DownConv(in_channels, out_channels, relu=True),
            )

    def forward(self, x):
        return self.conv_block(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DoubleConv(in_channels, out_channels, relu=True),
                nn.BatchNorm2d(out_channels),
                )
        else:
            self.up = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels , out_channels//2, kernel_size=4, stride=2, padding=1,),
                nn.BatchNorm2d(out_channels//2),
                )
            
        self.dp_fg = dropout
        self.dp = nn.Dropout(p=0.5)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.dp_fg:
            x1 = self.dp(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x1, x2], dim=1)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, out_channels, relu=True),
            )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
    	x = self.up(x)
        x = self.sigmoid(x) - 0.5
        x = self.relu(x)
        return x

# *********************** high pass filters ***********************
HPF=np.zeros([6,1,5,5],dtype=np.float32)
HPF[0, 0, :, :] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[1, 0, :, :] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[2, 0, :, :] = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[3, 0, :, :] = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],dtype=np.float32)
HPF[4, 0, :, :] =np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],dtype=np.float32)
HPF[5, 0, :, :] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)
#HPF = np.transpose(HPF, axes=(3, 2, 0, 1))

class HPFConv2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=6):
        super(HPFConv2d, self).__init__()
        self.in_channels = in_channels  ###
        self.out_channels = out_channels  ###
        hpf_weight = nn.Parameter(torch.Tensor(HPF), requires_grad=False)
        self.hpf_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=5, padding=2, bias=False)
        self.hpf_conv.weight = hpf_weight

    def forward(self, input):
        return self.hpf_conv(input)

class ConvTanBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, pool_size=5, pool_stride=2, abs=False, set_paras=False):
        super(ConvTanBlock, self).__init__()

        if not abs:
            self.ConvTBabs = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(2,2)),
                nn.BatchNorm2d(out_channels),
                nn.Tanh(),
                nn.AvgPool2d((pool_size, pool_size), (pool_stride, pool_stride))
            )
        else:
        	self.ConvTBabs = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(2,2)),
                ABS(),
                nn.BatchNorm2d(out_channels),
                nn.Tanh(),
                nn.AvgPool2d((pool_size, pool_size), (pool_stride, pool_stride))
            )
        if not set_paras:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.ConvTBabs(input)

class ConvReluBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pool_size=5, pool_stride=2, set_paras=False, f_output=False):
        super(ConvReluBlock, self).__init__()

        if not f_output:
        	self.ConvRB = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(2,2)),#(1,1)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((pool_size, pool_size), (pool_stride, pool_stride)),
            )
        else:
        	self.ConvRB = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(2,2)),#(1,1)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            )
        
        if not set_paras:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.ConvRB(input)
