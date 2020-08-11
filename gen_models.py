import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import torch
from module import *

class Generator_Unet(nn.Module):
    def __init__(self, input_nc=1, out_nc=1, bilinear=False):
        super(Generator_Unet, self).__init__()
        self.bil = bilinear
        self.input_nc = input_nc
        self.out_nc = out_nc
        factor = 2 if bilinear else 1

        self.inc = Down(self.input_nc, 16, inc=True)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 128)
        self.down5 = Down(128, 128)
        self.down6 = Down(128, 128)
        self.down7 = Down(128, 128)

        self.up1 = Up(128, 256//factor, self.bil, dropout=True)
        self.up2 = Up(256, 256//factor, self.bil, dropout=True)
        self.up3 = Up(256, 256//factor, self.bil, dropout=True)
        self.up4 = Up(256, 256//factor, self.bil)
        self.up5 = Up(256, 128//factor, self.bil)
        self.up6 = Up(128, 64//factor, self.bil)
        self.up7 = Up(64, 32//factor, self.bil)
        self.outc = OutConv(32, self.out_nc)

    def forward(self, input):
        x1 = self.inc(input)#(B, 16, 128, 128)
        x2 = self.down1(x1)#(B, 32, 64, 64)
        x3 = self.down2(x2)#(B, 64, 32, 32)
        x4 = self.down3(x3)#(B, 128, 16, 16)
        x5 = self.down4(x4)#(B, 128, 8, 8)
        x6 = self.down5(x5)#(B, 128, 4, 4)
        x7 = self.down6(x6)#(B, 128, 2, 2)
        x8 = self.down7(x7)#(B, 128, 1, 1)

        x = self.up1(x8, x7)#(B, 256, 2, 2)
        x = self.up2(x, x6)#(B, 256, 4, 4)
        x = self.up3(x, x5)#(B, 256, 8, 8)
        x = self.up4(x, x4)#(B, 256, 16, 16)
        x = self.up5(x, x3)#(B, 128, 32, 32)
        x = self.up6(x, x2)#(B, 64, 64, 64)
        x = self.up7(x, x1)#(B, 32, 128, 128)
        prob = self.outc(x)#(B, 1, 256, 256)
        return prob
