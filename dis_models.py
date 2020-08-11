import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
import torch
from module import *

class Discriminator_steg(nn.Module):
    def __init__(self, img_nc):
        super(Discriminator_steg, self).__init__()
        self.img_nc = img_nc
        self.Dis = nn.Sequential(
            HPFConv2d(self.img_nc, 6),
            ConvTanBlock(6, 8, abs=True),
            ConvTanBlock(8, 16),
            ConvReluBlock(16, 32),
            ConvReluBlock(32, 64),
            ConvReluBlock(64, 128, f_output=True),
            )
        self.FC = nn.Linear(in_features=128*16*16, out_features=2)

    def forward(self, input):
        x = self.Dis(input).view(input.size(0), -1)
        x = self.FC(x)
        return x
