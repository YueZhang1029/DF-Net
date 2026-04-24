# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
import torch
import torch.nn as nn
from networks.DWT_IDWT.DWT_IDWT_layer import *
import torch.nn.functional as F

class DWT(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(DWT, self).__init__()
        self.dwt = DWT_3D(wavename = wavename)

    def forward(self, input):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(input)
        H_all = torch.cat([LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        return LLL, H_all