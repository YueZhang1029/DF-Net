# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.DWT_downsample import *
from networks.networks_other import init_weights
from networks.utils import DirectionalUNetConv3D, SGMOE3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Conv3d(dim * 3, dim, kernel_size=1)  # Fuse outputs from W, H, D
        # self.outconv = nn.Conv3d(dim, dim*2, kernel_size=1)
    def forward(self, x):
        B, C = x.shape[:2] # x (B C D H W)
        assert C == self.dim
        n_tokens = x.shape[2:].numel()

        x_d = x
        img_dim_d = x_d.shape[2:]
        x_h = x.permute(0, 1, 3, 4, 2)
        img_dim_h = x_h.shape[2:]
        x_w = x.permute(0, 1, 4, 2, 3)
        img_dim_w = x_w.shape[2:]
        # print(x_d.size(),x_h.size())
        x_d_flat = x_d.reshape(B, C, n_tokens).transpose(-1, -2)
        x_h_flat = x_h.reshape(B, C, n_tokens).transpose(-1, -2)
        x_w_flat = x_w.reshape(B, C, n_tokens).transpose(-1, -2)
        # print(x_w_flat.size(), x_d_flat.size())

        x_d_norm = self.norm(x_d_flat)
        x_h_norm = self.norm(x_h_flat)
        x_w_norm = self.norm(x_w_flat)

        x_d_mamba = self.mamba(x_d_norm)
        x_h_mamba = self.mamba(x_h_norm)
        x_w_mamba = self.mamba(x_w_norm)

        out_d = x_d_mamba.transpose(-1, -2).reshape(B, C, *img_dim_d)
        out_h = x_h_mamba.transpose(-1, -2).reshape(B, C, *img_dim_h)
        out_w = x_w_mamba.transpose(-1, -2).reshape(B, C, *img_dim_w)
        out = torch.cat([out_d, out_h, out_w], dim=1)  # Concatenate along channel axis
        out = self.proj(out)
        # out = self.outconv(out)
        return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
#调整维度顺序
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class UNetEncoderWithMamba(nn.Module):
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2):
        super(UNetEncoderWithMamba, self).__init__()

        self.LN = LayerNorm(in_channels, eps=1e-6, data_format="channels_first")
        self.mlps = MlpChannel(in_channels, 4 * in_channels)
        self.conv = nn.Conv3d(in_channels, in_channels*2, kernel_size=1)
        # MambaLayer 替代传统卷积层
        self.mamba_layer = MambaLayer(
            dim=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        x1 = self.LN(x)
        # 用 MambaLayer 替代卷积层
        out1 = self.mamba_layer(x1)
        out_res = x + out1
        out = self.mlps(self.LN(out_res)) + out_res

        return out

class DFF(nn.Module):
    """
    轻量 Gate 融合（LF, HF）; HF 可为 3C 通道
    """
    def  __init__(self, in_channels, reduction=8):
        """
        Args
        ----
        in_channels : C  (LF 通道数)
        hf_ratio    : HF 通道倍数, 3 => HF=(B,3C,···)
        reduction   : 通道压缩比 r
        """
        super().__init__()

        d = max(in_channels // reduction, 4)
        # ② Per-Channel Gate (SE)
        self.shared_conv = nn.Sequential(
            nn.Conv3d(in_channels*2, d, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 分支特定的处理
        self.se_avg = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(d, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.se_max = nn.Sequential(
            nn.AdaptiveMaxPool3d(1),
            nn.Conv3d(d, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.act = nn.GELU()

    def forward(self, lf, hf):

        # —— 生成通道权重 w ∈ (0,1)——
        stat = torch.cat([lf, hf], dim=1)  # (B,2C,1,1,1) after GAP
        stat = self.shared_conv(stat)
        w1 = self.se_avg(stat)
        w2 = self.se_max(stat)# (B,C,1,1,1)

        # —— 融合 + 残差 ——
        fused = lf * w1 + hf * w2  # (B,C,D,H,W)
        out = self.act(self.norm(fused))
        return out


class DFNet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(DFNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # dual frequency encoder
        #stem
        self.conv0 = nn.Conv3d(self.in_channels, filters[0], kernel_size=3, padding=1)
        #frequency split and downsampling
        self.wavepool1 = nn.Sequential(*[DWT(wavename='haar')])

        #dual path block 1
        self.inconv1 = nn.Conv3d(filters[0]*7, filters[0], kernel_size=(1, 1, 1))
        self.conv1 = DirectionalUNetConv3D(filters[0], filters[1], self.is_batchnorm)
        self.ma_inconv1 = nn.Conv3d(filters[0], filters[1], kernel_size=(1, 1, 1))
        self.ma1 = UNetEncoderWithMamba(filters[1])
        self.fusion1 = DFF(filters[1])

        self.wavepool2 = nn.Sequential(*[DWT(wavename='haar')])

        #dual path block 2
        self.inconv2 = nn.Conv3d(filters[1]*7, filters[1], kernel_size=(1, 1, 1))
        self.conv2 = DirectionalUNetConv3D(filters[1], filters[2], self.is_batchnorm)
        self.ma_inconv2 = nn.Conv3d(filters[1], filters[2], kernel_size=(1, 1, 1))
        self.ma2 = UNetEncoderWithMamba(filters[2])
        self.fusion2 = DFF(filters[2])

        self.wavepool3 = nn.Sequential(*[DWT(wavename='haar')])

        #dual path block 3
        self.inconv3 = nn.Conv3d(filters[2]*7, filters[2], kernel_size=(1, 1, 1))
        self.conv3 = DirectionalUNetConv3D(filters[2], filters[3], self.is_batchnorm)
        self.ma_inconv3 = nn.Conv3d(filters[2], filters[3], kernel_size=(1, 1, 1))
        self.ma3 = UNetEncoderWithMamba(filters[3])
        self.fusion3 = DFF(filters[3])

        self.wavepool4 = nn.Sequential(*[DWT(wavename='haar')])

        #dual path block 4
        self.inconv4 = nn.Conv3d(filters[3]*7, filters[3], kernel_size=(1, 1, 1))
        self.conv4 = DirectionalUNetConv3D(filters[3], filters[4], self.is_batchnorm)
        self.ma_inconv4 = nn.Conv3d(filters[3], filters[4], kernel_size=(1, 1, 1))
        self.ma4 = UNetEncoderWithMamba(filters[4])
        self.fusion4 = DFF(filters[4])
        
        self.decoder4 = SGMOE3D(filters[4], filters[3], filters[3])
        self.decoder3 = SGMOE3D(filters[3], filters[2], filters[2])
        self.decoder2 = SGMOE3D(filters[2], filters[1], filters[1])
        self.decoder1 = SGMOE3D(filters[1], filters[0], filters[0])

        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)


        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        #stem
        conv0 = self.conv0(inputs)
  
        #split1
        ls1, hs1 = self.wavepool1(conv0)
        ls1 = self.ma_inconv1(ls1)
        hs1 = self.inconv1(hs1)
        
        #encoder block 1
        lf1 = self.ma1(ls1)
        hf1 = self.conv1(hs1)
        df1 = self.fusion1(lf1, hf1)
      
        #split2
        ls2, hs2 = self.wavepool2(df1)
        ls2 = self.ma_inconv2(ls2)
        hs2 = self.inconv2(hs2)
        #encoder block 2
        lf2 = self.ma2(ls2)
        hf2 = self.conv2(hs2)
        df2 = self.fusion2(lf2, hf2)
      
        #split3
        ls3, hs3 = self.wavepool3(df2)
        ls3 = self.ma_inconv3(ls3)
        hs3 = self.inconv3(hs3)
        #encoder block 3
        lf3 = self.ma3(ls3)
        hf3 = self.conv3(hs3)
        df3 = self.fusion3(lf3, hf3)

        #split4
        ls4, hs4 = self.wavepool4(df3)
        ls4 = self.ma_inconv4(ls4)
        hs4 = self.inconv4(hs4)
        #encoder block 4
        lf4 = self.ma4(ls4)
        hf4 = self.conv4(hs4)
        df4 = self.fusion4(lf4, hf4)
        
        center = self.dropout1(df4)

        #decoder
        d4 = self.decoder4(center, df3)
        d3 = self.decoder3(d4, df2)
        d2 = self.decoder2(d3, df1)
        d1 = self.decoder1(d2, conv0)
  
        final = self.final(d1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    from torchinfo import summary
    model = DFNet(in_channels=1, n_classes=2)
    summary(model, (2, 1, 128, 128, 128), device='cuda')
    print(model)
    input = torch.randn(2, 1, 128, 128, 128).cuda()
    macs, params = profile(model.cuda(), (input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
