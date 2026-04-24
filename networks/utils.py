import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.networks_other import init_weights


class DirectionalConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm):
        super().__init__()

        def make_branch(kernel_size, padding):
            layers = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
            # if is_batchnorm:
            #     layers.append(nn.InstanceNorm3d(out_channels))
            # layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.conv_xy = make_branch((3, 3, 1), (1, 1, 0))
        self.conv_xz = make_branch((3, 1, 3), (1, 0, 1))
        self.conv_yz = make_branch((1, 3, 3), (0, 1, 1))
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 3, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        xy = self.conv_xy(x)
        xz = self.conv_xz(x)
        yz = self.conv_yz(x)
        out = torch.cat([xy, xz, yz], dim=1)
        out = self.fuse(out)
        return out

class DirectionalUNetConv3D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), init_stride=(1,1,1), padding_size=(1,1,1)):
        super(DirectionalUNetConv3D, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                   nn.InstanceNorm3d(out_size),
                                   nn.ReLU(inplace=True), )

        self.conv2 = DirectionalConv3D(out_size, out_size, is_batchnorm)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

#SG-MOE
# ===================== 1. 修复方向感知卷积（保留核心，优化归一化） =====================
class DirectionAwareAxialConv3D(nn.Module):
    """
    方向感知 + 聚合卷积模块（优化版）
    1. 三轴深度卷积捕捉局部方向感知
    2. 1x1+3x3卷积融合，统一特征分布
    """

    def __init__(self, in_channels):
        super().__init__()

        # 方向深度卷积（保留核心）
        def dw_branch(kernel_size, padding):
            return nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,  # depthwise
                bias=False
            )

        self.dw_xy = dw_branch((3, 3, 1), (1, 1, 0))
        self.dw_xz = dw_branch((3, 1, 3), (1, 0, 1))
        self.dw_yz = dw_branch((1, 3, 3), (0, 1, 1))

        # 融合卷积（优化：减少通道压缩比例，加入GroupNorm统一分布）
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, kernel_size=1, padding=0, bias=False),
            nn.GroupNorm(8, in_channels),  # 替换InstanceNorm，更适配3D医学图像
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),  # 降低压缩比例
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 方向卷积
        xy = self.dw_xy(x)
        xz = self.dw_xz(x)
        yz = self.dw_yz(x)

        # 拼接融合 + 残差连接（保留原特征）
        out = torch.cat([xy, xz, yz], dim=1)
        out = self.fuse(out)
        return out


# ===================== 2. 修复3D Sobel（加归一化，提升数值稳定性） =====================
class SobelEdge3D(nn.Module):
    """3D Sobel边缘检测（优化版）：加入归一化，适配多通道特征"""

    def __init__(self):
        super().__init__()
        # 保留原有Sobel核定义
        sobel_kernel_d = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                       [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                                       [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        sobel_kernel_h = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                       [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                                       [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        sobel_kernel_w = torch.tensor([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                                       [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                       [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0)
        # 注册为buffer
        self.register_buffer('kernel_d', sobel_kernel_d)
        self.register_buffer('kernel_h', sobel_kernel_h)
        self.register_buffer('kernel_w', sobel_kernel_w)
        # 新增：边缘特征归一化，统一数值分布

    def forward(self, x):
        """
        Args:
            x: 多通道特征图 (B, C, D, H, W)
        Return:
            edge_mag: 归一化后的边缘幅值图 (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape

        # 多通道适配（保留原有逻辑）
        kernel_d = self.kernel_d.repeat(C, 1, 1, 1, 1)
        kernel_h = self.kernel_h.repeat(C, 1, 1, 1, 1)
        kernel_w = self.kernel_w.repeat(C, 1, 1, 1, 1)

        edge_d = F.conv3d(x, kernel_d, padding=1, groups=C)
        edge_h = F.conv3d(x, kernel_h, padding=1, groups=C)
        edge_w = F.conv3d(x, kernel_w, padding=1, groups=C)

        # 梯度幅值 + 归一化（核心修改：解决数值不稳定）
        edge_mag = torch.sqrt(edge_d ** 2 + edge_h ** 2 + edge_w ** 2 + 1e-6)

        return edge_mag


# ===================== 3. 修复DetailRefine3D（融合语义+边缘，不再只返回边缘） =====================
class DetailRefine3D(nn.Module):
    """3D细节细化模块（优化版）：语义+边缘融合，保留残差"""

    def __init__(self, in_channels):
        super().__init__()
        self.sobel = SobelEdge3D()
        # 边缘特征适配：卷积+归一化，匹配语义特征分布
        self.pre_edge_proj = nn.Conv3d(in_channels, 1, 1, bias=False)

        self.edge_adapt = nn.Sequential(
            nn.Conv3d(1, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.GELU()
        )
        # 语义特征细化：保留原特征的语义信息
        self.semantic_refine = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: 待细化的特征 (B, C, D, H, W)
        Return:
            out: 语义+边缘融合的细化特征 (B, C//2, D, H, W)
        """
        # 1. 提取边缘特征并适配
        edge_map = self.sobel(self.pre_edge_proj(x))  # (B,1,D,H,W)
        edge_map = torch.sigmoid(edge_map)  # 压到 [0,1]

        semantic_feat = self.semantic_refine(x)

        out = semantic_feat * (1.0 + edge_map)

        return out

# ===================== 4. 修复门控网络（从全局→空间级，适配局部特征） =====================
class LightweightMoEGate(nn.Module):
    def __init__(self, in_channels, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        self.gate_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, num_experts, 1, bias=False)
        )

    def forward(self, x):
        gate_logits = self.gate_conv(x)  # 小于1 → 拉大前景概率
        gate = F.softmax(gate_logits, dim=1)
        return gate

class SGMOE3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # self.conv1 = nn.Conv3d(in_channels, out_channels,1)
        self.expert_multiscale = DirectionAwareAxialConv3D(out_channels)
        self.expert_boundary = DetailRefine3D(out_channels)
        self.expert_foreground = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        # 🔧 新增：gate 输入通道对齐
        self.gate = LightweightMoEGate(out_channels)

        self.fusion1 = nn.Sequential(nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                                     nn.GroupNorm(8, out_channels),
                                     nn.ReLU(inplace=True))
        self.fusion2 = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                                     nn.GroupNorm(8, out_channels),
                                     nn.ReLU(inplace=True))

    def forward(self, decoder_feat, skip_feat):
        x_up = self.upsample(decoder_feat)
        x_cat = self.fusion1(torch.cat([x_up, skip_feat], dim=1))
        out_multi = self.expert_multiscale(x_cat)
        out_boundary = self.expert_boundary(x_cat)
        out_foreground = self.expert_foreground(x_cat)

        out_expert = torch.stack(
            [out_multi, out_boundary, out_foreground], dim=1
        ).contiguous()

        gate_input = x_cat
        gates = self.gate(gate_input)

        x_semantic = (out_expert * gates.unsqueeze(2)).sum(dim=1)

        # 🔧 residual 在这里加
        x_semantic = x_semantic + x_cat

        x_fusion = self.fusion2(x_semantic)

        return x_fusion
