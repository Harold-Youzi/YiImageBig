"""
RRDBNet 架构 - RealESRGAN_x4plus 使用的网络结构
内嵌实现，无需依赖 basicsr 库
完全匹配 RealESRGAN_x4plus.pth 权重键名
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """残差密集块 (Residual Dense Block)"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDB 网络 - 完全匹配 RealESRGAN_x4plus 权重

    键名映射:
        conv_first  -> 浅层特征提取
        body.N      -> RRDB 块序列
        conv_body   -> 主干残差卷积
        conv_up1    -> 上采样卷积1 (2x)
        conv_up2    -> 上采样卷积2 (2x)
        conv_hr     -> 高分辨率特征
        conv_last   -> 输出层
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                 num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        num_upscale = int(math.log2(scale)) if scale > 1 else 0

        # 浅层特征提取
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # RRDB 主干
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 上采样 (每次2倍，共 num_upscale 次)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if num_upscale > 1:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 高分辨率特征
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 输出
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body = self.body(feat)
        body = self.conv_body(body)
        feat = feat + body

        # 上采样 (2x * num_upscale = scale)
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        if hasattr(self, 'conv_up2'):
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
