# frequency_domain_adaptation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift


class FACTAdaptation(nn.Module):
    """Fourier-based Cross-domain Attention"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(FACTAdaptation, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # 频域注意力机制
        self.frequency_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

        # 跨域对齐模块
        self.cross_domain_align = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x, style_feat=None):
        # x: 输入特征 [B, C, H, W]
        B, C, H, W = x.shape

        # 傅里叶变换
        x_fft = fft2(x, dim=(-2, -1))
        x_amp = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        # 频域注意力
        freq_attn = self.frequency_attention(x_amp.unsqueeze(-1).unsqueeze(-1))
        freq_attn = F.interpolate(freq_attn, size=(H, W), mode='bilinear')

        # 应用频域注意力
        enhanced_amp = x_amp * freq_attn.squeeze()

        if style_feat is not None:
            # 如果有风格特征，进行跨域对齐
            style_fft = fft2(style_feat, dim=(-2, -1))
            style_amp = torch.abs(style_fft)

            # 频域特征融合
            fused_amp = torch.cat([enhanced_amp, style_amp], dim=1)
            aligned_amp = self.cross_domain_align(fused_amp)
        else:
            aligned_amp = enhanced_amp

        # 反傅里叶变换
        aligned_fft = aligned_amp * torch.exp(1j * x_phase)
        output = ifft2(aligned_fft, dim=(-2, -1)).real

        return output


class FDAAdaptation(nn.Module):
    """Fourier Domain Adaptation - 低频振幅交换"""

    def __init__(self, beta=0.01):
        super(FDAAdaptation, self).__init__()
        self.beta = beta  # 低频分量比例

    def low_freq_mutate(self, amp_src, amp_trg, beta=0.01):
        """交换低频振幅分量"""
        B, C, H, W = amp_src.shape

        # 计算低频掩码
        h_crop = int(H * beta)
        w_crop = int(W * beta)

        h_start = (H - h_crop) // 2
        w_start = (W - w_crop) // 2

        # 创建掩码
        mask = torch.zeros((H, W), device=amp_src.device)
        mask[h_start:h_start + h_crop, w_start:w_start + w_crop] = 1

        # 交换低频分量
        amp_src_mixed = amp_src.clone()
        amp_src_mixed[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
            amp_trg[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop]

        return amp_src_mixed

    def forward(self, x_src, x_trg=None):
        if x_trg is None:
            return x_src

        # 傅里叶变换
        fft_src = fft2(x_src, dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

        fft_trg = fft2(x_trg, dim=(-2, -1))
        amp_trg = torch.abs(fft_trg)

        # 交换低频振幅
        amp_mixed = self.low_freq_mutate(amp_src, amp_trg, self.beta)

        # 反傅里叶变换
        fft_mixed = amp_mixed * torch.exp(1j * pha_src)
        mixed = ifft2(fft_mixed, dim=(-2, -1)).real

        return mixed


class FMMAdaptation(nn.Module):
    """Fourier Moment Matching"""

    def __init__(self, momentum=0.1):
        super(FMMAdaptation, self).__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', None)
        self.register_buffer('running_cov', None)

    def update_stats(self, x):
        """更新运行统计量"""
        B, C, H, W = x.shape

        # 傅里叶变换
        x_fft = fft2(x, dim=(-2, -1))
        x_amp = torch.abs(x_fft)

        # 计算当前批次的统计量
        current_mean = x_amp.mean(dim=0, keepdim=True)
        x_centered = x_amp - current_mean
        current_cov = torch.bmm(x_centered.view(B, -1, 1),
                                x_centered.view(B, 1, -1)).mean(dim=0)

        # 更新运行统计量
        if self.running_mean is None:
            self.running_mean = current_mean
            self.running_cov = current_cov
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * current_mean
            self.running_cov = (1 - self.momentum) * self.running_cov + \
                               self.momentum * current_cov

    def whitening_transform(self, x, epsilon=1e-5):
        """白化变换"""
        B, C, H, W = x.shape

        # 特征值分解
        L, V = torch.linalg.eigh(self.running_cov)
        L = torch.clamp(L, min=epsilon)

        # 白化矩阵
        whitening_matrix = V @ torch.diag(1.0 / torch.sqrt(L)) @ V.t()

        # 应用白化
        x_flat = x.view(B, -1) - self.running_mean.view(1, -1)
        x_white = torch.mm(x_flat, whitening_matrix)

        return x_white.view(B, C, H, W)

    def coloring_transform(self, x_white, target_stats):
        """着色变换到目标分布"""
        target_mean, target_cov = target_stats

        # 特征值分解
        L, V = torch.linalg.eigh(target_cov)
        L = torch.clamp(L, min=1e-5)

        # 着色矩阵
        coloring_matrix = V @ torch.diag(torch.sqrt(L)) @ V.t()

        # 应用着色
        B, C, H, W = x_white.shape
        x_colored = torch.mm(x_white.view(B, -1), coloring_matrix)
        x_colored = x_colored + target_mean.view(1, -1)

        return x_colored.view(B, C, H, W)

    def forward(self, x, target_stats=None):
        # 更新源域统计量
        self.update_stats(x)

        if target_stats is None:
            return x

        # 傅里叶变换
        x_fft = fft2(x, dim=(-2, -1))
        x_amp, x_phase = torch.abs(x_fft), torch.angle(x_fft)

        # 白化
        x_white = self.whitening_transform(x_amp)

        # 着色到目标分布
        x_colored = self.coloring_transform(x_white, target_stats)

        # 保持相位不变，只调整振幅
        aligned_fft = x_colored * torch.exp(1j * x_phase)
        output = ifft2(aligned_fft, dim=(-2, -1)).real

        return output


class FSAAdaptation(nn.Module):
    """Fourier Self-Adaptation - 源数据无关"""

    def __init__(self, in_channels, init_weight=1.0, init_bias=0.0):
        super(FSAAdaptation, self).__init__()
        self.in_channels = in_channels

        # 可学习的自适应图像
        self.adaptive_image = nn.Parameter(
            torch.randn(1, in_channels, 224, 224) * 0.02
        )

        # 可学习的权重和偏置
        self.weight = nn.Parameter(torch.ones(1, in_channels, 1, 1) * init_weight)
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1) * init_bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # 调整自适应图像尺寸
        adaptive_img = F.interpolate(self.adaptive_image, size=(H, W),
                                     mode='bilinear', align_corners=False)
        adaptive_img = adaptive_img.expand(B, -1, -1, -1)

        # 傅里叶变换
        x_fft = fft2(x, dim=(-2, -1))
        x_amp, x_phase = torch.abs(x_fft), torch.angle(x_fft)

        adaptive_fft = fft2(adaptive_img, dim=(-2, -1))
        adaptive_amp = torch.abs(adaptive_fft)

        # 振幅插值
        mixed_amp = self.weight * x_amp + self.bias * adaptive_amp

        # 反傅里叶变换
        mixed_fft = mixed_amp * torch.exp(1j * x_phase)
        output = ifft2(mixed_fft, dim=(-2, -1)).real

        return output


class NoAdaptation(nn.Module):
    """无域自适应 - 基线"""

    def __init__(self):
        super(NoAdaptation, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x