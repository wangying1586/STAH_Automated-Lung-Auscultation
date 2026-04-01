"""
HaB_StarNet.py - StarNet backbone with HarmonicBridge integration
Based on Microsoft's StarNet (2024) - "Rewrite the Stars"
Adapted for HaB (HarmonicBridge) preprocessing structure
"""

import sys
import torch
import torch.nn as nn
import torch.nn.init as init
import timm
import math

# Import your HarmonicBridge module
sys.path.append('/home/wangying/Lung_sound_detection')
from feature_extractor.HarmonicBridge import HarmonicBridge


class StarOperation(nn.Module):
    """
    Star Operation: Element-wise multiplication that maps input to high-dimensional space
    Core innovation of StarNet architecture
    """

    def __init__(self, in_channels, expand_ratio=4):
        super(StarOperation, self).__init__()
        hidden_dim = in_channels * expand_ratio

        # Two parallel paths for star operation
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # Star operation: element-wise multiplication of two paths
        star_out = self.path1(x) * self.path2(x)
        out = self.out_proj(star_out)
        return x + out  # Residual connection


class StarBlock(nn.Module):
    """
    Star Block: Core building block of StarNet
    Combines depthwise conv, star operation, and feed-forward
    """

    def __init__(self, in_channels, kernel_size=7, expand_ratio=4):
        super(StarBlock, self).__init__()

        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.norm1 = nn.BatchNorm2d(in_channels)

        # Star Operation
        self.star_op = StarOperation(in_channels, expand_ratio)

        # Feed Forward Network
        hidden_dim = in_channels * expand_ratio
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # Depthwise convolution
        x = self.norm1(self.dwconv(x))

        # Star operation
        x = self.star_op(x)

        # Feed forward
        x = x + self.ffn(x)

        return x


class StarNet(nn.Module):
    """
    StarNet backbone implementation
    4-stage hierarchical architecture with star operations
    """

    def __init__(self, embed_dim=64, depths=[2, 2, 6, 2], num_classes=1000):
        super(StarNet, self).__init__()

        # Patch embedding (stem)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, embed_dim, 4, stride=4, bias=False),
            nn.BatchNorm2d(embed_dim)
        )

        # 4-stage architecture
        self.stages = nn.ModuleList()
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        for i, (dim, depth) in enumerate(zip(dims, depths)):
            if i > 0:
                # Downsampling layer
                downsample = nn.Sequential(
                    nn.Conv2d(dims[i - 1], dim, 2, stride=2, bias=False),
                    nn.BatchNorm2d(dim)
                )
            else:
                downsample = nn.Identity()

            # Star blocks
            blocks = nn.Sequential(*[
                StarBlock(dim) for _ in range(depth)
            ])

            stage = nn.Sequential(downsample, blocks)
            self.stages.append(stage)

        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)

        for stage in self.stages:
            x = stage(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class CustomStarNet(nn.Module):
    """
    StarNet with HarmonicBridge preprocessing
    Combines HaB frequency domain processing with StarNet's efficient architecture
    """

    def __init__(self, embed_dim=64, depths=[2, 2, 6, 2], num_classes=4):
        super(CustomStarNet, self).__init__()

        # HarmonicBridge preprocessing layer
        self.custom_conv = HarmonicBridge(in_channels=1, out_channels=1)
        self._reinitialize_wtconv2d_weights()

        # StarNet backbone
        self.backbone = StarNet(embed_dim=embed_dim, depths=depths, num_classes=num_classes)

        print(f"CustomStarNet initialized with embed_dim={embed_dim}, depths={depths}")

    def _reinitialize_wtconv2d_weights(self):
        """Reinitialize weights in HarmonicBridge module"""
        # Initialize wavelet filters (keep as non-trainable)
        if hasattr(self.custom_conv, 'wt_filter'):
            self.custom_conv.wt_filter = nn.Parameter(self.custom_conv.wt_filter, requires_grad=False)
            self.custom_conv.iwt_filter = nn.Parameter(self.custom_conv.iwt_filter, requires_grad=False)

        # Initialize base convolution layer
        if hasattr(self.custom_conv, 'base_conv'):
            init.kaiming_normal_(self.custom_conv.base_conv.weight, mode='fan_out', nonlinearity='relu')
            if self.custom_conv.base_conv.bias is not None:
                self.custom_conv.base_conv.bias.data.zero_()

        # Initialize wavelet domain convolution layers
        if hasattr(self.custom_conv, 'wavelet_convs'):
            for wavelet_conv in self.custom_conv.wavelet_convs:
                init.kaiming_normal_(wavelet_conv.weight, mode='fan_out', nonlinearity='relu')

        # Initialize normalization layer
        if hasattr(self.custom_conv, 'gn'):
            if isinstance(self.custom_conv.gn, nn.GroupNorm):
                pass  # GroupNorm default initialization is sufficient
            else:
                init.ones_(self.custom_conv.gn.weight)
                init.zeros_(self.custom_conv.gn.bias)

    def forward(self, x):
        # First pass through HarmonicBridge preprocessing
        x = self.custom_conv(x)
        # Then through StarNet backbone
        return self.backbone(x)

    @classmethod
    def create_model(cls, variant='S4', num_classes=4):
        """
        Factory method to create different StarNet variants

        Args:
            variant: 'T0', 'T1', 'S1', 'S2', 'S3', 'S4', 'B1', 'B2', 'B3'
            num_classes: Number of output classes
        """
        configs = {
            'T0': {'embed_dim': 32, 'depths': [1, 1, 3, 1]},
            'T1': {'embed_dim': 32, 'depths': [1, 1, 4, 1]},
            'S1': {'embed_dim': 48, 'depths': [1, 1, 6, 1]},
            'S2': {'embed_dim': 64, 'depths': [1, 1, 8, 1]},
            'S3': {'embed_dim': 64, 'depths': [2, 2, 8, 2]},
            'S4': {'embed_dim': 64, 'depths': [2, 2, 12, 2]},  # Recommended for lung sound
            'B1': {'embed_dim': 96, 'depths': [2, 2, 12, 2]},
            'B2': {'embed_dim': 96, 'depths': [3, 3, 15, 3]},
            'B3': {'embed_dim': 128, 'depths': [3, 3, 15, 3]},
        }

        if variant not in configs:
            raise ValueError(f"Unknown StarNet variant: {variant}")

        config = configs[variant]
        return cls(num_classes=num_classes, **config)


# Convenience functions for different StarNet variants
def starnet_t0(num_classes=4):
    """StarNet-T0 - Ultra lightweight"""
    return CustomStarNet.create_model('T0', num_classes)


def starnet_s4(num_classes=4):
    """StarNet-S4 - Recommended for lung sound (best performance/efficiency balance)"""
    return CustomStarNet.create_model('S4', num_classes)


def starnet_b1(num_classes=4):
    """StarNet-B1 - Higher capacity"""
    return CustomStarNet.create_model('B1', num_classes)


if __name__ == '__main__':
    print("Testing CustomStarNet with HarmonicBridge...")

    # Test different variants
    variants = ['T0', 'S4', 'B1']
    num_classes = 4

    for variant in variants:
        print(f"\nTesting StarNet-{variant}:")

        # Create model
        model = CustomStarNet.create_model(variant, num_classes=num_classes)
        model.eval()

        # Test input (batch_size=2, channels=1, height=128, width=1001)
        input_data = torch.randn(2, 1, 128, 1001)

        with torch.no_grad():
            output = model(input_data)

        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Input shape: {input_data.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    print("\n✅ StarNet + HaB integration test completed!")
    print("📝 Recommended: Use StarNet-S4 for best performance/efficiency balance")
    print("🔧 Usage: Replace MobileViTV2 with starnet_s4() in your training script")