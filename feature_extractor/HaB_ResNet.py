import sys

sys.path.append('/home/wangying/Lung_sound_detection')
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet50
from feature_extractor.HarmonicBridge import HarmonicBridge


class CustomResNet(nn.Module):
    def __init__(self, num_classes=None):
        super(CustomResNet, self).__init__()
        # 初始化WTConv2d模块
        self.custom_conv = HarmonicBridge(in_channels=1, out_channels=1)

        # 加载基础ResNet模型
        self.base_model = resnet50(pretrained=True)

        # 保存原始的第一层卷积
        self.original_conv = self.base_model.conv1

        # 调整原始卷积层的输入通道
        weight_data = self.original_conv.weight.data
        original_in_channels = weight_data.shape[1]
        new_in_channels = 1

        # 调整权重维度
        if new_in_channels < original_in_channels:
            weight_data = weight_data[:, :new_in_channels, :, :]
        elif new_in_channels > original_in_channels:
            additional_channels_weight = torch.randn(
                weight_data.shape[0],
                new_in_channels - original_in_channels,
                weight_data.shape[2],
                weight_data.shape[3]
            )
            weight_data = torch.cat([weight_data, additional_channels_weight], dim=1)

        # 更新原始卷积层的权重
        self.original_conv.weight.data = weight_data

        # 调整偏置项（如果存在）
        if self.original_conv.bias is not None:
            bias_data = self.original_conv.bias.data
            if new_in_channels < original_in_channels:
                bias_data = bias_data[:new_in_channels]
            elif new_in_channels > original_in_channels:
                additional_bias = torch.zeros(new_in_channels - original_in_channels)
                bias_data = torch.cat([bias_data, additional_bias], dim=0)
            self.original_conv.bias.data = bias_data

        # 将custom_conv和original_conv组合成新的第一层
        self.base_model.conv1 = nn.Sequential(
            self.custom_conv,
            self.original_conv
        )

        # 初始化WTConv2d的权重
        self._reinitialize_wtconv2d_weights()

        # 修改最后的分类层（如果指定了类别数）
        if num_classes is not None:
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_classes)
            init.kaiming_normal_(self.base_model.fc.weight, mode='fan_out', nonlinearity='relu')
            if self.base_model.fc.bias is not None:
                self.base_model.fc.bias.data.zero_()

    def _reinitialize_wtconv2d_weights(self):
        """重新初始化WTConv2d模块里的所有可学习权重参数"""
        # 初始化WTConv2d的滤波器（固定参数）
        if hasattr(self.custom_conv, 'wt_filter'):
            self.custom_conv.wt_filter = nn.Parameter(self.custom_conv.wt_filter, requires_grad=False)
            self.custom_conv.iwt_filter = nn.Parameter(self.custom_conv.iwt_filter, requires_grad=False)

        # 初始化基础卷积层
        if hasattr(self.custom_conv, 'base_conv'):
            init.kaiming_normal_(self.custom_conv.base_conv.weight, mode='fan_out', nonlinearity='relu')
            if self.custom_conv.base_conv.bias is not None:
                self.custom_conv.base_conv.bias.data.zero_()

        # 初始化基础缩放模块
        if hasattr(self.custom_conv, 'base_scale'):
            init.ones_(self.custom_conv.base_scale.weight)

        # 初始化小波域卷积层
        if hasattr(self.custom_conv, 'wavelet_convs'):
            for wavelet_conv in self.custom_conv.wavelet_convs:
                init.kaiming_normal_(wavelet_conv.weight, mode='fan_out', nonlinearity='relu')

        # 初始化小波域缩放模块
        if hasattr(self.custom_conv, 'wavelet_scale'):
            for wavelet_scale in self.custom_conv.wavelet_scale:
                init.ones_(wavelet_scale.weight)

        # 初始化归一化层
        if hasattr(self.custom_conv, 'gn'):
            if isinstance(self.custom_conv.gn, nn.GroupNorm):
                pass  # GroupNorm默认初始化已满足需求
            else:
                init.ones_(self.custom_conv.gn.weight)
                init.zeros_(self.custom_conv.gn.bias)

    def forward(self, x):
        return self.base_model(x)


if __name__ == '__main__':
    # 测试代码
    num_classes = 2
    input_data = torch.randn(2, 1, 128, 1001)  # 模拟输入数据

    model = CustomResNet(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    print("ResNet output shape:", output.shape)