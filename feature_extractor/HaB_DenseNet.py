import sys

sys.path.append('/home/wangying/Lung_sound_detection')
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import densenet121
from feature_extractor.HarmonicBridge import HarmonicBridge


class CustomDenseNet(nn.Module):
    def __init__(self, num_classes=None):
        super(CustomDenseNet, self).__init__()
        # 初始化 WTConv2d 预处理层
        self.custom_conv = HarmonicBridge(in_channels=1, out_channels=1)
        self._reinitialize_wtconv2d_weights()

        # 加载基础 DenseNet 模型
        self.base_model = densenet121(pretrained=True)

        # 调整第一层卷积的输入通道为1
        original_conv = self.base_model.features.conv0
        weight_data = original_conv.weight.data
        weight_data = weight_data[:, :1, :, :]  # 保留第一个通道的权重
        original_conv.weight.data = weight_data

        # 修改最后的分类层(如果指定了类别数)
        if num_classes is not None:
            num_ftrs = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(num_ftrs, num_classes)
            init.kaiming_normal_(self.base_model.classifier.weight, mode='fan_out', nonlinearity='relu')
            if self.base_model.classifier.bias is not None:
                self.base_model.classifier.bias.data.zero_()

    def _reinitialize_wtconv2d_weights(self):
        # 初始化 WTConv2d 的滤波器（保持为不可训练）
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
                pass
            else:
                init.ones_(self.custom_conv.gn.weight)
                init.zeros_(self.custom_conv.gn.bias)

    def forward(self, x):
        # 先通过 WTConv 预处理层
        x = self.custom_conv(x)
        # 再通过基础模型
        return self.base_model(x)


if __name__ == '__main__':
    # 测试代码
    num_classes = 2
    input_data = torch.randn(2, 1, 128, 1001)  # 模拟输入数据

    model = CustomDenseNet(num_classes=num_classes)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    print("DenseNet output shape:", output.shape)