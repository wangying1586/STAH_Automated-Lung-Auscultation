import sys

sys.path.append('/home/wangying/Lung_sound_detection')
import torch
import torch.nn as nn
from thop import profile

from feature_extractor.WTConv import WTConv2d, wavelet_transform, inverse_wavelet_transform


class WaveletTransformModule(nn.Module):
    def __init__(self, wt_filter):
        super().__init__()
        self.wt_filter = wt_filter

    def forward(self, x):
        return wavelet_transform(x, self.wt_filter)


class InverseWaveletTransformModule(nn.Module):
    def __init__(self, iwt_filter):
        super().__init__()
        self.iwt_filter = iwt_filter

    def forward(self, x):
        return inverse_wavelet_transform(x, self.iwt_filter)


def format_size(num):
    """将数字转换为人类可读的格式"""
    for unit in ['', 'K', 'M', 'G']:
        if abs(num) < 1000.0:
            return f"{num:.2f} {unit}"
        num /= 1000.0
    return f"{num:.2f} T"


def analyze_wtconv_complexity():
    """详细分析 WTConv 层的计算复杂度"""
    # 创建输入张量 [batch, channel, height, width]
    input_tensor = torch.randn(1, 1, 128, 1001)
    wtconv = WTConv2d(in_channels=1, out_channels=1)

    print("\nWTConv 层级分析结果:")
    print("=" * 50)

    computations = {}

    try:
        # 1. 分析小波变换
        wt_module = WaveletTransformModule(wtconv.wt_filter)
        macs_wt, params_wt = profile(wt_module, inputs=(input_tensor,))
        computations['Wavelet Transform'] = {'macs': macs_wt, 'params': params_wt}
        print(f"小波变换输出形状: {wt_module(input_tensor).shape}")

        # 2. 分析基础卷积
        macs_conv, params_conv = profile(wtconv.base_conv, inputs=(input_tensor,))
        computations['Base Convolution'] = {'macs': macs_conv, 'params': params_conv}

        # 3. 分析小波域卷积
        x_wt = wt_module(input_tensor)  # [batch, channel, 4, height/2, width/2]
        x_wt_reshaped = x_wt.view(x_wt.size(0), -1, x_wt.size(3), x_wt.size(4))  # [batch, channel*4, height/2, width/2]

        for i, conv in enumerate(wtconv.wavelet_convs):
            macs_wconv, params_wconv = profile(conv, inputs=(x_wt_reshaped,))
            computations[f'Wavelet Conv {i}'] = {'macs': macs_wconv, 'params': params_wconv}

        # 4. 分析逆小波变换
        iwt_module = InverseWaveletTransformModule(wtconv.iwt_filter)
        macs_iwt, params_iwt = profile(iwt_module, inputs=(x_wt,))
        computations['Inverse Wavelet Transform'] = {'macs': macs_iwt, 'params': params_iwt}

        # 5. 分析其他组件
        if hasattr(wtconv, 'gn'):
            macs_gn, params_gn = profile(wtconv.gn, inputs=(x_wt_reshaped,))
            computations['Group Normalization'] = {'macs': macs_gn, 'params': params_gn}

        if hasattr(wtconv, 'base_scale'):
            macs_scale, params_scale = profile(wtconv.base_scale, inputs=(input_tensor,))
            computations['Base Scale'] = {'macs': macs_scale, 'params': params_scale}

        # 打印各组件分析结果
        total_macs = 0
        total_params = 0

        print("\n各组件复杂度:")
        print("-" * 40)
        for layer_name, metrics in computations.items():
            macs = metrics['macs']
            params = metrics['params']
            total_macs += macs
            total_params += params

            print(f"\n{layer_name}:")
            print(f"MACs: {format_size(macs)}")
            print(f"Parameters: {format_size(params)}")

        print("\n总计:")
        print("-" * 40)
        print(f"Total MACs: {format_size(total_macs)}")
        print(f"Total Parameters: {format_size(total_params)}")

        # 打印整体模型信息
        print("\nWTConv 模型整体信息:")
        print("=" * 50)
        total_macs, total_params = profile(wtconv, inputs=(input_tensor,))
        print(f"Overall MACs: {format_size(total_macs)}")
        print(f"Overall Parameters: {format_size(total_params)}")

        # 打印模型参数分布
        print("\n参数分布:")
        print("-" * 40)
        for name, param in wtconv.named_parameters():
            print(f"{name}: {param.numel():,} parameters")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_wtconv_complexity()