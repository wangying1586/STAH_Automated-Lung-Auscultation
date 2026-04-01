#!/usr/bin/env python3
"""
模型参数量和FLOPS计算脚本
支持计算HarmonicBridge、EfficientNet-B4以及两者结合后的复杂度
"""

import sys
import torch
import torch.nn as nn
from torchinfo import summary
from ptflops import get_model_complexity_info
import warnings

warnings.filterwarnings('ignore')

# 添加模块路径
sys.path.append('/home/wangying/Lung_sound_detection')

# 导入自定义模块
from feature_extractor.HarmonicBridge import HarmonicBridge
from feature_extractor.EfficientNet import CustomEfficientNetWithLoad, load_efficientnet_model
from efficientnet_pytorch import EfficientNet


def format_number(num):
    """格式化数字显示，转换为M、G等单位"""
    if num >= 1e9:
        return f"{num / 1e9:.2f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.0f}"


def calculate_model_complexity(model, input_shape=(1, 128, 1001), model_name="Model"):
    """
    计算模型的参数量和FLOPS

    Args:
        model: PyTorch模型
        input_shape: 输入形状 (channels, height, width)
        model_name: 模型名称

    Returns:
        dict: 包含参数量和FLOPS的字典
    """
    print(f"\n{'=' * 60}")
    print(f"分析模型: {model_name}")
    print(f"{'=' * 60}")

    # 设置模型为评估模式
    model.eval()

    # 方法1: 使用torchinfo计算详细信息
    try:
        print(f"\n📊 使用torchinfo分析:")
        print("-" * 40)

        # 创建输入张量
        batch_size = 1
        input_tensor = torch.randn(batch_size, *input_shape)

        # 获取模型摘要
        model_stats = summary(
            model,
            input_size=(batch_size, *input_shape),
            verbose=0,
            col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"]
        )

        total_params = model_stats.total_params
        trainable_params = model_stats.trainable_params
        total_mult_adds = model_stats.total_mult_adds

        print(f"总参数量: {format_number(total_params)} ({total_params:,})")
        print(f"可训练参数: {format_number(trainable_params)} ({trainable_params:,})")
        print(f"推理FLOPs: {format_number(total_mult_adds)} ({total_mult_adds:,})")

    except Exception as e:
        print(f"torchinfo分析失败: {e}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_mult_adds = 0

    # 方法2: 使用ptflops计算FLOPS
    try:
        print(f"\n🔥 使用ptflops分析:")
        print("-" * 40)

        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )

        # MACs (Multiply-Accumulate Operations) ≈ FLOPs/2
        flops_ptflops = 2 * macs

        print(f"参数量: {format_number(params)} ({params:,})")
        print(f"MACs: {format_number(macs)} ({macs:,})")
        print(f"FLOPs: {format_number(flops_ptflops)} ({flops_ptflops:,})")

        # 如果torchinfo失败，使用ptflops的结果
        if total_mult_adds == 0:
            total_mult_adds = flops_ptflops
            total_params = params

    except Exception as e:
        print(f"ptflops分析失败: {e}")

    # 手动计算参数量（作为备用）
    manual_params = sum(p.numel() for p in model.parameters())
    manual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✋ 手动计算:")
    print("-" * 40)
    print(f"总参数量: {format_number(manual_params)} ({manual_params:,})")
    print(f"可训练参数: {format_number(manual_trainable)} ({manual_trainable:,})")

    # 计算模型大小（MB）
    param_size = total_params * 4 / (1024 * 1024)  # 假设float32，4字节每参数

    print(f"\n💾 存储信息:")
    print("-" * 40)
    print(f"模型大小 (Float32): {param_size:.2f} MB")
    print(f"模型大小 (Float16): {param_size / 2:.2f} MB")

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': total_mult_adds,
        'model_size_mb': param_size,
        'model_name': model_name
    }


def compare_models_efficiency(results_list):
    """
    比较多个模型的效率

    Args:
        results_list: 模型分析结果列表
    """
    print(f"\n{'=' * 80}")
    print("🏆 模型效率对比分析")
    print(f"{'=' * 80}")

    # 创建对比表格
    print(f"{'模型名称':<25} {'参数量':<12} {'FLOPs':<12} {'模型大小':<12} {'效率比':<10}")
    print("-" * 80)

    # 找到基准模型（通常是最简单的模型）
    baseline = min(results_list, key=lambda x: x['total_params'])
    baseline_params = baseline['total_params']
    baseline_flops = baseline['flops']

    for result in results_list:
        name = result['model_name']
        params = result['total_params']
        flops = result['flops']
        size_mb = result['model_size_mb']

        # 计算相对于基准的倍数
        param_ratio = params / baseline_params if baseline_params > 0 else 0
        flops_ratio = flops / baseline_flops if baseline_flops > 0 else 0

        print(f"{name:<25} {format_number(params):<12} {format_number(flops):<12} "
              f"{size_mb:.1f}MB {param_ratio:.1f}x")

    print("-" * 80)

    # 效率分析
    print(f"\n📈 效率分析:")
    print("-" * 40)

    # 找出最轻量的模型
    lightest = min(results_list, key=lambda x: x['total_params'])
    print(f"🏃 最轻量模型: {lightest['model_name']} ({format_number(lightest['total_params'])})")

    # 找出计算量最小的模型
    if all(r['flops'] > 0 for r in results_list):
        least_flops = min(results_list, key=lambda x: x['flops'])
        print(f"⚡ 计算最少模型: {least_flops['model_name']} ({format_number(least_flops['flops'])} FLOPs)")

    # 计算HarmonicBridge带来的额外开销
    hab_model = next((r for r in results_list if 'HarmonicBridge' in r['model_name']), None)
    base_model = next(
        (r for r in results_list if 'EfficientNet-B4' in r['model_name'] and 'HarmonicBridge' not in r['model_name']),
        None)

    if hab_model and base_model:
        hab_overhead_params = hab_model['total_params'] - base_model['total_params']
        hab_overhead_flops = hab_model['flops'] - base_model['flops'] if hab_model['flops'] > 0 and base_model[
            'flops'] > 0 else 0

        print(f"\n🔍 HarmonicBridge额外开销:")
        print("-" * 40)
        print(f"额外参数量: {format_number(hab_overhead_params)} ({hab_overhead_params:,})")
        if hab_overhead_flops > 0:
            print(f"额外FLOPs: {format_number(hab_overhead_flops)} ({hab_overhead_flops:,})")

        param_increase = (hab_overhead_params / base_model['total_params']) * 100
        print(f"参数增加比例: {param_increase:.2f}%")


def test_individual_models():
    """测试各个独立模型"""
    print("🚀 开始模型复杂度分析...")

    results = []

    # 定义输入形状 (对于肺音数据)
    input_shape = (1, 128, 1001)  # (channels, height, width)

    # 1. 分析HarmonicBridge
    try:
        print("\n" + "🌉 分析 HarmonicBridge 模块".center(60, "="))
        hab_model = HarmonicBridge(in_channels=1, out_channels=1)
        hab_result = calculate_model_complexity(hab_model, input_shape, "HarmonicBridge")
        results.append(hab_result)
    except Exception as e:
        print(f"❌ HarmonicBridge分析失败: {e}")

    # 2. 分析原始EfficientNet-B4
    try:
        print("\n" + "🏗️ 分析 EfficientNet-B4 原始模型".center(60, "="))

        # 创建原始EfficientNet-B4（适配单通道输入）
        efficientnet_model = load_efficientnet_model(num_classes=4)
        efficientnet_result = calculate_model_complexity(
            efficientnet_model, input_shape, "EfficientNet-B4 (单通道适配)"
        )
        results.append(efficientnet_result)
    except Exception as e:
        print(f"❌ EfficientNet-B4分析失败: {e}")

    # 3. 分析结合后的模型
    try:
        print("\n" + "🤝 分析 HarmonicBridge + EfficientNet-B4 结合模型".center(60, "="))

        # 创建结合模型
        combined_model = CustomEfficientNetWithLoad.from_pretrained('efficientnet-b4', num_classes=4)
        combined_result = calculate_model_complexity(
            combined_model, input_shape, "HarmonicBridge + EfficientNet-B4"
        )
        results.append(combined_result)
    except Exception as e:
        print(f"❌ 结合模型分析失败: {e}")

    # 4. 对比分析
    if len(results) >= 2:
        compare_models_efficiency(results)

    return results


def test_forward_pass():
    """测试前向传播"""
    print(f"\n{'=' * 60}")
    print("🧪 测试模型前向传播")
    print(f"{'=' * 60}")

    # 创建测试数据
    batch_size = 2
    input_data = torch.randn(batch_size, 1, 128, 1001)
    print(f"输入数据形状: {input_data.shape}")

    models_to_test = {
        "HarmonicBridge": lambda: HarmonicBridge(in_channels=1, out_channels=1),
        "EfficientNet-B4": lambda: load_efficientnet_model(num_classes=4),
        "HaB + EfficientNet-B4": lambda: CustomEfficientNetWithLoad.from_pretrained('efficientnet-b4', num_classes=4)
    }

    for model_name, model_creator in models_to_test.items():
        try:
            print(f"\n测试 {model_name}...")
            model = model_creator()
            model.eval()

            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if torch.cuda.is_available() and start_time:
                    start_time.record()

                output = model(input_data)

                if torch.cuda.is_available() and end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    print(f"✅ {model_name}: {input_data.shape} -> {output.shape} (推理时间: {inference_time:.2f}ms)")
                else:
                    print(f"✅ {model_name}: {input_data.shape} -> {output.shape}")

        except Exception as e:
            print(f"❌ {model_name}测试失败: {e}")


def generate_efficiency_report(results):
    """生成效率报告"""
    print(f"\n{'=' * 80}")
    print("📋 模型效率总结报告")
    print(f"{'=' * 80}")

    # 为ICBHI肺音分类任务提供建议
    print(f"\n🎯 针对ICBHI肺音分类任务的建议:")
    print("-" * 50)

    for result in results:
        name = result['model_name']
        params = result['total_params']
        size_mb = result['model_size_mb']

        if 'HarmonicBridge' in name and 'EfficientNet' in name:
            print(f"🏆 {name}:")
            print(f"   - 适用场景: 追求最高精度的研究场景")
            print(f"   - 优势: 频域特征增强 + 强大的CNN backbone")
            print(f"   - 参数量: {format_number(params)} ({params:,})")
            print(f"   - 模型大小: {size_mb:.1f}MB")

            if params < 10e6:
                print(f"   - 部署难度: 🟢 容易 (适合移动端)")
            elif params < 50e6:
                print(f"   - 部署难度: 🟡 中等 (适合云端/边缘)")
            else:
                print(f"   - 部署难度: 🔴 困难 (需要高性能设备)")

        elif 'HarmonicBridge' in name:
            print(f"🌉 {name}:")
            print(f"   - 适用场景: 轻量级频域特征提取")
            print(f"   - 优势: 小巧灵活，可与任意backbone结合")
            print(f"   - 参数量: {format_number(params)} ({params:,})")
            print(f"   - 建议: 可作为预处理模块使用")

        elif 'EfficientNet' in name:
            print(f"🏗️ {name}:")
            print(f"   - 适用场景: 传统CNN分类基线")
            print(f"   - 优势: 成熟稳定，部署简单")
            print(f"   - 参数量: {format_number(params)} ({params:,})")
            print(f"   - 建议: 适合快速原型和对比实验")

        print()

    print(f"💡 总结建议:")
    print("-" * 50)
    print("• 研究场景: 推荐使用HarmonicBridge + EfficientNet-B4组合")
    print("• 生产环境: 根据计算资源选择合适的模型规模")
    print("• 移动端: 考虑使用更轻量的backbone替代EfficientNet-B4")
    print("• 实时推理: 重点关注FLOPs指标和实际推理时间")


def main():
    """主函数"""
    print("🔬 深度学习模型复杂度分析工具")
    print("=" * 60)
    print("分析目标:")
    print("- HarmonicBridge 频域特征提取模块")
    print("- EfficientNet-B4 分类模型")
    print("- HarmonicBridge + EfficientNet-B4 组合模型")
    print("=" * 60)

    # 分析模型复杂度
    results = test_individual_models()

    # 测试前向传播
    test_forward_pass()

    # 生成效率报告
    if results:
        generate_efficiency_report(results)

    print(f"\n🎉 分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()