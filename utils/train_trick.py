# ===== 1. 更高效的学习率调度器 =====
# 可以替换原来的WarmupCosineAnnealingLR
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    结合了预热期、余弦退火和重启策略的学习率调度器
    """

    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1,
                 min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps  # 第一个周期的总步数
        self.cycle_mult = cycle_mult  # 周期乘数
        self.base_max_lr = max_lr  # 基础最大学习率
        self.max_lr = max_lr  # 当前最大学习率
        self.min_lr = min_lr  # 最小学习率
        self.warmup_steps = warmup_steps  # 预热步数
        self.gamma = gamma  # 学习率衰减率
        self.cur_cycle_steps = first_cycle_steps  # 当前周期步数
        self.cycle = 0  # 当前是第几个周期
        self.step_in_cycle = last_epoch  # 当前周期中的步数

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # 设置学习率
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # 线性预热
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            # 余弦退火
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            return [base_lr + (self.max_lr - base_lr) * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in
                    self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1

            # 检查是否需要进入下一个周期
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr = self.max_lr * self.gamma
        else:
            # 调整当前所在的周期和步数
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    # 计算当前所在的周期和步数（更复杂）
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** self.cycle
            else:
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# ===== 2. 梯度累积和梯度裁剪实现 =====
def train_with_gradient_accumulation(model, inputs, labels, criterion, optimizer,
                                     scaler, accumulation_steps, clip_grad_norm=1.0, use_amp=True):
    """
    使用梯度累积和梯度裁剪的训练步骤
    """
    # 梯度累积的批次大小
    micro_batch_size = len(inputs) // accumulation_steps
    # 确保梯度为空
    optimizer.zero_grad(set_to_none=True)

    # 梯度累积循环
    for i in range(accumulation_steps):
        # 计算当前micro-batch的索引
        start_idx = i * micro_batch_size
        end_idx = start_idx + micro_batch_size
        # 如果是最后一个batch，可能大小不同
        if i == accumulation_steps - 1:
            end_idx = len(inputs)

        micro_inputs = inputs[start_idx:end_idx]
        micro_labels = labels[start_idx:end_idx]

        # 前向传播（带混合精度）
        if use_amp:
            with autocast():
                outputs = model(micro_inputs)
                loss = criterion(outputs, micro_labels) / accumulation_steps

            # 反向传播
            scaler.scale(loss).backward()
        else:
            outputs = model(micro_inputs)
            loss = criterion(outputs, micro_labels) / accumulation_steps
            loss.backward()

    # 梯度裁剪和优化器步骤
    if use_amp:
        # 混合精度下的梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        # 标准精度下的梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

    return outputs, loss * accumulation_steps  # 返回最后一个micro-batch的输出和缩放后的损失


# ===== 3. 学习率预热和衰减函数 =====
def get_lr_with_warmup_and_decay(base_lr, step, warmup_steps, decay_steps,
                                 decay_rate=0.1, staircase=False, min_lr=0.0):
    """
    获取带预热和衰减的学习率

    Args:
        base_lr: 基础学习率
        step: 当前步数
        warmup_steps: 预热步数
        decay_steps: 衰减步数
        decay_rate: 衰减率
        staircase: 是否使用阶梯式衰减
        min_lr: 最小学习率

    Returns:
        当前步数对应的学习率
    """
    if step < warmup_steps:
        # 线性预热
        lr = base_lr * (step / max(1, warmup_steps))
    else:
        # 超过预热期后应用衰减
        step_after_warmup = step - warmup_steps
        if staircase:
            # 阶梯式衰减 (每decay_steps步衰减一次)
            decay_factor = decay_rate ** (step_after_warmup // decay_steps)
        else:
            # 连续衰减
            decay_factor = decay_rate ** (step_after_warmup / decay_steps)

        lr = base_lr * decay_factor

    # 确保学习率不低于最小值
    return max(min_lr, lr)


# ===== 4. 标签平滑的交叉熵损失 =====
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失
    """

    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ===== 5. 自适应梯度累积 =====
def adaptive_gradient_accumulation(batch_size, target_batch_size, min_accumulation=1, max_accumulation=16):
    """
    根据当前批量大小和目标批量大小计算梯度累积步数
    """
    steps = max(min_accumulation, min(max_accumulation, round(target_batch_size / batch_size)))
    return steps


# ===== 6. 指数移动平均权重 =====
class EMA:
    """
    模型参数的指数移动平均
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 注册模型参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA权重到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ===== 7. 学习率预热策略 =====
def apply_warmup(optimizer, current_step, warmup_steps, base_lr, warmup_method='linear'):
    """
    应用学习率预热

    Args:
        optimizer: 优化器
        current_step: 当前步数
        warmup_steps: 预热总步数
        base_lr: 基础学习率
        warmup_method: 预热方法 ('linear', 'exponential', 'constant')
    """
    if current_step >= warmup_steps:
        return

    if warmup_method == 'linear':
        # 线性预热
        lr = base_lr * (current_step / warmup_steps)
    elif warmup_method == 'exponential':
        # 指数预热
        warmup_factor = 0.001  # 起始学习率因子
        lr = base_lr * (warmup_factor ** (1 - current_step / warmup_steps))
    elif warmup_method == 'constant':
        # 常数预热（从很小的值开始）
        lr = base_lr * 0.1
    else:
        raise ValueError(f"不支持的预热方法: {warmup_method}")

    # 应用新的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ===== 8. 混合批量大小训练 =====
class MixedBatchSizeSampler:
    """
    混合批量大小的采样器，允许在训练过程中动态调整批量大小
    """

    def __init__(self, dataset, base_batch_size, epoch_size_multiplier=1.0,
                 batch_size_schedule=None, shuffle=True):
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.epoch_size = int(len(dataset) * epoch_size_multiplier)
        self.batch_size_schedule = batch_size_schedule or {}  # {epoch: batch_size}
        self.shuffle = shuffle
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_batch_size(self):
        # 获取当前epoch的批量大小
        for e in sorted(self.batch_size_schedule.keys(), reverse=True):
            if self.current_epoch >= e:
                return self.batch_size_schedule[e]
        return self.base_batch_size

    def __iter__(self):
        batch_size = self.get_batch_size()
        num_batches = self.epoch_size // batch_size
        total_samples = num_batches * batch_size

        # 生成索引
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
            # 如果需要的样本数超过数据集大小，则进行重复采样
            if total_samples > len(self.dataset):
                repeats = (total_samples + len(self.dataset) - 1) // len(self.dataset)
                indices = indices * repeats
            indices = indices[:total_samples]
        else:
            indices = list(range(len(self.dataset)))
            if total_samples > len(self.dataset):
                repeats = (total_samples + len(self.dataset) - 1) // len(self.dataset)
                indices = indices * repeats
            indices = indices[:total_samples]

        # 返回批次索引
        for i in range(0, total_samples, batch_size):
            yield indices[i:i + batch_size]

    def __len__(self):
        batch_size = self.get_batch_size()
        return self.epoch_size // batch_size


# ===== 9. 模型训练中的FP16梯度压缩 =====
def setup_fp16_gradient_compression(model):
    """设置FP16梯度压缩以减少通信开销"""
    for param in model.parameters():
        # 设置参数的梯度压缩
        param._register_hook(lambda grad: grad.half())


# ===== 10. 混合精度训练的缓存释放 =====
def clean_memory_in_amp_training():
    """在混合精度训练中定期清理内存"""
    # 清理PyTorch缓存
    torch.cuda.empty_cache()

    # 垃圾回收
    import gc
    gc.collect()


# ===== 11. 递增训练批量大小 =====
def get_progressive_batch_size(epoch, start_size=32, max_size=128, steps=5):
    """
    递增批量大小 - 从小批量开始，逐渐增加到目标批量大小
    有助于在训练初期稳定，后期加速
    """
    if steps <= 1:
        return max_size

    increment = (max_size - start_size) / (steps - 1)
    current_size = min(start_size + int(epoch * increment), max_size)
    # 确保batch size是8的倍数（有助于GPU优化）
    return max(8, current_size - (current_size % 8))


# ===== 12. 自适应权重衰减 =====
class AdaptiveWeightDecay:
    """
    自适应权重衰减 - 根据训练进度调整权重衰减率
    """

    def __init__(self, initial_wd=1e-4, min_wd=1e-5, max_wd=1e-3,
                 mode='linear', total_epochs=100):
        self.initial_wd = initial_wd
        self.min_wd = min_wd
        self.max_wd = max_wd
        self.mode = mode
        self.total_epochs = total_epochs

    def get_weight_decay(self, epoch):
        """获取当前epoch的权重衰减率"""
        progress = epoch / self.total_epochs

        if self.mode == 'linear':
            # 线性增加
            return self.min_wd + progress * (self.max_wd - self.min_wd)
        elif self.mode == 'cosine':
            # 余弦调度
            return self.min_wd + 0.5 * (self.max_wd - self.min_wd) * (1 + math.cos(math.pi * progress))
        elif self.mode == 'exponential':
            # 指数增加
            return self.min_wd * (self.max_wd / self.min_wd) ** progress
        else:
            return self.initial_wd

    def update_optimizer(self, optimizer, epoch):
        """更新优化器的权重衰减率"""
        new_wd = self.get_weight_decay(epoch)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = new_wd