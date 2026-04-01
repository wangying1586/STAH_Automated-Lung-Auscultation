import torch
import torch.nn as nn


""" -------------Randomized Quantization Aug核心思路----------------
   1.超参数设置：bins数量、区域折叠方式middle, inside_random, all_zeros、区域划分的间距模式random, uniform、数据增强应用概率
   2.先计算每个通道的H X W的特征像素点的最大值、最小值及bins划分节点数量
   3.依据步骤2的计算值，计算并获取百分位数位置作为bins的区域划分节点
   4.依据节点，做bins区域切分，依据区域折叠方式middle, inside_random, all_zeros计算bins区域的像素值更新方式(对所有bins均替换)
   （middle：取每个bins的左右端点的平均值替换该区域内的所有像素值）
   （inside_random：取每个bins的随机值替换该区域内的所有像素值）
   （all_zeros：直接用0替换该区域内的所有像素值）
   5.依据数据增强应用概率，去返回增强or原数据
"""

""" --------- 非均匀上采样合成少数类别样本 ---------
    1.超参数设置：bins数量、区域扩充方式middle, inside_random, right_copy、left-copy的一半值，1/3以内的随机个数值的插值上采样，区域划分的间距模式random, uniform、数据增强应用概率
    2.计算每个通道的H X W的特征像素点的最大值、最小值及bins划分节点数量
    3.依据步骤2的计算值，计算并获取百分位数位置作为bins的区域划分节点
    4.依据节点，做bins区域切分，依据区域重复和扩展上采样方式middle、inside_random、right_copy、left-copy计算bins区域的像素值扩充方式(对所有bins均进行重复或插值)
      先确定每个 bins 区域内的随机插值位置，其线性插值上采样个数限制在：总的 bins 区域内像素值个数的1/3以内的随机个数值
      （middle：取每个bins的左右端点的平均值插入该区域内的插值位置）
      （inside_random：取每个bins的随机值插入该区域内的插值位置）
      （right_copy：每个bins的插值位置的值都是 copy 其右边像素值）
      （left-copy：每个bins的插值位置的值都是 copy 其左边像素值）
    5.依据类别数量少的样本，运用上述上采样算法合成样本，返回合成后的样本。
"""
class RandomizedQuantizationAugModule(nn.Module):
    def __init__(self, region_num, collapse_to_val='inside_random', spacing='random', transforms_like=False, p_random_apply_rand_quant=1):
        """
        region_num: int; 区域数量，用于将数据范围划分为多个区域
        """
        super().__init__()
        self.region_num = region_num  # 初始化区域数量
        self.collapse_to_val = collapse_to_val  # 确定区域折叠值的方式，如'middle'、'inside_random'、'all_zeros'
        self.spacing = spacing  # 区域划分的间距模式，'random' 或 'uniform'
        self.transforms_like = transforms_like  # 是否类似图像变换处理数据
        self.p_random_apply_rand_quant = p_random_apply_rand_quant  # 随机应用数据增强的概率

    def get_params(self, x):
        """
        x: (C, H, W)· 输入数据的形状，C 为通道数，H 和 W 为高度和宽度（对于音频数据，可类比为其他维度含义）
        returns (C), (C), (C) 返回每个通道的最小值、最大值和区域百分位数数量
        """
        C, _, _ = x.size()  # 获取输入数据的通道数 C，这里假设输入数据为 [C, H, W] 形状，若不是需转换
        # 计算每个通道数据在所有像素点上的最小值和最大值
        min_val, max_val = x.view(C, -1).min(1)[0], x.view(C, -1).max(1)[0]
        # 初始化每个通道的区域百分位数数量为 (region_num - 1) 的张量，数据类型为整数
        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int()
        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W) 输入数据，可以是 [B, c, H, W] 批量数据形式或 [C, H, W] 单数据形式
        """
        EPSILON = 1
        # 如果随机应用数据增强概率不等于 1
        if self.p_random_apply_rand_quant!= 1:
            x_orig = x  # 保存原始输入数据

        # 如果不是类似图像变换的处理方式
        if not self.transforms_like:
            B, c, H, W = x.shape  # 获取输入数据的批量大小 B、通道数 c、高度 H 和宽度 W
            C = B * c  # 计算总通道数
            x = x.view(C, H, W)  # 将数据形状调整为 [C, H, W]
        else:
            C, H, W = x.shape  # 如果是类似图像变换的处理方式，直接获取数据的形状

        # 获取每个通道的最小值、最大值和区域百分位数数量
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x)
        # -> (C), (C), (C)

        """非均匀量化器 取百分位数位置作为bins的区域划分节点"""
        # 为每个通道生成区域百分位数
        if self.spacing == "random":
            # 随机生成区域百分位数，其总和为每个通道的区域百分位数数量总和
            region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)
        elif self.spacing == "uniform":
            # 均匀生成区域百分位数，在每个通道上重复特定范围的等差数列
            region_percentiles = torch.tile(torch.arange(1 / (total_region_percentile_number_per_channel[0] + 1), 1, step=1 / (total_region_percentile_number_per_channel[0] + 1), device=x.device), [C])
        # 将区域百分位数重塑为 [通道数, 区域数量 - 1] 的形状
        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])

        """非均匀量化器 根据bins的区域划分节点，将区域内像素值排序后，计算每个bins左右端点值，得到左右端点的平均值作为该bins的代理值
        （该方式只针对self.collapse_to_val == 'middle'情况需要计算）"""
        # 计算每个区域的结束位置（基于百分位数和数据范围）
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).view(C, 1) + min_val.view(C, 1)).view(C, -1, 1, 1)
        # 计算用于检查的有序区域右端点，在区域百分位数位置基础上加上最大值并添加一个小的正数EPSILON，然后进行排序
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        # 计算有序区域右端点，在区域百分位数位置基础上加上最大值并添加一个极小值（1e-6），然后进行排序
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.view(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        # 计算有序区域左端点，将最小值与区域百分位数位置进行拼接后排序
        ordered_region_left_ends = torch.cat([min_val.view(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]
        # 计算每个区域的中间点，即区域左右端点的平均值
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends) / 2

        """非均匀量化器 确定并获取待增强音频所属的bins编号"""
        # 确定每个像素点所属的区域，通过比较像素值与区域左右端点来判断（生成布尔类型的张量，维度为 [C, self.region_num, H, W]）
        is_inside_each_region = (x.view(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.view(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        # 进行合理性检查，确保每个像素都只属于一个区域（即每行的布尔值求和为1）
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        # 获取每个像素点所属的区域编号（维度变为 [C, 1, H, W]），通过取每行布尔值中为True的索引（即最大值所在位置的索引）来确定
        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)

        if self.collapse_to_val == 'middle':
            # 如果折叠值方式为'middle'，以区域中间点作为对应区域内所有值的代理值
            # 通过根据区域编号从扩展后的区域中间点张量中收集对应的值，来更新数据x
            proxy_vals = torch.gather(ordered_region_mid.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)
        elif self.collapse_to_val == 'inside_random':
            # 如果折叠值方式为'inside_random'，在每个区域内随机取点作为对应区域内所有值的代理值
            # 首先随机生成每个区域内的百分位数
            proxy_percentiles_per_region = torch.rand((total_region_percentile_number_per_channel + 1).sum(),
                                                      device=x.device)
            proxy_percentiles_per_channel = proxy_percentiles_per_region.reshape([-1, self.region_num])
            # 根据区域左右端点和随机生成的百分位数计算每个区域内的随机点位置
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.view(C, -1, 1, 1) * (
                        ordered_region_right_ends - ordered_region_left_ends)
            # 通过根据区域编号从扩展后的区域随机点张量中收集对应的值，来更新数据x
            proxy_vals = torch.gather(ordered_region_rand.expand([-1, -1, H, W]), 1, associated_region_id)[:, 0]
            x = proxy_vals.type(x.dtype)

        elif self.collapse_to_val == 'all_zeros':
            # 如果折叠值方式为'all_zeros'，将所有值都设为0作为代理值
            proxy_vals = torch.zeros_like(x, device=x.device)
            x = proxy_vals.type(x.dtype)
        else:
            raise NotImplementedError

        if not self.transforms_like:
            # 如果不是类似图像变换的处理方式，将数据形状还原为原始的批量数据形状 [B, c, H, W]
            x = x.view(B, c, H, W)

        if self.p_random_apply_rand_quant != 1:
            if not self.transforms_like:
                # 如果不是类似图像变换的处理方式，根据随机生成的概率决定是否使用原始数据替换增强后的数据
                x = torch.where(torch.rand([B, 1, 1, 1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)
            else:
                # 如果是类似图像变换的处理方式，同样根据随机生成的概率决定是否使用原始数据替换增强后的数据
                x = torch.where(torch.rand([C, 1, 1], device=x.device) < self.p_random_apply_rand_quant, x, x_orig)

        return x
