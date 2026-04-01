import sys
import os
import pywt
import pywt.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from torch.utils.data import DataLoader
from datasets.SPRSound_dataloader import SPRSoundDataset, collate_fn_train

# Configure matplotlib settings
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.size'] = 14
plt.style.use('seaborn-v0_8')
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.grid'] = False


class GroupBatchnorm2d(nn.Module):
    """
    Group Batch Normalization for 2D inputs
    """

    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """
    Create wavelet decomposition and reconstruction filters

    Args:
        wave: Wavelet type (e.g., 'db1')
        in_size: Input channel size
        out_size: Output channel size
        type: Tensor data type

    Returns:
        Tuple of decomposition and reconstruction filters
    """
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    """
    Apply wavelet transform on input tensor

    Args:
        x: Input tensor [batch_size, channels, height, width]
        filters: Wavelet decomposition filters

    Returns:
        Transformed tensor [batch_size, channels, 4, height//2, width//2]
    """
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    """
    Apply inverse wavelet transform on input tensor

    Args:
        x: Input tensor [batch_size, channels, 4, height_half, width_half]
        filters: Wavelet reconstruction filters

    Returns:
        Reconstructed tensor [batch_size, channels, height, width]
    """
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class ScaleModule(nn.Module):
    """
    Module for feature scaling
    """

    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class HarmonicBridge(nn.Module):
    """
    Harmonic Bridge: A wavelet-based feature extraction and processing module
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',
                 oup_channels=4, group_num=4, gate_threshold=0.5, torch_gn=True):
        super(HarmonicBridge, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.oup_channels = oup_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        # Create wavelet filters
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # Base convolution and scaling
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same',
                                   stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = ScaleModule([1, in_channels, 1, 1])

        # Wavelet domain convolutions and scaling
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same',
                       stride=1, dilation=1, groups=in_channels * 4, bias=False)
             for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
             for _ in range(self.wt_levels)]
        )

        # Normalization layer
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn \
            else GroupBatchnorm2d(c_num=oup_channels, group_num=group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

        # Pooling for frequency selection
        self.advavg = nn.AdaptiveAvgPool2d(1)

        # Optional stride implementation
        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def visualize_wavelet_components(self, x, save_path_prefix):
        """
        Visualize the four wavelet decomposition components

        Args:
            x: Input tensor with shape [batch_size, channels, 4, height, width]
            save_path_prefix: Directory to save visualizations
        """
        os.makedirs(save_path_prefix, exist_ok=True)
        batch_size, channels, _, height, width = x.shape
        component_names = ['Low-Low Frequency Component', 'Low-High Frequency Component',
                           'High-Low Frequency Component', 'High-High Frequency Component']
        for j in range(4):
            plt.rcParams["figure.figsize"] = (10, 5)
            ax = plt.subplot(111)
            for b in range(batch_size):
                img = x[b, :, j, :, :].squeeze().detach().cpu().numpy()
                if img.ndim == 1:
                    img = img.reshape(-1, 1)
                img = np.rot90(img, 2)
                ax.imshow(img, cmap='viridis', extent=[0, width, 0, height])
                ax.set_title(f'{component_names[j]}', fontsize=16)
                ax.axis('on')
                ax.set_xlabel('Time', fontsize=14)
                ax.set_ylabel('Mel Frequency', fontsize=14)
                save_path = os.path.join(save_path_prefix, f'wavelet_component_{b}_{component_names[j]}.png')
                plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
                plt.savefig(save_path)
                plt.close()

    def visualize_wavelet_components_importance(self, x, save_path_prefix):
        """
        Visualize importance of each wavelet component

        Args:
            x: Input tensor [batch_size, channels, height, width]
            save_path_prefix: Directory to save visualizations
        """
        os.makedirs(save_path_prefix, exist_ok=True)
        batch_size, _, height, width = x.shape
        component_names = ['Low-Low Frequency Component', 'Low-High Frequency Component',
                           'High-Low Frequency Component', 'High-High Frequency Component']
        for sample_idx in range(x.size(0)):
            sample_data = x[sample_idx]
            for channel_idx in range(len(component_names)):
                channel_data = sample_data[channel_idx]
                plt.rcParams["figure.figsize"] = (10, 5)
                plt.imshow(channel_data.detach().numpy(), origin='lower', aspect='auto', cmap='viridis',
                           extent=[0, width, 0, height])
                plt.xlabel('Time', fontsize=14)
                plt.ylabel('Mel Frequency', fontsize=14)
                plt.title(f'Sample {sample_idx}, Channel {component_names[channel_idx]}', fontsize=16)
                plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
                plt.savefig(
                    os.path.join(save_path_prefix, f'sample_{sample_idx}_channel_{component_names[channel_idx]}.png'))
                plt.close()

    def visualize_inverse_wavelet(self, x, save_path):
        """
        Visualize the result of inverse wavelet transform

        Args:
            x: Input tensor [batch_size, channels, height, width]
            save_path: Directory to save visualizations
        """
        os.makedirs(save_path, exist_ok=True)
        batch_size, channels, height, width = x.shape
        for b in range(batch_size):
            img = x[b, :, :, :].squeeze().detach().cpu().numpy()
            img = np.rot90(img, 2)
            plt.rcParams["figure.figsize"] = (20, 5)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(img, cmap='viridis', extent=[0, width, 0, height])
            ax.set_title('Inverse Wavelet Transform Result', fontsize=16)
            ax.axis('on')
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Mel Frequency', fontsize=14)
            save_path_b = os.path.join(save_path, f"Low-Low - inverse_wavelet_result_{b}.png")
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
            plt.savefig(save_path_b)
            plt.close(fig)

    def visualize_out1out2_combined(self, x, save_path):
        """
        Visualize combined results of out1 and out2

        Args:
            x: Input tensor [batch_size, channels, height, width]
            save_path: Path to save visualization
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        batch_size, channels, height, width = x.shape
        for b in range(batch_size):
            img = x[b, :, :, :].squeeze().detach().cpu().numpy()
            img = np.rot90(img, 2)
            plt.rcParams["figure.figsize"] = (20, 5)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(img, cmap='viridis', extent=[0, width, 0, height])
            ax.set_title(f'Combined Result - Batch {b}', fontsize=16)
            ax.axis('on')
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Mel Frequency', fontsize=14)
            single_save_path = os.path.join(os.path.dirname(save_path), f'out1+out2_combined_result_{b}.png')
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
            plt.savefig(single_save_path)
            plt.close(fig)

    def visualize_combined(self, x, save_path):
        """
        Visualize final combined result

        Args:
            x: Input tensor [batch_size, channels, height, width]
            save_path: Path to save visualization
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        batch_size, channels, height, width = x.shape
        for b in range(batch_size):
            img = x[b, :, :, :].squeeze().detach().cpu().numpy()
            img = np.rot90(img, 2)
            plt.rcParams["figure.figsize"] = (20, 5)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(img, cmap='viridis', extent=[0, width, 0, height])
            ax.set_title(f'Combined Result - Batch {b}', fontsize=16)
            ax.axis('on')
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Mel Frequency', fontsize=14)
            single_save_path = os.path.join(os.path.dirname(save_path), f'combined_result_{b}.png')
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
            plt.savefig(single_save_path)
            plt.close(fig)

    def important_frequency_vector_selection(self, x):
        """
        Select important frequency vectors using gating mechanism

        Args:
            x: Input tensor

        Returns:
            Tuple of tensors containing more and less important components
        """
        # Normalize input
        gn_x = self.gn(x)

        # Calculate relative importance weights
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)

        # Apply sigmoid activation to get importance scores
        reweights = self.sigmoid(gn_x * w_gamma)

        # Gate based on threshold
        w1 = torch.where(reweights > self.gate_threshold, torch.ones_like(reweights), reweights)
        w2 = torch.where(reweights > self.gate_threshold, torch.zeros_like(reweights), reweights)

        # Apply gating to input
        x_1 = w1 * x
        x_2 = w2 * x

        return x_1, x_2

    def more_or_less_important_frequency_vector_fuse(self, x_1, x_2):
        """
        Fuse more and less important frequency vectors

        Args:
            x_1: More important components
            x_2: Less important components

        Returns:
            Tuple of fused components
        """
        # Concatenate along channel dimension
        out = torch.cat([x_1, x_2], dim=1)

        # Apply adaptive weighting
        out = F.softmax(self.advavg(out), dim=1) * out

        # Split back into two components
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)

        return out1, out2

    def forward(self, x):
        """
        Forward pass through HarmonicBridge

        Args:
            x: Input tensor [batch_size, channels, height, width]

        Returns:
            Processed tensor
        """
        # Lists to store intermediate results
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        # Start with input as low-frequency component
        curr_x_ll = x

        # Forward wavelet transform through levels
        for i in range(self.wt_levels):
            # Store current shape
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # Pad if dimensions are odd
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # Apply wavelet transform
            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)

            # Extract low frequency component for next level
            curr_x_ll = curr_x[:, :, 0, :, :]

            # Reshape for convolution
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])

            # Apply wavelet domain convolution and scaling
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))

            # Select important frequency components
            more_important_curr_x_tag, less_important_curr_x_tag = self.important_frequency_vector_selection(curr_x_tag)

            # Fuse important components
            out1, out2 = self.more_or_less_important_frequency_vector_fuse(
                more_important_curr_x_tag, less_important_curr_x_tag)

            # Reshape back to original format
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # Store components for reconstruction
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        # Initialize for inverse transform
        next_x_ll = 0

        # Inverse wavelet transform (reconstruction)
        for i in range(self.wt_levels - 1, -1, -1):
            # Get stored components
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            # Add low frequency components
            curr_x_ll = curr_x_ll + next_x_ll

            # Concatenate for inverse transform
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            # Apply inverse wavelet transform
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            # Crop to original dimensions
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # Final reconstructed result
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # Apply base convolution
        x = self.base_scale(self.base_conv(x))

        # Add wavelet processed features
        x = x + x_tag

        # Upsample and fuse out1 and out2
        out1_upsampled = F.interpolate(out1, size=(curr_shape[2], curr_shape[3]),
                                       mode='bilinear', align_corners=True)
        out2_upsampled = F.interpolate(out2, size=(curr_shape[2], curr_shape[3]),
                                       mode='bilinear', align_corners=True)

        # Weighted fusion
        weight1, weight2 = 0.6, 0.4
        fused_tensor = weight1 * out1_upsampled + weight2 * out2_upsampled

        # Average channels
        result_tensor = torch.mean(fused_tensor, dim=1, keepdim=True)

        # Add fused result
        x = x + result_tensor

        # Apply stride if needed
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


if __name__ == '__main__':
    # Create a HarmonicBridge instance
    bridge = HarmonicBridge(in_channels=1, out_channels=1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load and Check Dataset Dimensions')
    parser.add_argument("--task", default=21, type=int, choices=[11, 12, 21, 22], help="task")
    parser.add_argument("--train_wav_path",
                        default='/home/wangying/Lung_sound_detection/SPRSound22_23/SPRSound22_23/restore_train_classification_wav',
                        type=str,
                        help="input wav path")
    parser.add_argument("--train_json_path",
                        default='/home/wangying/Lung_sound_detection/SPRSound22_23/SPRSound22_23/train_classification_json',
                        type=str,
                        help="input json dictionary")
    parser.add_argument("--valid_wav_path",
                        default='/home/wangying/Lung_sound_detection/SPRSound22_23/SPRSound22_23/restore_valid_classification_wav',
                        type=str,
                        help="input wav path")
    parser.add_argument("--valid_json_path",
                        default='/home/wangying/Lung_sound_detection/SPRSound22_23/SPRSound22_23/valid_classification_json',
                        type=str,
                        help="input json dictionary")
    parser.add_argument("--feature_type", default='log-mel', type=str,
                        choices=['MFCC', 'log-mel', 'mel', 'STFT'],
                        help="feature extraction type")
    args = parser.parse_args()

    # Load training dataset
    train_dataset = SPRSoundDataset(args.train_wav_path, args.train_json_path, args.task,
                                    mode='train', feature_type=args.feature_type)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_train)

    # Process and visualize data
    print("Training data loading (batch_size=8):")
    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1} - Data shape: {batch_data.shape}, Labels shape: {batch_labels.shape}")

        input_data = batch_data

        # Visualize input spectrograms
        n_mels, length = batch_data.shape[2], batch_data.shape[3]
        plt.rcParams["figure.figsize"] = (20, 5)
        fig, ax = plt.subplots()
        for i in range(batch_data.shape[0]):
            data = batch_data[i].squeeze().detach().cpu().numpy()
            ax.imshow(data, aspect='auto', origin='lower', extent=[0, length, 0, n_mels])
            ax.set_xlabel('Time', fontsize=14)
            ax.set_ylabel('Mel Frequency', fontsize=14)
            ax.set_title('Log-mel spectrogram of sample', fontsize=16)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9)
            plt.savefig(f'feature_extractor/visualization_HarmonicBridge/original_input_sample_{i}.png')
            plt.close()

        # Process through HarmonicBridge
        output = bridge(input_data)

        print(f"Input shape: {input_data.size()}, Labels: {batch_labels}")
        print(f"Output shape: {output.size()}")

        break