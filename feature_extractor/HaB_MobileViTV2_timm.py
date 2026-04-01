import sys

sys.path.append('/home/wangying/Lung_sound_detection')
import torch
import torch.nn as nn
import torch.nn.init as init
import timm
from feature_extractor.HarmonicBridge import HarmonicBridge


class CustomMobileViTV2(nn.Module):
    def __init__(self, model_name='mobilevitv2_100', num_classes=None, pretrained=True):
        """
        Custom MobileViTV2 with HarmonicBridge preprocessing

        Args:
            model_name: timm model name (e.g., 'mobilevitv2_100', 'mobilevitv2_075', 'mobilevitv2_050')
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(CustomMobileViTV2, self).__init__()

        # Initialize HarmonicBridge preprocessing layer
        self.custom_conv = HarmonicBridge(in_channels=1, out_channels=1)
        self._reinitialize_wtconv2d_weights()

        # Load base MobileViTV2 model from timm
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,  # Set input channels to 1 directly
            num_classes=num_classes if num_classes is not None else 1000
        )

        # If timm doesn't support in_chans=1 directly, we need to manually adjust
        # Check if the first layer needs adjustment
        self._adjust_first_layer_if_needed()

        # Initialize the final classification layer if num_classes is specified
        if num_classes is not None:
            self._initialize_classifier(num_classes)

    def _adjust_first_layer_if_needed(self):
        """Adjust the first convolution layer if needed"""
        # Find the first convolutional layer
        first_conv = None
        first_conv_name = None

        # Common names for first conv layers in MobileViT
        possible_names = ['stem.0', 'conv_stem', 'features.0.0', 'patch_embed.proj']

        for name in possible_names:
            try:
                module = self.base_model
                for part in name.split('.'):
                    module = getattr(module, part)
                if isinstance(module, nn.Conv2d):
                    first_conv = module
                    first_conv_name = name
                    break
            except AttributeError:
                continue

        # If we find the first conv and it has 3 input channels, adjust it
        if first_conv is not None and first_conv.in_channels == 3:
            print(f"Adjusting first conv layer: {first_conv_name}")

            # Get original weight and adjust for single channel
            original_weight = first_conv.weight.data
            # Use only the first channel or average across RGB channels
            new_weight = original_weight[:, :1, :, :]  # Take first channel
            # Alternative: new_weight = original_weight.mean(dim=1, keepdim=True)  # Average channels

            # Create new conv layer
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )

            # Copy adjusted weights
            new_conv.weight.data = new_weight
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data

            # Replace the layer
            module = self.base_model
            parts = first_conv_name.split('.')
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], new_conv)

    def _initialize_classifier(self, num_classes):
        """Initialize the final classifier layer"""
        # Find the classifier layer
        if hasattr(self.base_model, 'head'):
            classifier = self.base_model.head
        elif hasattr(self.base_model, 'classifier'):
            classifier = self.base_model.classifier
        elif hasattr(self.base_model, 'fc'):
            classifier = self.base_model.fc
        else:
            print("Warning: Could not find classifier layer to initialize")
            return

        if isinstance(classifier, nn.Linear):
            init.kaiming_normal_(classifier.weight, mode='fan_out', nonlinearity='relu')
            if classifier.bias is not None:
                classifier.bias.data.zero_()

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

        # Initialize base scaling module
        if hasattr(self.custom_conv, 'base_scale'):
            init.ones_(self.custom_conv.base_scale.weight)

        # Initialize wavelet domain convolution layers
        if hasattr(self.custom_conv, 'wavelet_convs'):
            for wavelet_conv in self.custom_conv.wavelet_convs:
                init.kaiming_normal_(wavelet_conv.weight, mode='fan_out', nonlinearity='relu')

        # Initialize wavelet domain scaling modules
        if hasattr(self.custom_conv, 'wavelet_scale'):
            for wavelet_scale in self.custom_conv.wavelet_scale:
                init.ones_(wavelet_scale.weight)

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
        # Then through the base model
        return self.base_model(x)

    @classmethod
    def create_model(cls, model_variant='100', num_classes=None, pretrained=True):
        """
        Factory method to create different MobileViTV2 variants

        Args:
            model_variant: '100', '075', '050', '125', '150', '175', '200'
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        model_name = f'mobilevitv2_{model_variant}'
        return cls(model_name=model_name, num_classes=num_classes, pretrained=pretrained)


# Convenience functions for different model variants
def mobilevitv2_050(num_classes=None, pretrained=True):
    """MobileViTV2-0.5"""
    return CustomMobileViTV2.create_model('050', num_classes, pretrained)


def mobilevitv2_075(num_classes=None, pretrained=True):
    """MobileViTV2-0.75"""
    return CustomMobileViTV2.create_model('075', num_classes, pretrained)


def mobilevitv2_100(num_classes=None, pretrained=True):
    """MobileViTV2-1.0"""
    return CustomMobileViTV2.create_model('100', num_classes, pretrained)


def mobilevitv2_125(num_classes=None, pretrained=True):
    """MobileViTV2-1.25"""
    return CustomMobileViTV2.create_model('125', num_classes, pretrained)


def mobilevitv2_150(num_classes=None, pretrained=True):
    """MobileViTV2-1.5"""
    return CustomMobileViTV2.create_model('150', num_classes, pretrained)


def mobilevitv2_175(num_classes=None, pretrained=True):
    """MobileViTV2-1.75"""
    return CustomMobileViTV2.create_model('175', num_classes, pretrained)


def mobilevitv2_200(num_classes=None, pretrained=True):
    """MobileViTV2-2.0"""
    return CustomMobileViTV2.create_model('200', num_classes, pretrained)


if __name__ == '__main__':
    # Test code
    print("Testing CustomMobileViTV2 with timm...")

    # Test different variants
    variants = ['050', '075', '100']
    num_classes = 2

    for variant in variants:
        print(f"\nTesting MobileViTV2-{variant}:")

        # Create model
        model = CustomMobileViTV2.create_model(variant, num_classes=num_classes, pretrained=False)
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

    # Test specific model creation functions
    print("\nTesting convenience functions:")
    models = {
        'MobileViTV2-0.5': mobilevitv2_050(num_classes=4),
        'MobileViTV2-0.75': mobilevitv2_075(num_classes=4),
        'MobileViTV2-1.0': mobilevitv2_100(num_classes=4),
    }

    test_input = torch.randn(1, 1, 128, 1001)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        print(f"{name}: {test_input.shape} -> {output.shape}")