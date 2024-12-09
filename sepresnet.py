from torch import add
from torch import nn


class SepResNetClassifier(nn.Module):
    def __init__(self, shape: str, num_classes=None):
        super().__init__()
        self.encoder = SepResNetEncoder(shape)
        if num_classes is not None:
            self.classifier = ClassificationHead(num_classes)
        else:
            self.classifier = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sep_conv_3x3_s2_1 = DepthwiseSeparableConv2d(
            256, 128, 
            kernel_size=3, stride=2, padding='same'
        )
        self.sep_conv_3x3_s2_2 = DepthwiseSeparableConv2d(
            128, 64, 
            kernel_size=3, stride=2, padding='same'
        )

        self.flat = nn.Flatten()
        self.dense_1 = nn.LazyLinear(32)
        self.dense_out = nn.Linear(32, num_classes)

    def forward(self, x):
        # reduce dimensions
        x = self.sep_conv_3x3_s2_1(x)
        x = self.sep_conv_3x3_s2_2(x)

        # flatten output
        x = self.flat(x)

        # fully connected layers with sigmoid activation
        x = self.dense_1(x)
        x = self.dense_out(x)
        return x # output raw logits


class SepResNetEncoder(nn.Module):
    _archs = {
        "18": {
            "block_1": {"in_channels":64, "out_channels":64, "n_blocks":2}, 
            "block_2": {"in_channels":64, "out_channels":128, "n_blocks":2, "stride":2}, 
            "block_3": {"in_channels":128, "out_channels":256, "n_blocks":2, "stride":2}, 
            "block_4": {"in_channels":256, "out_channels":512, "n_blocks":2, "stride":2}
        },
        "34": {
            "block_1": {"in_channels":64, "out_channels":64, "n_blocks":3}, 
            "block_2": {"in_channels":64, "out_channels":128, "n_blocks":4, "stride":2}, 
            "block_3": {"in_channels":128, "out_channels":256, "n_blocks":6, "stride":2}, 
            "block_4": {"in_channels":256, "out_channels":512, "n_blocks":3, "stride":2}
        },
        "50": {
            "block_1": {"in_channels":64, "mid_channels":64, "out_channels":256, "n_blocks":3}, 
            "block_2": {"in_channels":256, "mid_channels":128,  "out_channels":512, "n_blocks":4, "stride":2}, 
            "block_3": {"in_channels":512, "mid_channels":256,  "out_channels":1024, "n_blocks":6, "stride":2}, 
            "block_4": {"in_channels":1024, "mid_channels":512,  "out_channels":2048, "n_blocks":3, "stride":2}
        },
        "101": {
            "block_1": {"in_channels":64, "mid_channels":64, "out_channels":256, "n_blocks":3}, 
            "block_2": {"in_channels":256, "mid_channels":128,  "out_channels":512, "n_blocks":4, "stride":2}, 
            "block_3": {"in_channels":512, "mid_channels":256,  "out_channels":1024, "n_blocks":23, "stride":2}, 
            "block_4": {"in_channels":1024, "mid_channels":512,  "out_channels":2048, "n_blocks":3, "stride":2}
        },
        "152": {
            "block_1": {"in_channels":64, "mid_channels":64, "out_channels":256, "n_blocks":3}, 
            "block_2": {"in_channels":256, "mid_channels":128,  "out_channels":512, "n_blocks":8, "stride":2}, 
            "block_3": {"in_channels":512, "mid_channels":256,  "out_channels":1024, "n_blocks":36, "stride":2}, 
            "block_4": {"in_channels":1024, "mid_channels":512,  "out_channels":2048, "n_blocks":3, "stride":2}
        },
    }
    def __init__(self, shape: str):
        super().__init__()
        self.conv_1 = DepthwiseSeparableConv2d(
            3, 32, kernel_size=7, 
            stride=2, padding=3 # hardcoded as 3
        )
        self.max_pool = nn.MaxPool2d(
            3, stride=2, padding=1
        )
        arch_dict = self._archs[shape]

        self.block_1 = ResidualBlock(**arch_dict["block_1"])
        self.block_2 = ResidualBlock(**arch_dict["block_2"])
        self.block_3 = ResidualBlock(**arch_dict["block_3"])
        self.block_4 = ResidualBlock(**arch_dict["block_4"])

    def forward(self, x):
        x = self.conv_1(x)
        x = self.max_pool(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, n_blocks=2, stride=1, dilation=1):
        super().__init__()
        # collect params into a set of lists to zip and iterate over
        channel_list = [(in_channels, mid_channels, out_channels)] + ([(out_channels, mid_channels, out_channels)]*(n_blocks-1))
        stride_list = [stride] + ([1]*(n_blocks-1)) # only consider stride on first subblock
        dilation_list = [dilation]*n_blocks
        # get empty list to store subblocks in
        subblocks = []

        for (in_c, mid_c, out_c), s, d in zip(channel_list, stride_list, dilation_list):
            subblocks.append(ResidualSubBlock(in_c, out_c, mid_c, s, d))

        self.subblocks = nn.Sequential(*subblocks)

    def forward(self, x):
        return self.subblocks(x)


class ResidualSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, dilation=1):
        super().__init__()
        # if a number of mid channels is defined, use the bottleneck subblock version
        if mid_channels is not None:
            self.main = BottleneckMainPath(in_channels, mid_channels, out_channels, stride, dilation)
        else:
            self.main = StandardMainPath(in_channels, out_channels, stride, dilation)

        if stride == 1: # don't place a convolution in the shortcut path for identity blocks
            self.shortcut = nn.Identity()
        else: # otherwise, if strided block adjust n channels with a 1x1 conv
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        main_x = self.main(x)
        shortcut_x = self.shortcut(x)
        out_x = main_x.add(shortcut_x)
        return out_x


class BottleneckMainPath(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super().__init__(self)
        self.bn_1 = nn.BatchNorm2d(in_channels),
        self.act_1 = nn.ReLU(),
        self.conv_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)

        self.bn_2 = nn.BatchNorm2d(mid_channels),
        self.act_2 = nn.ReLU(),
        self.conv_2 = DepthwiseSeparableConv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, dilation=dilation)

        self.bn_3 = nn.BatchNorm2d(mid_channels),
        self.act_3 = nn.ReLU(),
        self.conv_3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.conv_1(x)

        x = self.bn_2(x)
        x = self.act_2(x)
        x = self.conv_2(x)

        x = self.bn_3(x)
        x = self.act_3(x)
        x = self.conv_3(x)
        return x


class StandardMainPath(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation):
        super().__init__(self)
        self.bn_1 = nn.BatchNorm2d(in_channels),
        self.act_1 = nn.ReLU(),
        self.conv_1 = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=stride, dilation=dilation, padding='same')

        self.bn_2 = nn.BatchNorm2d(out_channels),
        self.act_2 = nn.ReLU(),
        self.conv_2 = DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding='same')

    def forward(self, x):
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.conv_1(x)

        x = self.bn_2(x)
        x = self.act_2(x)
        x = self.conv_2(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    """
    referenced the discussion here: https://stackoverflow.com/a/65155106
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1):
        super().__init__()
        # get padding amount
        # if strides > 1, this will pad to in_channels/stride
        # this might break for some stuff but it works for 1/3/5/7
        pad_amount = (kernel_size // 2) + (dilation // 2) if padding == 'same' else 0

        # use the groups parameter to perform the convolution over each input
        # channel separately
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, groups=in_channels, 
            stride=stride, padding=pad_amount, dilation=dilation
        )
        # then combine the outputs with a pointwise (1x1) convolution
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return(x)


class SepResNet18(SepResNetClassifier):
    def __init__(self, num_classes=None):
        super().__init__("18", num_classes)

class SepResNet34(SepResNetClassifier):
    def __init__(self, num_classes=None):
        super().__init__("34", num_classes)

class SepResNet50(SepResNetClassifier):
    def __init__(self, num_classes=None):
        super().__init__("50", num_classes)

class SepResNet101(SepResNetClassifier):
    def __init__(self, num_classes=None):
        super().__init__("101", num_classes)

class SepResNet152(SepResNetClassifier):
    def __init__(self, num_classes=None):
        super().__init__("152", num_classes)


if __name__ == "__main__":
    from torchinfo import summary
    from argparse import ArgumentParser
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    
    parser = ArgumentParser("SepResNet [18, 34, 50, 101, 152]")
    parser.add_argument(
        "--version",
        nargs='?',
        help="Architecture version to test", 
        type=int,
        default=18
    )
    args = parser.parse_args()
    version = args.version

    input_shape = (3, 256, 256)

    if version == 18:
        model = SepResNet18(num_classes=1)
    elif version == 34:
        model = SepResNet34(num_classes=1)
    elif version == 50:
        model = SepResNet50(num_classes=1)
    elif version == 101:
        model = SepResNet101(num_classes=1)
    elif version == 152:
        model = SepResNet152(num_classes=1)
    else:
        raise ValueError(f"unrecognized model version '{version}', please choose from [18, 34, 50, 101, 152]")
    
    print(summary(model, input_size=(32,) + input_shape))