from torch import nn


class SepResNet50v2(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.input_shape = input_shape

        self.conv_7x7_s2 = nn.Conv2d(
            1, 32, kernel_size=7, 
            stride=2, padding='same'
        )

        self.sep_conv_3x3_s1 = DepthwiseSeparableConv2d(
            32, 64, kernel_size=3, 
            stride=1, padding='same'
        )

        self.res_block_1 = ResidualBlock(64, 3)
        self.res_block_2 = ResidualBlock(128, 4, stride=2)
        self.res_block_3 = ResidualBlock(256, 6, stride=2)
        self.res_block_4 = ResidualBlock(256, 3, dilation=2)

        self.classifier = ClassificationHead(num_classes)

    def forward(self, x):
        # return model output
        x = self.conv_7x7_s2(x)

        x = self.sep_conv_3x3_s1(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)

        x = self.classifier(x)
        return x


class SepResNet101v2(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.input_shape = input_shape

        self.conv_7x7_s2 = nn.Conv2d(
            1, 32, kernel_size=7, 
            stride=2, padding='same'
        )

        self.sep_conv_3x3_s1 = DepthwiseSeparableConv2d(
            32, 64, kernel_size=3, 
            stride=1, padding='same'
        )

        self.res_block_1 = ResidualBlock(64, 3)
        self.res_block_2 = ResidualBlock(128, 4, stride=2)
        self.res_block_3 = ResidualBlock(256, 23, stride=2)
        self.res_block_4 = ResidualBlock(256, 3, dilation=2)

        self.classifier = ClassificationHead(num_classes)

    def forward(self, x):
        # return model output
        x = self.conv_7x7_s2(x)

        x = self.sep_conv_3x3_s1(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)

        x = self.classifier(x)
        return x


class SepResNet152v2(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        self.input_shape = input_shape

        self.conv_7x7_s2 = nn.Conv2d(
            1, 32, kernel_size=7, 
            stride=2, padding='same'
        )

        self.sep_conv_3x3_s1 = DepthwiseSeparableConv2d(
            32, 64, kernel_size=3, 
            stride=1, padding='same'
        )

        self.res_block_1 = ResidualBlock(64, 3)
        self.res_block_2 = ResidualBlock(128, 8, stride=2)
        self.res_block_3 = ResidualBlock(256, 36, stride=2)
        self.res_block_4 = ResidualBlock(256, 3, dilation=2)

        self.classifier = ClassificationHead(num_classes)

    def forward(self, x):
        # return model output
        x = self.conv_7x7_s2(x)

        x = self.sep_conv_3x3_s1(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)

        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, filter_num, n, stride=1, dilation=1):
        super().__init__()
        block_list = []
        if stride == 2:
            # start with a strided identity block to halve dims
            block_list.append(StridedIdentityBlock(filter_num, filter_num, stride=stride))

            # get list of identity blocks
            for i in range(n - 1):
                block_list.append(IdentityBlock(filter_num, filter_num, dilation=dilation))

        else:
            # get list of identity blocks
            for i in range(n):
                block_list.append(IdentityBlock(filter_num, filter_num, dilation=dilation))

        # unpack block list into sequential model
        self.block = nn.Sequential(*block_list)

    def forward(self, x):
        return self.block(x)
    

class DepthwiseSeparableConv2d(nn.Module):
    """
    referenced the discussion here: https://stackoverflow.com/a/65155106
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1):
        super().__init__()
        # get padding amount
        # if strides > 1, this will pad to in_channels/stride
        # this might break for some stuff but it works for 1/3/5/7
        pad_amount = kernel_size // 2 if padding == 'same' else 0

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


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        # dims should go:
        # in_channels -> in_channels // 2 -> out_channels
        # should in_channels == out_channels?
        self.conv_1x1_s1 = nn.Conv2d(
            in_channels, in_channels // 2, # halve the channels in the first bottleneck conv
            kernel_size=1, stride=1, padding='same'
        )
        self.sep_conv_3x3_s1 = DepthwiseSeparableConv2d(
            in_channels // 2, out_channels, 
            kernel_size=3, stride=1, padding='same', dilation=dilation
        )

        self.bn_1 = nn.BatchNorm2d()
        self.bn_1 = nn.BatchNorm2d()

    def forward(self, x_in):
        # main path
        x = self.bn_1(x_in)
        x = nn.functional.relu(x)
        x = self.conv_1x1_s1(x)

        x = self.bn_2(x)
        x = nn.functional.relu(x)
        x = self.sep_conv_3x3_s1(x)

        # combined path
        x = nn.functional.add(x, x_in)
        return x


class StridedIdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, dilation=1):
        super().__init__()
        # default strides of 2, modify this if you need
        # dims should go:
        # in_channels -> in_channels // 2 -> out_channels
        # should in_channels == out_channels?
        self.conv_1x1_s2 = nn.Conv2d(
            in_channels, in_channels // 2, # halve the channels in the first bottleneck conv
            kernel_size=1, stride=stride, padding='same'
        )
        self.conv_1x1_s2_sc = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, padding='same'
        )
        self.sep_conv_3x3_s1 = DepthwiseSeparableConv2d(
            in_channels // 2, out_channels, 
            kernel_size=3, stride=1, padding='same', dilation=dilation
        )

        self.bn_1 = nn.BatchNorm2d()
        self.bn_1 = nn.BatchNorm2d()

    def forward(self, x_in):
        # main path
        x = self.bn_1(x_in)
        x = nn.functional.relu(x)
        x = self.conv_1x1_s1(x)

        x = self.bn_2(x)
        x = nn.functional.relu(x)
        x = self.sep_conv_3x3_s1(x)

        # shortcut path
        x_in = self.conv_1x1_s2_sc(x_in)

        # combined path
        x = nn.functional.add(x, x_in)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sep_conv_3x3_s2_1 = DepthwiseSeparableConv2d(
            512, 256, 
            kernel_size=3, stride=2, padding='same'
        )
        self.sep_conv_3x3_s2_2 = DepthwiseSeparableConv2d(
            256, 128, 
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

