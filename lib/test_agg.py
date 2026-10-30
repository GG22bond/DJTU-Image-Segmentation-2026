import torch
import torch.nn as nn
from torch.nn import init

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicConv2d(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x


class BasicConv2d_relu(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()

        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv3 = BasicConv2d(3 * channel, channel, 1)

        # self.conv1 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1

        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2

        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)

        x = self.conv3(x)

        # out = self.conv1(x)
        # return x, out
        return x


# class aggregationnew(nn.Module):
#
#     def __init__(self, channel):
#         super(aggregationnew, self).__init__()
#
#         self.relu = nn.ReLU(True)
#
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
#         self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
#         self.se = SEAttention(3*channel)
#         self.conv_upsample8 = BasicConv2d(3*channel, channel, 3, padding=1)
#         self.conv_upsample9 = BasicConv2d(channel, channel, 1)
#         self.conv = nn.Conv2d(channel, 1, 1)
#
#     def forward(self, x1, x2, x3):
#
#         x1_1 = x1
#         print("x1_1:", x1_1.size())
#         x2_1 = x2
#         print("x2_1:", x2_1.size())
#         x3_1 = x3
#         print("x3_1:", x3_1.size())
#
#         x1_2 = self.upsample(x1_1)
#         x1_3 = self.conv_upsample1(x1_2)
#         x1_4 = x1_3 * x2_1
#         x1_5 = self.conv_upsample2(x1_4)
#         print("x1_5:", x1_5.size())
#         x1_6 = x1_2 + x1_5
#         print("x1_6:", x1_6.size())
#         x1_6 = self.upsample(x1_6)
#         print("x1_6:", x1_6.size())
#
#
#         x2_2 = self.upsample(x2_1)
#         x2_3 = self.conv_upsample4(x2_2)
#         x2_4 = x3_1 * x2_3
#         x2_5 = self.conv_upsample5(x2_4)
#         x2_6 = x2_2 + x2_5
#         print("x2_6:", x2_6.size())
#
#
#         x3_2 = self.conv_upsample6(x3_1)
#         x3_3 = self.conv_upsample7(x3_2)
#         x3_4 = x3_1 + x3_3
#         print("x3_4:", x3_4.size())
#
#
#         x_cat = torch.cat((x1_6, x2_6, x3_4), dim=1)
#         print("x_cat:", x_cat.size())
#
#         x_cat = self.se(x_cat)
#         print("x_cat:", x_cat.size())
#
#         x = self.conv_upsample8(x_cat)
#         print("x:", x.size())
#         x = self.conv_upsample9(x)
#         print("x:", x.size())
#         out = self.conv(x)
#         print("out:", out.size())
#
#         return x, out


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class aggregationnew(nn.Module):
    def __init__(self, channel):
        super(aggregationnew, self).__init__()

        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                  channel_attention(channel))
        # self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_upsample2 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                            channel_attention(channel))
        # self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)

        # self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                            channel_attention(channel))

        # self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                            channel_attention(channel))

        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)

        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv3 = BasicConv2d(3 * channel, channel, 1)


    def forward(self, x1, x2, x3):
        x1_1 = x1

        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2

        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)

        x = self.conv3(x)

        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AMSF(nn.Module):
    def __init__(self, in_channels, out_channels):
    # def __init__(self, in_channels, out_channels, factor=4.0):
        super(AMSF, self).__init__()
        # dim = int(out_channels // factor)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(out_channels, out_channels, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(out_channels, out_channels, 3, padding=1)

        # self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        self.down = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=1)

        # self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_3x3 = BasicConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_5x5 = BasicConv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)

        # self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.conv_7x7 = BasicConv2d(out_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(out_channels)
        self.conv = BasicConv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x1, x2, x3):

        x1 = self.conv_upsample1(self.upsample(self.upsample(x1)))
        x2 = self.conv_upsample1(self.upsample(x2))

        x_fused = torch.cat([x1, x2, x3], dim=1)

        x_fused = self.down(x_fused)
        x_fused_c = x_fused * self.channel_attention(x_fused)
        # print("x_fused_c:", x_fused_c.shape)
        x_3x3 = self.conv_3x3(x_fused)
        # print("x_3x3:", x_3x3.shape)

        x_5x5 = self.conv_5x5(x_fused)
        # print("x_5x5:", x_5x5.shape)

        x_7x7 = self.conv_7x7(x_fused)
        # print("x_7x7:", x_7x7.shape)

        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.conv(x_fused_s + x_fused_c)

        return x_out


if __name__ == '__main__':
    from thop import clever_format, profile
    x = torch.randn(1, 64, 11, 11).cuda()
    y = torch.randn(1, 64, 22, 22).cuda()
    z = torch.randn(1, 64, 44, 44).cuda()

    model = AMSF(64*3, 64).cuda().eval()
    # model= aggregation(64).cuda().eval()

    out = model(x, y, z)
    print(out.shape)

    flops, params = profile(model, inputs=(x, y, z), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total params: %s' % (params))
    print('Total FLOPS: %s' % (flops))


# if __name__ == '__main__':
#
#     x4_rfb = torch.randn(1, 64, 11, 11).cuda()
#     x3_rfb = torch.randn(1, 64, 22, 22).cuda()
#     x2_rfb = torch.randn(1, 64, 44, 44).cuda()
#
#     model = aggregationnew(64)
#     print()
#     # model= aggregation(64)
#     model.cuda()
#     model.eval()
#
#     # out1, out2 = model(x4_rfb, x3_rfb, x2_rfb)
#     out_features = model(x4_rfb, x3_rfb, x2_rfb)
#     print("out_features:", out_features.size())
#
#     flops, params = profile(model, inputs=(x4_rfb, x3_rfb, x2_rfb), verbose=False)
#     flops, params = clever_format([flops * 2, params], "%.3f")
#
#     print('Total params: %s' % (params))
#     print('Total FLOPS: %s' % (flops))
