import torch
import torch.nn as nn
import torch.nn.functional as F
# from .Res2Net_v1b import res2net50_v1b_26w_4s
from transformers import AutoModel
# from .Res2Net_v1b import *
from thop import clever_format, profile
from DCNv4.modules import DCNv4


def conv_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True)
    )
    return model

def conv_block_2(in_dim, out_dim):
    model = nn.Sequential(
        conv_block(in_dim, out_dim),
        nn.Conv2d(out_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.BatchNorm2d(out_dim)
    )

    return model


class BasicConv2d(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x

class BasicConv2d_act(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_act, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DCNv4_Conv2d(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(DCNv4_Conv2d, self).__init__()

        # if in_planes != out_planes:
        #     self.conv1 = BasicConv2d(in_planes, out_planes, 1)

        self.dcnv4 = DCNv4(out_planes, kernel_size=kernel_size, stride=stride, pad=padding, dilation=dilation)
        # self.conv = nn.Conv2d(in_planes, out_planes,
        #                       kernel_size=kernel_size, stride=stride,
        #                       padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # if hasattr(self, 'conv1'):
        #     x = self.conv1(x)
        x = self.dcnv4(x, (x.size(2), x.size(3)))
        x = self.bn(x)
        x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

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


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d_act(in_channel, out_channel, 1),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            channel_attention(out_channel)

        )
        self.branch1 = nn.Sequential(
            BasicConv2d_act(in_channel, out_channel, 1),
            BasicConv2d_act(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d_act(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            channel_attention(out_channel)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d_act(in_channel, out_channel, 1),
            BasicConv2d_act(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d_act(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            channel_attention(out_channel)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d_act(in_channel, out_channel, 1),
            BasicConv2d_act(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d_act(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=7, dilation=7),
            channel_attention(out_channel)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

        self.sa = spatial_attention()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))

        x = self.sa(x)

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
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

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
        x = self.conv5(x)

        return x


# class aggregation(nn.Module):
#     # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
#     # used after MSF
#     def __init__(self, channel):
#         super(aggregation, self).__init__()
#
#         self.stem_conv = conv_block_2(channel * 4, channel)
#
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#
#         self.conv_upsample = BasicConv2d(channel, channel, 3, padding=1)
#
#         self.out_conv1 = conv_block(channel, channel)
#         self.global_map_output = nn.Conv2d(channel, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
#         self.out_conv2 = conv_block(channel * 2, channel)
#
#         self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
#         self.conv5 = nn.Conv2d(3*channel, 1, 1)
#         self.conv = conv_block(channel, 1)
#
#         self.alpha = nn.Parameter(torch.ones(1))
#         self.beta = nn.Parameter(torch.ones(1))
#
#     def forward(self, x1, x2, x3, x4):
#
#         x1_1 = self.upsample2(self.upsample2(self.upsample2(x1)))
#         # print("x1_1:", x1_1.size())
#         x1_1 = self.conv_upsample(x1_1)
#
#         x2_1 =  self.upsample2(self.upsample2(x2))
#         # print("x2_1:", x2_1.size())
#         x2_1 = self.conv_upsample(x2_1)
#
#         x3_1 = self.upsample2(x3)
#         # print("x3_1:", x3_1.size())
#         x3_1 = self.conv_upsample(x3_1)
#
#         x4_1 = x4
#         x4_1 = self.conv_upsample(x4_1)
#         # print("x4_1:", x4_1.size())
#
#         x = torch.cat((x1_1, x2_1, x3_1, x4_1), dim=1)
#         x = self.stem_conv(x)
#         x = self.out_conv1(x)
#         # print("x:", x.size())
#
#         global_map = self.global_map_output(x)
#         # print("global_map:", global_map.size())
#         reverse_global_map = 1 - torch.sigmoid(global_map)
#         # print("reverse_global_map:", reverse_global_map.size())
#
#         output = self.out_conv2(torch.cat([(x * torch.sigmoid(global_map)) * self.alpha, (x * reverse_global_map) * self.beta], dim=1))
#
#         output = self.conv(output)
#
#         # print("output:", output.size())
#
#         return output


class ASPPModule(nn.Module) :
    def __init__(self, in_channels, out_channels, act_fn):
        super(ASPPModule, self).__init__()

        self.stem_conv = conv_block_2(in_channels * 3, in_channels, act_fn)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv3x3_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv3x3_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.conv3x3_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(18, 18), dilation=(18, 18)),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

        self.out_conv1 = conv_block(out_channels, out_channels, act_fn)
        self.global_map_output = nn.Conv2d(out_channels, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.out_conv2 = conv_block(out_channels * 2, out_channels, act_fn)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, low, full, high):

        x = torch.cat([low, full, high], dim=1)

        stem_x = self.stem_conv(x)

        branch1 = self.conv1x1(stem_x)
        branch2 = self.conv3x3_6(stem_x)
        branch3 = self.conv3x3_12(stem_x)
        branch4 = self.conv3x3_18(stem_x)

        aspp = branch1 + branch2 + branch3 + branch4
        output = self.out_conv1(aspp)
        global_map = self.global_map_output(output)
        reverse_global_map = 1 - torch.sigmoid(global_map)

        output = self.out_conv2(torch.cat([(output * torch.sigmoid(global_map)) * self.alpha, (output * reverse_global_map) * self.beta], dim=1))

        return output, global_map

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class RFB_SDI_Seg(nn.Module):
    def __init__(self, channel=64):
        super(RFB_SDI_Seg, self).__init__()
        # ---- ResNet Backbone ----
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.mamba = AutoModel.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True)
        # self.resnet = timm.create_model('resnet50', pretrained=False, features_only = True)
        # ---- Receptive Field Block like module ----
        self.rfb1_1 = RFB_modified(96, channel)
        self.rfb2_1 = RFB_modified(192, channel)
        self.rfb3_1 = RFB_modified(384, channel)
        self.rfb4_1 = RFB_modified(768, channel)

        self.agg = aggregation(channel)

        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=1)

    def forward(self, x):

        _, features = self.mamba(x)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        # x1, x2, x3, x4 = self.resnet(x)
        # print("x1:", x1.size())  # [1, 64, 128, 128]
        # print("x2:", x2.size())  # [1, 256, 64, 64]
        # print("x3:", x3.size())  # [1, 512, 32, 32]
        # print("x4:", x4.size())  # [1, 1024, 16, 16]

        x1_rfb = self.rfb1_1(x1)
        # print("x1_rfb:", x1_rfb.size())

        x2_rfb = self.rfb2_1(x2)
        # print("x2_rfb", x2_rfb.size())  # [1, 128, 64, 64]

        x3_rfb = self.rfb3_1(x3)
        # print("x3_rfb", x3_rfb.size())  # [1, 128, 32, 32]

        x4_rfb = self.rfb4_1(x4)
        # print("x4_rfb", x4_rfb.size())

        # agg_fea = self.agg(x4_rfb, x3_rfb, x2_rfb, x1_rfb)
        agg_fea = self.agg(x4_rfb, x3_rfb, x2_rfb)
        # print("agg_fea:", agg_fea.size())
        out4_fea = F.interpolate(agg_fea, scale_factor=8, mode='bilinear')
        # print("out4_fea", out4_fea.size())

        out1 = self.up1(x4_rfb, x3_rfb)
        # print("out1:", out1.size())
        out1_fea = F.interpolate(self.side1(out1), scale_factor=16, mode='bilinear')
        # print("out1_fea:", out1_fea.size())

        out2 = self.up2(out1, x2_rfb)
        # print("out2:", out2.size())
        out2_fea = F.interpolate(self.side2(out2), scale_factor=8, mode='bilinear')
        # print("out2_fea:", out2_fea.size())

        out3 = self.up3(out2, x1_rfb)
        # print("out3:", out3.size())
        out3_fea = F.interpolate(self.head(out3), scale_factor=4, mode='bilinear')
        # print("out3_fea:", out3_fea.size())

        # return out4_fea, out3_fea, out2_fea, out1_fea
        return out4_fea, out1_fea, out2_fea, out3_fea

if __name__ == '__main__':

    input = torch.randn(1, 3, 256, 256).cuda()

    model = RFB_SDI_Seg()
    model.cuda()

    out = model(input)

    flops,params = profile(model.cuda(0), (input,), verbose=False)
    flops,params = clever_format([flops * 2, params], "%.3f")
    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))