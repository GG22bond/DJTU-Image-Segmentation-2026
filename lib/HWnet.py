import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s
# from .Res2Net_v1b import *
from thop import clever_format, profile
import math

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

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
        return x

class output_block(nn.Module):

    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.c1 = CBR(in_c, 128, kernel_size=3, padding=1)
        self.c2 = CBR(128, 64, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sig=nn.Sigmoid()

    def forward(self, x):

        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x=self.sig(x)
        return x


# class output_block(nn.Module):
#
#     def __init__(self, in_c, out_c=1):
#         super().__init__()
#
#         self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
#
#         self.fuse = CBR(in_c*3, in_c, kernel_size=3, padding=1)
#         self.c1 = CBR(in_c, 128, kernel_size=3, padding=1)
#         self.c2 = CBR(128, 64, kernel_size=1, padding=0)
#         self.c3 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x1, x2, x3):
#
#         x2 = self.up_2x2(x2)
#         # print("x2:", x2.size())
#         x3 = self.up_4x4(x3)
#         # print("x3:", x3.size())
#
#         x = torch.cat((x1, x2, x3), dim=1)
#
#         x=self.fuse(x)
#         # print("x", x.size())
#         x=self.up_2x2(x)
#         x = self.c1(x)
#         # print("x", x.size())
#         x = self.c2(x)
#         # print("x`", x.size())
#         x = self.c3(x)
#         # print("x``", x.size())
#         x=self.sig(x)
#         # print("x```", x.size())
#         return x

class Head(nn.Module):
    def __init__(self, large, middle, small):
        super(Head, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(large, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()

        )
        self.branch_bg = nn.Sequential(
            CBR(middle, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(small, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_large, f_middle, f_small):
        large = self.branch_fg(f_large)
        middle = self.branch_bg(f_middle)
        small = self.branch_uc(f_small)
        return large, middle, small

class HaarWaveletConv(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWaveletConv, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x):
        B, _, H, W = x.size()
        x = F.pad(x, [0, 1, 0, 1], value=0)
        out = F.conv2d(x, self.haar_weights, bias=None, stride=1, groups=self.in_channels) / 4.0
        out = out.reshape([B, self.in_channels, 4, H, W])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([B, self.in_channels * 4, H, W])

        # a (approximation): 低频信息，图像的平滑部分，代表了图像的整体结构。
        # h (horizontal): 水平方向的高频信息，捕捉水平方向上的边缘或变化。
        # v (vertical): 垂直方向的高频信息，捕捉垂直方向上的边缘或变化。
        # d (diagonal): 对角线方向的高频信息，捕捉对角线方向上的边缘或纹理。
        a, h, v, d = out.chunk(4, 1)

        # 低频，高频
        return a, h + v + d

class CDFAPreprocess(nn.Module):

    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        # self.ca = channel_attention(out_c)
        # self.sa = spatial_attention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), 1)

        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)
        # x = self.ca(x)
        # x = self.sa(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)

        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5


        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


class HWNet(nn.Module):
    def __init__(self, channel= 128):
        super(HWNet, self).__init__()

        # backbone
        self.backbone = res2net50_v1b_26w_4s(pretrained=True)

        self.rfb0 = RFB_modified(256, channel)
        self.rfb1 = RFB_modified(512, channel)
        self.rfb2 = RFB_modified(1024, channel)
        self.rfb3 = RFB_modified(2048, channel)

        self.hw0 = HaarWaveletConv(128)
        self.hw1 = HaarWaveletConv(128)
        self.hw2 = HaarWaveletConv(128)
        self.hw3 = HaarWaveletConv(128)

        self.process0 = CDFAPreprocess(256, 128, 1)
        self.process1 = CDFAPreprocess(512, 128, 1)
        self.process2 = CDFAPreprocess(1024, 128, 1)
        self.process3 = CDFAPreprocess(2048, 128, 1)

        self.cdaf0 = ContrastDrivenFeatureAggregation(256, 128, 4)
        self.cdaf1 = ContrastDrivenFeatureAggregation(256, 128, 4)
        self.cdaf2 = ContrastDrivenFeatureAggregation(256, 128, 4)
        self.cdaf3 = ContrastDrivenFeatureAggregation(128, 128, 4)

        self.decoder_small = decoder_block(128, 128, scale=2)
        self.decoder_middle = decoder_block(128, 128, scale=2)
        self.decoder_large = decoder_block(128, 128, scale=2)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.output = output_block(128, 1)


    def forward(self, x):

        x0, x1, x2, x3 = self.backbone(x)
        # print("x0:", x0.size())  # [1, 64, 128, 128]
        # print("x1:", x1.size())  # [1, 256, 64, 64]
        # print("x2:", x2.size())  # [1, 512, 32, 32]
        # print("x3:", x3.size())  # [1, 1024, 16, 16]

        # low3, high3 = self.hw3(x3)
        # low2, high2 = self.hw2(x2)
        # low1, high1 = self.hw1(x1)
        # low0, high0 = self.hw0(x0)

        # print("low3:", low3.size())
        # print("high3:", high3.size())
        # print("low2:", low2.size())
        # print("high2:", high2.size())
        # print("low1:", low1.size())
        # print("high1:", high1.size())
        # print("low0:", low0.size())
        # print("high0:", high0.size())

        rfb3 = self.rfb3(x3)
        rfb2 = self.rfb2(x2)
        rfb1 = self.rfb1(x1)
        rfb0 = self.rfb0(x0)

        # print("rfb3:", rfb3.size())
        # print("rfb2:", rfb2.size())
        # print("rfb1:", rfb1.size())
        # print("rfb0:", rfb0.size())

        low3, high3 = self.hw3(rfb3)
        low2, high2 = self.hw2(rfb2)
        low1, high1 = self.hw1(rfb1)
        low0, high0 = self.hw0(rfb0)

        # print("low3:", low3.size())
        # print("high3:", high3.size())
        # print("low2:", low2.size())
        # print("high2:", high2.size())
        # print("low1:", low1.size())
        # print("high1:", high1.size())
        # print("low0:", low0.size())
        # print("high0:", high0.size())

        # low_pro3 = self.process3(low3)
        # # print("low_pro3:", low_pro3.size())
        # high_pro3 = self.process3(high3)
        # # print("high_pro3:", high_pro3.size())
        #
        # low_pro2 = self.process2(low2)
        # # print("low_pro2:", low_pro2.size())
        # high_pro2 = self.process2(high2)
        # # print("high_pro2:", high_pro2.size())
        #
        # low_pro1 = self.process1(low1)
        # # print("low_pro1:", low_pro1.size())
        # high_pro1 = self.process1(high1)
        # # print("high_pro1:", high_pro1.size())
        #
        # low_pro0 = self.process0(low0)
        # # print("low_pro0:", low_pro0.size())
        # high_pro0 = self.process0(high0)
        # # print("high_pro0:", high_pro0.size())

        cdaf3 = self.cdaf3(rfb3, low3, high3)
        # print("cdaf3:", cdaf3.size())

        up_cdaf3 = self.up2(cdaf3)
        # print("up_cdaf3:", up_cdaf3.size())

        f_cat3 = torch.cat((rfb2, up_cdaf3), dim=1)
        # print("f_cat3:", f_cat3.size())

        cdaf2 = self.cdaf2(f_cat3, low2, high2)
        # print("cdaf2:", cdaf2.size())

        up_cdaf2 = self.up2(cdaf2)
        # print("up_cdaf2:", up_cdaf2.size())

        f_cat2 = torch.cat((rfb1, up_cdaf2), dim=1)
        # print("f_cat2:", f_cat2.size())

        cdaf1 = self.cdaf1(f_cat2, low1, high1)
        # print("cdaf1:", cdaf1.size())

        up_cdaf1 = self.up2(cdaf1)
        # print("up_cdaf1:", up_cdaf1.size())

        f_cat1 = torch.cat((rfb0, up_cdaf1), dim=1)
        # print("f_cat1:", f_cat1.size())

        cdaf0 = self.cdaf0(f_cat1, low0, high0)
        # print("cdaf0:", cdaf0.size())

        f_large = self.decoder_large(cdaf3, cdaf2)
        # print("f_large:", f_large.size())

        f_middle = self.decoder_middle(cdaf2, cdaf1)
        # print("f_middle:", f_middle.size())

        f_small = self.decoder_small(cdaf1, cdaf0)
        # print("f_small:", f_small.size())

        up_large = self.up4(f_large)
        # print("up_large:", up_large.size())

        up_middle = self.up2(f_middle)
        # print("up_middle:", up_middle.size())

        up_small = f_small
        # print("up_small:", up_small.size())

        large = self.output(up_large)
        large = F.interpolate(large, scale_factor=4, mode='bilinear')

        middle = self.output(up_middle)
        middle = F.interpolate(middle, scale_factor=4, mode='bilinear')

        small = self.output(up_small)
        small = F.interpolate(small, scale_factor=4, mode='bilinear')

        # print("large:", large.size())
        # print("middle:", middle.size())
        # print("small:", small.size())

        return large, middle, small

if __name__ == '__main__':

    input = torch.randn(1, 3, 320, 320).cuda()

    model = HWNet()
    model.cuda()

    out = model(input)
    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))