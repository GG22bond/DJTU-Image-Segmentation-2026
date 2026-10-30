import torch.nn as nn
import torch
from DCNv4.modules import DCNv4
from torch.nn import init
import timm
from transformers import AutoModel
from thop import clever_format, profile
import torch.nn.functional as F
# from MSA_Transformer import MSA_module
from MSA_Transformer_new_ln import MSA_module
# from .dynamic_tanh import convert_ln_to_dyt


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
        return x

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


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()

        self.silu = nn.SiLU(True)
        
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            channel_attention(out_channel)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            channel_attention(out_channel)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            DCNv4_Conv2d(out_channel, out_channel, 3, padding=7, dilation=7),
            channel_attention(out_channel)
        )

        # self.ema = EMA(out_channel)
        self.sa = spatial_attention()
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        # print("x_cat:", x_cat.size())

        # x_ema = self.ema(x_cat)
        x_sa = self.sa(x_cat)
        # print("x_se:", x_se.size())

        x = self.silu(x_sa + self.conv_res(x))
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

        self.conv1 = nn.Conv2d(channel, 1, 1)

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

        out = self.conv1(x)

        return x, out


class Fuse(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fuse, self).__init__()

        self.fuse = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channel))

    def forward(self, rfb, agg):
        E = torch.cat((rfb, agg), dim=1)
        E_fea = F.relu(self.fuse(E), inplace=True)
        return E_fea


class FormerNet(nn.Module):
    def __init__(self, channel=64):
        super(FormerNet, self).__init__()

        self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        # self.backbone = my_pvt_v2_b2()
        # path = './weight/backbone/pvt_v2_b2.pth'
        # save_model = torch.load(path, weights_only=True)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)

        self.rfb1_1 = RFB_modified(64, channel)
        self.rfb2_1 = RFB_modified(128, channel)
        self.rfb3_1 = RFB_modified(320, channel)
        self.rfb4_1 = RFB_modified(512, channel)

        # self.fuse4 = Fuse(channel * 2, channel)
        # self.fuse3 = Fuse(channel * 2, channel)
        # self.fuse2 = Fuse(channel * 2, channel)

        self.fuse4 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))
        self.fuse3 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))
        self.fuse2 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))

        self.agg = aggregation(channel)

        self.MSA4 = MSA_module(dim=channel)
        self.MSA3 = MSA_module(dim=channel)
        self.MSA2 = MSA_module(dim=channel)

        self.out4 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.out3 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)

    def forward(self, x):

        x1, x2, x3, x4 = self.backbone(x)

        # print("x1:", x1.size())
        # print("x2:", x2.size())
        # print("x3:", x3.size())
        # print("x4:", x4.size())

        x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        # print("x1_rfb", x1_rfb.size())
        # print("x2_rfb", x2_rfb.size())
        # print("x3_rfb", x3_rfb.size())
        # print("x4_rfb", x4_rfb.size())

        agg_fea, agg_out = self.agg(x4_rfb, x3_rfb, x2_rfb)
        # agg_fea = self.conv3(agg_fea)
        # print("agg_fea:", agg_fea.size())

        # agg_out = self.conv1(agg_fea)
        # print("agg_out:", agg_out.size())

        x3_rfb = F.interpolate(x3_rfb, scale_factor=2, mode='bilinear')
        # print("x3_rfb_:", x3_rfb.size())
        x4_rfb = F.interpolate(x4_rfb, scale_factor=4, mode='bilinear')
        # print("x4_rfb_:", x4_rfb.size())

        E4 = torch.cat((x4_rfb, agg_fea), dim=1)
        E3 = torch.cat((x3_rfb, agg_fea), dim=1)
        E2 = torch.cat((x2_rfb, agg_fea), dim=1)
        E4 = F.relu(self.fuse4(E4), inplace=True)
        E3 = F.relu(self.fuse3(E3), inplace=True)
        E2 = F.relu(self.fuse2(E2), inplace=True)

        # E4 = self.fuse4(x4_rfb, agg_fea)
        # E3 = self.fuse3(x3_rfb, agg_fea)
        # E2 = self.fuse2(x2_rfb, agg_fea)

        # print("E4:", E4.size())     # E4: torch.Size([1, 64, 44, 44])
        # print("E3:", E3.size())     # E3: torch.Size([1, 64, 44, 44])
        # print("E2:", E2.size())     # E2: torch.Size([1, 64, 44, 44])

        D4 = self.MSA4(agg_fea, E4, agg_out)
        # print("D4:", D4.size())
        out4 = self.out4(D4)

        # print("out4:", out4.size())

        D3 = self.MSA3(D4, E3, out4)
        # print("D3:", D3.size())
        out3 = self.out3(D3)

        # print("out3:", out3.size())

        D2 = self.MSA2(D3, E2, out3)
        # print("D2:", D2.size())
        out2 = self.out2(D2)

        # print("out2:", out2.size())

        agg_out = F.interpolate(agg_out, scale_factor=8, mode='bilinear')
        out4 = F.interpolate(out4, scale_factor=8, mode='bilinear')
        out3 = F.interpolate(out3, scale_factor=8, mode='bilinear')
        out2 = F.interpolate(out2, scale_factor=8, mode='bilinear')

        # print("agg_out:", agg_out.size())
        # print("out4:", out4.size())
        # print("out3:", out3.size())
        # print("out2:", out2.size())

        return agg_out, out4, out3, out2


if __name__ == '__main__':

    input = torch.randn(1, 3, 352, 352).cuda()

    model = FormerNet()

    print("Model:\n",model)
    model.cuda()
    model.eval()

    out = model(input)

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
