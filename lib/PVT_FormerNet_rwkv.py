import torch.nn as nn
import torch
# from DCNv4.modules import DCNv4
from torch.nn import init
from thop import clever_format, profile
import torch.nn.functional as F
from MSA_Transformer import MSA_module
from pvtv2 import my_pvt_v2_b2
from segmentation_models_pytorch.base.heads import SegmentationHead
from Vision_RWKV.rwkv import VRWKV_Bottleneck


class BasicConv2d(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.silu(x)
        return x

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


class FormerNet(nn.Module):
    def __init__(self, channel=64):
        super(FormerNet, self).__init__()

        # self.backbone = timm.create_model('pvt_v2_b2', pretrained=True, features_only=True)
        self.backbone = my_pvt_v2_b2()
        path = '../weight/backbone/pvt_v2_b2.pth'
        save_model = torch.load(path, weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # self.rfb1_1 = RFB_modified(64, channel)
        # self.rfb2_1 = RFB_modified(128, channel)
        # self.rfb3_1 = RFB_modified(320, channel)
        # self.rfb4_1 = RFB_modified(512, channel)

        self.rwkv2 = VRWKV_Bottleneck(n_embd=128,dim=channel,n_layer=12,layer_id=0,
                                      shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
                                      hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)

        self.rwkv3 = VRWKV_Bottleneck(n_embd=320,dim=channel,n_layer=12,layer_id=0,
                                      shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
                                      hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)

        self.rwkv4 = VRWKV_Bottleneck(n_embd=512,dim=channel,n_layer=12,layer_id=0,
                                      shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
                                      hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)


        self.fuse4 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))
        self.fuse3 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))
        self.fuse2 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))

        self.agg = aggregation(channel)
        # self.agg = AMSF(channel*3, channel)

        self.MSA4 = MSA_module(dim=channel)
        self.MSA3 = MSA_module(dim=channel)
        self.MSA2 = MSA_module(dim=channel)

        self.out4 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.out3 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(channel, 1, kernel_size=3, padding=1)

        self.agg_head = SegmentationHead(channel, 1, kernel_size=3)
        self.out4_head = SegmentationHead(channel, 1, kernel_size=3)
        self.out3_head = SegmentationHead(channel, 1, kernel_size=3)
        self.out2_head = SegmentationHead(channel, 1, kernel_size=3)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        # print("x1:", x1.size())
        # print("x2:", x2.size())
        # print("x3:", x3.size())
        # print("x4:", x4.size())

        # x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rwkv2(x2)
        x3_rfb = self.rwkv3(x3)
        x4_rfb = self.rwkv4(x4)

        # print("x1_rfb", x1_rfb.size())
        # print("x2_rfb", x2_rfb.size())
        # print("x3_rfb", x3_rfb.size())
        # print("x4_rfb", x4_rfb.size())

        agg_fea = self.agg(x4_rfb, x3_rfb, x2_rfb)
        agg_out = self.agg_head(agg_fea)

        x3_rfb = F.interpolate(x3_rfb, scale_factor=2, mode='bilinear')
        # print("x3_rfb_:", x3_rfb.size())
        x4_rfb = F.interpolate(x4_rfb, scale_factor=4, mode='bilinear')
        # print("x4_rfb_:", x4_rfb.size())

        E4 = torch.cat((x4_rfb, agg_fea), dim=1)
        E3 = torch.cat((x3_rfb, agg_fea), dim=1)
        E2 = torch.cat((x2_rfb, agg_fea), dim=1)

        # print("E4:", E4.size())
        # print("E3:", E3.size())
        # print("E2:", E2.size())

        E4 = F.relu(self.fuse4(E4), inplace=True)
        E3 = F.relu(self.fuse3(E3), inplace=True)
        E2 = F.relu(self.fuse2(E2), inplace=True)

        # print("E4:", E4.size())     # E4: torch.Size([1, 64, 44, 44])
        # print("E3:", E3.size())     # E3: torch.Size([1, 64, 44, 44])
        # print("E2:", E2.size())     # E2: torch.Size([1, 64, 44, 44])

        D4 = self.MSA4(agg_fea, E4, agg_out)
        # print("D4:", D4.size())
        out4 = self.out4_head(D4)

        # print("out4:", out4.size())

        D3 = self.MSA3(D4, E3, out4)
        # print("D3:", D3.size())
        out3 = self.out3_head(D3)

        # print("out3:", out3.size())

        D2 = self.MSA2(D3, E2, out3)
        # print("D2:", D2.size())
        out2 = self.out2_head(D2)

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

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    print("agg_out:", out[0].size())
    print("out4:", out[1].size())
    print("out3:", out[2].size())
    print("out2:", out[3].size())

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))


