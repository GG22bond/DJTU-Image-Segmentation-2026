import torch.nn as nn
import torch
from thop import clever_format, profile
import torch.nn.functional as F
from MSA_Transformer import MSA_module
from pvtv2 import my_pvt_v2_b2
from segmentation_models_pytorch.base.heads import SegmentationHead
# from Vision_RWKV.rwkv import VRWKV_Bottleneck

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

        x_5x5 = self.conv_5x5(x_fused)

        x_7x7 = self.conv_7x7(x_fused)

        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)

        x_out = self.conv(x_fused_s + x_fused_c)

        return x_out

################  MCA 模块  #################################
####  https://github.com/ChengLong1222/MSRAformer-main
####  https://doi.org/10.1016/j.compbiomed.2022.106274

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight * x


class MConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.mconv1 = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.mconv2 = BasicConv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2)
        self.mconv3 = BasicConv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3)

        self.decay_weight2 = nn.Parameter(torch.zeros(1, 1, 1))
        self.decay_weight3 = nn.Parameter(torch.zeros(1, 1, 1))

        self.se = SEWeightModule(out_channel)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid
        self.catc = BasicConv2d(3 * out_channel, out_channel, 1, 1)
        self.res_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                                       nn.BatchNorm2d(out_channel))
        self.res_back_conv = nn.Conv2d(out_channel, in_channel, 1, 1)

        nn.init.trunc_normal_(self.decay_weight2, a=0.5, b=0.75)
        nn.init.trunc_normal_(self.decay_weight3, a=0.25, b=0.5)

    def forward(self, x):
        ideneity = x
        x1 = self.mconv1(x)
        x2 = self.mconv2(x)
        x3 = self.mconv3(x)

        x1_ch = x1
        x2_ch = self.decay_weight2 * x2
        x3_ch = self.decay_weight3 * x3

        x_cat = torch.cat((x1_ch, x2_ch, x3_ch), dim=1)
        x_catc = self.catc(x_cat)
        x_se = self.se(x_catc)

        out = self.relu(x_se + self.res_conv(ideneity))

        return out

################  MCA 模块  #################################


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


        self.mconv2 = MConv(128, channel)
        self.mconv3 = MConv(320, channel)
        self.mconv4 = MConv(512, channel)


        # self.rwkv2 = VRWKV_Bottleneck(n_embd=128,dim=channel,n_layer=12,layer_id=0,
        #                               shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
        #                               hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)
        #
        # self.rwkv3 = VRWKV_Bottleneck(n_embd=320,dim=channel,n_layer=12,layer_id=0,
        #                               shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
        #                               hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)
        #
        # self.rwkv4 = VRWKV_Bottleneck(n_embd=512,dim=channel,n_layer=12,layer_id=0,
        #                               shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
        #                               hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)


        self.fuse4 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))
        self.fuse3 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))
        self.fuse2 = nn.Sequential(nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel))

        self.agg = AMSF(3 * channel, channel)

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

        print("x2:", x2.shape)
        print("x3:", x3.shape)
        print("x4:", x4.shape)

        # x2_rfb = self.rwkv2(x2)
        # x3_rfb = self.rwkv3(x3)
        # x4_rfb = self.rwkv4(x4)

        x2_rfb = self.mconv2(x2)
        x3_rfb = self.mconv3(x3)
        x4_rfb = self.mconv4(x4)

        print("x2_rfb:", x2_rfb.shape)
        print("x3_rfb:", x3_rfb.shape)
        print("x4_rfb:", x4_rfb.shape)

        agg_fea = self.agg(x4_rfb, x3_rfb, x2_rfb)
        agg_out = self.agg_head(agg_fea)

        x3_rfb = F.interpolate(x3_rfb, scale_factor=2, mode='bilinear')

        x4_rfb = F.interpolate(x4_rfb, scale_factor=4, mode='bilinear')

        E4 = torch.cat((x4_rfb, agg_fea), dim=1)
        E3 = torch.cat((x3_rfb, agg_fea), dim=1)
        E2 = torch.cat((x2_rfb, agg_fea), dim=1)

        E4 = F.relu(self.fuse4(E4), inplace=True)
        E3 = F.relu(self.fuse3(E3), inplace=True)
        E2 = F.relu(self.fuse2(E2), inplace=True)

        D4 = self.MSA4(agg_fea, E4, agg_out)

        out4 = self.out4_head(D4)

        D3 = self.MSA3(D4, E3, out4)

        out3 = self.out3_head(D3)

        D2 = self.MSA2(D3, E2, out3)

        out2 = self.out2_head(D2)

        agg_out = F.interpolate(agg_out, scale_factor=8, mode='bilinear')
        out4 = F.interpolate(out4, scale_factor=8, mode='bilinear')
        out3 = F.interpolate(out3, scale_factor=8, mode='bilinear')
        out2 = F.interpolate(out2, scale_factor=8, mode='bilinear')


        return agg_out, out4, out3, out2


if __name__ == '__main__':
    input = torch.randn(1, 3, 352, 352).cuda()

    model = FormerNet()

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))




