import torch.nn as nn
import torch
from thop import clever_format, profile
import torch.nn.functional as F
from MSA_Transformer import MSA_module
from pvtv2 import my_pvt_v2_b2, my_pvt_v2_b3, my_pvt_v2_b4
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


        self.rwkv2 = VRWKV_Bottleneck(n_embd=128,dim=channel,n_layer=12,layer_id=0,
                                      shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
                                      hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)

        self.rwkv3 = VRWKV_Bottleneck(n_embd=320,dim=channel,n_layer=12,layer_id=0,
                                      shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
                                      hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)

        self.rwkv4 = VRWKV_Bottleneck(n_embd=512,dim=channel,n_layer=12,layer_id=0,
                                      shift_mode='q_shift', channel_gamma=1/4, shift_pixel=1,
                                      hidden_rate=4, init_mode='fancy', drop_path=0.1, k_norm=True)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.agg = AMSF(3 * channel, channel)

        self.agg_head = SegmentationHead(channel, 1, kernel_size=3)


    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        x2_rfb = self.rwkv2(x2)
        x3_rfb = self.rwkv3(x3)
        x4_rfb = self.rwkv4(x4)

        agg_fea = self.agg(x4_rfb, x3_rfb, x2_rfb)
        agg_out = self.agg_head(agg_fea)

        lateral_map_5 = F.interpolate(agg_out, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # print("lateral_map_5", lateral_map_5.size())  # 1, 1, 352, 352

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(agg_out, scale_factor=0.25, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        # print("x:", x.size())
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # print("lateral_map_4", lateral_map_4.size())  # 1, 1, 352, 352

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 320, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        # print("lateral_map_3", lateral_map_3.size())   # 1, 1, 352, 352

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 128, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # print("lateral_map_2", lateral_map_2.size())  # 1, 1, 352, 352

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2



if __name__ == '__main__':
    input = torch.randn(1, 3, 352, 352).cuda()

    model = FormerNet()

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    print("out0:", out[0].size())
    print("out1:", out[1].size())
    print("out2:", out[2].size())
    print("out3:", out[3].size())

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))




