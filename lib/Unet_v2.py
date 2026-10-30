import os.path
import warnings
import torch
from torch import nn
from pvtv2 import my_pvt_v2_b2
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
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


class Encoder(nn.Module):
    def __init__(self, pretrain_path):
        super().__init__()
        self.backbone = my_pvt_v2_b2()

        if pretrain_path is None:
            warnings.warn('please provide the pretrained pvt model. Not using pretrained model.')
        elif not os.path.isfile(pretrain_path):
            warnings.warn(f'path: {pretrain_path} does not exists. Not using pretrained model.')
        else:
            print(f"using pretrained file: {pretrain_path}")
            save_model = torch.load(pretrain_path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)

            self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)  # (x: 3, 352, 352)
        return f1, f2, f3, f4


class SDI1(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):

        # print("anchor_shape:", anchor.shape)
        ans = torch.ones_like(anchor)
        # print("ans:", ans.shape)
        target_size = anchor.shape[-1]
        # print("target_size:", target_size)

        for i, x in enumerate(xs):
            # print("#####################################################################################")
            # print("x_shape:", x.shape, "x.shape[-1]:",x.shape[-1])
            # print("\n")

            if x.shape[-1] > target_size:
                # print("x.shape[-1] > target_size:","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
                # print("x.shape:", x.shape)

            elif x.shape[-1] < target_size:
                # print("x.shape[-1] < target_size:","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True)
                # print("x.shape:", x.shape)

            else:
                pass

            ans = ans * self.convs[i](x)

        return ans


class SDI2(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):

        # print("anchor_shape:", anchor.shape)
        ans = torch.ones_like(anchor)
        # print("ans:", ans.shape)
        target_size = anchor.shape[-1]
        # print("target_size:", target_size)

        for i, x in enumerate(xs):

            # print("#####################################################################################")
            # print("x_shape:", x.shape, "x.shape[-1]:",x.shape[-1])
            # print("\n")

            if x.shape[-1] > target_size:
                # print("x.shape[-1] > target_size,","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
                # print("x.shape:", x.shape)

            elif x.shape[-1] < target_size:
                # print("x.shape[-1] < target_size, ","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True)
                # print("x.shape:", x.shape)

            else:
                pass

            ans = ans * self.convs[i](x)

        return ans


class SDI3(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):

        # print("anchor_shape:", anchor.shape)
        ans = torch.ones_like(anchor)
        # print("ans:", ans.shape)
        target_size = anchor.shape[-1]
        # print("target_size:", target_size)

        for i, x in enumerate(xs):

            # print("#####################################################################################")
            # print("x_shape:", x.shape, "x.shape[-1]:",x.shape[-1])
            # print("\n")

            if x.shape[-1] > target_size:
                # print("x.shape[-1] > target_size,","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
                # print("x.shape:", x.shape)

            elif x.shape[-1] < target_size:
                # print("x.shape[-1] < target_size, ","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True)
                # print("x.shape:", x.shape)

            else:
                pass

            ans = ans * self.convs[i](x)

        return ans

class SDI4(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        # print("anchor_shape:", anchor.shape)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                # print("x.shape[-1] > target_size,","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))

            elif x.shape[-1] < target_size:
                # print("x.shape[-1] < target_size, ","x.shape:", x.shape, "x.shape[-1]:",x.shape[-1], "target_size:", target_size)
                x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=True)

            # print("x", x.shape)

            ans = ans * self.convs[i](x)

        return ans


class UNetV2(nn.Module):
    """
    use SpatialAtt + ChannelAtt
    """
    def __init__(self, channel=32, n_classes=1, pretrained_path="./weight/backbone/pvt_v2_b2.pth"):
        super().__init__()
        # self.deep_supervision = deep_supervision

        self.encoder = Encoder(pretrained_path)

        self.ca_1 = ChannelAttention(64)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(128)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(320)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(512)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(64, channel, 1)
        self.Translayer_2 = BasicConv2d(128, channel, 1)
        self.Translayer_3 = BasicConv2d(320, channel, 1)
        self.Translayer_4 = BasicConv2d(512, channel, 1)

        self.sdi_1 = SDI1(channel)
        self.sdi_2 = SDI2(channel)
        self.sdi_3 = SDI3(channel)
        self.sdi_4 = SDI4(channel)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, n_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1,
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)

    def forward(self, x):

        f1, f2, f3, f4 = self.encoder(x)

        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)


        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)


        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)


        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)


        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)


        seg_outs1 = self.seg_outs[0](f41)
        seg_outs1 = F.interpolate(seg_outs1, scale_factor=32, mode='bilinear')
        # print("seg_outs1", seg_outs1.shape)

        y = self.deconv2(f41) + f31
        seg_outs2 = self.seg_outs[1](y)
        seg_outs2 = F.interpolate(seg_outs2, scale_factor=16, mode='bilinear')
        # print("seg_outs2", seg_outs2.shape)

        y = self.deconv3(y) + f21
        seg_outs3 = self.seg_outs[2](y)
        seg_outs3 = F.interpolate(seg_outs3, scale_factor=8, mode='bilinear')
        # print("seg_outs3:", seg_outs3.shape)

        y = self.deconv4(y) + f11
        seg_outs4 = self.seg_outs[3](y)
        seg_outs4 = F.interpolate(seg_outs4, scale_factor=4, mode='bilinear')
        # print("seg_outs4:",seg_outs4.shape)

        return seg_outs1, seg_outs2, seg_outs3, seg_outs4


if __name__ == "__main__":
    from thop import clever_format, profile

    x = torch.rand(1, 3, 352, 352).cuda()

    pretrained_path = "../weight/backbone/pvt_v2_b2.pth"

    model = UNetV2(n_classes=1, pretrained_path=pretrained_path)

    model.cuda()
    model.eval()

    out = model(x)

    print("out1:", out[0].shape)
    print("out2:", out[1].shape)
    print("out3:", out[2].shape)
    print("out4:", out[3].shape)

    flops, params = profile(model.cuda(0), (x,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")
    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    print("Peak GPU Memory: %.3f GB" % max_memory)



