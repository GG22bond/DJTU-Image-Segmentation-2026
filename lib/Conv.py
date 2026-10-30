import torch
import torch.nn as nn
import torch.nn.functional as F
from DCNv4.modules import DCNv4
from timm.models.layers import CondConv2d
from FDConv import FDConv


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


class DW_BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(DW_BasicConv2d, self).__init__()
        self.dwconv = nn.Conv2d(in_planes,
                              in_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=in_planes, bias=False)

        self.pwconv = nn.Conv2d(in_planes,
                                out_planes,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=1, bias=False)

        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.pwconv(x)
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
        # self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        # if hasattr(self, 'conv1'):
        #     x = self.conv1(x)
        x = self.dcnv4(x, (x.size(2), x.size(3)))
        x = self.bn(x)
        # x = self.silu(x)
        return x


class DynamicConv_layer(nn.Module):
    """ Dynamic Conv layer
    """

    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                                    groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x


class DynamicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, num_experts=4):

        super().__init__()
        self.dynamic_conv = DynamicConv_layer(in_features=in_planes, out_features=out_planes, kernel_size=kernel_size,
                                              stride=stride, padding=padding, dilation=dilation,
                                              groups=groups, bias=bias, num_experts=num_experts)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.dynamic_conv(x)
        x = self.bn(x)
        return x


class BasicFDConv(nn.Module):  # CBR
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicFDConv, self).__init__()
        self.conv = FDConv(in_planes, out_planes,
                           kernel_size=kernel_size, stride=stride,kernel_num=8,
                           padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.silu(x)
        return x


if __name__ == '__main__':

    from thop import clever_format, profile

    input = torch.randn(1, 192, 44, 44).cuda()

    model = BasicFDConv(3*64, 64, 1)
    model.cuda()
    model.eval()

    out = model(input)

    print("output shape:", out.shape)

    flops, params = profile(model, inputs=(input, ), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total params: %s' % (params))
    print('Total FLOPS: %s' % (flops))



