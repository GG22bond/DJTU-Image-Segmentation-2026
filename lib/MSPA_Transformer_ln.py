import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from thop import clever_format, profile
from dynamic_tanh import convert_ln_to_dyt

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                nn.AdaptiveAvgPool1d,
                nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape):

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        super().__init__(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True)

    def forward(self, x):

        return super().forward(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):

        # print("in:", x.shape)

        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        # print("out:", out.shape)

        return out

    def initialize(self):
        weight_init(self)


# class MSPA_head(nn.Module):
#
#     def __init__(self, mode='dilation', dim=64, num_heads=8, e=0.5, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
#         super().__init__()
#
#         self.c = int(dim * e)
#
#         self.cv1 = BasicConv2d(dim, dim, 1, 1)
#         self.cv2 = BasicConv2d(dim, dim, 1)
#
#         self.norm1 = LayerNorm(self.c, LayerNorm_type)
#         self.attn = Attention(self.c, num_heads, bias, mode)
#         self.norm2 = LayerNorm(self.c, LayerNorm_type)
#         self.ffn = FeedForward(self.c, ffn_expansion_factor, bias)
#
#     def forward(self, x, mask=None):
#
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)   # a: torch.Size([1, 32, 44, 44]), b: torch.Size([1, 32, 44, 44])
#         b = b + self.attn(self.norm1(b), mask)
#         b = b + self.ffn(self.norm2(b))
#         return self.cv2(torch.cat((a, b), 1))
#
#     def initialize(self):
#         weight_init(self)


class MSPA_head(nn.Module):

    def __init__(self, mode='dilation', dim=64, num_heads=8, e=0.5, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        self.c = int(dim * e)

        self.cv1 = BasicConv2d(dim, dim, 1, 1)
        self.cv2 = BasicConv2d(dim, dim, 1)

        self.norm1 = convert_ln_to_dyt(LayerNorm(self.c, LayerNorm_type))
        self.norm2 = convert_ln_to_dyt(LayerNorm(self.c, LayerNorm_type))
        self.attn1 = Attention(self.c, num_heads, bias, mode)
        self.ffn1 = FeedForward(self.c, ffn_expansion_factor, bias)

        self.norm3 = convert_ln_to_dyt(LayerNorm(self.c, LayerNorm_type))
        self.norm4 = convert_ln_to_dyt(LayerNorm(self.c, LayerNorm_type))
        self.attn2 = Attention(self.c, num_heads, bias, mode)
        self.ffn2 = FeedForward(self.c, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):

        a, b = self.cv1(x).split((self.c, self.c), dim=1)   # a: torch.Size([1, 32, 44, 44]), b: torch.Size([1, 32, 44, 44])

        a = a + self.attn1(self.norm1(a), mask=mask)
        a = a + self.ffn1(self.norm2(a))

        b = b + self.attn2(self.norm3(b), mask)
        b = b + self.ffn2(self.norm4(b))

        return self.cv2(torch.cat((a, b), 1))

    def initialize(self):
        weight_init(self)


class MSPA_module(nn.Module):
    def __init__(self, dim):
        super(MSPA_module, self).__init__()
        self.B_TA = MSPA_head(dim=dim)
        self.F_TA = MSPA_head(dim=dim)
        self.TA = MSPA_head(dim=dim)
        self.Fuse = nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x, side_x, mask):
        N, C, H, W = x.shape
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)
        xf = self.F_TA(x, mask_d)
        xb = self.B_TA(x, 1 - mask_d)
        x = self.TA(x)
        x = torch.cat((xb, xf, x), 1)
        x = x.view(N, 3 * C, H, W)
        x = self.Fuse(x)
        D = self.Fuse2(side_x + side_x * x)
        return D

    def initialize(self):
        weight_init(self)

if __name__ == '__main__':

    in_1 = torch.randn(1, 64, 44, 44).cuda()
    in_2 = torch.randn(1, 64, 44, 44).cuda()
    in_3 = torch.randn(1, 1, 44, 44).cuda()

    model = MSPA_module(64)

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(in_1, in_2,  in_3)
    print("out_features:", out.size())

    flops, params = profile(model, inputs=(in_1, in_2,  in_3), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total params: %s' % (params))
    print('Total FLOPS: %s' % (flops))


