import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from timm.layers import LayerNorm2d
from thop import clever_format, profile
# from dynamic_tanh import convert_ln_to_dyt


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
        nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d,
        nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
#
#     def initialize(self):
#         weight_init(self)


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
        return out

    def initialize(self):
        weight_init(self)


class MUl_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(MUl_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.qkv1conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv2conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv3conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim*3, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        q_3 = self.qkv1conv_3(self.qkv_0(x))
        k_3 = self.qkv2conv_3(self.qkv_1(x))
        v_3 = self.qkv3conv_3(self.qkv_2(x))

        if mask is not None:
            q_3 = q_3 * mask
            k_3 = k_3 * mask

        q_3 = rearrange(q_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_3 = rearrange(k_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_3 = rearrange(v_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_3 = torch.nn.functional.normalize(q_3, dim=-1)
        k_3 = torch.nn.functional.normalize(k_3, dim=-1)
        attn_3 = (q_3 @ k_3.transpose(-2, -1)) * self.temperature
        attn_3 = attn_3.softmax(dim=-1)
        out_3 = (attn_3 @ v_3)
        out_3 = rearrange(out_3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_5 = self.qkv1conv_5(self.qkv_0(x))
        k_5 = self.qkv2conv_5(self.qkv_1(x))
        v_5 = self.qkv3conv_5(self.qkv_2(x))

        q_5 = rearrange(q_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_5 = rearrange(k_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_5 = rearrange(v_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_5 = torch.nn.functional.normalize(q_5, dim=-1)
        k_5 = torch.nn.functional.normalize(k_5, dim=-1)
        attn_5 = (q_5 @ k_5.transpose(-2, -1)) * self.temperature
        attn_5 = attn_5.softmax(dim=-1)
        out_5 = (attn_5 @ v_5)
        out_5 = rearrange(out_5, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_7 = self.qkv1conv_7(self.qkv_0(x))
        k_7 = self.qkv2conv_7(self.qkv_1(x))
        v_7 = self.qkv3conv_7(self.qkv_2(x))

        q_7 = rearrange(q_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_7 = rearrange(k_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_7 = rearrange(v_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_7 = torch.nn.functional.normalize(q_7, dim=-1)
        k_7 = torch.nn.functional.normalize(k_7, dim=-1)
        attn_7 = (q_7 @ k_7.transpose(-2, -1)) * self.temperature
        attn_7 = attn_7.softmax(dim=-1)
        out_7 = (attn_7 @ v_7)
        out_7 = rearrange(out_7, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(torch.cat((out_3,out_5,out_7),1))
        return out

    def initialize(self):
        weight_init(self)

class MSA_head(nn.Module):
    def __init__(self, mode='dilation', dim=64, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        # self.norm1 = convert_ln_to_dyt(LayerNorm(dim, LayerNorm_type))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class MSA_module(nn.Module):
    def __init__(self, dim):
        super(MSA_module, self).__init__()
        self.B_TA = MSA_head(dim=dim)
        self.F_TA = MSA_head(dim=dim)
        self.TA = MSA_head(dim=dim)
        self.Fuse = nn.Conv2d(3 * dim, dim, kernel_size=3, padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),
                                   nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

    def forward(self, x, side_x, mask):
        N, C, H, W = x.shape
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')
        mask_d = mask.detach()
        mask_d = torch.sigmoid(mask_d)

        xf = self.F_TA(x, mask_d)
        # print("xf:", xf.size())
        xb = self.B_TA(x, 1 - mask_d)
        # print("xb:", xb.size())
        x = self.TA(x)
        # print("x:", x.size())

        x = torch.cat((xb, xf, x), 1)
        # print("x_cat:", x.size())

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

    model = MSA_module(64)

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(in_1, in_2,  in_3)
    print("out_features:", out.size())

    flops, params = profile(model, inputs=(in_1, in_2,  in_3), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total params: %s' % (params))
    print('Total FLOPS: %s' % (flops))
