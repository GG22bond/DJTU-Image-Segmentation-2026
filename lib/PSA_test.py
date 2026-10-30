import torch.nn as nn
import torch
from einops import rearrange
import numbers
from thop import clever_format, profile
import torch.nn.functional as F

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


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


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
    def __init__(self, dim, num_heads, bias):
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

        print("att_in:",x.shape)

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

        print("att_out:",out.shape)

        return out

    def initialize(self):
        weight_init(self)


class MSA_head(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):

        # norm_x = self.norm1(x)
        # print("norm_x:",norm_x.shape) # norm_x: torch.Size([1, 64, 44, 44])

        x = x + self.attn(self.norm1(x), mask)

        # ffn_x = self.ffn(self.norm2(x))
        # print("ffn_out:",ffn_x.shape)  # ffn_out: torch.Size([1, 64, 44, 44])

        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class PSA(nn.Module):

    def __init__(self, c1, c2, bias=False, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = BasicConv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = BasicConv2d(2 * self.c, c1, 1)

        self.attn = Attention(self.c, num_heads=8, bias=bias)
        self.ffn = nn.Sequential(BasicConv2d(self.c, self.c * 2, 1), BasicConv2d(self.c * 2, self.c, 1))

    def forward(self, x):

        print("PSA_in:",x.shape)

        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class PSA_(nn.Module):

    def __init__(self, c1, c2, e=0.5, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        assert c1 == c2
        self.c = int(c1 * e)

        self.cv1 = BasicConv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = BasicConv2d(2 * self.c, c1, 1)

        self.norm1 = LayerNorm(self.c, LayerNorm_type)
        self.attn = Attention(self.c, num_heads=8, bias=bias)
        self.norm2 = LayerNorm(self.c, LayerNorm_type)
        self.ffn = FeedForward(self.c, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):

        print("PSA_in:", x.shape)
        # x1 = self.cv1(x)
        # print("cv1:",x1.shape)  # cv1: torch.Size([1, 64, 44, 44])

        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # print("a:",a.shape)  a: torch.Size([1, 32, 44, 44])
        # print("b:",b.shape)  b: torch.Size([1, 32, 44, 44])

        # att = self.attn(b)
        # # print("att:",att.shape)  att: torch.Size([1, 32, 44, 44])

        b = b + self.attn(self.norm1(b), mask)
        print("b1:", b.shape)

        # ffn = self.ffn(self.norm2(b))
        # print("ffn:", ffn.shape)

        b = b + self.ffn(self.norm2(b))
        print("b2:", b.shape)
        return self.cv2(torch.cat((a, b), 1))

    def initialize(self):
        weight_init(self)


if __name__ == '__main__':
    input = torch.randn(1, 64, 44, 44).cuda()

    # model = Attention(64, 8, bias=False)
    model = PSA_(64, 64)
    # model = MSA_head(dim=64)
    model.cuda()
    model.eval()

    # print("Model:\n", model)

    out = model(input)

    print("out:",out.shape)

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))