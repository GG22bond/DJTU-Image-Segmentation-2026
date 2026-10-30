# import torch
# import einops
# from torch import nn
# import numpy as np
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
#
#
# def weight_init(module):
#     for n, m in module.named_children():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
#             nn.init.ones_(m.weight)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.Sequential):
#             weight_init(m)
#         elif isinstance(m, (
#                 nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
#                 nn.AdaptiveAvgPool1d,
#                 nn.Sigmoid, nn.Identity)):
#             pass
#         else:
#             m.initialize()
#
# class LayerNormProxy(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         x = einops.rearrange(x, 'b c h w -> b h w c')
#         x = self.norm(x)
#         return einops.rearrange(x, 'b h w c -> b c h w')
#
#     def initialize(self):
#         weight_init(self)
#
# class DAttention(nn.Module):
#     # Vision Transformer with Deformable Attention CVPR2022
#     # fixed_pe=True need adujust 640x640
#     def __init__(
#             self, channel, q_size, n_heads=8, n_groups=4,
#             attn_drop=0.0, proj_drop=0.0, stride=1,
#             offset_range_factor=4, use_pe=True, dwc_pe=True,
#             no_off=False, fixed_pe=False, ksize=3, log_cpb=False, kv_size=None
#     ):
#         super().__init__()
#         n_head_channels = channel // n_heads
#         self.dwc_pe = dwc_pe
#         self.n_head_channels = n_head_channels
#         self.scale = self.n_head_channels ** -0.5
#         self.n_heads = n_heads
#         self.q_h, self.q_w = q_size
#         # self.kv_h, self.kv_w = kv_size
#         self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
#         self.nc = n_head_channels * n_heads
#         self.n_groups = n_groups
#         self.n_group_channels = self.nc // self.n_groups
#         self.n_group_heads = self.n_heads // self.n_groups
#         self.use_pe = use_pe
#         self.fixed_pe = fixed_pe
#         self.no_off = no_off
#         self.offset_range_factor = offset_range_factor
#         self.ksize = ksize
#         self.log_cpb = log_cpb
#         self.stride = stride
#         kk = self.ksize
#         pad_size = kk // 2 if kk != stride else 0
#
#         self.conv_offset = nn.Sequential(
#             nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
#             LayerNormProxy(self.n_group_channels),
#             nn.GELU(),
#             nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
#         )
#         if self.no_off:
#             for m in self.conv_offset.parameters():
#                 m.requires_grad_(False)
#
#         self.proj_q = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#
#         self.proj_k = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#
#         self.proj_v = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#
#         self.proj_out = nn.Conv2d(
#             self.nc, self.nc,
#             kernel_size=1, stride=1, padding=0
#         )
#
#         self.proj_drop = nn.Dropout(proj_drop, inplace=True)
#         self.attn_drop = nn.Dropout(attn_drop, inplace=True)
#
#         if self.use_pe and not self.no_off:
#             if self.dwc_pe:
#                 self.rpe_table = nn.Conv2d(
#                     self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
#             elif self.fixed_pe:
#                 self.rpe_table = nn.Parameter(
#                     torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
#                 )
#                 trunc_normal_(self.rpe_table, std=0.01)
#             elif self.log_cpb:
#                 # Borrowed from Swin-V2
#                 self.rpe_table = nn.Sequential(
#                     nn.Linear(2, 32, bias=True),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(32, self.n_group_heads, bias=False)
#                 )
#             else:
#                 self.rpe_table = nn.Parameter(
#                     torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
#                 )
#                 trunc_normal_(self.rpe_table, std=0.01)
#         else:
#             self.rpe_table = None
#
#     @torch.no_grad()
#     def _get_ref_points(self, H_key, W_key, B, dtype, device):
#
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
#             torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
#             indexing='ij'
#         )
#         ref = torch.stack((ref_y, ref_x), -1)
#         ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
#         ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
#         ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
#
#         return ref
#
#     @torch.no_grad()
#     def _get_q_grid(self, H, W, B, dtype, device):
#
#         ref_y, ref_x = torch.meshgrid(
#             torch.arange(0, H, dtype=dtype, device=device),
#             torch.arange(0, W, dtype=dtype, device=device),
#             indexing='ij'
#         )
#         ref = torch.stack((ref_y, ref_x), -1)
#         ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
#         ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
#         ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
#
#         return ref
#
#     def forward(self, x, mask=None):
#
#         B, C, H, W = x.size()
#         dtype, device = x.dtype, x.device
#
#         q = self.proj_q(x)
#
#         if mask is not None:
#             q = q * mask
#
#         # print("q:",q.shape)
#
#         q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
#         offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
#         Hk, Wk = offset.size(2), offset.size(3)
#         n_sample = Hk * Wk
#
#         if self.offset_range_factor >= 0 and not self.no_off:
#             offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
#             offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
#
#         offset = einops.rearrange(offset, 'b p h w -> b h w p')
#         reference = self._get_ref_points(Hk, Wk, B, dtype, device)
#
#         if self.no_off:
#             offset = offset.fill_(0.0)
#
#         if self.offset_range_factor >= 0:
#             pos = offset + reference
#         else:
#             pos = (offset + reference).clamp(-1., +1.)
#
#         if self.no_off:
#             x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
#             assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
#         else:
#             pos = pos.type(x.dtype)
#             print("pos:",pos.shape)
#             print("x:", x.shape)
#             x_sampled = F.grid_sample(
#                 input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
#                 grid=pos[..., (1, 0)],  # y, x -> x, y
#                 mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
#
#         print("x_sampled:",x_sampled.shape)
#
#         k_sampled = x_sampled.clone()
#         # print("k_sampled:",k_sampled.shape)
#         # v_sampled = x_sampled.clone()
#
#         if mask is not None:
#             print("k_sampled:", k_sampled.shape)
#             print("mask:", mask.shape)
#             mask_resized = F.interpolate(mask, size=(Hk, Wk), mode="nearest")
#             mask_resized = mask_resized.repeat_interleave(B * self.n_groups, dim=0)
#             k_sampled = k_sampled * mask_resized
#             print("mask_resized:", mask_resized.shape)
#         # print("k_sampled:",k_sampled.shape)
#
#         x_sampled = x_sampled.reshape(B, C, 1, n_sample)
#         # print("x_sampled:",x_sampled.shape)
#         k_sampled = k_sampled.reshape(B, C, 1, n_sample)
#         # print("k_sampled:",k_sampled.shape)
#
#         q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
#         k = self.proj_k(k_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
#         # print("k:", k.shape)
#         v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
#
#         attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
#         attn = attn.mul(self.scale)
#
#         if self.use_pe and (not self.no_off):
#
#             if self.dwc_pe:
#                 residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
#                                                                               H * W)
#             elif self.fixed_pe:
#                 rpe_table = self.rpe_table
#                 attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
#                 attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
#             elif self.log_cpb:
#                 q_grid = self._get_q_grid(H, W, B, dtype, device)
#                 displacement = (
#                             q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
#                                                                                                    n_sample,
#                                                                                                    2).unsqueeze(1)).mul(
#                     4.0)  # d_y, d_x [-8, +8]
#                 displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
#                 attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
#                 attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
#             else:
#                 rpe_table = self.rpe_table
#                 rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
#                 q_grid = self._get_q_grid(H, W, B, dtype, device)
#                 displacement = (
#                             q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
#                                                                                                    n_sample,
#                                                                                                    2).unsqueeze(1)).mul(
#                     0.5)
#                 attn_bias = F.grid_sample(
#                     input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
#                                            g=self.n_groups),
#                     grid=displacement[..., (1, 0)],
#                     mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns
#
#                 attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
#                 attn = attn + attn_bias
#
#         if mask is not None:
#             mask_flat = mask.reshape(B, 1, H * W)  # for query
#             attn = attn.masked_fill(mask_flat == 0, float('-inf'))
#
#         attn = F.softmax(attn, dim=2)
#         attn = self.attn_drop(attn)
#
#         out = torch.einsum('b m n, b c n -> b c m', attn, v)
#
#         if self.use_pe and self.dwc_pe:
#             out = out + residual_lepe
#         out = out.reshape(B, C, H, W)
#
#         y = self.proj_drop(self.proj_out(out))
#
#         return y
#
#     def initialize(self):
#         weight_init(self)
#
# if __name__ == '__main__':
#
#     from thop import clever_format, profile
#
#     in_1 = torch.randn(1, 64, 44, 44).cuda()
#     in_2 = torch.randn(1, 1, 44, 44).cuda()
#
#     model = DAttention(64, (64,64), 8)
#
#     print("Model:\n", model)
#
#     model.cuda()
#     model.eval()
#
#     out = model(in_1, in_2)
#     print("out_features:", out.size())
#
#     flops, params = profile(model, inputs=(in_1, in_2), verbose=False)
#     flops, params = clever_format([flops * 2, params], "%.3f")
#
#     print('Total params: %s' % (params))
#     print('Total FLOPS: %s' % (flops))

import torch
import einops
from torch import nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


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


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

    def initialize(self):
        weight_init(self)


class DAttention(nn.Module):
    # Vision Transformer with Deformable Attention CVPR2022
    # fixed_pe=True need adujust 640x640
    def __init__(
            self, channel, q_size, n_heads=8, n_groups=4,
            attn_drop=0.0, proj_drop=0.0, stride=1,
            offset_range_factor=4, use_pe=True, dwc_pe=True,
            no_off=False, fixed_pe=False, ksize=3, log_cpb=False, kv_size=None
    ):
        super().__init__()
        n_head_channels = channel // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x, mask=None):
        """
        x: (B, C, H, W)
        mask: (B, 1, H, W)  0=无效, 1=有效
        """
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # ===== Q =====
        q = self.proj_q(x)  # (B, C, H, W)
        if mask is not None:
            q = q * mask  # mask query

        # ===== offset =====
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w',
                                 g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # (B*g, 2, Hk, Wk)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)],
                                        device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        # ===== deformable sampling =====
        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk
        else:
            pos = pos.type(x.dtype)
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],
                mode='bilinear', align_corners=True
            )  # (B*g, Cg, Hk, Wk)

        # ===== K/V =====
        k_sampled = x_sampled.clone()
        v_sampled = x_sampled.clone()

        if mask is not None:
            # resize mask -> (B,1,Hk,Wk)
            mask_resized = F.interpolate(mask, size=(Hk, Wk), mode="nearest")
            # repeat for groups
            mask_resized = mask_resized.repeat_interleave(self.n_groups, dim=0)
            k_sampled = k_sampled * mask_resized
            # v_sampled = v_sampled * mask_resized  # 如果你也想屏蔽掉 Value

        # reshape for proj_k/proj_v
        k_sampled = k_sampled.reshape(B, C, 1, n_sample)
        v_sampled = v_sampled.reshape(B, C, 1, n_sample)

        # ===== proj & reshape =====
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(k_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(v_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        # ===== Attention =====
        attn = torch.einsum('b c m, b c n -> b m n', q, k) * self.scale  # (B*h, HW, Ns)

        # --- masked_fill ---
        if mask is not None:
            # query mask: (B, HW)
            mask_q = mask.reshape(B, -1)  # (B, HW)
            # key mask: (B, Ns)
            mask_k = mask_resized.reshape(B, -1)  # (B*g, Ns) → 需要合并 group
            mask_k = mask_k.view(B, self.n_groups, -1).mean(1)  # (B, Ns) 简单合并

            # expand to heads
            mask_q = mask_q.unsqueeze(1).repeat(1, self.n_heads, 1).reshape(B * self.n_heads, 1, H * W)
            mask_k = mask_k.unsqueeze(1).repeat(1, self.n_heads, 1).reshape(B * self.n_heads, 1, n_sample)

            # where mask_q=0 or mask_k=0 → -inf
            attn = attn.masked_fill(mask_q.transpose(1, 2) == 0, float('-inf'))
            attn = attn.masked_fill(mask_k == 0, float('-inf'))

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        # ===== weighted sum =====
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        if self.use_pe and self.dwc_pe:
            residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(
                B * self.n_heads, self.n_head_channels, H * W
            )
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))
        return y

    def initialize(self):
        weight_init(self)


if __name__ == '__main__':
    from thop import clever_format, profile

    in_1 = torch.randn(1, 64, 44, 44).cuda()
    in_2 = torch.randn(1, 1, 44, 44).cuda()

    model = DAttention(64, (64, 64), 8)

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(in_1, in_2)
    print("out_features:", out.size())

    flops, params = profile(model, inputs=(in_1,in_2), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    print('Total params: %s' % (params))
    print('Total FLOPS: %s' % (flops))


# import torch
# import einops
# from torch import nn
# import numpy as np
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
#
#
# class LayerNormProxy(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         x = einops.rearrange(x, 'b c h w -> b h w c')
#         x = self.norm(x)
#         return einops.rearrange(x, 'b h w c -> b c h w')
#
#
# class MaskedDAttention(nn.Module):
#     """
#     Deformable Attention with robust masking options.
#
#     mask_mode:
#         - 'none': 不使用mask。
#         - 'mul':  通过逐点相乘(mask)抑制 Q/K/V（Q, K, V 的具体规则见代码）。
#         - 'attn': 在注意力矩阵（对键Ns维度softmax之前）用 -inf 做 key padding mask；输出再乘以 query mask（避免被mask的query产生非零输出）。
#         - 'both': 同时启用 'mul' + 'attn'。
#
#     说明：
#         * 支持对 mask 做 grid_sample，确保它与可变形采样位置一一对应（用于 K/V 与 key-padding-mask）。
#         * 当某些 query 的所有 key 都被mask时，softmax(-inf) 会导致 NaN，这里做了行级保护：将该行注意力设为0，并在输出阶段用 q_mask 将输出置零。
#     """
#     def __init__(
#         self,
#         channel,
#         q_size,
#         n_heads=8,
#         n_groups=4,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         stride=1,
#         offset_range_factor=4,
#         use_pe=True,
#         dwc_pe=True,
#         no_off=False,
#         fixed_pe=False,
#         ksize=3,
#         log_cpb=False,
#         kv_size=None,
#         mask_mode='both',  # 'none' | 'mul' | 'attn' | 'both'
#     ):
#         super().__init__()
#         assert channel % n_heads == 0, "channel must be divisible by n_heads"
#         assert n_heads % n_groups == 0, "n_heads must be divisible by n_groups"
#
#         n_head_channels = channel // n_heads
#         self.dwc_pe = dwc_pe
#         self.n_head_channels = n_head_channels
#         self.scale = self.n_head_channels ** -0.5
#         self.n_heads = n_heads
#         self.q_h, self.q_w = q_size
#         self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
#         self.nc = n_head_channels * n_heads
#         self.n_groups = n_groups
#         self.n_group_channels = self.nc // self.n_groups
#         self.n_group_heads = self.n_heads // self.n_groups
#         self.use_pe = use_pe
#         self.fixed_pe = fixed_pe
#         self.no_off = no_off
#         self.offset_range_factor = offset_range_factor
#         self.ksize = ksize
#         self.log_cpb = log_cpb
#         self.stride = stride
#         self.mask_mode = mask_mode
#
#         kk = self.ksize
#         pad_size = kk // 2 if kk != stride else 0
#
#         # offset 预测（分组深度可分离卷积）
#         self.conv_offset = nn.Sequential(
#             nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
#             LayerNormProxy(self.n_group_channels),
#             nn.GELU(),
#             nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
#         )
#         if self.no_off:
#             for m in self.conv_offset.parameters():
#                 m.requires_grad_(False)
#
#         # 1x1 投影
#         self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
#         self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
#         self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
#         self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
#
#         self.proj_drop = nn.Dropout(proj_drop, inplace=True)
#         self.attn_drop = nn.Dropout(attn_drop, inplace=True)
#
#         # 位置偏置
#         if self.use_pe and not self.no_off:
#             if self.dwc_pe:
#                 self.rpe_table = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
#             elif self.fixed_pe:
#                 self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w))
#                 trunc_normal_(self.rpe_table, std=0.01)
#             elif self.log_cpb:
#                 self.rpe_table = nn.Sequential(
#                     nn.Linear(2, 32, bias=True),
#                     nn.ReLU(inplace=True),
#                     nn.Linear(32, self.n_group_heads, bias=False)
#                 )
#             else:
#                 self.rpe_table = nn.Parameter(torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1))
#                 trunc_normal_(self.rpe_table, std=0.01)
#         else:
#             self.rpe_table = None
#
#     @torch.no_grad()
#     def _get_ref_points(self, H_key, W_key, B, dtype, device):
#         ref_y, ref_x = torch.meshgrid(
#             torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
#             torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
#             indexing='ij'
#         )
#         ref = torch.stack((ref_y, ref_x), -1)
#         ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
#         ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
#         ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # (B*g, Hk, Wk, 2)
#         return ref
#
#     @torch.no_grad()
#     def _get_q_grid(self, H, W, B, dtype, device):
#         ref_y, ref_x = torch.meshgrid(
#             torch.arange(0, H, dtype=dtype, device=device),
#             torch.arange(0, W, dtype=dtype, device=device),
#             indexing='ij'
#         )
#         ref = torch.stack((ref_y, ref_x), -1)
#         ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
#         ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
#         ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # (B*g, H, W, 2)
#         return ref
#
#     def _masked_softmax_over_keys(self, attn, key_mask):
#         """attn: (B*h, HW, Ns); key_mask: (B*h, 1, Ns) bool
#         稳定版 softmax：对被mask的key置 -inf；对全mask的行输出置0。
#         """
#         # 将非mask位置为True，mask位置为False
#         if key_mask.dtype != torch.bool:
#             key_mask = key_mask > 0.5
#
#         # 用 -inf 屏蔽 keys
#         attn = attn.masked_fill(~key_mask, float('-inf'))
#         # 数值稳定化
#         maxv = torch.amax(attn, dim=2, keepdim=True)
#         attn = attn - maxv
#         attn = torch.exp(attn)
#         attn = attn * key_mask.float()
#         denom = attn.sum(dim=2, keepdim=True)
#         zero_rows = (denom == 0)
#         denom = denom + zero_rows.float()  # 避免除0
#         attn = attn / denom
#         # 对全mask的行，softmax结果设为0
#         attn = torch.where(zero_rows, torch.zeros_like(attn), attn)
#         return attn
#
#     def forward(self, x, mask=None):
#         """
#         x:    (B, C, H, W)
#         mask: (B, 1, H, W) / (B, H, W) / (B, C, H, W) — 会自动广播并转换到 (B,1,H,W)
#         """
#         B, C, H, W = x.size()
#         dtype, device = x.dtype, x.device
#
#         # 1) 生成 Q
#         q = self.proj_q(x)  # (B, C, H, W)
#
#         # 规范化 mask 形状
#         if mask is not None:
#             if mask.dim() == 3:
#                 mask = mask.unsqueeze(1)
#             if mask.size(1) != 1:
#                 mask = mask[:, :1, ...]  # 只取第一通道作为有效mask
#             mask = mask.to(dtype=q.dtype, device=q.device)
#
#         # mul 模式：先对 Q 做逐点抑制
#         if self.mask_mode in ('mul', 'both') and mask is not None:
#             q = q * mask
#
#         # 2) 预测 offset
#         q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
#         offset = self.conv_offset(q_off).contiguous()  # (B*g, 2, Hk, Wk)
#         Hk, Wk = offset.size(2), offset.size(3)
#         n_sample = Hk * Wk
#
#         if self.offset_range_factor >= 0 and not self.no_off:
#             offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
#             offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
#
#         offset = einops.rearrange(offset, 'b p h w -> b h w p')
#         reference = self._get_ref_points(Hk, Wk, B, dtype, device)
#
#         if self.no_off:
#             offset = offset.fill_(0.0)
#
#         if self.offset_range_factor >= 0:
#             pos = offset + reference
#         else:
#             pos = (offset + reference).clamp(-1., +1.)
#
#         # 3) 采样特征 / 采样mask
#         if self.no_off:
#             x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
#             assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk
#             if mask is not None:
#                 mask_sampled = F.avg_pool2d(mask, kernel_size=self.stride, stride=self.stride)
#         else:
#             x_group = x.reshape(B * self.n_groups, self.n_group_channels, H, W)
#             x_sampled = F.grid_sample(x_group, grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)
#             if mask is not None:
#                 mask_group = mask.expand(B, 1, H, W)
#                 mask_group = mask_group.repeat_interleave(self.n_groups, dim=0)  # (B*g,1,H,W)
#                 mask_sampled = F.grid_sample(mask_group, grid=pos[..., (1, 0)], mode='nearest', align_corners=True)
#
#         # 4) 整理回 (B, C, 1, Ns)，方便用 1x1 conv 线性投影
#         x_sampled = einops.rearrange(x_sampled, '(b g) c h w -> b (g c) 1 (h w)', b=B, g=self.n_groups)
#         if mask is not None:
#             mask_sampled = einops.rearrange(mask_sampled, '(b g) 1 h w -> b g 1 (h w)', b=B, g=self.n_groups)
#             # 保守做法：只有当所有组在该位置都有效时，才视为有效（取 min）
#             key_mask_flat = mask_sampled.min(dim=1).values  # (B,1,Ns)
#         else:
#             key_mask_flat = None
#
#         # mul 模式：对 K/V 逐点抑制（在投影前或后都可以；这里在投影前）
#         if self.mask_mode in ('mul', 'both') and key_mask_flat is not None:
#             x_sampled = x_sampled * key_mask_flat
#
#         # 5) 得到 K/V，并reshape到多头形式
#         k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
#         v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
#
#         # Q reshape
#         q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
#
#         # 6) 注意力
#         attn = torch.einsum('b c m, b c n -> b m n', q, k)  # (B*h, HW, Ns)
#         attn = attn.mul(self.scale)
#
#         # 7) 位置偏置
#         if self.use_pe and (not self.no_off):
#             if self.dwc_pe:
#                 residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
#             elif self.fixed_pe:
#                 rpe_table = self.rpe_table  # (h, HW, Ns)
#                 attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
#                 attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
#             elif self.log_cpb:
#                 q_grid = self._get_q_grid(H, W, B, dtype, device)
#                 displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(4.0)
#                 displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
#                 attn_bias = self.rpe_table(displacement)  # (B*g, HW, Ns, h_g)
#                 attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
#             else:
#                 rpe_table = self.rpe_table
#                 rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
#                 q_grid = self._get_q_grid(H, W, B, dtype, device)
#                 displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
#                 attn_bias = F.grid_sample(
#                     input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
#                     grid=displacement[..., (1, 0)],
#                     mode='bilinear', align_corners=True
#                 )  # (B*g, h_g, HW, Ns)
#                 attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
#                 attn = attn + attn_bias
#
#         # 8) attn 级别的 key padding mask（-inf），并对被mask的 query 做行级抑制
#         if self.mask_mode in ('attn', 'both') and mask is not None:
#             # key mask: (B*h, 1, Ns)
#             key_mask_bh = key_mask_flat.expand(B, 1, n_sample)
#             key_mask_bh = key_mask_bh.repeat_interleave(self.n_heads, dim=0)
#             attn = self._masked_softmax_over_keys(attn, key_mask_bh)
#         else:
#             attn = F.softmax(attn, dim=2)
#         attn = self.attn_drop(attn)
#
#         # 9) 输出
#         out = torch.einsum('b m n, b c n -> b c m', attn, v)  # (B*h, C_head, HW)
#
#         if self.use_pe and self.dwc_pe:
#             out = out + residual_lepe
#         out = out.reshape(B, C, H, W)
#
#         # 如果 query 被mask，强制其输出也为0（防止被其它未mask key“拉起”）
#         if self.mask_mode in ('attn', 'both') and mask is not None:
#             out = out * mask
#
#         y = self.proj_drop(self.proj_out(out))
#         return y
#
#
# if __name__ == '__main__':
#     from thop import clever_format, profile
#
#     in_1 = torch.randn(1, 64, 44, 44).cuda()
#     in_2 = torch.randn(1, 1, 44, 44).cuda()
#
#     model = MaskedDAttention(64, (64, 64), 8)
#
#     print("Model:\n", model)
#
#     model.cuda()
#     model.eval()
#
#     out = model(in_1, in_2)
#     print("out_features:", out.size())
#
#     flops, params = profile(model, inputs=(in_1, in_2), verbose=False)
#     flops, params = clever_format([flops * 2, params], "%.3f")
#
#     print('Total params: %s' % (params))
#     print('Total FLOPS: %s' % (flops))


