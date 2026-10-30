"""
Unet and Unet++
    Reference:
        https://github.com/qubvel-org/segmentation_models.pytorch
        https://arxiv.org/abs/1505.04597
        https://arxiv.org/abs/1807.10165
"""

import torch
import segmentation_models_pytorch as smp
from torch import nn

__all__ = ['UNet', 'UnetPlusPlus']

class UNet(nn.Module):
    def __init__(self,
                 encoder_name="resnet50",
                 encoder_weights="imagenet",
                 decoder_channels=(512, 256, 128, 64, 32),
                 classes=1):
        super().__init__()

        self.model = smp.Unet(encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              decoder_channels=decoder_channels,
                              classes=classes)

    def forward(self, input):
        output = self.model(input)
        return output


class UnetPlusPlus(nn.Module):
    def __init__(self,
                 encoder_name="resnet50",
                 encoder_weights="imagenet",
                 decoder_channels=(512, 256, 128, 64, 32),
                 classes=1):
        super().__init__()

        self.model = smp.UnetPlusPlus(encoder_name=encoder_name,
                                      encoder_weights=encoder_weights,
                                      decoder_channels=decoder_channels,
                                      classes=classes)

    def forward(self, input):
        output = self.model(input)
        return output


if __name__ == '__main__':

    from thop import clever_format, profile

    input = torch.randn(1, 3, 352, 352).cuda()
    model = UNet()
    # model = UnetPlusPlus()

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    print("output:", out.shape)

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    print("Peak GPU Memory: %.3f GB" % max_memory)


# import torch
# import segmentation_models_pytorch as smp
# from torch import nn
# from thop import clever_format, profile
# model = smp.Unet("resnet50", encoder_weights="imagenet", decoder_channels=(512, 256, 128, 64, 32),classes=1)
## model = smp.UnetPlusPlus("resnet50", encoder_weights="imagenet", decoder_channels=(512, 256, 128, 64, 32),classes=1)
#
# model.cuda()
# model.eval()
#
# input = torch.randn(1, 3, 352, 352).cuda()
#
# out = model(input)
#
# print("out", out.shape)
#
# flops, params = profile(model.cuda(0), (input,), verbose=False)
# flops, params = clever_format([flops * 2, params], "%.3f")
#
# print('Total FLOPS: %s' % (flops))
# print('Total params: %s' % (params))



