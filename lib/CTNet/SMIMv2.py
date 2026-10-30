# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 16:48
# @Author  : Jinwu Hu
# @FileName: SMIMv2.py
# @Blog    :https://blog.csdn.net/Fhujinwu
#Self-Multiscale Interaction Module
import torch
import torch.nn as nn
import torch.nn.functional as F


def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


class SMIModule(nn.Module):
    def __init__(self,out_channel = 32):
        super(SMIModule, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample

        self.h2l_0 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(out_channel)
        self.bnh_0 = nn.BatchNorm2d(out_channel)

        self.h2h_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(out_channel)
        self.bnh_1 = nn.BatchNorm2d(out_channel)

        self.h2h_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bnh_2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2:]

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x



if __name__ == '__main__':
    model = SMIModule().cuda()
    input_tensor = torch.randn(1, 32, 352, 352).cuda()

    prediction = model(input_tensor)
    print(prediction.size())
