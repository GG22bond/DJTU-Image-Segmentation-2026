import torch
import torch.nn as nn
import torch.nn.functional as F
from MIT import mit_b3
from CIMv4 import CIModule
from BasicConv import BasicConv2d
from SMIMv2 import SMIModule
from Contrastive_head import ProjectionHead


#  https://github.com/SegmentationBLWX/sssegmentation/blob/main/scripts/ckpts/download_pretrained_weights.py
class PolypModule(nn.Module):
    def __init__(self, channel=32):
        super(PolypModule, self).__init__()
        # print("Modelv13")
        self.backbone = mit_b3()  # [64, 128, 320, 512]
        path = '../../weight/backbone/mit_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # print(state_dict)
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)


        self.Translayer1 = BasicConv2d(64, channel, 1)
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(320, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)
        self.contrasthead = ProjectionHead(dim=32)

        self.Smim = SMIModule(out_channel=channel)
        self.cim = CIModule(out_channel=channel)

        self.uper =  nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2, padding=1),
                                   BasicConv2d(in_planes=channel, out_planes=channel,kernel_size=3,stride=1,padding=1))

        self.prediction = nn.Sequential(BasicConv2d(in_planes=channel,out_planes=channel,kernel_size=3, stride=1,padding=1),
                                        nn.Conv2d(channel, 1, 1))


    def forward(self, x):

        #feature extraction
        cswin = self.backbone(x)

        x1 = self.Translayer1(cswin[0])
        x2 = self.Translayer2(cswin[1])
        x3 = self.Translayer3(cswin[2])
        x4 = self.Translayer4(cswin[3])

        print("x4:", x4.shape)

        emb = self.contrasthead(x4)

        print("emb:", emb.shape)

        x4 = self.Smim(x4)
        x3 = self.Smim(x3)
        x2 = self.Smim(x2)


        x_4_3_2 = self.cim(x4,x3,x2)
        # print("x_4_3_2:", x_4_3_2.shape)

        x1 = x1 + self.uper(x_4_3_2)
        prediction1 = self.prediction(x1)
        prediction1_4 = F.interpolate(prediction1, scale_factor=4, mode='bilinear', align_corners=False)
        # return prediction1_4
        return prediction1_4, emb


if __name__ == '__main__':

    from thop import clever_format, profile

    input = torch.randn(1, 3, 352, 352).cuda()

    model = PolypModule()

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    print("prediction1_4:", out[0].size())
    print("emb:", out[1].size())

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    print("Peak GPU Memory: %.3f GB" % max_memory)



