import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import Res2Net50, weight_init

class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        # self.args    = args
        self.bkbone  = Res2Net50()
        self.linear5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)
        # self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
        pred = self.predict(pred)
        pred = F.interpolate(pred, scale_factor=8, mode='bilinear')
        return pred

    # def initialize(self):
    #     if self.args.snapshot:
    #         self.load_state_dict(torch.load(self.args.snapshot))
    #     else:
    #         weight_init(self)


if __name__ == '__main__':

    from thop import clever_format, profile

    input = torch.randn(1, 3, 352, 352).cuda()

    model = Model()

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    print("out:", out.size())

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    print("Peak GPU Memory: %.3f GB" % max_memory)

