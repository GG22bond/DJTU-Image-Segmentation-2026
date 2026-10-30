import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from pvtv2 import my_pvt_v2_b2
from decoders import WBANet
from mmseg.ops import resize


logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class WBANetout(nn.Module):
    def __init__(self, n_class=1):
        super(WBANetout, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        self.backbone = my_pvt_v2_b2()  # [64, 128, 320, 512]
        path = '../../weight/backbone/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        # decoder initialization
        self.decoder = WBANet(channels=[512, 320, 128, 64])
        
        # Prediction heads initialization
        # self.out_head1 = nn.Conv2d(512, n_class, 1).to('cuda')
        # self.out_head2 = nn.Conv2d(320, n_class, 1).to('cuda')
        # self.out_head3 = nn.Conv2d(128, n_class, 1).to('cuda')
        # self.out_head4 = nn.Conv2d(64, n_class, 1).to('cuda')
        self.out_head1 = nn.Conv2d(512, n_class, 1).to('cuda')
        self.out_head2 = nn.Conv2d(320, n_class, 1).to('cuda')
        self.out_head3 = nn.Conv2d(128, n_class, 1).to('cuda')
        self.out_head4 = nn.Conv2d(64, n_class, 1).to('cuda')
        # self.conv_dw = WTConv2d(
        #     3,
        #     3,
        # )
        # self.contrasthead = ProjectionHead(dim=32)
        # self.Translayer4 = BasicConv2d(512, 32, 1)


    def forward(self, x):

        r,g,b=x[:,0,:,:],x[:,1,:,:],x[:,2,:,:]
        gray_images=0.299*r+0.587*g+0.114*b
        threshold=180/255.0
        brightness_images=(gray_images>=threshold).float()
        brightness_images=brightness_images.unsqueeze(1)


        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        # transformer backbone as encoder
        xall = self.backbone(x)
        x1, x2, x3, x4=xall
        # emb = self.contrasthead(self.Translayer4(x4))
        # rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = laplacian
        _xb_1 = resize(brightness_images, size=x1.size()[2:], mode='bilinear', align_corners=False)
        _xb_2 = resize(brightness_images, size=x2.size()[2:], mode='bilinear', align_corners=False)
        _xb_3 = resize(brightness_images, size=x3.size()[2:], mode='bilinear', align_corners=False)
        _xb_4 = resize(brightness_images, size=x4.size()[2:], mode='bilinear', align_corners=False)


        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4+_xb_4, [x3+_xb_3, x2+_xb_2, x1+_xb_1])
        # x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3 , x2 , x1 ])
        
        # prediction heads  
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')

        return p1, p2, p3, p4


        
if __name__ == '__main__':

    from thop import clever_format, profile

    input = torch.randn(1, 3, 352, 352).cuda()

    model = WBANetout()

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    print("agg_out:", out[0].size())
    print("out4:", out[1].size())
    print("out3:", out[2].size())
    print("out2:", out[3].size())

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    print("Peak GPU Memory: %.3f GB" % max_memory)


