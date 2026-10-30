import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from pvtv2 import my_pvt_v2_b2
from decoders import GCASCADE, GCASCADE_Cat


logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
    

                
class PVT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu', skip_aggregation='additive'):
        super(PVT_GCASCADE, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class
        
        # conv block to convert single channel to 3 channels
        self.conv_1cto3c = nn.Sequential(
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
        
        self.channels = [512, 320, 128, 64]
        
        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1]*2, self.channels[2]*2, self.channels[3]*2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size=img_size, k=k, padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
                     ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)
        

    def forward(self, x):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv_1cto3c(x)
        
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])
        
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

    model = PVT_GCASCADE()

    print("Model:\n", model)

    model.cuda()
    model.eval()

    out = model(input)

    print("out0:", out[0].size())
    print("out1:", out[1].size())
    print("out2:", out[2].size())
    print("out3:", out[3].size())

    flops, params = profile(model.cuda(0), (input,), verbose=False)
    flops, params = clever_format([flops * 2, params], "%.3f")

    max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB

    print('Total FLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    print("Peak GPU Memory: %.3f GB" % max_memory)



