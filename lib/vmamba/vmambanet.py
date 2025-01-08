import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.vmamba.vmamba import VSSM, Backbone_VSSM
from lib.backbone.Decoder import DenseDecoder
from lib.backbone.LightRFB import LightRFB

class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output

class conbine_feature(nn.Module):
    def __init__(self, high_channel=128, low_channel=128, middle_channel=16):
        super(conbine_feature, self).__init__()
        self.up2_high = DilatedParallelConvBlockD2(high_channel, middle_channel)
        self.up2_low = nn.Conv2d(low_channel, middle_channel, 1, stride=1, padding=0, bias=False)
        self.up2_bn2 = nn.BatchNorm2d(middle_channel)
        self.up2_act = nn.PReLU(middle_channel)
        self.refine = nn.Sequential(
            nn.Conv2d(middle_channel, middle_channel, 3, padding=1, bias=False), 
            nn.BatchNorm2d(middle_channel), nn.PReLU())

    def forward(self, high_fea, low_fea=None):
        high_fea = self.up2_high(high_fea)
        if low_fea is not None:
            low_fea = self.up2_bn2(self.up2_low(low_fea))
            refine_feature = self.refine(self.up2_act(high_fea + low_fea))
        else:
            refine_feature = self.refine(self.up2_act(high_fea))
        return refine_feature

class VmambaNet(nn.Module):
    def __init__(self, pretrained=True):
        super(VmambaNet, self).__init__()

        # Tiny
        self.mamba_backbone = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            dims=96,
            depths=[2, 2, 15, 2],
            ssm_d_state=1,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            ssm_conv=3,
            ssm_conv_bias=False,
            forward_type="v05_noz",
            mlp_ratio=4.0,
            downsample_version="v3",
            patchembed_version="v2",
            drop_path_rate=0.3,
            norm_layer="ln2d",
        )

        if pretrained:
            ckpt = torch.load("/data/cgl/pretrained/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth")
            ckpt_dict = ckpt['state_dict']
            #model_dict = {key: value for key, value in ckpt_dict.items() if (key in ckpt_dict and key.split('.')[])}
            model_dict = {}
            for ckpt_key, value in ckpt_dict.items():
                key_list = ckpt_key.split('.')[1::]
                key = ".".join(key_list)
                model_dict[key] = value
            self.mamba_backbone.load_state_dict(model_dict, strict=False)
        
        self.High_RFB = LightRFB(channels_in=768, channels_mid=384, channels_out=32)
        self.Low_RFB = LightRFB(channels_in=384, channels_mid=192, channels_out=32)

        self.decoder = conbine_feature(32, 32, 16)
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))
        #self.decoder = DenseDecoder(self.embed_dim)
        
    def forward(self, x):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        x1, x2, x3, x4 = self.mamba_backbone(x)

        x3 = self.Low_RFB(x3)
        x4 = self.High_RFB(x4)
        x4 = F.interpolate(x4, size=(x3.shape[-2], x3.shape[-1]), mode="bilinear", align_corners=False)

        out = self.decoder(x4, x3)
        out = F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        return out

