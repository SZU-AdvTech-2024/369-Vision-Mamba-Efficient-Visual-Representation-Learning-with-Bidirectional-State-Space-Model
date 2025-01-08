from lib.mamba.models_mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as Vim
from timm.models.vision_transformer import VisionTransformer, _cfg

import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from timm.models.layers import trunc_normal_
from lib.backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbone.LightRFB import LightRFB
from lib.backbone.pvt_v2 import pvt_v2_b2, pvt_v2_b5
from lib.backbone.CompressEncoder import CompressConv
from lib.backbone.Decoder import DenseDecoder
from lib.mamba.udfe import UDFE

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

class VimNet(nn.Module):
    def __init__(self, pretrained=True, img_size=(224, 224)):
        super(VimNet, self).__init__()

        # Vim-S
        self.mamba_backbone = Vim(img_size=img_size)

        if pretrained:
            ckpt = torch.load("/media/cgl/pretrained/vim_s_midclstok_ft_81p6acc.pth")
            ckpt_dict = ckpt['model']
            # model_dict = {key: value for key, value in ckpt_dict.items() if (key in ckpt_dict and key.split('.')[])}
            # model_dict = {}
            # for ckpt_key, value in ckpt_dict.items():
            #     key_list = ckpt_key.split('.')[1::]
            #     key = ".".join(key_list)
            #     model_dict[key] = value
            # del ckpt_dict["pos_embed"]
            self.mamba_backbone.load_state_dict(ckpt_dict, strict=False)
        
        # self.High_RFB = LightRFB(channels_in=768, channels_mid=384, channels_out=32)
        self.Low_RFB = LightRFB(channels_in=384, channels_mid=384, channels_out=32)

        self.decoder = conbine_feature(32, 32, 16)
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(16, 1, kernel_size=1, bias=False))
        # self.decoder = DenseDecoder(32)
        
    def forward(self, x):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        # x = self.mamba_backbone(x)
        x, tokens = self.mamba_backbone(x)
        x = torch.cat([x[:,:tokens], x[:,tokens+1:]], dim=1)
        x = x.reshape(-1, 384, 27, 27)

        x = self.Low_RFB(x)
        x = F.interpolate(x, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        out = self.decoder(x, x)
        out = self.SegNIN(out)
        # out = F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)

        # out = self.decoder(x)
        # out = torch.sigmoid(F.interpolate(out, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False))

        return out
