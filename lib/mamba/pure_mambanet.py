from lib.mamba.models_mamba import VisionMamba
from timm.models.vision_transformer import VisionTransformer, _cfg
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbone.LightRFB import LightRFB

from lib.backbone.Decoder import DenseDecoder


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        #self.drop = nn.Dropout(drop)

        self.conv1 = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)

        self.apply(self._init_weights)
    
    def dwconv(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        #x = self.drop(x)
        return x

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
    def __init__(self, high_channel=128, low_channel=128, middle_channel=32):
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

class PureMambaNet(nn.Module):
    def __init__(self, f_num=5, img_size=(224, 224), **kwargs):
        super(PureMambaNet, self).__init__()
        self.f_num = f_num + 1
        self.fea_channels = 32

        mamba_channels = [96, 192, 384]
        self.mamba_chs = mamba_channels
        self.mamba_layer1 = self.make_mamba_layer(patch_size=4, img_size=img_size, channels=3, d_state=1, d_conv=3, embed_dim=mamba_channels[0], depth=2)
        self.size1 = tuple(elem // 4 for elem in img_size)
        self.mamba_layer2 = self.make_mamba_layer(patch_size=2, img_size=self.size1, channels=mamba_channels[0], d_state=1, d_conv=3, embed_dim=mamba_channels[1], depth=2)
        self.size2 = tuple(elem // 2 for elem in self.size1)
        self.mamba_layer3 = self.make_mamba_layer(patch_size=2, img_size=self.size2, channels=mamba_channels[1], d_state=1, d_conv=3, embed_dim=mamba_channels[2], depth=4)
        self.size3 = tuple(elem // 2 for elem in self.size2)

        self.mamba_layer4 = self.make_mamba_layer(patch_size=16, img_size=img_size, channels=3, d_state=1, d_conv=3, embed_dim=mamba_channels[2], depth=4)

        self.High_RFB = LightRFB(channels_in=384, channels_mid=128, channels_out=self.fea_channels)  # out 32
        self.Low_RFB = LightRFB(channels_in=384, channels_mid=128, channels_out=self.fea_channels)

        ## Decoder ##
        middle_channel = 16
        self.decoder = conbine_feature(self.fea_channels, self.fea_channels, middle_channel)
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False))
    
    def make_mamba_layer(self, patch_size, img_size, channels, d_state, d_conv, embed_dim, depth, **kwargs):
        return VisionMamba(
            patch_size=patch_size, img_size=img_size, channels=channels, d_state=d_state, d_conv=d_conv, conv_bias=False,
            stride=patch_size, embed_dim=embed_dim, depth=depth, drop_path_rate = 0.,
            residual_in_fp32=True, fused_add_norm=True, final_pool_type='all', 
            if_abs_pos_embed=False, bimamba_type="v2", if_cls_token=False, 
            if_devide_out=True, use_middle_cls_token=False, **kwargs
        )

    def forward(self, x):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        B = origin_shape[0] * origin_shape[1]

        # branch 1
        x_1 = self.mamba_layer1(x)
        x_1 = x_1.reshape(B, self.mamba_chs[0], *self.size1)
        x_1 = self.mamba_layer2(x_1)
        x_1 = x_1.reshape(B, self.mamba_chs[1], *self.size2)
        x_1 = self.mamba_layer3(x_1)
        x_1 = x_1.reshape(B, self.mamba_chs[2], *self.size3)
        # branch 2
        x_2 = self.mamba_layer4(x)
        x_2 = x_2.reshape(B, self.mamba_chs[2], *self.size3)

        x_1 = self.Low_RFB(x_1)
        x_2 = self.High_RFB(x_2)
        # Interpolate

        out = self.decoder(x_2, x_1)
        out = torch.sigmoid(
            F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear",
                          align_corners=False))
        # out = self.decoder(global_x4)
        # out = torch.sigmoid(F.interpolate(out, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False))
        return out

