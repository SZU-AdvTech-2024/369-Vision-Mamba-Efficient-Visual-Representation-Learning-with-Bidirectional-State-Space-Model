from lib.mamba.models_mamba import VisionMamba
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
from lib.vmamba.vmamba import VSSM, Backbone_VSSM
from lib.backbone.CompressEncoder import CompressConv
from lib.backbone.Decoder import DenseDecoder
from lib.mamba.udfe import UDFE

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

class combine_feature(nn.Module):
    def __init__(self, high_channel=128, low_channel=128, middle_channel=32):
        super(combine_feature, self).__init__()
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

class MambaNet(nn.Module):
    def __init__(self, f_num=5, img_size=(224, 224), mlp_ratio=2.0, **kwargs):
        super(MambaNet, self).__init__()

        #### Res2Net-50 ####
        # self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        #### PVT-V2-B2 ####
        self.feature_extractor = pvt_v2_b2()
        self.load_model("/media/cgl/pretrained/pvt_v2_b2.pth")
        #### Vmamba-S ####
        # self.feature_extractor = Backbone_VSSM(
        #     out_indices=(0, 1, 2, 3),
        #     dims=96,
        #     depths=[2, 2, 15, 2],
        #     ssm_d_state=1,
        #     ssm_dt_rank="auto",
        #     ssm_ratio=2.0,
        #     ssm_conv=3,
        #     ssm_conv_bias=False,
        #     forward_type="v05_noz",
        #     mlp_ratio=4.0,
        #     downsample_version="v3",
        #     patchembed_version="v2",
        #     drop_path_rate=0.3,
        #     norm_layer="ln2d",
        # )
        # self.load_vmamba("/media/cgl/pretrained/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth")

        self.fea_channels = 32
        self.embed_dim = 384
        self.patch_size = 11
        self.f_num = f_num+1
        self.img_size = (img_size[0] // 16 // 2, img_size[1] // 16 // 2 * self.f_num)
        self.udfe_size = (img_size[0] // 4 // 2, img_size[1] // 4 // 2)

        # self.High_RFB = LightRFB(channels_in=2048, channels_mid=512, channels_out=self.fea_channels)
        # self.Low_RFB = LightRFB(channels_in=1024, channels_mid=256, channels_out=self.fea_channels)
        # self.First_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=self.fea_channels)
        self.High_RFB = LightRFB(channels_in=512, channels_mid=256, channels_out=self.fea_channels)
        self.Low_RFB = LightRFB(channels_in=320, channels_mid=160, channels_out=self.fea_channels)
        self.First_RFB = LightRFB(channels_in=128, channels_mid=64, channels_out=self.fea_channels)
        # self.High_RFB = LightRFB(channels_in=768, channels_mid=768, channels_out=self.fea_channels)
        # self.Low_RFB = LightRFB(channels_in=384, channels_mid=384, channels_out=self.fea_channels)
        # self.First_RFB = LightRFB(channels_in=192, channels_mid=192, channels_out=self.fea_channels)

        self.spacetime_extractor = VisionMamba(
            patch_size=self.patch_size, img_size=self.img_size, channels=self.fea_channels, d_state=1, d_conv=3, conv_bias=False,
            stride=self.patch_size, embed_dim=self.embed_dim, depth=2, drop_path_rate=0., rms_norm=False,
            residual_in_fp32=True, fused_add_norm=True, final_pool_type='all', 
            if_abs_pos_embed=False, bimamba_type="v2", if_cls_token=False, 
            if_devide_out=True, use_middle_cls_token=False, **kwargs
        )
        self.spacetime_extractor.default_cfg = _cfg()
        self.spacetime_outlayer = Mlp(in_features=self.embed_dim, hidden_features=int(self.embed_dim * mlp_ratio), out_features=self.fea_channels)
        
        self.mask_extract = nn.Conv2d(self.fea_channels, 1, kernel_size=3, stride=1, padding=1)

        # for baseline #
        middle_channel = 16
        # self.decoder = combine_feature(512, 320, middle_channel)
        self.decoder = combine_feature(self.fea_channels, self.fea_channels, middle_channel)
        self.SegNIN = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False))

        self.udfe = UDFE(img_size=self.udfe_size, channels=self.fea_channels, embed_dim=self.embed_dim, f_num=self.f_num)
        self.decoder2 = combine_feature(self.fea_channels, self.fea_channels, middle_channel)
        self.SegNIN2 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False))
    
    def load_model(self, ckpt):
        pretrained_dict = torch.load(ckpt)
        model_dict = self.feature_extractor.state_dict()
        print("Load pretrained parameters from {}".format(ckpt))
        # for k, v in pretrained_dict.items():
        #     if (k in model_dict):
        #         print("load:%s"%k)
        #     else:
        #         print("jump over:%s"%k)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)        
        self.feature_extractor.load_state_dict(model_dict)
        print("PVTv2 Loaded!")
    
    def load_vmamba(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        ckpt_dict = ckpt['state_dict']
        #model_dict = {key: value for key, value in ckpt_dict.items() if (key in ckpt_dict and key.split('.')[])}
        model_dict = {}
        for ckpt_key, value in ckpt_dict.items():
            key_list = ckpt_key.split('.')[1::]
            key = ".".join(key_list)
            model_dict[key] = value
        self.feature_extractor.load_state_dict(model_dict, strict=False)
    
    def forward(self, x, mask_on=False, sdpm_on=True, udfe_on=True, mode='eval'):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        ### res2net-50 ###
        # x = self.feature_extractor.conv1(x)
        # x = self.feature_extractor.bn1(x)
        # x = self.feature_extractor.relu(x)
        # x = self.feature_extractor.maxpool(x)
        # x = self.feature_extractor.layer1(x)
        # x = self.feature_extractor.layer2(x)
        # x1_f = self.feature_extractor.layer3(x)
        # x2_f = self.feature_extractor.layer4(x1_f)
        ### pvtv2-b5 and vmamba-s ###
        _, x, x1_f, x2_f = self.feature_extractor(x)
        
        x2 = self.High_RFB(x2_f)  # [B, fea_channels, 16, 32]
        x1 = self.Low_RFB(x1_f)   # [B, fea_channels, 32, 64]
        x0 = self.First_RFB(x)    # [B, fea_channels, 64, 128]

        B, C, H, W = x2.shape
        if sdpm_on:
            x2 = x2.reshape(origin_shape[1], origin_shape[0], C, H, W)
            x2_frames = []

            # Add random mask
            mask_flag = False
            if mask_on:
                if random.random() > 0.5:
                    mask_flag = True
                    mask_idx = random.randint(1, origin_shape[1]-1)
            for i in range(origin_shape[1]):
                if mask_flag and i == mask_idx:
                    mask_tensor = torch.zeros(x2[i].shape, dtype=torch.float32).to(x2[i].device)
                    x2_frames.append(mask_tensor)
                else:
                    x2_frames.append(x2[i])         # x2[i]: [B, fea_channels, 16, 32]
            x2_cat = torch.cat(x2_frames, dim=3)    #x2_cat: [B, fea_channels, 16, 192]
            x2_cat = self.spacetime_extractor(x2_cat)
            x2_cat = x2_cat.reshape(B, x2_cat.shape[2], self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size // self.f_num)
            x2_cat = F.interpolate(x2_cat, size=(x2.shape[-2], x2.shape[-1]), mode='nearest')
            x2_cat = self.spacetime_outlayer(x2_cat.reshape(B, H*W, x2_cat.shape[1]), H, W)

            ### Add Mask Guidance ###
            mask_guide = F.interpolate(self.mask_extract(x1), size=(x2.shape[-2], x2.shape[-1]), mode="bilinear", align_corners=False)
            x2_cat = (1+torch.sigmoid(mask_guide)).expand(-1, C, -1, -1).mul(x2_cat.reshape(B, C, H, W)) + x2.reshape(B, C, H, W)
        else:
            mask_guide = F.interpolate(self.mask_extract(x1), size=(x2.shape[-2], x2.shape[-1]), mode="bilinear", align_corners=False)
            x2_cat = x2

        ## decoder 1 ##
        x2 = F.interpolate(x2_cat, size=(x1.shape[-2], x1.shape[-1]), mode="bilinear", align_corners=False)
        # x2 = F.interpolate(x2_cat, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        # x1 = F.interpolate(x1, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        decoder_out = self.decoder(x2, x1)
        # f_maps = decoder_out
        decoder_out = self.SegNIN(decoder_out)
        # out = decoder_out
        out = F.interpolate(decoder_out, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        mask_guide = F.interpolate(mask_guide, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        
        ###### Ultimate Dynamic Feature Extraction ######
        if udfe_on:
            masks = torch.sigmoid(F.interpolate(decoder_out, size=(x0.shape[-2], x0.shape[-1]), mode='bilinear', align_corners=False))
            _, x0_C, x0_H, x0_W = x0.shape
            x0 = x0.reshape(origin_shape[0], origin_shape[1], x0_C, x0_H, x0_W)
            masks = masks.reshape(origin_shape[0], origin_shape[1], 1, x0_H, x0_W)
            dynamic_f = self.udfe(x0, masks)
            sdpm_out = F.interpolate(x2_cat, size=(dynamic_f.shape[-2], dynamic_f.shape[-1]), mode="bilinear", align_corners=False)
            # sdpm_out = F.interpolate(x2_cat, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
            # dynamic_f =  F.interpolate(dynamic_f.reshape(B, x0_C, x0_H, x0_W), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
            # udfe_out = self.decoder2(sdpm_out, dynamic_f)
            udfe_out = self.decoder2(sdpm_out, dynamic_f.reshape(B, x0_C, x0_H, x0_W))
            udfe_out = self.SegNIN2(udfe_out)
            f_maps = torch.sigmoid(udfe_out)
            udfe_out = F.interpolate(udfe_out, size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        
        # TODO: out可以采取独立分开的策略，也可以采取融合的策略
        if mode == 'train':
            assert mode in ['train', 'eval'], "mode should be train or eval."
            # return out, mask_guide
            if udfe_on:
                # return torch.sigmoid(out), torch.sigmoid(mask_guide), torch.sigmoid(udfe_out)
                return out, mask_guide, udfe_out
            else:
                # return torch.sigmoid(out), torch.sigmoid(mask_guide)
                return out, mask_guide
        else:
            # return f_maps
            if udfe_on:
                # return torch.sigmoid(out)
                return torch.sigmoid(udfe_out)
                # return torch.sigmoid(mask_guide)
            else:
                return torch.sigmoid(out)

    def backbone_forward(self, x):
    # def forward(self, x):
        origin_shape = x.shape
        x = x.view(-1, *origin_shape[2:])
        ### Res2Net50 ###
        # x = self.feature_extractor.conv1(x)
        # x = self.feature_extractor.bn1(x)
        # x = self.feature_extractor.relu(x)
        # x = self.feature_extractor.maxpool(x)
        # x = self.feature_extractor.layer1(x)
        # x1 = self.feature_extractor.layer2(x)
        # x2 = self.feature_extractor.layer3(x1)
        ### pvtv2 ####
        _, x, x1, x2 = self.feature_extractor(x)
        # f_map = x1

        # B, C, H, W = x1.shape
        # x1 = x1.reshape(origin_shape[0], origin_shape[1], C, H, W)
        # x1 = x1[:, 1:self.f_num, ...].reshape(-1, C, H, W)

        # B, C, H, W = x2.shape
        # x2 = x2.reshape(origin_shape[0], origin_shape[1], C, H, W)
        # x2 = x2[:, 1:self.f_num, ...].reshape(-1, C, H, W)

        x2 = F.interpolate(x2, size=(x1.shape[-2], x1.shape[-1]), mode="bilinear", align_corners=False)

        out = self.decoder(x2.clone(), x1.clone())
        f_map = out
        out = F.interpolate(self.SegNIN(out), size=(origin_shape[-2], origin_shape[-1]), mode="bilinear", align_corners=False)
        # return out
        return f_map