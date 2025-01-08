from lib.mamba.models_mamba import VisionMamba
from timm.models.vision_transformer import VisionTransformer, _cfg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbone.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.backbone.LightRFB import LightRFB

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
        B, HxW, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
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

class MambaNet2(nn.Module):
    def __init__(self, f_num=5, img_size=(224, 224), mlp_ratio=2.0, **kwargs):
        super(MambaNet2, self).__init__()

        self.f_num = f_num+1
        self.patch_size = 14
        self.fea_channels = 32

        self.feature_extractor = res2net50_v1b_26w_4s(pretrained=True)
        
        # self.High_RFB = LightRFB(channels_in=768, channels_mid=128, channels_out=self.fea_channels)
        # self.Low_RFB = LightRFB(channels_in=512, channels_mid=128, channels_out=self.fea_channels)
        self.High_RFB = LightRFB(channels_in=1024, channels_mid=128, channels_out=self.fea_channels)
        self.Low_RFB = LightRFB(channels_in=192, channels_mid=128, channels_out=self.fea_channels)

        self.mamba_channels = [96, 192, 384, 768]
        self.sizes1 = img_size
        self.mamba1 = self.make_mamba_layer(patch_size=4, img_size=self.sizes1, channels=3, d_state=16, d_conv=3, embed_dim=self.mamba_channels[0], depth=3, **kwargs)
        self.sizes2 = tuple(elem // 4 for elem in self.sizes1)
        self.mamba2 = self.make_mamba_layer(patch_size=2, img_size=self.sizes2, channels=self.mamba_channels[0], d_state=16, d_conv=3, embed_dim=self.mamba_channels[1], depth=4, **kwargs)
        self.sizes3 = tuple(elem // 2 for elem in self.sizes2)
        # self.mamba3 = self.make_mamba_layer(patch_size=2, img_size=self.sizes3, channels=self.mamba_channels[1], d_state=64, d_conv=3, embed_dim=self.mamba_channels[2], depth=6, **kwargs)
        # self.sizes4 = tuple(elem // 2 for elem in self.sizes3)
        # self.mamba4 = self.make_mamba_layer(patch_size=2, img_size=self.sizes4, channels=self.mamba_channels[2], d_state=64, d_conv=3, embed_dim=self.mamba_channels[3], depth=3, **kwargs)
        # self.final_sizes = tuple(elem // 2 for elem in self.sizes4)

        self.select_gate1 = LightRFB(channels_in=256, channels_mid=128, channels_out=self.mamba_channels[0])
        self.select_gate2 = LightRFB(channels_in=512, channels_mid=256, channels_out=self.mamba_channels[1])
        # self.select_gate3 = LightRFB(channels_in=1024, channels_mid=1024, channels_out=self.mamba_channels[2])
        # self.select_gate4 = LightRFB(channels_in=2048, channels_mid=1024, channels_out=self.mamba_channels[3])

       
        middle_channel = 16 
        self.decoder = conbine_feature(self.fea_channels, self.fea_channels, middle_channel)
        self.SegNIN = nn.Conv2d(middle_channel, 1, kernel_size=1, bias=False)

    def make_mamba_layer(self, patch_size, img_size, channels, d_state, d_conv, embed_dim, depth, **kwargs):
        return VisionMamba(
            patch_size=patch_size, img_size=img_size, channels=channels, d_state=d_state, d_conv=d_conv, conv_bias=False,
            stride=patch_size, embed_dim=embed_dim, depth=depth, drop_path_rate = 0.,
            residual_in_fp32=True, fused_add_norm=True, final_pool_type='all', 
            if_abs_pos_embed=False, bimamba_type="v2", if_cls_token=False, 
            if_devide_out=True, use_middle_cls_token=False, **kwargs
        )

    def get_firstframe(self, x, B, N, channels, sizes):
        x = x.reshape(B, N, channels, *sizes)
        x = x[:, 0, ...]
        return x

    def get_seqframes(self, x, B, N, channels, sizes):
        x = x.reshape(B, N, channels, *sizes)
        x = x[:, 1:self.f_num, ...]
        return x

    def fuse_frames(self, x1, x2, channels, sizes):
        x2 = torch.cat((x1.unsqueeze(1), x2), dim=1)
        x2 = x2.reshape(-1, channels, *(sizes))
        return x2
        
    def forward(self, x):
        B, N, C, H, W  = x.shape

        # first frame
        # x_f1 = x[:, 0, ...]
        
        # all frames
        x_allf = x.view(-1, C, H, W)

        ### first frame from all frames ###
        x = self.feature_extractor.conv1(x_allf)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        x = self.feature_extractor.layer1(x)
        x_f1 = self.get_firstframe(x, B, N, 256, self.sizes2)
        x_f1_s1 = self.select_gate1(x_f1)
        x = self.feature_extractor.layer2(x)
        x_f2 = self.get_firstframe(x, B, N, 512, self.sizes3)
        x_f1_s2 = self.select_gate2(x_f2)
        x = self.feature_extractor.layer3(x)
        high_feature = x
        # x_f3 = self.get_firstframe(x, B, N, 1024, self.sizes4)
        # x_f1_s3 = self.select_gate3(x_f3)

        ### single first frame ###
        # x_f1 = self.feature_extractor.conv1(x_f1)
        # x_f1 = self.feature_extractor.bn1(x_f1)
        # x_f1 = self.feature_extractor.relu(x_f1)
        # x_f1 = self.feature_extractor.maxpool(x_f1)
        # x_f1 = self.feature_extractor.layer1(x_f1)       # [B, 256, 56, 112]
        # x_f1_s1 = self.select_gate1(x_f1)
        # x_f1 = self.feature_extractor.layer2(x_f1)    # [B, 512, 28, 56]
        # x_f1_s2 = self.select_gate2(x_f1)
        # x_f1 = self.feature_extractor.layer3(x_f1)    # [B, 1024, 14, 28]
        # x_f1_s3 = self.select_gate3(x_f1)
        # x_f1 = self.feature_extractor.layer4(x_f1)    # [B, 2048, 7, 14]
        # x_f1_s4 = self.select_gate4(x_f1)

        # rest frame mamba backbone
        x_fs_s1 = self.mamba1(x_allf)
        x_fs_s1 = self.get_seqframes(x_fs_s1, B, N, self.mamba_channels[0], self.sizes2)
        x_fs_s2 = self.fuse_frames(x_f1_s1, x_fs_s1, self.mamba_channels[0], self.sizes2)
        
        x_fs_s2 = self.mamba2(x_fs_s2)
        x_fs_s2 = self.get_seqframes(x_fs_s2, B, N, self.mamba_channels[1], self.sizes3)
        x_fs_s3 = self.fuse_frames(x_f1_s2, x_fs_s2, self.mamba_channels[1], self.sizes3)
        low_feature = x_fs_s3

        # x_fs_s3 = self.mamba3(x_fs_s3)
        # x_fs_s3 = self.get_seqframes(x_fs_s3, B, N, self.mamba_channels[2], self.sizes4)
        # x_fs_s4 = self.fuse_frames(x_f1_s3, x_fs_s3, self.mamba_channels[2], self.sizes4)
        
        # Reduce the channel dimension.
        high_feature = self.High_RFB(high_feature)
        low_feature = self.Low_RFB(low_feature)

        # Resize high-level feature to the same as low-level feature.
        high_feature = F.interpolate(high_feature, size=(low_feature.shape[-2], low_feature.shape[-1]),
                                     mode="bilinear",
                                     align_corners=False)

        out = self.decoder(high_feature, low_feature)
        out = torch.sigmoid(F.interpolate(self.SegNIN(out), size=(H, W), mode="bilinear", align_corners=False))
        return out