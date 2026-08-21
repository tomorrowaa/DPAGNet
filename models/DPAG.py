"""
import torch
import torch.nn as nn
import torch.nn.functional as TF
from models.common import Conv, CAB
"""
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from models.common import Conv, CAB, CALayer


class Conv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv_Relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DeConv_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeConv_Relu, self).__init__()
        self.deconv = Conv_Relu(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = TF.upsample(x, scale_factor=2)
        return self.deconv(x)

class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y




class GLAB_PEN(nn.Module):
    def __init__(self, channel, n_feats, dim=32, config=[2, 2, 2, 2, 2], drop_path_rate=0.0, input_resolution=128):
        super(GLAB_PEN, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 16
        self.window_size = 8

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = nn.Sequential(
            nn.Conv2d(channel, dim, 3, 1, 1, bias=False)
        )

        begin = 0
        self.m_down1 = nn.Sequential(*(
            [ConvTransBlock(
                dim // 2, dim // 2, self.head_dim, self.window_size,
                dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution
            ) for i in range(config[0])]
            + [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]
        ))

        begin += config[0]
        self.m_down2 = nn.Sequential(*(
            [ConvTransBlock(
                dim, dim, self.head_dim, self.window_size,
                dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2
            ) for i in range(config[1])]
            + [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]
        ))

        begin += config[1]
        self.m_body = nn.Sequential(*[
            ConvTransBlock(
                2 * dim, 2 * dim, self.head_dim, self.window_size,
                dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 4
            ) for i in range(config[2])
        ])

        begin += config[2]
        self.m_up2 = nn.Sequential(*(
            [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False)]
            + [ConvTransBlock(
                dim, dim, self.head_dim, self.window_size,
                dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution // 2
            ) for i in range(config[3])]
        ))

        begin += config[3]
        self.m_up1 = nn.Sequential(*(
            [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False)]
            + [ConvTransBlock(
                dim // 2, dim // 2, self.head_dim, self.window_size,
                dpr[i + begin], 'W' if not i % 2 else 'SW', input_resolution
            ) for i in range(config[4])]
        ))

        self.tail = Conv(dim, n_feats, 3)
        self.out2 = PCM(channel, n_feats, 3, bias=False)

    def forward(self, x0):
        h, w = x0.size()[-2:]

        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0_pad = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0_pad)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)

        x = self.m_body(x3)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)

        x = self.tail(x)
        x = x[..., :h, :w]

        feats, out = self.out2(x, x0)
        return out, feats




## Residual Channel Attention Network (RCAN)
class RCAB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, reduction, n_blocks, bias=False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        self.conv1 = Conv(in_feat, out_feat, 3)
        self.cab = nn.Sequential(*[CAB(out_feat, kernel_size=kernel_size, reduction=reduction, bias=bias, act=act) for _ in range(n_blocks)])
        self.conv2 = Conv(out_feat, out_feat, 3)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.cab(x) + x
        x = self.conv2(x1)
        return x


class PCM(nn.Module):
    def __init__(self, channels, n_feat, kernel_size, bias):
        super(PCM, self).__init__()
        self.conv1 = Conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = Conv(n_feat, channels, kernel_size, bias=bias)
        self.conv3 = Conv(channels, n_feat, kernel_size, bias=bias)
        self.conv4 = Conv(n_feat, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = self.conv3(img)
        x3 = torch.sigmoid(self.conv4(x1))
        x2 = x3 * x2
        x1 = x + x2
        return x1, img


class PEN(nn.Module):
    def __init__(self, channel, n_feats):
        super(PEN, self).__init__()
        self.enc1 = nn.Sequential(Conv_Relu(channel, n_feats, 3, 1), Conv(n_feats, n_feats, 3))
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(Conv_Relu(n_feats, n_feats * 2, 3, 1), Conv(n_feats * 2, n_feats * 2, 3))
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = Conv_Relu(n_feats * 2, n_feats * 4, 3, 1)
        self.conv3 = Conv(n_feats * 4, n_feats * 4, 3)

        self.up32 = DeConv_Relu(n_feats * 4, n_feats * 2, 3, 1)
        self.dec2 = nn.Sequential(Conv_Relu(n_feats * 4, n_feats * 4, 3, 1), Conv(n_feats * 4, n_feats * 2, 3))

        self.up21 = DeConv_Relu(n_feats * 2, n_feats, 3, 1)
        self.dec1 = nn.Sequential(Conv_Relu(n_feats * 2, n_feats * 2, 3, 1), Conv(n_feats * 2, n_feats, 3))

        self.out2 = PCM(channel, n_feats, 3, bias=False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pooling1(enc1))
        neck = self.enc3(self.pooling2(enc2))
        neck = self.conv3(neck)

        up2 = self.up32(neck)
        cat2 = torch.cat([up2, enc2], 1)
        dec2 = self.dec2(cat2)

        up1 = self.up21(dec2)
        cat1 = torch.cat([up1, enc1], 1)
        dec1 = self.dec1(cat1)

        feats, out = self.out2(dec1,x)
        return out, feats


class PGB(nn.Module):
    def __init__(self, in_channel=3, f_channel=64, g_channel=1):
        super(PGB, self).__init__()
        self.conv1 = Conv(g_channel, 1, 1)
        self.conv2 = Conv(g_channel, f_channel, 1)
        self.conv3 = Conv(in_channel, f_channel, 3)
        self.conv4 = Conv(f_channel, f_channel, 3)
        self.conv5 = Conv(f_channel, f_channel, 3)

    def forward(self, img, guide_f):
        guide_mul = torch.sigmoid(self.conv1(guide_f))
        guide_add = self.conv2(guide_f)
        x = self.conv3(img)
        x = x * guide_mul
        x = self.conv4(x)
        x = x + guide_add
        x = self.conv5(x)
        return x
class MultiStrengthPriorFusion(nn.Module):
    def __init__(self, feat_channels=64, delta=0.3, gamma=2.0):
        super(MultiStrengthPriorFusion, self).__init__()
        self.delta = delta
        self.gamma = gamma

        # 从 illumination feature 里预测一个 1-channel gate
        self.illum_head = nn.Sequential(
            Conv(feat_channels, feat_channels // 2, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 根据三路 prior + gate 预测融合权重
        self.weight_net = nn.Sequential(
            Conv(feat_channels * 3 + 1, feat_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, 3, 1, padding=0)
        )


    def forward(self, pc_feat, illum_feat):
        # illum_map: [B,1,H,W]
        illum_map = self.illum_head(illum_feat)

        # Bread 风格：暗区 gate 更强
        noise_gate = torch.exp(-self.gamma * illum_map)

        # 三个强度版本
        p1 = pc_feat * (1.0 - self.delta * noise_gate)   # 弱去噪 / 更保守
        p2 = pc_feat                                     # 中等强度
        p3 = pc_feat * (1.0 + self.delta * noise_gate)   # 强去噪 / 更激进

        # 融合权重
        fusion_logits = self.weight_net(torch.cat([p1, p2, p3, noise_gate], dim=1))
        fusion_weight = torch.softmax(fusion_logits, dim=1)

        w1 = fusion_weight[:, 0:1, :, :]
        w2 = fusion_weight[:, 1:2, :, :]
        w3 = fusion_weight[:, 2:3, :, :]

        fused_feat = w1 * p1 + w2 * p2 + w3 * p3
        return fused_feat

class ExposureAwareGate(nn.Module):
    def __init__(self, feat_channels=64, tau_dark=0.35, tau_bright=0.75,
                 sharpness=10.0, gate_min=0.10):
        super(ExposureAwareGate, self).__init__()

        self.tau_dark = tau_dark
        self.tau_bright = tau_bright
        self.sharpness = sharpness
        self.gate_min = gate_min

        self.gate_net = nn.Sequential(
            Conv(feat_channels + 1, feat_channels // 2, 3),
            nn.ReLU(inplace=True),
            Conv(feat_channels // 2, feat_channels // 4, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def _norm_y(self, y):
        # 兼容你现在的 0~255 数据，也兼容以后改成 0~1
        if y.detach().max() > 2.0:
            return y / 255.0
        return y

    def forward(self, y, illum_feat):
        y_norm = self._norm_y(y).clamp(0.0, 1.0)

        # 暗区 mask：Y 越小越接近 1
        dark_mask = torch.sigmoid(self.sharpness * (self.tau_dark - y_norm))

        # 高亮 mask：Y 越大越接近 1
        bright_mask = torch.sigmoid(self.sharpness * (y_norm - self.tau_bright))

        # 正常曝光区域：不暗也不过亮
        normal_mask = torch.sigmoid(self.sharpness * (y_norm - self.tau_dark)) * \
                      torch.sigmoid(self.sharpness * (self.tau_bright - y_norm))

        # 手工曝光先验：暗区允许增强，高亮区抑制增强
        base_gate = dark_mask * (1.0 - bright_mask)

        # 学习型曝光 gate
        learn_gate = self.gate_net(torch.cat([illum_feat, y_norm], dim=1))

        # 二者融合
        gate = 0.5 * base_gate + 0.5 * learn_gate * (1.0 - bright_mask)

        # 不让 gate 完全为 0，避免正常区域完全不能修复颜色/噪声
        gate = self.gate_min + (1.0 - self.gate_min) * gate
        gate = gate.clamp(0.0, 1.0)

        aux = {
            "exposure_gate": gate,
            "dark_mask": dark_mask,
            "normal_mask": normal_mask,
            "bright_mask": bright_mask,
            "y_norm": y_norm
        }

        return gate, aux

class IEN(nn.Module):
    def __init__(self, in_channel=3, f_channel=48, w_channel=48):
        super(IEN, self).__init__()

        self.layer0 = nn.Sequential(
                      Conv(in_channel, f_channel, 3),
                      Conv(f_channel, f_channel, 3)
        )

        self.para = torch.nn.Parameter(torch.ones(w_channel, 1, 1))

        self.guide1 = PGB(f_channel, f_channel, w_channel)
        self.layer1 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide2 = PGB(f_channel, f_channel, w_channel)
        self.layer2 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide3 = PGB(f_channel, f_channel, w_channel)
        self.layer3 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide4 = PGB(f_channel, f_channel, w_channel)
        self.layer4 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide5 = PGB(f_channel, f_channel, w_channel)
        self.layer5 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.guide6 = PGB(f_channel, f_channel, w_channel)
        self.layer6 = RCAB(in_feat=f_channel, out_feat=f_channel, kernel_size=3, reduction=8, n_blocks=3, bias=False,
                           act=nn.ReLU(True))
        self.out = Conv(f_channel, 3, 3)

    def forward(self, img, illu, rest, exposure_gate=None):
        x = self.layer0(img)
        res_illu = illu - self.para * rest
    
        x = self.guide1(x, res_illu)
        x = self.layer1(x)
    
        x = self.guide2(x, rest)
        x = self.layer2(x)
    
        x = self.guide3(x, res_illu)
        x = self.layer3(x)
    
        x = self.guide4(x, rest)
        x = self.layer4(x)
    
        x = self.guide5(x, res_illu)
        x = self.layer5(x)
    
        x = self.guide6(x, rest)
        x = self.layer6(x)
    
        pred = self.out(x)
    
      
        if exposure_gate is not None:
            gate = exposure_gate.repeat(1, img.shape[1], 1, 1)
            pred = img + gate * (pred - img)
    
        return pred


class Illum_YCRCB_Denoise_IN(nn.Module):
    def __init__(self):
        super(Illum_YCRCB_Denoise_IN, self).__init__()
       

        self.illu = GLAB_PEN(
            channel=1,
            n_feats=64,
            dim=32,
            config=[2, 2, 2, 2, 2],
            drop_path_rate=0.0,
            input_resolution=128
        )
        
        self.denoise = PEN(channel=2, n_feats=64)
        
        self.ms_fuse = MultiStrengthPriorFusion(
            feat_channels=64,
            delta=0.3,
            gamma=2.0
        )
        
        self.exp_gate = ExposureAwareGate(
            feat_channels=64,
            tau_dark=0.35,
            tau_bright=0.75,
            sharpness=10.0,
            gate_min=0.10
        )
        
        self.FuN = IEN(3, 64, 64)



    def forward(self, x, return_aux=False):
        y = x[:, 0, :, :].unsqueeze(1)
        uv = x[:, 1:3, :, :]
    
        out_y, out_y_feats = self.illu(y)
        out_uv, out_uv_feats = self.denoise(uv)
    
        fused_uv_feats = self.ms_fuse(out_uv_feats, out_y_feats)
    
        exposure_gate, aux = self.exp_gate(y, out_y_feats)
    
        out = self.FuN(
            x,
            out_y_feats,
            fused_uv_feats,
            exposure_gate=exposure_gate
        )
    
        if return_aux:
            return out_y, out_uv, out, aux
    
        return out_y, out_uv, out

        
        




