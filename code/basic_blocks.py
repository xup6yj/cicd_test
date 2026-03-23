# -*- coding: utf-8 -*-
# from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from cc_attention import *

kx = ky = kz = 3
px = py = pz = 1
epsilon = 1e-7

norm_mode = 'bn'
num_groups = None
ch_per_group = 16


class Norm(nn.Module):
    def __init__(self, channel):
        super(Norm, self).__init__()
        if norm_mode == 'bn':
            self.norm = nn.BatchNorm3d(channel)
        elif norm_mode == 'gn':
            if num_groups is not None and ch_per_group is not None:
                raise ValueError('Can only choose one, num_groups or ch_per_group')
            if num_groups is not None:
                assert channel%num_groups == 0, 'channel%%num_groups != 0'
                self.norm = nn.GroupNorm(num_groups, channel)
            elif ch_per_group is not None:
                assert channel%ch_per_group == 0, 'channel%%ch_per_group != 0'
                self.norm = nn.GroupNorm(channel//ch_per_group, channel)
            else:
                raise ValueError('Please choose one, num_groups or ch_per_group')
        else:
            raise ValueError('Unknown normalization mode')

    def forward(self, x):
        return self.norm(x)


class Upsample(nn.Module):
    def __init__(self, scale=2, z=False):
        super(Upsample, self).__init__()
        z_scale = scale if z else 1
        self.scale = (scale, scale, z_scale)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2):
        super(DeConvBlock, self).__init__()
        k = (scale, scale, 1)
        s = (scale, scale, 1)
        self.deconv = nn.ConvTranspose3d(in_channel, out_channel, kernel_size=k, stride=s, padding=0)
    def forward(self, x):
        x = self.deconv(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2, deconv=False):
        super(UpConvBlock, self).__init__()
        if deconv:
            self.up = nn.Sequential(
                DeConvBlock(in_channel, out_channel, scale))
        else:
            layers = [Upsample(scale)]
            if in_channel != out_channel:
                layers.append(nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0))
            self.up = nn.Sequential(*layers)

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, n=2):
        super(ConvBlock, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        layers = []
        for _ in range(n):
            layers += [
                nn.Conv3d(in_channel, out_channel, kernel_size=k, stride=1, padding=p, bias=bias),
                Norm(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, channel, bias=True, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=k, stride=1, padding=p, bias=bias),
            Norm(channel),
			nn.ReLU(inplace=True))

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1


# Recurrent CNN
class RCNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, num_rcnn=2, t=2):
        super(RCNNBlock, self).__init__()
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        for _ in range(num_rcnn):
            layers.append(RecurrentBlock(out_channel, bias=bias, t=t))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.nn(out)
        return out


# Recurrent Residual CNN
class RRCNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, num_rcnn=2, t=2):
        super(RRCNNBlock, self).__init__()
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        for _ in range(num_rcnn):
            layers.append(RecurrentBlock(out_channel, bias=bias, t=t))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.nn(x1)
        return x1 + x2


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, bias=True):
        super(AttentionBlock, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm3d(f_int))
        self.w_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm3d(f_int))
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm3d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x * psi


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class BAMChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(BAMChannelGate, self).__init__()
        gate_c = [Flatten()]
        gate_channels = [gate_channel]
        gate_channels += [gate_channel//reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels)-2):
            gate_c += [
                nn.Linear(gate_channels[i], gate_channels[i+1]),
                nn.BatchNorm1d(gate_channels[i+1]),
                nn.ReLU(inplace=True)
            ]
        gate_c.append(nn.Linear(gate_channels[-2], gate_channels[-1]))
        self.gate_c = nn.Sequential(*gate_c)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        att = self.gate_c(avg_pool)
        att = att.reshape(att.shape[0], att.shape[1], 1, 1, 1).expand_as(x)
        return att


class BAMSpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4, dim='3d'):
        k = (kx, ky, kz if dim == '3d' else 1)
        p = (dilation_val, dilation_val, dilation_val if dim == '3d' else 0)
        super(BAMSpatialGate, self).__init__()
        gate_s = [
            nn.Conv3d(gate_channel, gate_channel//reduction_ratio, kernel_size=1),
            nn.BatchNorm3d(gate_channel//reduction_ratio),
            nn.ReLU(inplace=True)
        ]
        for _ in range(dilation_conv_num):
            gate_s += [
                nn.Conv3d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=k, padding=p, dilation=dilation_val),
                nn.BatchNorm3d(gate_channel//reduction_ratio),
                nn.ReLU(inplace=True)
            ]
        gate_s.append(nn.Conv3d(gate_channel//reduction_ratio, 1, kernel_size=1))
        self.gate_s = nn.Sequential(*gate_s)

    def forward(self, x):
        att = self.gate_s(x).expand_as(x)
        return att


class BAM(nn.Module):
    def __init__(self, gate_channel, dim='3d'):
        super(BAM, self).__init__()
        self.channel_att = BAMChannelGate(gate_channel)
        self.spatial_att = BAMSpatialGate(gate_channel, dim=dim)

    def forward(self, x):
        att_c = self.channel_att(x)
        att_s = self.spatial_att(x)
        scale = 1 + torch.sigmoid(att_c+att_s)
        return x * scale


class BAMAPBlock(nn.Module):
    def __init__(self, channel, dim='3d'):
        super(BAMAPBlock, self).__init__()
        k = (kx, ky, kz if dim == '3d' else 1)
        p = (px, py, pz if dim == '3d' else 0)
        self.conv = nn.Conv3d(channel*2, channel, kernel_size=k, stride=1, padding=p)
        self.bam = BAM(channel, dim=dim)

    def forward(self, x1, x2):
        out = self.conv(torch.cat((x1, x2), dim=1))
        out = self.bam(out)
        return out


class CBAMChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=10):
        super(CBAMChannelGate, self).__init__()
        self.gate_channel = gate_channel
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channel, gate_channel//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channel//reduction_ratio, gate_channel))

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        max_pool = F.adaptive_max_pool3d(x, 1)
        channel_att = self.mlp(avg_pool) + self.mlp(max_pool)
        scale = torch.sigmoid(channel_att).reshape(channel_att.shape[0], channel_att.shape[1], 1, 1, 1).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        channel_max = x.max(dim=1)[0].unsqueeze(1)
        channel_mean = x.mean(dim=1).unsqueeze(1)
        return torch.cat((channel_max, channel_mean), dim=1)


class CBAMSpatialGate(nn.Module):
    def __init__(self):
        super(CBAMSpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, stride=1, padding=(7-1)//2, bias=False),
            nn.BatchNorm3d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = CBAMChannelGate(gate_channel, reduction_ratio)
        self.spatial_gate = CBAMSpatialGate()

    def forward(self, x):
        x_out = self.channel_gate(x)
        x_out = self.spatial_gate(x_out)
        return x_out


class CBAMAPBlock(nn.Module):
    def __init__(self, channel, level, reduction_ratio, fixed_kernel=True):
        super(CBAMAPBlock, self).__init__()
        kernel = 3 if fixed_kernel else 9-2*level
        self.conv = nn.Conv3d(channel*2, channel, kernel_size=kernel, stride=1, padding=(kernel-1)//2)
        self.cbam = CBAM(channel, reduction_ratio)

    def forward(self, x1, x2):
        out = self.conv(torch.cat((x1, x2), dim=1))
        out = self.cbam(out)
        return out


# Classification Guided Module
class CGM(nn.Module):
    def __init__(self, in_channel):
        super(CGM, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channel, 1, kernel_size=(1, 1, 1)),
            nn.AdaptiveAvgPool3d((50, 50, 1)))
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(2500, 2))

    def forward(self, x):
        out = self.net(x)
        out = self.classifier(out)
        return out


class ResConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, n=2, dim='3d'):
        super(ResConvBlock, self).__init__()
        k = (kx, ky, kz if dim == '3d' else 1)
        p = (px, py, pz if dim == '3d' else 0)
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        for _ in range(n):
            layers += [
                nn.Conv3d(out_channel, out_channel, kernel_size=k, stride=1, padding=p, bias=bias),
                Norm(out_channel),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv(x1)
        return x1 + x2



class AFFModule(nn.Module):
    def __init__(self, channel):
        super(AFFModule, self).__init__()
        self.se_block = SEBlock(channel*2)
        self.conv_1x1 = nn.Conv3d(channel*2, channel, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.activation = nn.Sigmoid()

    def forward(self, x, x3d):
        concat = torch.cat((x, x3d), dim=1)
        att = self.conv_1x1(self.se_block(concat))
        global_context = self.avg_pool(att)
        weights = self.activation(global_context)
        return x + x3d*weights


class DAF(nn.Module):
    def __init__(self, channel):
        super(DAF, self).__init__()
        self.se_block = SEBlock(channel*2)
        self.conv_1x1 = nn.Conv3d(channel*2, channel, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.activation = nn.Sigmoid()
        self.se_block_2d = SEBlock(channel)

    def forward(self, x, x3d):
        concat = torch.cat((x, x3d), dim=1)
        att = self.conv_1x1(self.se_block(concat))
        global_context = self.avg_pool(att)
        weights = self.activation(global_context)
        return self.se_block_2d(x) + x3d*weights


class DTBlock(nn.Module):
    def __init__(self, channel):
        super(DTBlock, self).__init__()
        self.att2d = SEBlock(channel)
        self.att3d = SEBlock(channel)

    def forward(self, x, x3d):
        return self.att2d(x) + self.att3d(x3d)


class ZattDTBlock(nn.Module):
    def __init__(self, channel, alpha=1):
        super(ZattDTBlock, self).__init__()
        self.att2d = SEBlock(channel)
        self.att3d = SEBlock(channel)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, None))
        self.query_conv = nn.Conv3d(channel, channel//8, kernel_size=1)
        self.key_conv = nn.Conv3d(channel, channel//8, kernel_size=1)
        self.value_conv = nn.Conv3d(channel, channel, kernel_size=1)
        self.alpha = alpha

    def forward(self, x, x3d):
        b, c, h, w, d = x.shape
        fusion = x + x3d
        q = self.avg_pool(self.query_conv(fusion)).view(b, c//8, d).permute(0, 2, 1)
        k = self.avg_pool(self.key_conv(fusion)).view(b, c//8, d)
        score = torch.bmm(q, k)
        attn = F.softmax(score.view(-1, score.shape[-1]), dim=1).view(score.shape[0], -1, score.shape[-1])
        attn = torch.permute(attn, (0, 2, 1))
        context = torch.bmm(self.value_conv(fusion).view(b, c*h*w, d), attn).view(fusion.shape)
        return self.alpha*context + fusion


class RConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, n=2):
        super(RConvBlock, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv_1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        layers = []
        layers += [
            nn.Conv3d(out_channel, out_channel, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias),
            Norm(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channel, out_channel, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0), bias=bias),
            Norm(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=bias),
            Norm(out_channel),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv(x1)
        return x1 + x2


class Bottleneck(nn.Module):
    def __init__(self, in_channel, growth_rate, bias=False):
        super(Bottleneck, self).__init__()
        out_channel = 4 * growth_rate
        self.bn1 = nn.BatchNorm3d(in_channel)
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv2 = nn.Conv3d(out_channel, growth_rate, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), dim=1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channel)
        self.conv1 = nn.Conv3d(in_channel, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class DCBlock(nn.Module):
    def __init__(self, channel, growth_rate, num_blocks, bottleneck):
        super(DCBlock, self).__init__()
        layers = []
        for _ in range(int(num_blocks)):
            if bottleneck:
                layers.append(Bottleneck(channel, growth_rate))
            else:
                layers.append(SingleLayer(channel, growth_rate))
            channel += growth_rate
        self.dense = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.dense(x)


class MFSEBlock(nn.Module):
    # Mixed Fusion Squeeze-and-Excitation
    def __init__(self, channel):
        super(MFSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.maxout = nn.AdaptiveMaxPool3d(1)
        self.fc_avg_pool = nn.Sequential(
            nn.Linear(channel, channel//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//2, channel, bias=False),
            nn.Sigmoid())
        self.fc_maxout = nn.Sequential(
            nn.Linear(channel, channel//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//2, channel, bias=False),
            nn.Sigmoid())
        self.conv_1x1 = nn.Conv3d(3*channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c = x.shape[:2]
        y1 = self.avg_pool(x).reshape(b, c)
        y1 = self.fc_avg_pool(y1).reshape(b, c, 1, 1, 1)
        y2 = self.maxout(x).reshape(b, c)
        y2 = self.fc_maxout(y2).reshape(b, c, 1, 1, 1)
        y = torch.cat(((y1+y2), (y1*y2), torch.maximum(y1, y2)), dim=1)
        y = self.conv_1x1(y)
        return x * y


class MAGM(nn.Module):
    # Multi-level Attention Gate Module
    def __init__(self, in_channel, level):
        super(MAGM, self).__init__()
        self.level = level
        self.resample = []
        for i in range(1, level):
            channel = in_channel * (2**i)
            self.resample.append(nn.Sequential(
                UpConvBlock(channel, in_channel, 2**i, deconv=True)))
        self.resample = nn.ModuleList(self.resample)

        self.conv_1x1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, *x):
        assert len(x) == self.level, f'The number of input tensors should be equal to {self.level}.'
        tensors = [x[0]]
        for i in range(1, len(x)):
            tensors.append(self.resample[i-1](x[i]))
        attention_coef = torch.stack(tensors, dim=0).sum(dim=0)
        attention_coef = self.conv_1x1(attention_coef)
        return x[0] * attention_coef


class CDB(nn.Module):
    def __init__(self, channel, level, cur_level, pyramid=False):
        super(CDB, self).__init__()
        self.pyramid = pyramid
        self.pyramid_level = level - cur_level - 1
        total_channel = (level-cur_level)*channel if pyramid else channel
        self.conv_sub_1x1 = nn.Conv3d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.mfse0 = MFSEBlock(channel)
        if pyramid:
            for i in range(1, level-cur_level):
                ks = 2**i + 1
                p = (ks-1) // 2
                setattr(self, f'conv{i}', nn.Conv3d(channel, channel, kernel_size=(ks, ks, kz), stride=1, padding=(p, p, pz)))
                setattr(self, f'mfse{i}', MFSEBlock(channel))
        self.conv_apb_1x1 = nn.Conv3d(total_channel, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        x = self.conv_sub_1x1(x1-x2)
        if self.pyramid:
            branches = [self.mfse0(x)]
            for i in range(1, self.pyramid_level+1):
                branches.append(getattr(self, f'mfse{i}')(getattr(self, f'conv{i}')(x)))
            x = torch.cat(branches, dim=1)
            x = self.conv_apb_1x1(x)
        else:
            x = self.conv_apb_1x1(self.mfse0(x))
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_avg_pool = nn.Sequential(
            nn.Linear(in_channel, in_channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel//reduction, in_channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        avg_pool = self.avg_pool(x).reshape(b, c)
        y = self.fc_avg_pool(avg_pool).reshape(b, c, 1, 1, 1)
        return x * y


class SEPP(nn.Module):
    def __init__(self, in_channel, dilation):
        super(SEPP, self).__init__()
        k = (kx, ky, 1)
        p = (px, py, 1)
        self.atrous0 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=k, dilation=dilation, stride=1, padding=p),
            ConvBlock(in_channel, in_channel, n=1))
        self.atrous1 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=k, dilation=dilation*2, stride=1, padding=p),
            ConvBlock(in_channel, in_channel, n=1))
        self.atrous2 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=k, dilation=dilation*4, stride=1, padding=p),
            ConvBlock(in_channel, in_channel, n=1))

        self.se0 = SEBlock(in_channel)
        self.se1 = SEBlock(in_channel)
        self.se2 = SEBlock(in_channel)
            
        self.conv1x1_0 = ConvBlock(in_channel, in_channel, n=1)
        self.conv1x1_1 = ConvBlock(in_channel, in_channel, n=1)
        self.conv1x1_2 = ConvBlock(in_channel, in_channel, n=1)
        self.con1x1_final = nn.Conv3d(in_channel*3, in_channel, kernel_size=1)
    
    def forward(self, x):
        b, c = x.shape[:2]

        atrous0 = self.atrous0(x)
        atrous1 = self.atrous1(x)
        atrous2 = self.atrous2(x)

        dilation0 = x * self.se0(atrous0).reshape(b, c, 1, 1, 1)
        dilation1 = x * self.se1(atrous1).reshape(b, c, 1, 1, 1)
        dilation2 = x * self.se2(atrous2).reshape(b, c, 1, 1, 1)

        branch0 = self.conv1x1_0(dilation0)
        branch1 = self.conv1x1_1(dilation1)
        branch2 = self.conv1x1_2(dilation2)

        out = torch.cat((branch0, branch1, branch2), dim=1)

        return self.con1x1_final(out)


class SEPPAPBlock(nn.Module):
    def __init__(self, channel):
        super(SEPPAPBlock, self).__init__()
        self.sepp = SEPP(channel, dilation=1)
        self.conv = nn.Conv3d(channel*2, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        out = self.conv(torch.cat((x1, x2), dim=1))
        out = self.sepp(out)
        return out


# https://github.com/xvjiarui/GCNet/blob/a9fcc88c4bd3a0b89de3678b4629c9dfd190575f/mmdet/ops/gcb/context_block.py#L13
class GCBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add', )):
        super(GCBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        b, c, h, w, d = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(b, c, h*w*d) # [N, C, H*W*D]
            input_x = input_x.unsqueeze(1) # [N, 1, C, H*W*D]
            context_mask = self.conv_mask(x) # [N, 1, H, W, D]
            context_mask = context_mask.view(b, 1, h*w*d) # [N, 1, H*W*D]
            context_mask = self.softmax(context_mask) # [N, 1, H*W*D]
            context_mask = context_mask.unsqueeze(-1) # [N, 1, H*W*D, 1]
            context = torch.matmul(input_x, context_mask) # [N, 1, C, 1]
            context = context.unsqueeze(-1).view(b, c, 1, 1, 1) # [N, C, 1, 1, 1]
        else:
            context = self.avg_pool(x) # [N, C, 1, 1, 1]
        return context

    def forward(self, x):
        context = self.spatial_pool(x) # [N, C, 1, 1, 1]
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context)) # [N, C, 1, 1, 1]
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context) # [N, C, 1, 1, 1]
            out = out + channel_add_term
        return out


class ASPP(nn.Module):
    def __init__(self, channel, rate=1):
        super(ASPP, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.atrous0 = nn.Conv3d(channel, channel, kernel_size=k, padding=p, dilation=rate)
        self.atrous1 = nn.Conv3d(channel, channel, kernel_size=k, padding=p, dilation=rate*2)
        self.atrous2 = nn.Conv3d(channel, channel, kernel_size=k, padding=p, dilation=rate*4)
        self.conv1x1 = nn.Conv3d(channel*4, channel, kernel_size=1)

    def forward(self, x):
        _, _, h, w, d = x.shape
        tmp = x
        x0 = self.atrous0(x)
        x0 = F.interpolate(x0, size=(h, w, d), mode='trilinear', align_corners=False)
        x1 = self.atrous0(x)
        x1 = F.interpolate(x1, size=(h, w, d), mode='trilinear', align_corners=False)
        x2 = self.atrous0(x)
        x2 = F.interpolate(x2, size=(h, w, d), mode='trilinear', align_corners=False)
        concat = torch.cat((tmp, x0, x1, x2), dim=1)
        out = self.conv1x1(concat)
        return out


class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels//reduction, in_channels, 1),
            nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv3d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


from pywt import dwt2
class DWTBlock(nn.Module):
    def __init__(self, channels):
        super(DWTBlock, self).__init__()
        self.conv1x1 = nn.Conv3d(channels+4, channels, kernel_size=1)
        self.attention = SCSEBlock(channels)

    def to_tensor(self, LL, LH, HL, HH):
        return torch.from_numpy(LL).cuda(), \
               torch.from_numpy(LH).cuda(), \
               torch.from_numpy(HL).cuda(), \
               torch.from_numpy(HH).cuda()

    def forward(self, x1, x2):
        LL, (LH, HL, HH) = dwt2(x2.cpu(), wavelet='haar', mode='symmetric', axes=(-3, -2))
        LL, LH, HL, HH = self.to_tensor(LL, LH, HL, HH)
        features = torch.cat((LL, LH, HL, HH, x1), dim=1)
        return self.attention(self.conv1x1(features)), LL


class DWTBlockV2(nn.Module):
    def __init__(self, channels):
        super(DWTBlockV2, self).__init__()
        self.conv0 = ConvBlock(4, channels)
        self.conv1 = nn.Conv3d(channels*2, channels, kernel_size=3, padding=1)
        self.attention = SCSEBlock(channels)

    def to_tensor(self, LL, LH, HL, HH):
        return torch.from_numpy(LL).cuda(), \
               torch.from_numpy(LH).cuda(), \
               torch.from_numpy(HL).cuda(), \
               torch.from_numpy(HH).cuda()

    def forward(self, x1, x2):
        LL, (LH, HL, HH) = dwt2(x2.cpu(), wavelet='haar', mode='symmetric', axes=(-3, -2))
        LL, LH, HL, HH = self.to_tensor(LL, LH, HL, HH)
        dwt_features = self.conv0(torch.cat((LL, LH, HL, HH), dim=1))
        features = self.conv1(torch.cat((dwt_features, x1), dim=1))
        return self.attention(features), LL


class DWTBlockv3(nn.Module):
    def __init__(self, channels):
        super(DWTBlockv3, self).__init__()
        self.bn = Norm(4)
        self.relu = nn.ReLU(inplace=True)
        self.attention = BAM(channels)

    def to_tensor(self, LL, LH, HL, HH):
        return torch.from_numpy(LL).cuda(), \
               torch.from_numpy(LH).cuda(), \
               torch.from_numpy(HL).cuda(), \
               torch.from_numpy(HH).cuda()

    def forward(self, x1, x2):
        LL, (LH, HL, HH) = dwt2(x2.cpu(), wavelet='haar', mode='symmetric', axes=(-3, -2))
        LL, LH, HL, HH = self.to_tensor(LL, LH, HL, HH)
        dwt_features = torch.cat((LL, LH, HL, HH), dim=1)
        dwt_features = self.relu(self.bn(dwt_features))
        features = torch.cat((dwt_features, x1), dim=1)
        return self.attention(features), LL


class DWTInitBlock(nn.Module):
    def __init__(self):
        super(DWTInitBlock, self).__init__()
        num_dwt_features = 4
        self.conv = nn.Conv3d(num_dwt_features, num_dwt_features*3, kernel_size=3, padding=1)
    
    def to_tensor(self, LL, LH, HL, HH):
        return torch.from_numpy(LL).cuda(), \
               torch.from_numpy(LH).cuda(), \
               torch.from_numpy(HL).cuda(), \
               torch.from_numpy(HH).cuda()

    def forward(self, x):
        LL, (LH, HL, HH) = dwt2(x.cpu(), wavelet='haar', mode='symmetric', axes=(-3, -2))
        LL, LH, HL, HH = self.to_tensor(LL, LH, HL, HH)
        dwt_features = torch.cat((LL, LH, HL, HH), dim=1)
        init = self.conv(dwt_features)
        return torch.cat((init, dwt_features), dim=1)


class APBlock(nn.Module):
    def __init__(self, channel):
        super(APBlock, self).__init__()
        self.conv = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bam = BAM(channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.bam(out)
        return out


class APBlockv2(nn.Module):
    def __init__(self, channel):
        super(APBlockv2, self).__init__()
        self.conv = nn.Conv3d(channel*5, channel, kernel_size=3, stride=1, padding=1)
        self.bam = BAM(channel)

    def forward(self, LL, LH, HL, HH, x):
        out = self.conv(torch.cat((LL, LH, HL, HH, x), dim=1))
        out = self.bam(out)
        return out


class APBlockv3(nn.Module):
    def __init__(self, channel):
        super(APBlockv3, self).__init__()
        self.bam = BAM(channel)

    def forward(self, x):
        out = self.bam(x)
        return out


# https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/xception.py
class DWSConv(nn.Module):
    '''
    Depthwise separable convolution layer.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 3 int
        Convolution window size.
    stride : int or tuple/list of 3 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 3 int, default 0
        Padding value for convolution layer.
    '''
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0):
        super(DWSConv, self).__init__()
        self.dw_conv = nn.Conv3d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channel,
            bias=False)
        self.pw_conv = nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            bias=False)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DWSConvBlock(nn.Module):
    '''
    Depthwise separable convolution block with batchnorm and ReLU pre-activation.
    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 3 int
        Convolution window size.
    stride : int or tuple/list of 3 int
        Strides of the convolution.
    padding : int or tuple/list of 3 int
        Padding value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    '''
    def __init__(self, in_channel, out_channel, stride=1, activate=True):
        super(DWSConvBlock, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.activate = activate
        # if self.activate:
        #     self.activ = nn.ReLU(inplace=False)
        self.conv1 = DWSConv(in_channel=in_channel, out_channel=out_channel, kernel_size=k, stride=stride, padding=p)
        self.bn = Norm(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # if self.activate:
        #     x = self.activ(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResDWSConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResDWSConvBlock, self).__init__()
        self.conv1x1 = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)    
        self.dws = DWSConvBlock(out_channel, out_channel)
        self.conv3x3 = nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=(2, 2, 1), padding=1)

    def forward(self, x, skip):
        x = self.conv1x1(x)
        x = self.dws(x)
        skip = self.conv3x3(skip)
        return x + skip


class PACBlock2d(nn.Module):
    def __init__(self, channel, ratio=6, bias=True):
        super(PACBlock2d, self).__init__()
        k = (kx, ky, kz)
        self.pac0 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=k,
                dilation=(ratio, ratio, 1),
                padding=(ratio, ratio, 1),
                bias=bias))
        self.pac1 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=k,
                dilation=(ratio*2, ratio*2, 1),
                padding=(ratio*2, ratio*2, 1),
                bias=bias))
        self.pac2 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=k,
                dilation=(ratio*3, ratio*3, 1),
                padding=(ratio*3, ratio*3, 1),
                bias=bias))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=1))
        self.conv1x1 = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=1)

    def forward(self, x):
        h, w, d = x.shape[2:]
        pac0 = self.pac0(x)
        pac1 = self.pac1(x)
        pac2 = self.pac2(x)
        gap = F.interpolate(self.gap(x), size=(h, w, d), mode='trilinear', align_corners=False)
        conv1x1 = self.conv1x1(x)
        out = torch.cat((pac0, pac1, pac2, gap, conv1x1), dim=1)
        return out


class PACBlock2dv2(nn.Module):
    def __init__(self, channel, ratio=6, bias=True):
        super(PACBlock2dv2, self).__init__()
        k = (kx, ky, kz)
        self.pac0 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=k,
                dilation=(ratio, ratio, ratio),
                padding=(ratio, ratio, ratio),
                bias=bias))
        self.pac1 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=k,
                dilation=(ratio, ratio, ratio),
                padding=(ratio, ratio, ratio),
                bias=bias))
        self.pac2 = nn.Sequential(
            nn.Conv3d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=k,
                dilation=(ratio, ratio, ratio),
                padding=(ratio, ratio, ratio),
                bias=bias))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=1))
        self.conv1x1 = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=1)

    def forward(self, x):
        h, w, d = x.shape[2:]
        pac0 = self.pac0(x)
        pac1 = self.pac1(pac0)
        pac2 = self.pac2(pac1)
        gap = F.interpolate(self.gap(x), size=(h, w, d), mode='trilinear', align_corners=False)
        conv1x1 = self.conv1x1(x)
        out = torch.cat((pac0, pac1, pac2, gap, conv1x1), dim=1)
        return out


class DAB(nn.Module):
    def __init__(self, channels):
        super(DAB, self).__init__()
        self.dws_conv = DWSConv(channels, channels//2, kernel_size=3, padding=1)
        self.conv0 = nn.Conv3d(channels//2, channels//2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv1 = nn.Conv3d(channels//2, channels//2, kernel_size=(1, 3, 1), padding=(0, 1, 0))
        self.conv2 = nn.Conv3d(channels//2, channels//2, kernel_size=(1, 1, 3), padding=(0, 0, 1))
        self.diconv0 = nn.Conv3d(channels//2, channels//2, kernel_size=(3, 1, 1), padding=(2, 0, 0), dilation=(2, 1, 1))
        self.diconv1 = nn.Conv3d(channels//2, channels//2, kernel_size=(1, 3, 1), padding=(0, 2, 0), dilation=(1, 2, 1))
        self.diconv2 = nn.Conv3d(channels//2, channels//2, kernel_size=(1, 1, 3), padding=(0, 0, 2), dilation=(1, 1, 2))

    def forward(self, x):
        x = self.dws_conv(x)
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        diconv0 = self.diconv0(x)
        diconv1 = self.diconv1(diconv0)
        diconv2 = self.diconv2(diconv1)
        return torch.cat((conv2, diconv2), dim=1)


class IRLB(nn.Module):
    def __init__(self, channels):
        super(IRLB, self).__init__()
        self.conv0 = nn.Conv3d(channels, channels, kernel_size=1)
        self.dws = DWSConv(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)

        self.relu0 = nn.ReLU6(inplace=True)
        self.relu1 = nn.ReLU6(inplace=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        relu0 = self.relu0(conv0)
        dws = self.dws(relu0)
        relu1 = self.relu1(dws)
        conv1 = self.conv1(relu1)


class ResPath(nn.Module):
    def __init__(self, channel):
        super(ResPath, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv0 = nn.Conv3d(channel, channel, kernel_size=k, padding=p)
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=k, padding=p)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=k, padding=p)
        self.conv3 = nn.Conv3d(channel, channel, kernel_size=k, padding=p)

        self.conv0_1x1 = nn.Conv3d(channel, channel, kernel_size=1)
        self.conv1_1x1 = nn.Conv3d(channel, channel, kernel_size=1)
        self.conv2_1x1 = nn.Conv3d(channel, channel, kernel_size=1)
        self.conv3_1x1 = nn.Conv3d(channel, channel, kernel_size=1)
    
    def forward(self, x):
        x0 = self.conv0(x)
        conv0_1x1 = self.conv0_1x1(x)
        x0 += conv0_1x1
        x1 = self.conv1(x0)
        conv1_1x1 = self.conv1_1x1(x0)
        x1 += conv1_1x1
        x2 = self.conv2(x1)
        conv2_1x1 = self.conv2_1x1(x1)
        x2 += conv2_1x1
        x3 = self.conv3(x2)
        conv3_1x1 = self.conv3_1x1(x2)
        x3 += conv3_1x1
        return x3


# masks is normal CSF
class AssistedExcitationLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(AssistedExcitationLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x, masks):
        avg = x.mean(dim=1).unsqueeze(dim=1)
        attention = avg * (1-masks) * self.alpha
        return x + attention.expand_as(x)


# masks is inv CSF
class AssistedExcitationLayerInv(nn.Module):
    def __init__(self, alpha=1.0):
        super(AssistedExcitationLayerInv, self).__init__()
        self.alpha = alpha

    def forward(self, x, masks):
        avg = x.mean(dim=1).unsqueeze(dim=1)
        attention = avg * masks * self.alpha
        return x + attention.expand_as(x)


class CCABlock(nn.Module):
    def __init__(self, channel):
        super(CCABlock, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv = nn.Conv3d(channel*2, channel, kernel_size=k, stride=1, padding=p)
        self.cc_attention = CrissCrossAttention3D2(channel)

    def forward(self, x1, x2, recurrence=2):
        diff = x1 - x2
        out = self.conv(torch.cat((diff, x2), dim=1))
        for _ in range(recurrence):
            out = self.cc_attention(out)
        return out


class MGAP(nn.Module):
    def __init__(self, channel, alpha=1.0):
        super(MGAP, self).__init__()
        k = (kx, ky, kz)
        p = (px, py, pz)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.conv = nn.Conv3d(channel*2, channel, kernel_size=k, stride=1, padding=p)
        self.bam = BAM(channel)

    def forward(self, x1, x2, masks):
        diff = x1 - x2
        out = self.conv(torch.cat((diff, x2), dim=1))
        out = self.bam(out)
        avg_diff = diff.mean(dim=1).unsqueeze(dim=1)
        avg_diff = avg_diff * masks * self.alpha
        return out + avg_diff.expand_as(out)


if __name__ == '__main__':
    import torchsummaryX
    device = 'cuda'
    # model = Fuchigami(1, 2, 32, cgm=True).to(device)
    model = CCABlock(16).to(device)
    x = torch.zeros(4, 16, 64, 64, 8).to(device)
    outputs = model(x, x)