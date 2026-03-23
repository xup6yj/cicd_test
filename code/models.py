# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *
from basic_blocks import *

bias = True
# directly flip the image
# x_flip = tF.vflip(bchwd2bcdhw(x))
# x_flip = bcdhw2bchwd(x_flip)


class Module(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, init_ch=32, deconv=False, cgm=False):
        super(Module, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.init_ch = init_ch
        self.deconv = deconv
        self.cgm = cgm
        self.compare_left_right = False


class UNetPP(Module):
    def __init__(self, **kwargs):
        super(UNetPP, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv00 = ConvBlock(in_ch, init_ch)
        self.conv10 = ConvBlock(init_ch, init_ch*2)
        self.conv20 = ConvBlock(init_ch*2, init_ch*4)
        self.conv30 = ConvBlock(init_ch*4, init_ch*8)
        self.conv40 = ConvBlock(init_ch*8, init_ch*16)

        self.up10 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up20 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up30 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up40 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.conv01 = ConvBlock(init_ch*2, init_ch, bias=bias)
        self.conv11 = ConvBlock(init_ch*4, init_ch*2, bias=bias)
        self.conv21 = ConvBlock(init_ch*8, init_ch*4, bias=bias)
        self.conv31 = ConvBlock(init_ch*16, init_ch*8, bias=bias)

        self.up11 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up21 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up31 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)

        self.conv02 = ConvBlock(init_ch*3, init_ch)
        self.conv12 = ConvBlock(init_ch*6, init_ch*2)
        self.conv22 = ConvBlock(init_ch*12, init_ch*4)

        self.up12 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up22 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)

        self.conv03 = ConvBlock(init_ch*4, init_ch)
        self.conv13 = ConvBlock(init_ch*8, init_ch*2)

        self.up13 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)

        self.conv04 = ConvBlock(init_ch*5, init_ch)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.maxpool(x00))
        x01 = self.conv01(torch.cat([self.up10(x10), x00], dim=1))

        x20 = self.conv20(self.maxpool(x10))
        x11 = self.conv11(torch.cat([self.up20(x20), x10], dim=1))
        x02 = self.conv02(torch.cat([self.up11(x11), x00, x01], dim=1))

        x30 = self.conv30(self.maxpool(x20))
        x21 = self.conv21(torch.cat([self.up30(x30), x20], dim=1))
        x12 = self.conv12(torch.cat([self.up21(x21), x10, x11], dim=1))
        x03 = self.conv03(torch.cat([self.up12(x12), x00, x01, x02], dim=1))

        x40 = self.conv40(self.maxpool(x30))
        x31 = self.conv31(torch.cat([self.up40(x40), x30], dim=1))
        x22 = self.conv22(torch.cat([self.up31(x31), x20, x21], dim=1))
        x13 = self.conv13(torch.cat([self.up22(x22), x10, x11, x12], dim=1))
        x04 = self.conv04(torch.cat([self.up13(x13), x00, x01, x02, x03], dim=1))

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x40)
        out = self.activation(self.conv_1x1(x04))
        return has_roi, out


class UNet3P(Module):
    def __init__(self, **kwargs):
        super(UNet3P, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        cat_ch = init_ch
        agg_ch = cat_ch * 5
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv0 = ConvBlock(in_ch, init_ch)
        self.conv1 = ConvBlock(init_ch, init_ch*2)
        self.conv2 = ConvBlock(init_ch*2, init_ch*4)
        self.conv3 = ConvBlock(init_ch*4, init_ch*8)
        self.conv4 = ConvBlock(init_ch*8, init_ch*16)

        k = (kx, ky, kz)
        p = (px, py, pz)
        self.conv0_cat_0 = nn.Conv3d(init_ch, cat_ch, kernel_size=k, stride=1, padding=p)
        self.conv0_cat_1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            nn.Conv3d(init_ch, cat_ch, kernel_size=k, stride=1, padding=p))
        self.conv0_cat_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(4, 4, 1), stride=(4, 4, 1)),
            nn.Conv3d(init_ch, cat_ch, kernel_size=k, stride=1, padding=p))
        self.conv0_cat_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(8, 8, 1), stride=(8, 8, 1)),
            nn.Conv3d(init_ch, cat_ch, kernel_size=k, stride=1, padding=p))

        self.conv1_cat_1 = nn.Conv3d(init_ch*2, cat_ch, kernel_size=k, stride=1, padding=p)
        self.conv1_cat_2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            nn.Conv3d(init_ch*2, cat_ch, kernel_size=k, stride=1, padding=p))
        self.conv1_cat_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(4, 4, 1), stride=(4, 4, 1)),
            nn.Conv3d(init_ch*2, cat_ch, kernel_size=k, stride=1, padding=p))

        self.conv2_cat_2 = nn.Conv3d(init_ch*4, cat_ch, kernel_size=k, stride=1, padding=p)
        self.conv2_cat_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            nn.Conv3d(init_ch*4, cat_ch, kernel_size=k, stride=1, padding=p))

        self.conv3_cat_3 = nn.Conv3d(init_ch*8, cat_ch, kernel_size=k, stride=1, padding=p)

        self.up1_cat_0 = nn.Sequential(
            Upsample(scale=2),
            nn.Conv3d(agg_ch, cat_ch, kernel_size=k, stride=1, padding=p))
        self.up2_cat_0 = nn.Sequential(
            Upsample(scale=4),
            nn.Conv3d(agg_ch, cat_ch, kernel_size=k, stride=1, padding=p))
        self.up3_cat_0 = nn.Sequential(
            Upsample(scale=8),
            nn.Conv3d(agg_ch, cat_ch, kernel_size=k, stride=1, padding=p))
        self.up4_cat_0 = nn.Sequential(
            Upsample(scale=16),
            nn.Conv3d(init_ch*16, cat_ch, kernel_size=k, stride=1, padding=p))

        self.up2_cat_1 = nn.Sequential(
            Upsample(scale=2),
            nn.Conv3d(agg_ch, cat_ch, kernel_size=k, stride=1, padding=p))
        self.up3_cat_1 = nn.Sequential(
            Upsample(scale=4),
            nn.Conv3d(agg_ch, cat_ch, kernel_size=k, stride=1, padding=p))
        self.up4_cat_1 = nn.Sequential(
            Upsample(scale=8),
            nn.Conv3d(init_ch*16, cat_ch, kernel_size=k, stride=1, padding=p))

        self.up3_cat_2 = nn.Sequential(
            Upsample(scale=2),
            nn.Conv3d(agg_ch, cat_ch, kernel_size=k, stride=1, padding=p, bias=True))
        self.up4_cat_2 = nn.Sequential(
            Upsample(scale=4),
            nn.Conv3d(init_ch*16, cat_ch, kernel_size=k, stride=1, padding=p, bias=True))

        self.up4_cat_3 = nn.Sequential(
            Upsample(scale=2),
            nn.Conv3d(init_ch*16, cat_ch, kernel_size=k, stride=1, padding=p, bias=True))

        self.up_conv0 = ConvBlock(agg_ch, agg_ch, n=1)
        self.up_conv1 = ConvBlock(agg_ch, agg_ch, n=1)
        self.up_conv2 = ConvBlock(agg_ch, agg_ch, n=1)
        self.up_conv3 = ConvBlock(agg_ch, agg_ch, n=1)

        self.conv_out = nn.Conv3d(agg_ch, out_ch, kernel_size=k, stride=1, padding=p)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        d3 = torch.cat((
            self.conv0_cat_3(x0),
            self.conv1_cat_3(x1),
            self.conv2_cat_3(x2),
            self.conv3_cat_3(x3),
            self.up4_cat_3(x4)),
            dim=1)
        d3 = self.up_conv3(d3)

        d2 = torch.cat((
            self.conv0_cat_2(x0),
            self.conv1_cat_2(x1),
            self.conv2_cat_2(x2),
            self.up3_cat_2(d3),
            self.up4_cat_2(x4)),
            dim=1)
        d2 = self.up_conv2(d2)

        d1 = torch.cat((
            self.conv0_cat_1(x0),
            self.conv1_cat_1(x1),
            self.up2_cat_1(d2),
            self.up3_cat_1(d3),
            self.up4_cat_1(x4)),
            dim=1)
        d1 = self.up_conv1(d1)

        d0 = torch.cat((
            self.conv0_cat_0(x0),
            self.up1_cat_0(d1),
            self.up2_cat_0(d2),
            self.up3_cat_0(d3),
            self.up4_cat_0(x4)),
            dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_out(d0))
        return has_roi, out


class BaseUNet(Module):
    def __init__(self, **kwargs):
        super(BaseUNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv0 = ConvBlock(in_ch, init_ch)
        self.conv1 = ConvBlock(init_ch, init_ch*2)
        self.conv2 = ConvBlock(init_ch*2, init_ch*4)
        self.conv3 = ConvBlock(init_ch*4, init_ch*8)
        self.conv4 = ConvBlock(init_ch*8, init_ch*16)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)


class UNet(BaseUNet):
    def __init__(self, **kwargs):
        super(UNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        d3 = torch.cat((x3, self.up4(x4)), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((x2, self.up3(d3)), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((x1, self.up2(d2)), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((x0, self.up1(d1)), dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class Fuchigami(Module):
    def __init__(self, **kwargs):
        super(Fuchigami, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv0 = ConvBlock(in_ch, init_ch)
        self.conv1 = ConvBlock(init_ch, init_ch*2)
        self.conv2 = ConvBlock(init_ch*2, init_ch*4)
        self.conv3 = ConvBlock(init_ch*8, init_ch*8)
        self.conv4 = ConvBlock(init_ch*8, init_ch*16)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        mp_x2 = self.maxpool(x2)
        mp_x2_flip = tF.vflip(bchwd2bcdhw(mp_x2))
        mp_x2_flip = bcdhw2bchwd(mp_x2_flip)
        merge = torch.cat((mp_x2, mp_x2_flip), dim=1)
        x3 = self.conv3(merge)
        x4 = self.conv4(self.maxpool(x3))

        d3 = torch.cat((x3, self.up4(x4)), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((x2, self.up3(d3)), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((x1, self.up2(d2)), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((x0, self.up1(d1)), dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class AttentionUNet(BaseUNet):
    def __init__(self, **kwargs):
        super(AttentionUNet, self).__init__(**kwargs)
        init_ch = self.init_ch
        self.att0 = AttentionBlock(init_ch, init_ch, init_ch//2)
        self.att1 = AttentionBlock(init_ch*2, init_ch*2, init_ch)
        self.att2 = AttentionBlock(init_ch*4, init_ch*4, init_ch*2)
        self.att3 = AttentionBlock(init_ch*8, init_ch*8, init_ch*4)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        up4 = self.up4(x4)
        x3 = self.att3(g=up4, x=x3)
        d3 = torch.cat((x3, up4), dim=1)
        d3 = self.up_conv3(d3)

        up3 = self.up3(d3)
        x2 = self.att2(g=up3, x=x2)
        d2 = torch.cat((x2, up3), dim=1)
        d2 = self.up_conv2(d2)

        up2 = self.up2(d2)
        x1 = self.att1(g=up2, x=x1)
        d1 = torch.cat((x1, up2), dim=1)
        d1 = self.up_conv1(d1)

        up1 = self.up1(d1)
        x0 = self.att0(g=up1, x=x0)
        d0 = torch.cat((x0, up1), dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class BaseRUNet(Module):
    def __init__(self, num_rcnn=2, t=2, **kwargs):
        super(BaseRUNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.rcnn0 = RCNNBlock(in_ch, init_ch, num_rcnn=num_rcnn, t=t)
        self.rcnn1 = RCNNBlock(init_ch, init_ch*2, num_rcnn=num_rcnn, t=t)
        self.rcnn2 = RCNNBlock(init_ch*2, init_ch*4, num_rcnn=num_rcnn, t=t)
        self.rcnn3 = RCNNBlock(init_ch*4, init_ch*8, num_rcnn=num_rcnn, t=t)
        self.rcnn4 = RCNNBlock(init_ch*8, init_ch*16, num_rcnn=num_rcnn, t=t)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_rcnn0 = RCNNBlock(init_ch*2, init_ch, num_rcnn=num_rcnn, t=t)
        self.up_rcnn1 = RCNNBlock(init_ch*4, init_ch*2, num_rcnn=num_rcnn, t=t)
        self.up_rcnn2 = RCNNBlock(init_ch*8, init_ch*4, num_rcnn=num_rcnn, t=t)
        self.up_rcnn3 = RCNNBlock(init_ch*16, init_ch*8, num_rcnn=num_rcnn, t=t)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)


class RUNet(BaseRUNet):
    def __init__(self, **kwargs):
        super(RUNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.rcnn0(x)
        x1 = self.rcnn1(self.maxpool(x0))
        x2 = self.rcnn2(self.maxpool(x1))
        x3 = self.rcnn3(self.maxpool(x2))
        x4 = self.rcnn4(self.maxpool(x3))

        d3 = torch.cat((x3, self.up4(x4)), dim=1)
        d3 = self.up_rcnn3(d3)
        d2 = torch.cat((x2, self.up3(d3)), dim=1)
        d2 = self.up_rcnn2(d2)
        d1 = torch.cat((x1, self.up2(d2)), dim=1)
        d1 = self.up_rcnn1(d1)
        d0 = torch.cat((x0, self.up1(d1)), dim=1)
        d0 = self.up_rcnn0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class BaseR2UNet(Module):
    def __init__(self, num_rcnn=2, t=2, **kwargs):
        super(BaseR2UNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.rrcnn0 = RRCNNBlock(in_ch, init_ch, num_rcnn=num_rcnn, t=t)
        self.rrcnn1 = RRCNNBlock(init_ch, init_ch*2, num_rcnn=num_rcnn, t=t)
        self.rrcnn2 = RRCNNBlock(init_ch*2, init_ch*4, num_rcnn=num_rcnn, t=t)
        self.rrcnn3 = RRCNNBlock(init_ch*4, init_ch*8, num_rcnn=num_rcnn, t=t)
        self.rrcnn4 = RRCNNBlock(init_ch*8, init_ch*16, num_rcnn=num_rcnn, t=t)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_rrcnn0 = RRCNNBlock(init_ch*2, init_ch, num_rcnn=num_rcnn, t=t)
        self.up_rrcnn1 = RRCNNBlock(init_ch*4, init_ch*2, num_rcnn=num_rcnn, t=t)
        self.up_rrcnn2 = RRCNNBlock(init_ch*8, init_ch*4, num_rcnn=num_rcnn, t=t)
        self.up_rrcnn3 = RRCNNBlock(init_ch*16, init_ch*8, num_rcnn=num_rcnn, t=t)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)


class R2UNet(BaseR2UNet):
    def __init__(self, **kwargs):
        super(R2UNet, self).__init__(**kwargs)

    def forward(self, x):
        x0 = self.rrcnn0(x)
        x1 = self.rrcnn1(self.maxpool(x0))
        x2 = self.rrcnn2(self.maxpool(x1))
        x3 = self.rrcnn3(self.maxpool(x2))
        x4 = self.rrcnn4(self.maxpool(x3))

        d3 = torch.cat((self.up4(x4), x3), dim=1)
        d3 = self.up_rrcnn3(d3)
        d2 = torch.cat((self.up3(d3), x2), dim=1)
        d2 = self.up_rrcnn2(d2)
        d1 = torch.cat((self.up2(d2), x1), dim=1)
        d1 = self.up_rrcnn1(d1)
        d0 = torch.cat((self.up1(d1), x0), dim=1)
        d0 = self.up_rrcnn0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class ResUNet(Module):
    def __init__(self, **kwargs):
        super(ResUNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv0 = ResConvBlock(in_ch, init_ch, dim='2d')
        self.conv1 = ResConvBlock(init_ch, init_ch*2, dim='2d')
        self.conv2 = ResConvBlock(init_ch*2, init_ch*4, dim='2d')
        self.conv3 = ResConvBlock(init_ch*4, init_ch*8, dim='2d')
        self.conv4 = ResConvBlock(init_ch*8, init_ch*16, dim='2d')

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ResConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ResConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ResConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ResConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
            # self.activation = nn.Identity()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))

        d3 = torch.cat((self.up4(x4), x3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((self.up3(d3), x2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((self.up2(d2), x1), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((self.up1(d1), x0), dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class APUNet(Module):
    def __init__(self, **kwargs):
        super(APUNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv0 = ResConvBlock(in_ch, init_ch)
        self.conv1 = ResConvBlock(init_ch, init_ch*2)
        self.conv2 = ResConvBlock(init_ch*2, init_ch*4)
        self.conv3 = ResConvBlock(init_ch*4, init_ch*8)
        self.conv4 = ResConvBlock(init_ch*8, init_ch*16)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ResConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ResConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ResConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ResConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        self.compare_left_right = True
        self.diff0 = BAMAPBlock(init_ch)
        self.diff1 = BAMAPBlock(init_ch*2)
        self.diff2 = BAMAPBlock(init_ch*4)
        self.diff3 = BAMAPBlock(init_ch*8)

    def forward(self, x, x_flip):
        x0 = self.conv0(x)
        x0_flip = self.conv0(x_flip)
        x1 = self.conv1(self.maxpool(x0))
        x1_flip = self.conv1(self.maxpool(x0_flip))
        x2 = self.conv2(self.maxpool(x1))
        x2_flip = self.conv2(self.maxpool(x1_flip))
        x3 = self.conv3(self.maxpool(x2))
        x3_flip = self.conv3(self.maxpool(x2_flip))
        
        x4 = self.conv4(self.maxpool(x3))

        diff0 = self.diff0(x0_flip, x0)
        diff1 = self.diff1(x1_flip, x1)
        diff2 = self.diff2(x2_flip, x2)
        diff3 = self.diff3(x3_flip, x3)

        d3 = torch.cat((self.up4(x4), diff3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((self.up3(d3), diff2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((self.up2(d2), diff1), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((self.up1(d1), diff0), dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class CSFAPUNet(Module):
    def __init__(self, **kwargs):
        super(CSFAPUNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv0 = ResConvBlock(in_ch, init_ch)
        self.conv1 = ResConvBlock(init_ch, init_ch*2)
        self.conv2 = ResConvBlock(init_ch*2, init_ch*4)
        self.conv3 = ResConvBlock(init_ch*4, init_ch*8)
        self.conv4 = ResConvBlock(init_ch*8, init_ch*16)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ResConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ResConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ResConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ResConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        self.compare_left_right = True
        self.diff0 = BAMAPBlock(init_ch)
        self.diff1 = BAMAPBlock(init_ch*2)
        self.diff2 = BAMAPBlock(init_ch*4)
        self.diff3 = BAMAPBlock(init_ch*8)

        self.assisted0 = AssistedExcitationLayerInv()
        self.assisted1 = AssistedExcitationLayerInv()
        self.assisted2 = AssistedExcitationLayerInv()
        self.assisted3 = AssistedExcitationLayerInv()

    def forward(self, x, x_flip, layers):
        x0 = self.conv0(x)
        x0_flip = self.conv0(x_flip)
        x1 = self.conv1(self.maxpool(x0))
        x1_flip = self.conv1(self.maxpool(x0_flip))
        x2 = self.conv2(self.maxpool(x1))
        x2_flip = self.conv2(self.maxpool(x1_flip))
        x3 = self.conv3(self.maxpool(x2))
        x3_flip = self.conv3(self.maxpool(x2_flip))
        
        x4 = self.conv4(self.maxpool(x3))

        diff0 = self.diff0(x0_flip, x0)
        diff1 = self.diff1(x1_flip, x1)
        diff2 = self.diff2(x2_flip, x2)
        diff3 = self.diff3(x3_flip, x3)

        diff0 = self.assisted0(diff0, layers[0])
        diff1 = self.assisted1(diff1, layers[1])
        diff2 = self.assisted2(diff2, layers[2])
        diff3 = self.assisted3(diff3, layers[3])

        d3 = torch.cat((self.up4(x4), diff3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((self.up3(d3), diff2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((self.up2(d2), diff1), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((self.up1(d1), diff0), dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


class RoughAffineZ(nn.Module):
    def __init__(self, init_ch=16, ds=0):
        super(RoughAffineZ, self).__init__()
        self.downsample = False
        if ds > 0:
            self.downsample = True
            self.avgpool = nn.AvgPool3d(kernel_size=(ds, ds, 1), stride=(ds, ds, 1))
        self.conv0 = ConvBlock(1, init_ch)
        self.conv1 = ConvBlock(init_ch, init_ch*2)
        self.conv2 = ConvBlock(init_ch*2, init_ch*4)
        self.conv3 = ConvBlock(init_ch*4, init_ch*8)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.pooling = nn.AdaptiveAvgPool3d((2, 2, 2))
        last_ch = init_ch * 8
        self.linear = nn.Sequential(
            nn.Linear(8*last_ch, last_ch//4),
            nn.LeakyReLU(),
            nn.Linear(last_ch//4, last_ch//16),
            nn.LeakyReLU(),
            nn.Linear(last_ch//16, 3))

    def forward(self, x):
        if self.downsample:
            x = self.avgpool(x)
        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        xp = self.pooling(x3)
        p = self.linear(xp.reshape(x.shape[0], -1))
        return p


class EISNet(Module):
    def __init__(self, **kwargs):
        super(EISNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        features_num = 2
        deconv = True
        
        self.compare_left_right = True
        self.cgm = CGM(init_ch*8) if self.cgm else None

        self.conv0 = ConvBlock(in_ch, init_ch)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.conv1 = ConvBlock(init_ch, init_ch*2)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.conv2 = ConvBlock(init_ch*2, init_ch*4)

        self.cdb0 = CDB(channel=init_ch, level=3, cur_level=0)
        self.cdb1 = CDB(channel=init_ch*2, level=3, cur_level=1)
        self.cdb2 = CDB(channel=init_ch*4, level=3, cur_level=2)

        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.conv3 = ConvBlock(init_ch*4, init_ch*8)
        self.conv4 = ConvBlock(init_ch*8, init_ch*8)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)

        self.upconv1 = ConvBlock(init_ch*features_num, init_ch)
        self.upconv2 = ConvBlock(init_ch*2*features_num, init_ch*2)
        self.upconv3 = ConvBlock(init_ch*4*features_num, init_ch*4)

        self.magm = MAGM(in_channel=init_ch, level=4)
        self.conv1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x, x_flip):
        x0 = self.conv0(x)
        x0_pool = self.maxpool0(x0)
        x1 = self.conv1(x0_pool)
        x1_pool = self.maxpool1(x1)
        x2 = self.conv2(x1_pool)

        x0_flip = self.conv0(x_flip)
        x0_flip_pool = self.maxpool0(x0_flip)
        x1_flip = self.conv1(x0_flip_pool)
        x1_flip_pool = self.maxpool1(x1_flip)
        x2_flip = self.conv2(x1_flip_pool)

        cdb0 = self.cdb0(x0, x0_flip)
        cdb1 = self.cdb1(x1, x1_flip)
        cdb2 = self.cdb2(x2, x2_flip)

        x2_pool = self.maxpool2(x2)
        x3 = self.conv3(x2_pool)
        x4 = self.conv4(x3)

        up3 = torch.cat((self.up3(x4), cdb2), dim=1)
        upconv3 = self.upconv3(up3)
        up2 = torch.cat((self.up2(upconv3), cdb1), dim=1)
        upconv2 = self.upconv2(up2)
        up1 = torch.cat((self.up1(upconv2), cdb0), dim=1)
        upconv1 = self.upconv1(up1)

        out = self.magm(upconv1, upconv2, upconv3, x4)
        out = self.activation(self.conv1x1(out))

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)

        return has_roi, out


class MGAPDAFAPUNet(Module):
    def __init__(self, **kwargs):
        super(MGAPDAFAPUNet, self).__init__(**kwargs)
        in_ch = self.in_ch
        out_ch = self.out_ch
        init_ch = self.init_ch
        deconv = self.deconv
        self.cgm = CGM(init_ch*16) if self.cgm else None

        self.conv0 = ResConvBlock(in_ch, init_ch, dim='2d')
        self.conv1 = ResConvBlock(init_ch, init_ch*2, dim='2d')
        self.conv2 = ResConvBlock(init_ch*2, init_ch*4, dim='2d')
        self.conv3 = ResConvBlock(init_ch*4, init_ch*8, dim='2d')
        self.conv4 = ResConvBlock(init_ch*8, init_ch*16, dim='2d')

        self.conv0_3d = ResConvBlock(in_ch, init_ch)
        self.conv1_3d = ResConvBlock(init_ch, init_ch*2)
        self.conv2_3d = ResConvBlock(init_ch*2, init_ch*4)

        self.up1 = UpConvBlock(init_ch*2, init_ch, deconv=deconv)
        self.up2 = UpConvBlock(init_ch*4, init_ch*2, deconv=deconv)
        self.up3 = UpConvBlock(init_ch*8, init_ch*4, deconv=deconv)
        self.up4 = UpConvBlock(init_ch*16, init_ch*8, deconv=deconv)

        self.up_conv0 = ResConvBlock(init_ch*2, init_ch)
        self.up_conv1 = ResConvBlock(init_ch*4, init_ch*2)
        self.up_conv2 = ResConvBlock(init_ch*8, init_ch*4)
        self.up_conv3 = ResConvBlock(init_ch*16, init_ch*8)

        self.conv_1x1 = nn.Conv3d(init_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        if out_ch == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        self.fusion1 = DAF(init_ch*2)
        self.fusion2 = DAF(init_ch*4)

        self.compare_left_right = True
        self.diff0 = MGAP(init_ch)
        self.diff1 = MGAP(init_ch*2)
        self.diff2 = MGAP(init_ch*4)
        self.diff3 = MGAP(init_ch*8)

    def forward(self, x, x_flip, layers):
        x0_3d = self.conv0_3d(x)
        x0_flip_3d = self.conv0_3d(x_flip)
        x1_3d = self.conv1_3d(self.maxpool(x0_3d))
        x1_flip_3d = self.conv1_3d(self.maxpool(x0_flip_3d))
        x2_3d = self.conv2_3d(self.maxpool(x1_3d))
        x2_flip_3d = self.conv2_3d(self.maxpool(x1_flip_3d))

        x0 = self.conv0(x)
        x1 = self.conv1(self.maxpool(x0))
        x1_fusion = self.fusion1(x1, x1_3d)
        x2 = self.conv2(self.maxpool(x1_fusion))
        x2_fusion = self.fusion2(x2, x2_3d)
        x3 = self.conv3(self.maxpool(x2_fusion))

        x0_flip = self.conv0(x_flip)
        x1_flip = self.conv1(self.maxpool(x0_flip))
        x1_fusion_flip = self.fusion1(x1_flip, x1_flip_3d)
        x2_flip = self.conv2(self.maxpool(x1_fusion_flip))
        x2_fusion_flip = self.fusion2(x2_flip, x2_flip_3d)
        x3_flip = self.conv3(self.maxpool(x2_fusion_flip))
        
        x4 = self.conv4(self.maxpool(x3))

        diff0 = self.diff0(x0_flip, x0, layers[0])
        diff1 = self.diff1(x1_fusion_flip, x1_fusion, layers[1])
        diff2 = self.diff2(x2_fusion_flip, x2_fusion, layers[2])
        diff3 = self.diff3(x3_flip, x3, layers[3])

        d3 = torch.cat((self.up4(x4), diff3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = torch.cat((self.up3(d3), diff2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = torch.cat((self.up2(d2), diff1), dim=1)
        d1 = self.up_conv1(d1)
        d0 = torch.cat((self.up1(d1), diff0), dim=1)
        d0 = self.up_conv0(d0)

        has_roi = None
        if self.cgm is not None:
            has_roi = self.cgm(x4)
        out = self.activation(self.conv_1x1(d0))
        return has_roi, out


if __name__ == '__main__':
    import torchsummaryX
    device = 'cuda'
    # model = Fuchigami(1, 2, 32, cgm=True).to(device)
    x = torch.zeros(4, 1, 128, 128, 8)

    model = MGAPDAFAPUNet()
    layers = []
    layers.append(x)
    for i in range(4):
        _, _, h, w, d = layers[i].shape
        csf_masks = torch.nn.functional.interpolate(layers[i], size=(h//2, w//2, d), mode='nearest')
        layers.append(csf_masks)

    if model.compare_left_right:
        torchsummaryX.summary(model, x, x, layers)
    else:
        torchsummaryX.summary(model, x)