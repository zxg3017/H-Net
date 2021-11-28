import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from functools import partial
from segmentation_models_pytorch import create_model
import segmentation_models_pytorch as smp

nonlinearity = partial(F.relu, inplace=True)

BN_EPS = 1e-4  #1e-4  #1e-5


class H_Net_V1(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_V1, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e4_up = nn.ConvTranspose2d(516, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self, x):

        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x) # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1) # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3) # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)

        # CE_Net Encoder part
        rx = x
        e0 = self.firstconv(rx) #[64,256,256]
        e0 = self.firstbn(e0) #[64,256,256]
        e0 = self.firstrelu(e0) #[64,256,256]
        pe0 = self.firstmaxpool(e0) #[64,128,128]
        e1 = self.encoder1(pe0) #[64,128,128]
        e2 = self.encoder2(e1) #[128,64,64]
        e3 = self.encoder3(e2) #[256,32,32]
        e4 = self.encoder4(e3) #[512,16,16]

        # Center of CE_Net
        # e4 = self.CAC_Ce(e4)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)
        # the center part
        e4_up = self.e4_up(e4)
        CAC_out = self.CAC(out4)
        CAC_out = e4_up + CAC_out
        cet_out = self.CAC_conv4(CAC_out)
        r1_cat = torch.cat([e3, cet_out], dim=1)
        up_out = self.rc_up1(r1_cat)
        up5 = self.up5(up_out)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + self.d3_conv(out3) # [128,64,64]
        d2 = self.decoder2(d3) + self.d2_conv(out2) # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1) # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out),F.sigmoid(M_Net_out),F.sigmoid(ave_out),cet_out,M_Net_out,ave_out

class H_Net_V2(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_V2, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e4_up = nn.ConvTranspose2d(516, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self, x):

        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x) # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1) # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3) # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)

        # CE_Net Encoder part
        rx = x
        e0 = self.firstconv(rx) #[64,256,256]
        e0 = self.firstbn(e0) #[64,256,256]
        e0 = self.firstrelu(e0) #[64,256,256]
        pe0 = self.firstmaxpool(e0) #[64,128,128]
        e1 = self.encoder1(pe0) #[64,128,128]
        e2 = self.encoder2(e1) #[128,64,64]
        e3 = self.encoder3(e2) #[256,32,32]
        e4 = self.encoder4(e3) #[512,16,16]

        # Center of CE_Net
        e4 = self.CAC_Ce(e4)
        # e4 = self.dblock(e4)
        e4 = self.spp(e4)
        # the center part
        e4_up = self.e4_up(e4)
        CAC_out = self.CAC(out4)
        CAC_out = e4_up + CAC_out
        cet_out = self.CAC_conv4(CAC_out)
        r1_cat = torch.cat([e3, cet_out], dim=1)
        up_out = self.rc_up1(r1_cat)
        up5 = self.up5(up_out)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + self.d3_conv(out3) # [128,64,64]
        d2 = self.decoder2(d3) + self.d2_conv(out2) # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1) # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out),F.sigmoid(M_Net_out),F.sigmoid(ave_out),cet_out,M_Net_out,ave_out

class H_Net_134(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_134, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock134(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e4_up = nn.ConvTranspose2d(516, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = DACblock134(256)
        self.CAC_Ce = DACblock134(512)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self, x):

        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x) # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1) # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3) # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)

        # CE_Net Encoder part
        rx = x
        e0 = self.firstconv(rx) #[64,256,256]
        e0 = self.firstbn(e0) #[64,256,256]
        e0 = self.firstrelu(e0) #[64,256,256]
        pe0 = self.firstmaxpool(e0) #[64,128,128]
        e1 = self.encoder1(pe0) #[64,128,128]
        e2 = self.encoder2(e1) #[128,64,64]
        e3 = self.encoder3(e2) #[256,32,32]
        e4 = self.encoder4(e3) #[512,16,16]

        # Center of CE_Net
        e4 = self.CAC_Ce(e4)
        # e4 = self.dblock(e4)
        e4 = self.spp(e4)
        # the center part
        e4_up = self.e4_up(e4)
        CAC_out = self.CAC(out4)
        CAC_out = e4_up + CAC_out
        cet_out = self.CAC_conv4(CAC_out)
        r1_cat = torch.cat([e3, cet_out], dim=1)
        up_out = self.rc_up1(r1_cat)
        up5 = self.up5(up_out)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + self.d3_conv(out3) # [128,64,64]
        d2 = self.decoder2(d3) + self.d2_conv(out2) # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1) # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out),F.sigmoid(M_Net_out),F.sigmoid(ave_out),cet_out,M_Net_out,ave_out

class H_Net_137(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_137, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock137(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e4_up = nn.ConvTranspose2d(516, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = DACblock137(256)
        self.CAC_Ce = DACblock137(512)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self, x):

        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x) # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1) # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3) # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)

        # CE_Net Encoder part
        rx = x
        e0 = self.firstconv(rx) #[64,256,256]
        e0 = self.firstbn(e0) #[64,256,256]
        e0 = self.firstrelu(e0) #[64,256,256]
        pe0 = self.firstmaxpool(e0) #[64,128,128]
        e1 = self.encoder1(pe0) #[64,128,128]
        e2 = self.encoder2(e1) #[128,64,64]
        e3 = self.encoder3(e2) #[256,32,32]
        e4 = self.encoder4(e3) #[512,16,16]

        # Center of CE_Net
        e4 = self.CAC_Ce(e4)
        # e4 = self.dblock(e4)
        e4 = self.spp(e4)
        # the center part
        e4_up = self.e4_up(e4)
        CAC_out = self.CAC(out4)
        CAC_out = e4_up + CAC_out
        cet_out = self.CAC_conv4(CAC_out)
        r1_cat = torch.cat([e3, cet_out], dim=1)
        up_out = self.rc_up1(r1_cat)
        up5 = self.up5(up_out)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + self.d3_conv(out3) # [128,64,64]
        d2 = self.decoder2(d3) + self.d2_conv(out2) # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1) # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out),F.sigmoid(M_Net_out),F.sigmoid(ave_out),cet_out,M_Net_out,ave_out


class H_Net_139(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_139, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock139(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e4_up = nn.ConvTranspose2d(516, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = DACblock139(256)
        self.CAC_Ce = DACblock139(512)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self, x):

        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x) # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1) # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3) # conv4 [256,64,64]
        # cet_out = self.center(out)
        # CAC_out = self.CAC(out4)

        # CE_Net Encoder part
        rx = x
        e0 = self.firstconv(rx) #[64,256,256]
        e0 = self.firstbn(e0) #[64,256,256]
        e0 = self.firstrelu(e0) #[64,256,256]
        pe0 = self.firstmaxpool(e0) #[64,128,128]
        e1 = self.encoder1(pe0) #[64,128,128]
        e2 = self.encoder2(e1) #[128,64,64]
        e3 = self.encoder3(e2) #[256,32,32]
        e4 = self.encoder4(e3) #[512,16,16]

        # Center of CE_Net
        e4 = self.CAC_Ce(e4)
        # e4 = self.dblock(e4)
        e4 = self.spp(e4)
        # the center part
        e4_up = self.e4_up(e4)
        CAC_out = self.CAC(out4)
        CAC_out = e4_up + CAC_out
        cet_out = self.CAC_conv4(CAC_out)
        r1_cat = torch.cat([e3, cet_out], dim=1)
        up_out = self.rc_up1(r1_cat)
        up5 = self.up5(up_out)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + self.d3_conv(out3) # [128,64,64]
        d2 = self.decoder2(d3) + self.d2_conv(out2) # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1) # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out),F.sigmoid(M_Net_out),F.sigmoid(ave_out),cet_out,M_Net_out,ave_out

class H_Net(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e4_up = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_Ce = CACblock_with_inception(512)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self, x):

        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x) # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1) # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3) # conv4 [256,64,64]
        # cet_out = self.center(out)

        # CE_Net Encoder part
        rx = x
        e0 = self.firstconv(rx) #[64,256,256]
        e0 = self.firstbn(e0) #[64,256,256]
        e0 = self.firstrelu(e0) #[64,256,256]
        pe0 = self.firstmaxpool(e0) #[64,128,128]
        e1 = self.encoder1(pe0) #[64,128,128]
        e2 = self.encoder2(e1) #[128,64,64]
        e3 = self.encoder3(e2) #[256,32,32]
        e4 = self.encoder4(e3) #[512,16,16]

        # Center of CE_Net
        e4 = self.CAC_Ce(e4)
        # the center part
        e4_up = self.e4_up(e4)
        CAC_out = self.CAC(out4)
        CAC_out = e4_up + CAC_out
        cet_out = self.CAC_conv4(CAC_out)
        r1_cat = torch.cat([e3, cet_out], dim=1)
        up_out = self.rc_up1(r1_cat)
        up5 = self.up5(up_out)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + self.d3_conv(out3) # [128,64,64]
        d2 = self.decoder2(d3) + self.d2_conv(out2) # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1) # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out) /2

        return F.sigmoid(cet_out),F.sigmoid(M_Net_out),F.sigmoid(ave_out),cet_out,M_Net_out,ave_out

class H_Net_ori(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(H_Net_ori, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(3, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(3, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # ce_net encoder part
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # the center of M_Net
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the center of ce_Net
        self.dblock = DACblock(512)  # 空洞卷积
        self.spp = SPPblock(512)  # 池化后再上采样

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_up0 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.e3_up = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.e2_up = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rc_up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_up4 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(64, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(32, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        #
        self.d3_conv = nn.Conv2d(384, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.d2_conv = nn.Conv2d(192, 64, kernel_size=1, padding=0, stride=1, bias=True)
        self.d1_conv = nn.Conv2d(96, 64, kernel_size=1, padding=0, stride=1, bias=True)

        # the decoder of ce_net
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 解码部分

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)  # 逆卷积
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)


    def forward(self, x):

        # M_Net Encoder Part
        l_x = x
        _, _, img_shape, _ = l_x.size()
        x_2 = F.upsample(l_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(l_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(l_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out1 = self.down1(l_x) # conv1 [32,512,512]
        out1 = torch.cat([self.conv2(x_2), out1], dim=1)
        conv2, out2 = self.down2(out1) # conv2 [64,256,256]
        out2 = torch.cat([self.conv3(x_3), out2], dim=1)
        conv3, out3 = self.down3(out2)  # conv3 [128,128,128]
        out3 = torch.cat([self.conv4(x_4), out3], dim=1)
        conv4, out4 = self.down4(out3) # conv4 [256,64,64]
        # cet_out = self.center(out)

        # CE_Net Encoder part
        rx = x
        e0 = self.firstconv(rx) #[64,256,256]
        e0 = self.firstbn(e0) #[64,256,256]
        e0 = self.firstrelu(e0) #[64,256,256]
        pe0 = self.firstmaxpool(e0) #[64,128,128]
        e1 = self.encoder1(pe0) #[64,128,128]
        e2 = self.encoder2(e1) #[128,64,64]
        e3 = self.encoder3(e2) #[256,32,32]
        e4 = self.encoder4(e3) #[512,16,16]


        # the center part
        CAC_out = self.CAC(out4)
        cet_out = self.CAC_conv4(CAC_out)
        r1_cat = torch.cat([e3, cet_out], dim=1)
        up_out = self.rc_up1(r1_cat)
        up5 = self.up5(up_out)
        r2_cat = torch.cat([e2, up5], dim=1)
        up_out1 = self.rc_up2(r2_cat)
        up6 = self.up6(up_out1)
        r3_cat = torch.cat([e1, up6], dim=1)
        up_out2 = self.rc_up3(r3_cat)
        up7 = self.up7(up_out2)
        r4_cat = torch.cat([e0, up7], dim=1)
        up_out3 = self.rc_up4(r4_cat)
        up8 = self.up8(up_out3)
        M_Net_out = self.side_8(up8)

        # center of CE_Net
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Encoder of CE_Net
        d4 = self.decoder4(e4) + out4 # [256,32,32]
        d3 = self.decoder3(d4) + self.d3_conv(out3) # [128,64,64]
        d2 = self.decoder2(d3) + self.d2_conv(out2) # [64,128,128]
        d1 = self.decoder1(d2) + self.d1_conv(out1) # [64,256,256]

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        cet_out = self.finalconv3(out)

        ave_out = (cet_out + M_Net_out)/2
        return F.sigmoid(ave_out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out
class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class DACblock134(nn.Module):
    def __init__(self, channel):
        super(DACblock134, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
class DACblock137(nn.Module):
    def __init__(self, channel):
        super(DACblock137, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class DACblock139(nn.Module):
    def __init__(self, channel):
        super(DACblock139, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=9, padding=9)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
class M_Net(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(M_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(in_ch, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(in_ch, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(in_ch, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_down1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.rc_down2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_down3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_down4 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)
        #
        # # attention blocks
        # self.attentionblock5 = GridAttentionBlock(in_channels=512)
        # self.attentionblock6 = GridAttentionBlock(in_channels=256)
        # self.attentionblock7 = GridAttentionBlock(in_channels=128)
        # self.attentionblock8 = GridAttentionBlock(in_channels=64)


    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        cet_out = self.center(out)

        up_out = self.rc_down1(cet_out)
        r1_cat = torch.cat([conv4, up_out], dim=1)
        up5 = self.up5(r1_cat)

        up_out = self.rc_down2(up5)
        r2_cat = torch.cat([conv3,up_out],dim = 1)
        up6 = self.up6(r2_cat)
        up_out = self.rc_down3(up6)
        r3_cat = torch.cat([conv2,up_out],dim = 1)
        up7 = self.up7(r3_cat)
        up_out = self.rc_down4(up7)
        r4_cat = torch.cat([conv1,up_out],dim = 1)
        up8 = self.up8(r4_cat)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]

class M_Net_CAC_with_CAM(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(M_Net_CAC_with_CAM, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(in_ch, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(in_ch, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(in_ch, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_down1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.rc_down2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_down3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_down4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_conv4 = M_Conv(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the CAM block
        self.p_CAM = nn.MaxPool2d(16)
        # self.CAM_conv = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.CAM_conv = M_Conv(1, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x,cam_x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        # cet_out = self.center(out)

        CAC_out = self.CAC(out)

        # the CAC block concat with our CAM
        cam = self.p_CAM(cam_x)
        cam_conv = self.CAM_conv(cam)
        cam_cac = torch.cat([CAC_out,cam_conv],dim=1)

        cet_out = self.CAC_conv4(cam_cac)
        up_out = self.rc_down1(cet_out)
        r1_cat = torch.cat([conv4, up_out], dim=1)
        up5 = self.up5(r1_cat)


        up_out = self.rc_down2(up5)
        r2_cat = torch.cat([conv3,up_out],dim = 1)
        up6 = self.up6(r2_cat)
        up_out = self.rc_down3(up6)
        r3_cat = torch.cat([conv2,up_out],dim = 1)
        up7 = self.up7(r3_cat)
        up_out = self.rc_down4(up7)
        r4_cat = torch.cat([conv1,up_out],dim = 1)
        up8 = self.up8(r4_cat)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_CAC_with_CAM_Slide(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(M_Net_CAC_with_CAM_Slide, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(1, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(1, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(1, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_down1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.rc_down2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_down3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_down4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_conv4 = M_Conv(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the CAM block
        self.p_CAM = nn.MaxPool2d(16)
        # self.CAM_conv = M_Conv(3, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.CAM_conv = M_Conv(1, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x,cam_x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(cam_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(cam_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(cam_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        # cet_out = self.center(out)

        CAC_out = self.CAC(out)

        # the CAC block concat with our CAM
        cam = self.p_CAM(cam_x)
        cam_conv = self.CAM_conv(cam)
        cam_cac = torch.cat([CAC_out,cam_conv],dim=1)

        cet_out = self.CAC_conv4(cam_cac)
        up_out = self.rc_down1(cet_out)
        r1_cat = torch.cat([conv4, up_out], dim=1)
        up5 = self.up5(r1_cat)


        up_out = self.rc_down2(up5)
        r2_cat = torch.cat([conv3,up_out],dim = 1)
        up6 = self.up6(r2_cat)
        up_out = self.rc_down3(up6)
        r3_cat = torch.cat([conv2,up_out],dim = 1)
        up7 = self.up7(r3_cat)
        up_out = self.rc_down4(up7)
        r4_cat = torch.cat([conv1,up_out],dim = 1)
        up8 = self.up8(r4_cat)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]
class M_Net_CAC_slide(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(M_Net_CAC_slide, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(1, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(1, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(1, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_down1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.rc_down2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_down3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_down4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)
        #
        # # attention blocks
        # self.attentionblock5 = GridAttentionBlock(in_channels=512)
        # self.attentionblock6 = GridAttentionBlock(in_channels=256)
        # self.attentionblock7 = GridAttentionBlock(in_channels=128)
        # self.attentionblock8 = GridAttentionBlock(in_channels=64)


    def forward(self, x,cam_x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(cam_x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(cam_x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(cam_x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        # cet_out = self.center(out)

        CAC_out = self.CAC(out)
        cet_out = self.CAC_conv4(CAC_out)
        up_out = self.rc_down1(cet_out)
        r1_cat = torch.cat([conv4, up_out], dim=1)
        up5 = self.up5(r1_cat)


        up_out = self.rc_down2(up5)
        r2_cat = torch.cat([conv3,up_out],dim = 1)
        up6 = self.up6(r2_cat)
        up_out = self.rc_down3(up6)
        r3_cat = torch.cat([conv2,up_out],dim = 1)
        up7 = self.up7(r3_cat)
        up_out = self.rc_down4(up7)
        r4_cat = torch.cat([conv1,up_out],dim = 1)
        up8 = self.up8(r4_cat)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class M_Net_CAC(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(M_Net_CAC, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(in_ch, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(in_ch, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(in_ch, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_down1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.rc_down2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_down3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_down4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception(256)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)
        #
        # # attention blocks
        # self.attentionblock5 = GridAttentionBlock(in_channels=512)
        # self.attentionblock6 = GridAttentionBlock(in_channels=256)
        # self.attentionblock7 = GridAttentionBlock(in_channels=128)
        # self.attentionblock8 = GridAttentionBlock(in_channels=64)


    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        # cet_out = self.center(out)

        CAC_out = self.CAC(out)
        cet_out = self.CAC_conv4(CAC_out)
        up_out = self.rc_down1(cet_out)
        r1_cat = torch.cat([conv4, up_out], dim=1)
        up5 = self.up5(r1_cat)


        up_out = self.rc_down2(up5)
        r2_cat = torch.cat([conv3,up_out],dim = 1)
        up6 = self.up6(r2_cat)
        up_out = self.rc_down3(up6)
        r3_cat = torch.cat([conv2,up_out],dim = 1)
        up7 = self.up7(r3_cat)
        up_out = self.rc_down4(up7)
        r4_cat = torch.cat([conv1,up_out],dim = 1)
        up8 = self.up8(r4_cat)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]

class M_Net_CAC_with_Coarse(nn.Module):
    def __init__(self,  in_ch, out_ch, bn=True, BatchNorm=False):
        super(M_Net_CAC_with_Coarse, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(in_ch, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(in_ch, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(in_ch, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # self.rw_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rc_down1 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.rc_down2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.rc_down3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.rc_down4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # the CAC block
        self.CAC = CACblock_with_inception_blocks(256)
        self.CAC_conv4 = M_Conv(256, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        # self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)
        #
        # # attention blocks
        # self.attentionblock5 = GridAttentionBlock(in_channels=512)
        # self.attentionblock6 = GridAttentionBlock(in_channels=256)
        # self.attentionblock7 = GridAttentionBlock(in_channels=128)
        # self.attentionblock8 = GridAttentionBlock(in_channels=64)


    def forward(self, x,cam):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        # cet_out = self.center(out)

        CAC_out = self.CAC(out)
        cet_out = self.CAC_conv4(CAC_out)
        up_out = self.rc_down1(cet_out)
        r1_cat = torch.cat([conv4, up_out], dim=1)
        up5 = self.up5(r1_cat)


        up_out = self.rc_down2(up5)
        r2_cat = torch.cat([conv3,up_out],dim = 1)
        up6 = self.up6(r2_cat)
        up_out = self.rc_down3(up6)
        r3_cat = torch.cat([conv2,up_out],dim = 1)
        up7 = self.up7(r3_cat)
        up_out = self.rc_down4(up7)
        r4_cat = torch.cat([conv1,up_out],dim = 1)
        up8 = self.up8(r4_cat)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]


class AG_Net_EASPP(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(AG_Net_EASPP, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(in_ch, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(in_ch, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(in_ch, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the CAC block
        # self.CAC = CACblock_with_inception_blocks(512)
        self.CAC = CACblock_with_inception(512)
        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)

        # attention blocks
        self.attentionblock5 = GridAttentionBlock(in_channels=512)
        self.attentionblock6 = GridAttentionBlock(in_channels=256)
        self.attentionblock7 = GridAttentionBlock(in_channels=128)
        self.attentionblock8 = GridAttentionBlock(in_channels=64)


    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        CAC_out = self.CAC(out)

        FG = torch.cat([self.conv4(x_4), conv4], dim=1)
        N, C, H, W= FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, CAC_out, FG,self.attentionblock5(FG_small,out))
        up5 = self.up5(out)

        FG = torch.cat([self.conv3(x_3), conv3], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, up5, FG,self.attentionblock6(FG_small,up5))
        up6 = self.up6(out)

        FG = torch.cat([self.conv2(x_2), conv2], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, up6, FG,self.attentionblock7(FG_small,up6))
        up7 = self.up7(out)

        FG = torch.cat([conv1, conv1], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, up7, FG,self.attentionblock8(FG_small,up7))
        up8 = self.up8(out)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]

class AG_Net(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, BatchNorm=False):
        super(AG_Net, self).__init__()

        # mutli-scale simple convolution
        self.conv2 = M_Conv(in_ch, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv3 = M_Conv(in_ch, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.conv4 = M_Conv(in_ch, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the down convolution contain concat operation
        self.down1 = M_Encoder(in_ch, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 512
        self.down2 = M_Encoder(64 + 32, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 256
        self.down3 = M_Encoder(128 + 64, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 128
        self.down4 = M_Encoder(256 + 128, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)  # 64

        # the center
        self.center = M_Encoder(256, 512, kernel_size=3, pooling=False)

        # the up convolution contain concat operation
        self.up5 = M_Decoder_my_10(512, 256, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up6 = M_Decoder_my_10(256, 128, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up7 = M_Decoder_my_10(128, 64, kernel_size=3, bn=bn, BatchNorm=BatchNorm)
        self.up8 = M_Decoder_my_10(64, 32, kernel_size=3, bn=bn, BatchNorm=BatchNorm)

        # the sideoutput
        self.side_5 = nn.Conv2d(256, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_6 = nn.Conv2d(128, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_7 = nn.Conv2d(64, out_ch, kernel_size=1, padding=0, stride=1, bias=True)
        self.side_8 = nn.Conv2d(32, out_ch, kernel_size=1, padding=0, stride=1, bias=True)

        self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)

        # attention blocks
        self.attentionblock5 = GridAttentionBlock(in_channels=512)
        self.attentionblock6 = GridAttentionBlock(in_channels=256)
        self.attentionblock7 = GridAttentionBlock(in_channels=128)
        self.attentionblock8 = GridAttentionBlock(in_channels=64)


    def forward(self, x):
        _, _, img_shape, _ = x.size()
        x_2 = F.upsample(x, size=(int(img_shape / 2), int(img_shape / 2)), mode='bilinear')
        x_3 = F.upsample(x, size=(int(img_shape / 4), int(img_shape / 4)), mode='bilinear')
        x_4 = F.upsample(x, size=(int(img_shape / 8), int(img_shape / 8)), mode='bilinear')
        conv1, out = self.down1(x)
        out = torch.cat([self.conv2(x_2), out], dim=1)
        conv2, out = self.down2(out)
        out = torch.cat([self.conv3(x_3), out], dim=1)
        conv3, out = self.down3(out)
        out = torch.cat([self.conv4(x_4), out], dim=1)
        conv4, out = self.down4(out)
        out = self.center(out)

        FG = torch.cat([self.conv4(x_4), conv4], dim=1)
        N, C, H, W= FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, out, FG,self.attentionblock5(FG_small,out))
        up5 = self.up5(out)

        FG = torch.cat([self.conv3(x_3), conv3], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, up5, FG,self.attentionblock6(FG_small,up5))
        up6 = self.up6(out)

        FG = torch.cat([self.conv2(x_2), conv2], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, up6, FG,self.attentionblock7(FG_small,up6))
        up7 = self.up7(out)

        FG = torch.cat([conv1, conv1], dim=1)
        N, C, H, W = FG.size()
        FG_small = F.upsample(FG, size=(int(H/2), int(W/2)), mode='bilinear')
        out = self.gf(FG_small, up7, FG,self.attentionblock8(FG_small,up7))
        up8 = self.up8(out)

        side_5 = F.upsample(up5, size=(img_shape, img_shape), mode='bilinear')
        side_6 = F.upsample(up6, size=(img_shape, img_shape), mode='bilinear')
        side_7 = F.upsample(up7, size=(img_shape, img_shape), mode='bilinear')
        side_8 = F.upsample(up8, size=(img_shape, img_shape), mode='bilinear')

        side_5 = self.side_5(side_5)
        side_6 = self.side_6(side_6)
        side_7 = self.side_7(side_7)
        side_8 = self.side_8(side_8)

        ave_out = (side_5+side_6+side_7+side_8)/4
        return [ave_out, side_5, side_6, side_7, side_8]

class CACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(CACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.conv7x7 = nn.Conv2d(channel, channel, kernel_size=7, dilation=1, padding=3)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = nonlinearity(self.conv7x7(self.conv1x1(x)))
        dilate5_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class CACblock_with_inception(nn.Module): # 1X1,3X3,5X5
    def __init__(self, channel):
        super(CACblock_with_inception, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        # self.conv7x7 = nn.Conv2d(channel, channel, kernel_size=7, dilation=1, padding=3)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        # dilate4_out = nonlinearity(self.conv7x7(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=False, BatchNorm=False, is_relu=True, num_groups=32):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        if BatchNorm:
            self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if is_bn:
            if out_channels//num_groups==0:
                num_groups=1
            self.gn  =nn.GroupNorm(num_groups, out_channels, eps=BN_EPS)
        self.is_bn = is_bn
        self.is_BatchNorm=BatchNorm
        if is_relu is False: self.relu=None

    def forward(self,x):
        x = self.conv(x)
        if self.is_BatchNorm: x = self.bn(x)
        if self.is_bn: x = self.gn(x)
        if self.relu is not None: x = self.relu(x)
        return x


class StackEncoder (nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackEncoder, self).__init__()
        padding=(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self,x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoder (nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3, dilation=1, bn=False, BatchNorm=False, num_groups=32):
        super(StackDecoder, self).__init__()
        padding=(dilation*kernel_size-1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels+x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        y = F.upsample(x, size=(H,W),mode='bilinear')
        #y = F.upsample(x, scale_factor=2,mode='bilinear')
        y = torch.cat([y,x_big],1)
        y = self.decode(y)
        return  y


class M_Encoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Encoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )
        self.pooling = pooling

    def forward(self, x):
        conv = self.encode(x)
        if self.pooling:
            pool = F.max_pool2d(conv, kernel_size=2, stride=2)
            return conv,pool
        else:
            return conv


class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, pooling=True, bn=False, BatchNorm=False, num_groups=32):
        super(M_Conv, self).__init__()
        padding =(dilation*kernel_size-1)//2
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,kernel_size=kernel_size, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        conv = self.encode(x)
        return conv


class M_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self, x_big, x):
        N,C,H,W = x_big.size()
        out = F.upsample(x, size=(H,W),mode='bilinear')
        out = torch.cat([x_big,out], dim=1)
        if self.deconv:
            out = self.deconv(out)
        else:
            out = self.decode(out)
        return out


class M_Decoder_my_10(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1, deconv = False, bn=False, BatchNorm=False, num_groups=32):
        super(M_Decoder_my_10, self).__init__()
        padding =(dilation*kernel_size-1)//2
        if deconv:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                             stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
                ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1,is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            )
        else:
            self.deconv = False

        self.decode = nn.Sequential(
            ConvBnRelu2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
            ConvBnRelu2d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=1, groups=1, is_bn=bn,BatchNorm=BatchNorm, num_groups=num_groups),
        )

    def forward(self,x):
        x = self.decode(x)
        return x

class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        return sigm_psi_f

class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        ## N
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # l_a = torch.abs(l_a)
        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        ## mean_attention
        mean_a = self.boxfilter(l_a) / N
        ## mean_a^2xy
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        ## mean_tax
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        ## mean_ay
        mean_ay = self.boxfilter(l_a * lr_y) / N
        ## mean_a^2x^2
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        ## mean_ax
        mean_ax = self.boxfilter(l_a * lr_x) / N

        ## A
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        ## b
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N


        ## mean_A; mean_b
        mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return (mean_A*hr_x+mean_b).float()
def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

if __name__ == '__main__':
    from torchstat import stat

    # model = H_Net(3,2,bn=True, BatchNorm=False)
    model = create_model(arch='DeepLabV3Plus', encoder_name="efficientnet-b0", encoder_weights= "imagenet",
                          in_channels = 3, classes = 2)

    eff = smp.encoders.get_encoder(name='efficientnet-b0', in_channels=3, depth=4, weights=None)
    a = torch.rand((2, 3, 512, 512))
    print(model.encoder._blocks)