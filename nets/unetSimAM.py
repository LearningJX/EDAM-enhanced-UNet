import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from nets.vgg import VGG16

# from nets.SimAM import SimAM1
# from nets.SimAM import SimAM2
# from nets.SimAM3 import SimAM3_4
from nets.SimAM3 import SimAM3_5

# from nets.CBAM import CBAMBlock

# from nets.ECA_AM import ECAAttention


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=4, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        # # Simple attention model
        # self.simAM1 = SimAM3_5(item_weight=torch.tensor([1.00, 0.00], dtype=torch.float, requires_grad=True)).to(torch.device('cuda:0'))
        self.simAM2 = SimAM3_5(item_weight=torch.tensor([0.66, 0.33], dtype=torch.float, requires_grad=True)).to(torch.device('cuda:0'))
        self.simAM3 = SimAM3_5(item_weight=torch.tensor([0.33, 0.66], dtype=torch.float, requires_grad=True)).to(torch.device('cuda:0'))
        self.simAM4 = SimAM3_5(item_weight=torch.tensor([0.00, 1.00], dtype=torch.float, requires_grad=True)).to(torch.device('cuda:0'))

        # # CBAM attention model
        # self.CBAM1 = CBAMBlock(channel=64, reduction=8, kernel_size=3).to(torch.device('cuda:0'))
        # self.CBAM2 = CBAMBlock(channel=128, reduction=8, kernel_size=3).to(torch.device('cuda:0'))

        # # ECA attention model
        # self.ECA = ECAAttention(kernel_size=3).to(torch.device('cuda:0'))

    def forward(self, inputs):
        """
        # # Insert self-attention module to original model totally
        # feat1 = self.simAM(self.vgg.features[:4](inputs))
        # feat2 = self.simAM(self.vgg.features[4:9](feat1))
        # feat3 = self.simAM(self.vgg.features[9:16](feat2))
        # feat4 = self.simAM(self.vgg.features[16:23](feat3))
        # feat5 = self.simAM(self.vgg.features[23:-1](feat4))

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final
        """
        """
        # # Original model
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final
        """
        # # Insert self-attention module to the x layer of model
        feat1 = self.vgg.features[:4](inputs)
        feat2 = self.vgg.features[4:9](feat1)
        feat3 = self.vgg.features[9:16](feat2)
        feat4 = self.vgg.features[16:23](feat3)
        feat5 = self.vgg.features[23:-1](feat4)

        up4 = self.up_concat4(self.simAM4(feat4), feat5)
        up3 = self.up_concat3(self.simAM3(feat3), up4)
        up2 = self.up_concat2(self.simAM2(feat2), up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final

    # def _initialize_weights(self, *stages):
    #     for modules in stages:
    #         for module in modules.modules():
    #             if isinstance(module, nn.Conv2d):
    #                 nn.init.kaiming_normal_(module.weight)
    #                 if module.bias is not None:
    #                     module.bias.data.zero_()
    #             elif isinstance(module, nn.BatchNorm2d):
    #                 module.weight.data.fill_(1)
    #                 module.bias.data.zero_()

