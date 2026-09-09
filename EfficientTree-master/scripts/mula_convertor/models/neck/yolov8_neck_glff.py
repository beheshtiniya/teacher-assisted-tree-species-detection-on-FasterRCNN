import torch.nn as nn
from ..backbone.common import  Conv,Concat, C2f,Transpose
from utils.general import make_divisible
import torch.nn.functional as F

import torch


class LocalAttention(nn.Module):

    def __init__(self, channels=64, scales=[3, 5, 7]):
        super(LocalAttention, self).__init__()
        self.scales = scales
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=s, padding=s // 2, groups=channels),
                nn.Conv2d(channels, 1, kernel_size=1)
            ) for s in scales])
        self.fusion_weights = nn.Parameter(torch.ones(len(scales)), requires_grad=True)

    def forward(self, x):
        attention_maps = []
        for branch in self.branches:
            attn = branch(x)  # [B, 1, H, W]
            attention_maps.append(attn)
        # 归一化融合权重
        weights = F.softmax(self.fusion_weights, dim=0)
        # 加权融合多尺度注意力图
        local_att = torch.zeros_like(attention_maps[0])
        for i, attn in enumerate(attention_maps):
            local_att += weights[i] * attn
        local_att = torch.sigmoid(local_att)
        return x * local_att


class GlobalAttention(nn.Module):

    def __init__(self, channels=64):

        super(GlobalAttention, self).__init__()

        self.global_att = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, channels//8, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels//8, channels, kernel_size=1),
                    nn.Sigmoid())

    def forward(self, x):
        global_att = self.global_att(x)
        return  x * global_att




class GLFF(nn.Module):

    def __init__(self,ch_l=64,ch_g=64):

        super(GLFF, self).__init__()
        self.local_att = LocalAttention(channels=ch_l)
        self.global_att = GlobalAttention(channels=ch_g)
        self.concat = Concat()

    def forward(self, x, y):

        x_att = self.local_att(x)
        y_att = self.global_att(y)
        fusion = self.concat([x_att,y_att])

        return  fusion


class YoloV8Neck_glff(nn.Module):

    def __init__(self, cfg):
        super(YoloV8Neck_glff, self).__init__()
        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        input_p3, input_p4, input_p5 = cfg.Model.Neck.in_channels
        output_p3, output_p4, output_p5 = cfg.Model.Neck.out_channels


        self.channels = {
            'input_p3': input_p3,
            'input_p4': input_p4,
            'input_p5': input_p5,
            'output_p3': output_p3,
            'output_p4': output_p4,
            'output_p5': output_p5,
        }
        self.re_channels_out()

        self.input_p3 = self.channels['input_p3'] #256
        self.input_p4 = self.channels['input_p4'] #512
        self.input_p5 = self.channels['input_p5'] #768

        self.output_p3 = self.channels['output_p3'] #256
        self.output_p4 = self.channels['output_p4'] #512
        self.output_p5 = self.channels['output_p5'] #768

        if cfg.Model.Neck.activation == 'SiLU':
            CONV_ACT = 'silu'
            C_ACT = 'silu'
        elif cfg.Model.Neck.activation == 'ReLU':
            CONV_ACT = 'relu'
            C_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'
            C_ACT = 'hard_swish'

        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")  # 10
        self.la1 = LocalAttention(channels=self.input_p5)
        self.ga1 = GlobalAttention(channels=self.input_p4)
        self.concat1 = Concat()
        self.C1 = C2f(self.input_p5 + self.input_p4, self.input_p4, self.get_depth(3), False, 1, 0.5, C_ACT)  # 12

        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")  # 13
        self.la2 = LocalAttention(channels=self.input_p4)
        self.ga2 = GlobalAttention(channels=self.input_p3)
        self.concat2 = Concat()
        self.C2 = C2f(self.input_p4 + self.input_p3, self.output_p3, self.get_depth(3), False, 1, 0.5, C_ACT)  # 15

        self.conv3 = Conv(self.output_p3, self.output_p3, 3, 2, None, 1, CONV_ACT)
        self.la3 = LocalAttention(channels=self.input_p3)
        self.ga3 = GlobalAttention(channels=self.input_p4)
        self.concat3 = Concat()
        self.C3 = C2f(self.output_p3 + self.input_p4, self.output_p4, self.get_depth(3), False, 1, 0.5, C_ACT)  # 20

        self.conv4 = Conv(self.output_p4, self.output_p4, 3, 2, None, 1, CONV_ACT)
        self.la4 = LocalAttention(channels=self.input_p4)
        self.ga4 = GlobalAttention(channels=self.input_p5)
        self.concat4 = Concat()
        self.C4 = C2f(self.output_p4 + self.input_p5, self.output_p5, self.get_depth(3), False, 1, 0.5, C_ACT)  # 23


    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels.items():
            self.channels[k] = self.get_width(v)

    def forward(self, inputs):
        P3, P4, P5 = inputs

        P4 = self.ga1(P4)
        P3 = self.ga2(P3)
        P5 = self.ga4(P5)

        x1 = self.upsample1(P5)  # 10
        x1 = self.la1(x1)
        x1 = self.concat1([x1, P4])
        x1 = self.C1(x1)  # 12

        x2 = self.upsample2(x1)  # 13
        x2 = self.la2(x2)
        x2 = self.concat2([x2, P3])
        x2 = self.C2(x2)  # 15

        x3 = self.conv3(x2)  # 16
        x3 = self.la3(x3)
        x3 = self.concat3([x3, self.ga3(x1)])  # 17
        x3 = self.C3(x3)  # 18

        x4 = self.conv4(x3)  # 19
        x4 = self.la4(x4)
        x4 = self.concat4([x4, P5])
        x4 = self.C4(x4)  # 21

        return [x2, x3, x4]


