import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer,\
                     ConvModule, DepthwiseSeparableConvModule, constant_init,kaiming_init


from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNetV1d


@BACKBONES.register_module()
class PoseNet(nn.Module):

    def __init__(self,
                 num_input_frames=2,
                 backbone_feat_as_input=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6')
                 ):
        super(PoseNet, self).__init__()
        self.num_input_frames = num_input_frames
        self.conv.cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg

        self.convs = {}
        self.conv[0] = ConvModule(
            in_channels=3*num_input_frames,
            out_channels=16,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

