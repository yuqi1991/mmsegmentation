import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class PoseHead(nn.Module):
    def __init__(self, backbone_channels,
                 input_frame_cnt=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):

        super(PoseHead, self).__init__()

        self.backbone_channels = backbone_channels
        self.num_input_features = input_frame_cnt
        self.num_frames_to_predict_for = input_frame_cnt - 1

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.backbone_channels[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(input_frame_cnt * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * self.num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation