import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.utils import get_root_logger
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class PoseHead(nn.Module):
    def __init__(self, backbone_channels,
                 input_frame_cnt=2,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):

        super(PoseHead, self).__init__()

        self.backbone_channels = backbone_channels
        self.input_frame_cnt = input_frame_cnt
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

    def init_weights(self, pretrained=None):
        """Initialize weights of classification layer."""
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.PReLU):
                    constant_init(m, 0)
