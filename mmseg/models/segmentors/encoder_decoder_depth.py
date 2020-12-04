import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoderDepth(BaseSegmentor):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 pose_net=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoderDepth, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if pose_net is not None:
            self.pose_net = builder.build_head(pose_net)
            self.with_pose_net = True
        else:
            self.with_pose_net = False

        self.decode_head = builder.build_head(decode_head)

        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg if train_cfg is not None else dict()
        self.test_cfg = test_cfg if test_cfg is not None else dict()

        self.sensor_mode = self.train_cfg.get('sensor_mode','monocular')
        self.train_mode = self.train_cfg.get('mode','self-sup')

        self.init_weights(pretrained=pretrained)
        assert self.with_decode_head

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoderDepth, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_pose_net:
            self.pose_net.init_weights()

        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, ):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self,img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        feats = self.extract_feat(img)
        out = self._decode_head_forward_test(feats, img_metas)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, **kwargs):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     self.train_cfg, **kwargs)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_depth=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_depth,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_depth, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        depth_pred = self.encode_decode(img, None)

        return depth_pred

    def forward_train(self, img, img_metas, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        side_img = kwargs.get('side_img', None)
        gt_depth = kwargs.get('gt_depth', None)

        if self.sensor_mode == 'binocular' and side_img is not None:
            input_data = torch.cat([img,side_img],dim=1)
        else:
            input_data = img

        x = self.extract_feat(input_data)

        if self.with_pose_net:
            if self.pose_net.backbone_feat_as_input:
                pose = self.pose_net(x)
            else:
                assert len(kwargs['ref_img']) >= self.pose_net.input_frame_cnt
                pose = self.pose_net(img, ref_img=kwargs['ref_img'])
            kwargs.update(dict(pose=pose))

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas, **kwargs)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, **kwargs)
            losses.update(loss_aux)

        return losses

    def inference(self, img, img_meta, rescale):
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)

        depth_logit = self.encode_decode(img, img_meta)
        if rescale:
            depth_logit = resize(
                depth_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        flip = img_meta[0]['flip']
        output = depth_logit
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = depth_logit.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = depth_logit.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        depth_pred = self.inference(img, img_meta, rescale)
        # seg_pred = depth_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = depth_pred.unsqueeze(0)
            return depth_pred
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred


    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        depth_pred = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_depth_pred = self.inference(imgs[i], img_metas[i], rescale)
            depth_pred += cur_depth_pred
        depth_pred /= len(imgs)
        # seg_pred = seg_logit.argmax(dim=1)
        depth_pred = depth_pred.cpu().numpy()
        # unravel batch dim
        depth_pred = list(depth_pred)
        return depth_pred


    # def predict_poses(self, imgs,img_metas,**kwargs):
    #     """Predict poses between input frames for monocular sequences.
    #     """
    #     outputs = {}
    #     if self.num_pose_frames == 2:
    #         # In this setting, we compute the pose to each source frame via a
    #         # separate forward pass through the pose network.
    #
    #         pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
    #
    #         for f_i in self.opt.frame_ids[1:]:
    #             if f_i != "s":
    #                 # To maintain ordering we always pass frames in temporal order
    #                 if f_i < 0:
    #                     pose_inputs = [pose_feats[f_i], pose_feats[0]]
    #                 else:
    #                     pose_inputs = [pose_feats[0], pose_feats[f_i]]
    #
    #                 pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
    #                 axisangle, translation = self.models["pose_decoder"](pose_inputs)
    #                 outputs[("axisangle", 0, f_i)] = axisangle
    #                 outputs[("translation", 0, f_i)] = translation
    #
    #                 # Invert the matrix if the frame id is negative
    #                 outputs[("cam_T_cam", 0,
    #                          f_i)] = transformation_from_parameters(axisangle[:, 0],
    #                                                                 translation[:, 0],
    #                                                                 invert=(f_i < 0))
    #
    #     else:
    #         # Here we input all frames to the pose net (and predict all poses) together
    #         pose_inputs = torch.cat(
    #             [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
    #         pose_inputs = [self.models["pose_encoder"](pose_inputs)]
    #         axisangle, translation = self.models["pose_decoder"](pose_inputs)
    #
    #         for i, f_i in enumerate(self.opt.frame_ids[1:]):
    #             if f_i != "s":
    #                 outputs[("axisangle", 0, f_i)] = axisangle
    #                 outputs[("translation", 0, f_i)] = translation
    #                 outputs[("cam_T_cam", 0,
    #                          f_i)] = transformation_from_parameters(axisangle[:, i], translation[:,
    #                                                                                  i])
    #
    #     return outputs