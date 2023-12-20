# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.models import ConvFCBBoxHead
from mmdet.models.layers import multiclass_nms
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes


@MODELS.register_module()
class FVLMConvFCBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 alpha=0.35,
                 beta=0.65,
                 clip_temp=50.0,
                 cls_temp=50.0,
                 learn_cls_temp=True,
                 cls_embeddings_path=None, bg_embedding='learn',
                 invalid_classes=None,
                 *args, **kwargs):
        super(FVLMConvFCBBoxHead, self).__init__(*args, **kwargs)
        if learn_cls_temp:
            self.cls_temp = nn.Parameter(torch.tensor(cls_temp))
        else:
            self.cls_temp = cls_temp
        self.clip_temp = clip_temp
        self.alpha = alpha
        self.beta = beta
        assert self.with_cls
        assert self.reg_class_agnostic
        assert not self.custom_cls_channels

        if invalid_classes is not None:
            self.register_buffer('invalid_classes', torch.tensor(invalid_classes))
        else:
            self.invalid_classes = None

        cls_embeddings = torch.from_numpy(np.load(cls_embeddings_path)).float()
        self.learn_bg = False
        if bg_embedding == 'zero':
            assert self.num_classes == cls_embeddings.shape[0]
            self.register_buffer('cls_embeddings', cls_embeddings)
            self.register_buffer('bg_embedding',
                                 torch.zeros_like(cls_embeddings[:1]))
        elif bg_embedding == 'learn':
            assert self.num_classes == cls_embeddings.shape[0]
            self.register_buffer('cls_embeddings', cls_embeddings)
            self.bg_embedding = nn.Linear(1, cls_embeddings.shape[1])
            self.init_cfg += [
                dict(
                    type='Xavier', distribution='uniform',
                    override=dict(name='bg_embedding')),
            ]
            self.learn_bg = True
        elif bg_embedding == 'clip':
            assert (self.num_classes + 1) == cls_embeddings.shape[0]
            self.register_buffer('cls_embeddings', cls_embeddings[:-1])
            self.register_buffer('bg_embedding', cls_embeddings[-1:])
        else:
            raise ValueError(f"{bg_embedding} not supported.")

        del self.fc_cls
        cls_predictor_cfg_ = self.cls_predictor_cfg.copy()
        cls_predictor_cfg_.update(
            in_features=self.cls_last_dim, out_features=cls_embeddings.shape[1])
        self.fc_cls = MODELS.build(cls_predictor_cfg_)

        class_weight = self.loss_cls.class_weight[:-1] + [1.0]
        self.register_buffer('class_weight', torch.tensor(class_weight), persistent=False)

    def pred_cls_logits(self, region_embeddings, clip_embeddings=None):
        region_embeddings = F.normalize(region_embeddings, dim=-1)
        if self.learn_bg:
            input_ones = region_embeddings.new_ones(1, 1)
            bg_embedding = self.bg_embedding(input_ones)
            bg_embedding = F.normalize(bg_embedding, p=2, dim=-1)   # normalize
        else:
            bg_embedding = self.bg_embedding
        cls_embeddings = torch.cat([self.cls_embeddings, bg_embedding])
        cls_logits = self.cls_temp * region_embeddings @ cls_embeddings.T
        assert cls_logits.shape[1] == self.num_classes + 1

        if self.training:
            return cls_logits
        else:
            if self.invalid_classes is not None:
                cls_logits[:, self.invalid_classes > 0] = float('-inf')
            cls_scores = torch.softmax(cls_logits, dim=-1)
            assert clip_embeddings is not None
            clip_embeddings = F.normalize(clip_embeddings, p=2, dim=-1)
            clip_logits = self.clip_temp * clip_embeddings @ cls_embeddings.T
            if self.invalid_classes is not None:
                clip_logits[:, self.invalid_classes > 0] = float('-inf')
            clip_scores = torch.softmax(clip_logits, dim=-1)

            base_idx = self.class_weight > 0.0
            novel_idx = torch.logical_not(base_idx)

            cls_scores[:, base_idx] = (cls_scores[:, base_idx] ** (1 - self.alpha)
                                       * clip_scores[:, base_idx] ** self.alpha)
            cls_scores[:, novel_idx] = (cls_scores[:, novel_idx] ** (1 - self.beta)
                                        * clip_scores[:, novel_idx] ** self.beta)

            return cls_scores

    def forward(self, x: Tuple[Tensor], clip_embeddings=None):
        region_embeddings, bbox_pred = super().forward(x)
        cls_score = self.pred_cls_logits(region_embeddings,
                                         clip_embeddings=clip_embeddings)
        return cls_score, bbox_pred

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]

        # some loss (Seesaw loss..) may have custom activation
        # if self.custom_cls_channels:
        #     scores = self.loss_cls.get_activation(cls_score)
        # else:
        #     scores = F.softmax(
        #         cls_score, dim=-1) if cls_score is not None else None
        scores = cls_score

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results

