# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.roi_heads import StandardRoIHead
from mmengine.structures import BaseDataElement
from ovdet.methods.builder import OVD
from typing import List, Tuple
import torch
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from mmdet.models.utils import empty_instances


@MODELS.register_module()
class OVDStandardRoIHead(StandardRoIHead):
    def __init__(self, clip_cfg=None, ovd_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_cfg is None:
            self.clip = None
        else:
            self.clip = MODELS.build(clip_cfg)
        if ovd_cfg is not None:
            for k, v in ovd_cfg.items():
                # self.register_module(k, OVD.build(v))   # not supported in pt1.8.1
                setattr(self, k, OVD.build(v))

    def _bbox_forward(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, self.clip)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def run_ovd(self, x, batch_data_samples, rpn_results_list, ovd_name, batch_inputs,
                *args, **kwargs):
        ovd_method = getattr(self, ovd_name)

        sampling_results_list = list(map(ovd_method.sample, rpn_results_list, batch_data_samples))
        if isinstance(sampling_results_list[0], BaseDataElement):
            rois = bbox2roi([res.bboxes for res in sampling_results_list])
        else:
            sampling_results_list_ = []
            bboxes = []
            for sampling_results in sampling_results_list:
                bboxes.append(torch.cat([res.bboxes for res in sampling_results]))
                sampling_results_list_ += sampling_results
            rois = bbox2roi(bboxes)
            sampling_results_list = sampling_results_list_

        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.vision_to_language(bbox_feats)

        return ovd_method.get_losses(region_embeddings, sampling_results_list, self.clip, batch_inputs)


@MODELS.register_module()
class FVLMStandardRoIHead(StandardRoIHead):
    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor,
                      clip_x=None, clip_pool=None) -> dict:
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        if self.training:
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
        else:
            clip_embeddings = clip_pool(clip_x, rois)
            cls_score, bbox_pred = self.bbox_head(bbox_feats, clip_embeddings=clip_embeddings)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                clip_x=None,
                clip_pool=None) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        # TODO: nms_op in mmcv need be enhanced, the bbox result may get
        #  difference when not rescale in bbox_head

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale,
            clip_x=clip_x,
            clip_pool=clip_pool
        )

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale)

        return results_list

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False,
                     clip_x=None,
                     clip_pool=None
                     ) -> InstanceList:
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head.predict_box_type,
                num_classes=self.bbox_head.num_classes,
                score_per_cls=rcnn_test_cfg is None)

        bbox_results = self._bbox_forward(x, rois, clip_x=clip_x, clip_pool=clip_pool)

        # split batch bbox prediction back to each image
        cls_scores = bbox_results['cls_score']
        bbox_preds = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_preds will be None
        if bbox_preds is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_preds, torch.Tensor):
                bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
            else:
                bbox_preds = self.bbox_head.bbox_pred_split(
                    bbox_preds, num_proposals_per_img)
        else:
            bbox_preds = (None, ) * len(proposals)

        result_list = self.bbox_head.predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale)
        return result_list
