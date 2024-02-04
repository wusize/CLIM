# Copyright (c) OpenMMLab. All rights reserved.
from .centernet2 import CenterNet2
import torch
from torch import Tensor
from mmdet.structures import SampleList
from typing import Dict
from mmdet.registry import MODELS


@MODELS.register_module()
class Detic(CenterNet2):

    def __init__(self,
                 batch2ovd=None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch2ovd = dict() if batch2ovd is None else batch2ovd

    def run_ovd(self, x, inputs, data_samples, ovd_name):
        losses = dict()
        if self.with_rpn:
            with torch.no_grad():
                rpn_results_list = self.rpn_head_predict(x, data_samples)
        else:
            assert data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in data_samples
            ]
        if isinstance(ovd_name, str):
            ovd_name = [ovd_name]
        for _ovd_name in ovd_name:
            losses.update(self.roi_head.run_ovd(x, data_samples, rpn_results_list,
                                                _ovd_name, inputs))
        return losses

    def rpn_head_predict(self, x, batch_data_samples):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.rpn_head(x)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        predictions = self.rpn_head.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg, rescale=False)
        return predictions

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        if not isinstance(multi_batch_inputs, dict):
            multi_batch_inputs = dict(det_batch=multi_batch_inputs)
            multi_batch_data_samples = dict(det_batch=multi_batch_data_samples)

        # detection losses
        losses = super().loss(multi_batch_inputs.pop('det_batch'),
                              multi_batch_data_samples.pop('det_batch'))

        multi_batch_features = {k: self.extract_feat(v)
                                for k, v in multi_batch_inputs.items()}

        for batch_name, ovd_name in self.batch2ovd.items():
            batch_inputs = multi_batch_inputs.get(batch_name)
            batch_data_samples = multi_batch_data_samples.get(batch_name)
            batch_features = multi_batch_features.get(batch_name)
            loss_ovd = self.run_ovd(batch_features,
                                    batch_inputs,
                                    batch_data_samples,
                                    ovd_name)
            for k, v in loss_ovd.items():
                losses.update({f'{batch_name}.{k}': v})
        return losses
