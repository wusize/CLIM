from typing import Tuple
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.models.detectors import TwoStageDetector
from mmdet.structures import SampleList


@MODELS.register_module()
class FVLM(TwoStageDetector):
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x[:self.neck.in_channels] = self.neck(x[:self.neck.in_channels])
        return x[:self.neck.in_channels]

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.backbone(batch_inputs)
        clip_x = x[self.neck.in_channels]
        if self.with_neck:
            x = self.neck(x[:self.neck.in_channels])
        else:
            x = x[:self.neck.in_channels]

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale,
            clip_x=clip_x, clip_pool=self.backbone.clip_pool
        )

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
