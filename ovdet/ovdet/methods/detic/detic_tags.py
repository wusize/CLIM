import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.structures.bbox import bbox_overlaps
from mmengine.structures import InstanceData
from ovdet.methods.builder import OVD
from ovdet.methods.detic.utils import bboxes_area, bboxes_clamp
import random
import copy


@OVD.register_module()
class DeticTags(nn.Module):
    def __init__(self,
                 tag_embeddings_path=None,
                 sampling_cfg=dict(topk=128, iof_thr=0.5),
                 base_batch_size=32,
                 bce_bias=None, norm_temp=50.0, tag_weight=0.1,
                 tag_neg_weight=1.0):
        super(DeticTags, self).__init__()
        self.base_batch_size = base_batch_size
        self.norm_temp = norm_temp
        self.sampling_cfg = sampling_cfg
        self.tag_weight = tag_weight

        if bce_bias is None:
            self.bce_bias = 0.0
        else:
            self.bce_bias = nn.Parameter(torch.ones(1) * bce_bias)
        if tag_embeddings_path.endswith('npy'):
            tag_embeddings = torch.from_numpy(np.load(tag_embeddings_path)).float()
        else:
            tag_embeddings = torch.load(tag_embeddings_path).float()
        tag_embeddings = F.normalize(tag_embeddings, dim=-1)
        self.register_buffer('tag_embeddings', tag_embeddings)
        self.tag_neg_weight = tag_neg_weight

    @staticmethod
    def sort_by_box_ids(proposals, box_ids):
        sorted_box_ids, sorted_order = box_ids.sort()
        return proposals[sorted_order], sorted_box_ids

    @staticmethod
    def sample_on_topk(proposals, box_ids, image_boxes, metainfo):
        max_size_proposals = []
        for box_id, image_box in enumerate(image_boxes):
            box_id_proposals = proposals[box_ids == box_id]
            if len(box_id_proposals) == 0:
                data_dict = dict(bboxes=image_box[None])
                if hasattr(proposals, "labels"):
                    data_dict.update(labels=torch.tensor([0], dtype=torch.int64,
                                                         device=image_boxes.device))
                if hasattr(proposals, "scores"):
                    data_dict.update(scores=torch.tensor([1], device=image_boxes.device))
                box_proposal = InstanceData(**data_dict)
            else:
                box_id_proposals.bboxes = bboxes_clamp(box_id_proposals.bboxes, image_box)
                max_size_idx = bboxes_area(box_id_proposals.bboxes).argmax().item()
                box_proposal = box_id_proposals[max_size_idx:max_size_idx+1]
            max_size_proposals.append(box_proposal)
        max_size_proposals = InstanceData.cat(max_size_proposals)
        max_size_proposals.set_metainfo(metainfo)

        return max_size_proposals

    def obtain_topk_proposal(self, proposals, box_ids):
        if hasattr(proposals, 'scores') and len(box_ids) > 0:
            unique_box_ids = box_ids.unique()
            topk_proposals = []
            topk_box_ids = []
            for box_id in unique_box_ids:
                cand_box_ids = box_ids[box_ids == box_id]
                cand_proposals = proposals[box_ids == box_id]

                num = min(len(cand_proposals), self.sampling_cfg['topk'])
                _, topk_inds = cand_proposals.scores.topk(num)

                topk_proposals.append(cand_proposals[topk_inds])
                topk_box_ids.append(cand_box_ids[topk_inds])
            return InstanceData.cat(topk_proposals), torch.cat(topk_box_ids)
        else:
            return proposals, box_ids

    # TODO: input topk proposals
    def sample(self, rpn_results, batch_data_sample, **kwargs):
        image_boxes = batch_data_sample.gt_instances.bboxes
        rpn_results, box_ids = self.assign_proposal2image_boxes(rpn_results, image_boxes)
        rpn_results, box_ids = self.obtain_topk_proposal(rpn_results, box_ids)
        rpn_results, box_ids = self.sort_by_box_ids(rpn_results, box_ids)
        sampling_result = self.sample_on_topk(rpn_results, box_ids, image_boxes,
                                              batch_data_sample.metainfo)

        return sampling_result

    def assign_proposal2image_boxes(self, proposals, image_boxes):
        iofs = bbox_overlaps(proposals.bboxes, image_boxes, mode='iof')
        max_iofs, box_ids = iofs.max(dim=1)
        valid = max_iofs > self.sampling_cfg.get('iof_thr', 0.5)
        return proposals[valid], box_ids[valid]

    def get_losses(self, region_embeddings, sampling_results, *args, **kwargs):
        region_embeddings = F.normalize(region_embeddings, dim=-1)

        similarity_matrix_1 = self.norm_temp * region_embeddings @ self.tag_embeddings.T
        similarity_matrix_1 += self.bce_bias

        label_matrix_1 = torch.zeros_like(similarity_matrix_1)
        tags_per_image = [tags_ for batch in sampling_results for tags_ in batch.tags]
        for i, tags_ in enumerate(tags_per_image):
            label_matrix_1[i][tags_] = 1.0

        loss_tags = F.binary_cross_entropy_with_logits(similarity_matrix_1,
                                                       label_matrix_1, reduction='none')

        # num_pos = label_matrix_1.sum(dim=1, keepdim=True).repeat(1, label_matrix_1.shape[1]).clamp(min=1.0)
        # tag_loss_weight = torch.where(label_matrix_1 > 0, self.tag_pos_weight / num_pos,
        #                               torch.ones_like(num_pos))
        loss_tags = loss_tags * label_matrix_1 + \
                    self.tag_neg_weight * loss_tags * (1. - label_matrix_1)
        loss_tags = loss_tags.sum(-1).mean()
        if self.base_batch_size is not None:
            loss_tags *= (label_matrix_1.shape[0] / self.base_batch_size)

        return dict(loss_tags=loss_tags * self.tag_weight)


@OVD.register_module()
class DeticTagsWithComposition(DeticTags):
    def __init__(self, num_groups=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_groups = num_groups

    def _sample_pseudo_regions(self, M, N):
        num_groups = self.num_groups
        assert num_groups <= M * N
        assert M >= 2 and N >= 2
        sampled = set()
        while len(sampled) < num_groups:
            m0, m1 = sorted(random.choices(range(M), k=2))
            n0, n1 = sorted(random.choices(range(N), k=2))
            if m0 == m1 and n0 == n1:
                continue
            sampled_sub_images = []
            for m_ in range(m0, m1+1):
                for n_ in range(n0, n1+1):
                    sampled_sub_images.append(m_ * N + n_)
            sampled.add(tuple(sampled_sub_images))
        return list(sampled)

    def sample_on_topk(self, proposals, box_ids, image_boxes, metainfo):
        metainfo = copy.deepcopy(metainfo)
        max_size_proposals = []
        for box_id, image_box in enumerate(image_boxes):
            box_id_proposals = proposals[box_ids == box_id]
            if len(box_id_proposals) == 0:
                data_dict = dict(bboxes=image_box[None])
                if hasattr(proposals, "labels"):
                    data_dict.update(labels=torch.tensor([0], dtype=torch.int64,
                                                         device=image_boxes.device))
                if hasattr(proposals, "scores"):
                    data_dict.update(scores=torch.tensor([1], device=image_boxes.device))
                box_proposal = InstanceData(**data_dict)
            else:
                box_id_proposals.bboxes = bboxes_clamp(box_id_proposals.bboxes, image_box)
                max_size_idx = bboxes_area(box_id_proposals.bboxes).argmax().item()
                box_proposal = box_id_proposals[max_size_idx:max_size_idx+1]
            max_size_proposals.append(box_proposal)
        max_size_proposals = InstanceData.cat(max_size_proposals)

        if image_boxes.shape[0] > 1:
            M = int(image_boxes.shape[0] ** 0.5)
            assert M**2 == image_boxes.shape[0]

            sampled_groups = self._sample_pseudo_regions(M, M)

            # TODO: mosaic, if the number of image boxes is larger than one, add a global box
            tags, image_ids = metainfo['tags'], metainfo['image_ids']
            max_size_boxes = max_size_proposals.bboxes
            added_proposals = []
            for sampled_group in sampled_groups:
                max_size_boxes_ = max_size_boxes[list(sampled_group)]
                x0y0 = max_size_boxes_.view(-1, 2).min(dim=0).values
                x1y1 = max_size_boxes_.view(-1, 2).max(dim=0).values

                added_box = torch.cat([x0y0, x1y1])
                added_proposal = InstanceData(bboxes=added_box[None],
                                              labels=torch.tensor([0], dtype=torch.int64,
                                                                  device=image_boxes.device),
                                              scores=torch.tensor([1], device=image_boxes.device))
                added_proposals.append(added_proposal)
                image_ids.append([image_ids[pseudo_region_idx] for pseudo_region_idx in sampled_group])
                added_tag = []
                for pseudo_region_idx in sampled_group:
                    added_tag += tags[pseudo_region_idx]
                tags.append(added_tag)

            metainfo['tags'], metainfo['image_ids'] = tags, image_ids
            max_size_proposals = InstanceData.cat([max_size_proposals] + added_proposals)

        max_size_proposals.set_metainfo(metainfo)

        return max_size_proposals
