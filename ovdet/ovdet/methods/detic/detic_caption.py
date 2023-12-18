import torch
import torch.nn as nn
import torch.nn.functional as F
from ovdet.methods.queues import Queues
from ovdet.models.vlms.clip import clip as CLIP
from mmengine.structures import InstanceData
from ovdet.methods.builder import build_queue, OVD
from ovdet.methods.detic.utils import multi_apply
import random
import copy


@OVD.register_module()
class DeticCaption(nn.Module):
    def __init__(self,
                 base_batch_size=32,
                 bce_bias=0.0, norm_temp=50.0, caption_weight=1.0,
                 max_caps=5,
                 queue_cfg=dict(lengths=[256], id_length=16, ),
                 cap_neg_weight=0.125):
        super(DeticCaption, self).__init__()
        self.base_batch_size = base_batch_size
        self.norm_temp = norm_temp
        if 'type' in queue_cfg:
            self.queues = build_queue(queue_cfg)
        else:
            self.queues = Queues(**queue_cfg)
        self.caption_weight = caption_weight
        self.id_length = queue_cfg.get('id_length')

        if bce_bias is None:
            self.bce_bias = 0.0
        else:
            self.bce_bias = nn.Parameter(torch.ones(1) * bce_bias)
        self.max_caps = max_caps
        self.cap_neg_weight = cap_neg_weight

    @staticmethod    # The caption version of detic use image boxes
    def sample(rpn_results, batch_data_sample, **kwargs):
        image_boxes = batch_data_sample.gt_instances.bboxes
        return InstanceData(
            bboxes=image_boxes,
            labels=torch.tensor([0] * len(image_boxes), device=image_boxes.device, dtype=torch.int64),
            scores=torch.tensor([1.0] * len(image_boxes), device=image_boxes.device),
            metainfo=batch_data_sample.metainfo
        )

    @torch.no_grad()
    def get_caption_features(self, captions, device, clip_model):
        captions = [cap for b in captions for cap in b]
        tokens = CLIP.tokenize_dynamic(captions, truncate=True).to(device)
        caption_features = clip_model.text_encoder.encode_text(tokens, normalize=True).float()
        return caption_features

    def get_losses(self, region_embeddings, sampling_results, clip_model,
                   *args, **kwargs):
        region_embeddings = F.normalize(region_embeddings, dim=-1)
        device = region_embeddings.device
        region_image_ids = list(map(self._parse_image_ids_per_sample,
                                    sampling_results))
        all_captions, caption_image_ids = multi_apply(self._parse_captions_per_sample,
                                                      [v.captions for v in sampling_results],
                                                      region_image_ids)
        region_image_ids = torch.cat(region_image_ids).to(device)
        caption_image_ids = torch.cat(caption_image_ids).to(device)
        assert region_image_ids.shape[0] == region_embeddings.shape[0]

        clip_model.eval()
        clip_caption_features = self.get_caption_features(all_captions, device, clip_model)

        global_clip_caption_features = self.queues.get_queue('clip_caption_features')
        contrast_clip_caption_features = torch.cat([clip_caption_features,
                                                    global_clip_caption_features[..., :-self.id_length]], dim=0)
        contrast_clip_caption_image_ids = torch.cat([caption_image_ids,
                                                     global_clip_caption_features[..., -self.id_length:]], dim=0)

        # matrix 0
        image_id_match_matrix = region_image_ids[:, None] == contrast_clip_caption_image_ids[None]
        label_matrix_0 = image_id_match_matrix.float().sum(dim=-1).clamp(max=1.0)
        similarity_matrix_0 = self.norm_temp * region_embeddings @ contrast_clip_caption_features.T
        similarity_matrix_0 += self.bce_bias
        loss_caption = F.binary_cross_entropy_with_logits(similarity_matrix_0, label_matrix_0,
                                                          reduction='none')
        # loss_caption = loss_caption * label_matrix_0 + \
        #                self.cap_neg_weight * loss_caption * (1. - label_matrix_0)
        # adjust positive weights in the case of multi-label target
        num_pos_labels = label_matrix_0.sum(-1, keepdim=True)
        assert (num_pos_labels > 0).all()
        pos_weights = torch.ones_like(loss_caption) / num_pos_labels
        neg_weights = torch.ones_like(loss_caption) * self.cap_neg_weight
        loss_caption *= torch.where(label_matrix_0 > 0.0, pos_weights, neg_weights)

        loss_caption = loss_caption.sum(-1).mean()
        clip_caption_features_update = torch.cat([clip_caption_features, caption_image_ids], dim=-1)
        queue_update = dict(clip_caption_features=clip_caption_features_update)
        self.queues.dequeue_and_enqueue(queue_update)
        if self.base_batch_size is not None:
            loss_caption *= (label_matrix_0.shape[0] / self.base_batch_size)
        losses = dict(loss_caption=loss_caption * self.caption_weight)

        return losses

    def _parse_image_ids_per_sample(self, sampling_result):
        image_ids = sampling_result.metainfo['image_ids']
        image_ids_tensor = list(map(self._image_id2tensor, image_ids))
        return torch.stack(image_ids_tensor)

    def _image_id2tensor(self, img_id):
        template = -torch.ones(self.id_length)
        if isinstance(img_id, (list, tuple)):
            num_imgs_ = len(img_id)
            template[:num_imgs_] = torch.tensor(img_id)
        else:
            template[:] = img_id
        return template

    def _parse_captions_per_sample(self, pseudo_region_captions, pseudo_region_image_ids):
        caption_image_ids = []    # use image ids to obtain labels
        all_captions = []
        assert len(pseudo_region_image_ids) == len(pseudo_region_captions)

        for caps, image_id in zip(pseudo_region_captions, pseudo_region_image_ids):
            num_caps_per_pseudo_region = len(caps)
            if num_caps_per_pseudo_region == 0:
                continue
            num_sampled_caps = min(self.max_caps, num_caps_per_pseudo_region)
            all_captions += random.sample(caps, k=num_sampled_caps)
            caption_image_ids.append(image_id[None].repeat(num_sampled_caps, 1))

        caption_image_ids = torch.cat(caption_image_ids)

        return all_captions, caption_image_ids


@OVD.register_module()
class DeticCaptionWithComposition(DeticCaption):
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

    def sample(self, rpn_results, batch_data_sample, **kwargs):
        image_boxes = batch_data_sample.gt_instances.bboxes
        metainfo = copy.deepcopy(batch_data_sample.metainfo)
        if image_boxes.shape[0] > 1:
            M = int(image_boxes.shape[0] ** 0.5)
            assert M**2 == image_boxes.shape[0]
            sampled_groups = self._sample_pseudo_regions(M, M)

            # TODO: mosaic, if the number of image boxes is larger than one, add a global box
            captions, image_ids = metainfo['captions'], metainfo['image_ids']
            added_boxes = []
            for sampled_group in sampled_groups:
                sampled_boxes = image_boxes[list(sampled_group)]
                x0y0 = sampled_boxes.view(-1, 2).min(dim=0).values
                x1y1 = sampled_boxes.view(-1, 2).max(dim=0).values
                added_boxes.append(torch.cat([x0y0, x1y1]))
                captions.append([])
                image_ids.append([image_ids[pseudo_region_id] for pseudo_region_id in sampled_group])

            metainfo['image_ids'] = image_ids
            metainfo['captions'] = captions

            added_boxes = torch.stack(added_boxes)
            image_boxes = torch.cat([image_boxes, added_boxes])

        return InstanceData(
            bboxes=image_boxes,
            labels=torch.tensor([0] * len(image_boxes), device=image_boxes.device, dtype=torch.int64),
            scores=torch.tensor([1.0] * len(image_boxes), device=image_boxes.device),
            metainfo=metainfo
        )
