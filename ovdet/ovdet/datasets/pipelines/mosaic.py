# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv
import numpy as np
from mmdet.datasets.transforms import Mosaic, CachedMosaic
from mmcv.transforms.utils import cache_randomness
from numpy import random
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type

@TRANSFORMS.register_module()
class MosaicWithCaption(Mosaic):
    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []

        mosaic_tags = []
        mosaic_captions = []
        mosaic_image_ids = []

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
                mosaic_tags += results['tags']
                mosaic_captions += results['captions']
                mosaic_image_ids += results['image_ids']
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])
                mosaic_tags += results['mix_results'][i - 1]['tags']
                mosaic_captions += results['mix_results'][i - 1]['captions']
                mosaic_image_ids += results['mix_results'][i - 1]['image_ids']

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        inside_inds = np.where(inside_inds)[0]
        results['tags'] = [mosaic_tags[idx] for idx in inside_inds]
        results['captions'] = [mosaic_captions[idx] for idx in inside_inds]
        results['image_ids'] = [mosaic_image_ids[idx] for idx in inside_inds]

        return results


@TRANSFORMS.register_module()
class CachedMosaicWithCaption(CachedMosaic):

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return results

        if random.uniform(0, 1) > self.prob:
            return results
        indices = self.get_indexes(self.results_cache)

        return self._transform(results, indices)

    def _transform(self, results, indices):
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []

        mosaic_tags = []
        mosaic_captions = []
        mosaic_image_ids = []

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
                mosaic_tags += results['tags']
                mosaic_captions += results['captions']
                mosaic_image_ids += results['image_ids']
            else:
                results_patch = copy.deepcopy(mix_results[i - 1])
                mosaic_tags += mix_results[i - 1]['tags']
                mosaic_captions += mix_results[i - 1]['captions']
                mosaic_image_ids += mix_results[i - 1]['image_ids']

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        is_image_box = mosaic_bboxes_labels >= 0

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        image_boxes_inside_inds = inside_inds[is_image_box]

        image_boxes_inside_inds = np.where(image_boxes_inside_inds)[0]
        results['tags'] = [mosaic_tags[idx] for idx in image_boxes_inside_inds]
        results['captions'] = [mosaic_captions[idx] for idx in image_boxes_inside_inds]
        results['image_ids'] = [mosaic_image_ids[idx] for idx in image_boxes_inside_inds]

        return results


@TRANSFORMS.register_module()
class MultiChoicesMosaic(CachedMosaicWithCaption):
    _allowed_choices = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4),
                        (4, 3), (4, 4)]

    def __init__(self,
                 random_rescale=None,
                 choices=None, post_process=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if choices is None:
            self.choices = self._allowed_choices
        else:
            self.choices = [choice for choice in choices if choice in self._allowed_choices]
            assert len(self.choices) > 0
        self.random_rescale = random_rescale

        if post_process is None:
            self.post_process = [[] for _ in range(len(self.choices))]
        else:
            assert len(post_process) == len(self.choices)
            self.post_process = []
            for idx in range(len(self.choices)):
                post_process_ = [TRANSFORMS.build(trans) for trans in post_process[idx]]
                self.post_process.append(post_process_)

    def _rescale_sub_image(self, results):
        img = results['img']
        gt_bboxes = results['gt_bboxes']
        h, w = img.shape[:2]
        # keep_ratio resize
        scale_ratio = min(self.img_scale[1] / h, self.img_scale[0] / w)
        img = mmcv.imresize(img, (int(w * scale_ratio), int(h * scale_ratio)))

        gt_bboxes.rescale_([scale_ratio, scale_ratio])

        return img, gt_bboxes

    def _random_rescale_results(self, results):
        img = results['img']
        gt_bboxes = results['gt_bboxes']
        h, w = img.shape[:2]
        # keep_ratio resize
        ratio_low, ratio_high = min(self.random_rescale), max(self.random_rescale)
        scale_ratio = random.uniform(ratio_low, ratio_high)
        img = mmcv.imresize(img, (int(w * scale_ratio), int(h * scale_ratio)))
        gt_bboxes.rescale_([scale_ratio, scale_ratio])
        results['img'] = img
        results['gt_bboxes'] = gt_bboxes

        return results

    def mosaic_1x2(self, results_left, results_right, align_up):
        mosaic_img = np.full(
            (int(self.img_scale[1]), int(self.img_scale[0] * 2), 3),
            self.pad_val, dtype=results_left['img'].dtype)

        img_left, gt_bboxes_left = self._rescale_sub_image(results_left)
        img_right, gt_bboxes_right = self._rescale_sub_image(results_right)

        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        # left
        if align_up:
            y_0 = 0
        else:
            y_0 = self.img_scale[1] - h_left
        x_0 = self.img_scale[0] - w_left
        gt_bboxes_left.translate_([x_0, y_0])
        mosaic_img[y_0:y_0 + h_left, x_0:x_0 + w_left] = img_left

        # right
        if align_up:
            y_0 = 0
        else:
            y_0 = self.img_scale[1] - h_right
        x_0 = self.img_scale[0]
        gt_bboxes_right.translate_([x_0, y_0])
        mosaic_img[y_0:y_0 + h_right, x_0:x_0 + w_right] = img_right

        mosaic_bboxes = gt_bboxes_left.cat([gt_bboxes_left, gt_bboxes_right], 0)
        mosaic_bboxes_labels = np.concatenate([results_left['gt_bboxes_labels'],
                                               results_right['gt_bboxes_labels']], 0)
        mosaic_ignore_flags = np.concatenate([results_left['gt_ignore_flags'],
                                              results_right['gt_ignore_flags']], 0)
        mosaic_tags = results_left['tags'] + results_right['tags']
        mosaic_captions = results_left['captions'] + results_right['captions']
        mosaic_image_ids = results_left['image_ids'] + results_right['image_ids']

        results = results_left

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        results['tags'] = mosaic_tags
        results['captions'] = mosaic_captions
        results['image_ids'] = mosaic_image_ids

        return results

    def mosaic_1x3(self, results_left, results_mid, results_right, align_up):
        mosaic_img = np.full(
            (int(self.img_scale[1]), int(self.img_scale[0] * 3), 3),
            self.pad_val, dtype=results_left['img'].dtype)

        img_left, gt_bboxes_left = self._rescale_sub_image(results_left)
        img_mid, gt_bboxes_mid = self._rescale_sub_image(results_mid)
        img_right, gt_bboxes_right = self._rescale_sub_image(results_right)

        h_left, w_left = img_left.shape[:2]
        h_mid, w_mid = img_mid.shape[:2]
        h_right, w_right = img_right.shape[:2]

        # left
        if align_up:
            y_0 = 0
        else:
            y_0 = self.img_scale[1] - h_left
        x_0 = self.img_scale[0] - w_left
        gt_bboxes_left.translate_([x_0, y_0])
        mosaic_img[y_0:y_0 + h_left, x_0:x_0 + w_left] = img_left

        # mid
        if align_up:
            y_0 = 0
        else:
            y_0 = self.img_scale[1] - h_mid
        x_0 = random.randint(self.img_scale[0], 2 * self.img_scale[0] - w_mid + 1)
        gt_bboxes_mid.translate_([x_0, y_0])
        mosaic_img[y_0:y_0 + h_mid, x_0:x_0 + w_mid] = img_mid

        # right
        if align_up:
            y_0 = 0
        else:
            y_0 = self.img_scale[1] - h_right
        x_0 = self.img_scale[0] * 2
        gt_bboxes_right.translate_([x_0, y_0])
        mosaic_img[y_0:y_0 + h_right, x_0:x_0 + w_right] = img_right

        mosaic_bboxes = gt_bboxes_left.cat([gt_bboxes_left, gt_bboxes_mid, gt_bboxes_right], 0)
        mosaic_bboxes_labels = np.concatenate([results_left['gt_bboxes_labels'],
                                               results_mid['gt_bboxes_labels'],
                                               results_right['gt_bboxes_labels']], 0)
        mosaic_ignore_flags = np.concatenate([results_left['gt_ignore_flags'],
                                              results_mid['gt_ignore_flags'],
                                              results_right['gt_ignore_flags']], 0)
        mosaic_tags = results_left['tags'] + results_mid['tags'] + results_right['tags']
        mosaic_captions = results_left['captions'] + results_mid['captions'] + results_right['captions']
        mosaic_image_ids = results_left['image_ids'] + results_mid['image_ids'] + results_right['image_ids']

        results = results_left

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        results['tags'] = mosaic_tags
        results['captions'] = mosaic_captions
        results['image_ids'] = mosaic_image_ids

        return results

    def mosaic_1x4(self, results_0, results_1, results_2, results_3, align_up):
        results_left = self.mosaic_1x2(results_0, results_1, align_up=align_up)
        results_right = self.mosaic_1x2(results_2, results_3, align_up=align_up)

        mosaic_img = np.concatenate([results_left['img'],
                                     results_right['img']], axis=1)

        left_boxes = results_left['gt_bboxes']
        right_boxes = results_right['gt_bboxes']
        right_boxes.translate_([2 * self.img_scale[0], 0.0])

        mosaic_bboxes = left_boxes.cat([left_boxes, right_boxes], 0)
        mosaic_bboxes_labels = np.concatenate([results_left['gt_bboxes_labels'],
                                               results_right['gt_bboxes_labels']], 0)
        mosaic_ignore_flags = np.concatenate([results_left['gt_ignore_flags'],
                                              results_right['gt_ignore_flags']], 0)
        mosaic_tags = results_left['tags'] + results_right['tags']
        mosaic_captions = results_left['captions'] + results_right['captions']
        mosaic_image_ids = results_left['image_ids'] + results_right['image_ids']

        results = results_left

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        results['tags'] = mosaic_tags
        results['captions'] = mosaic_captions
        results['image_ids'] = mosaic_image_ids

        return results

    def mosaic_4xN(self, results_list, N):
        assert len(results_list) == 4 * N
        assert N in [4, 3]
        results_up = self.mosaic_2xN(results_list[:2 * N], N=N, random_align=True)
        results_down = self.mosaic_2xN(results_list[2 * N:4 * N], N=N, random_align=True)

        mosaic_img = np.concatenate([results_up['img'],
                                     results_down['img']], axis=0)

        up_boxes = results_up['gt_bboxes']
        down_boxes = results_down['gt_bboxes']
        down_boxes.translate_([0.0, 2 * self.img_scale[1]])

        mosaic_bboxes = up_boxes.cat([up_boxes, down_boxes], 0)
        mosaic_bboxes_labels = np.concatenate([results_up['gt_bboxes_labels'],
                                               results_down['gt_bboxes_labels']], 0)
        mosaic_ignore_flags = np.concatenate([results_up['gt_ignore_flags'],
                                              results_down['gt_ignore_flags']], 0)
        mosaic_tags = results_up['tags'] + results_down['tags']
        mosaic_captions = results_up['captions'] + results_down['captions']
        mosaic_image_ids = results_up['image_ids'] + results_down['image_ids']

        results = results_up

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        results['tags'] = mosaic_tags
        results['captions'] = mosaic_captions
        results['image_ids'] = mosaic_image_ids

        return results

    def mosaic_3xN(self, results_list, N):
        assert len(results_list) == 3 * N
        assert N in [2, 3, 4]
        mosaic_method = getattr(self, f'mosaic_1x{N}')

        results_up = mosaic_method(*results_list[:N], align_up=False)
        results_mid = mosaic_method(*results_list[N:2 * N], align_up=random.choice([True, False]))
        results_down = mosaic_method(*results_list[2 * N:3 * N], align_up=True)

        mosaic_img = np.concatenate([results_up['img'],
                                     results_mid['img'],
                                     results_down['img']], axis=0)

        up_boxes = results_up['gt_bboxes']
        mid_boxes = results_mid['gt_bboxes']
        mid_boxes.translate_([0.0, self.img_scale[1]])
        down_boxes = results_down['gt_bboxes']
        down_boxes.translate_([0.0, 2 * self.img_scale[1]])

        mosaic_bboxes = up_boxes.cat([up_boxes, mid_boxes, down_boxes], 0)
        mosaic_bboxes_labels = np.concatenate([results_up['gt_bboxes_labels'],
                                               results_mid['gt_bboxes_labels'],
                                               results_down['gt_bboxes_labels']], 0)
        mosaic_ignore_flags = np.concatenate([results_up['gt_ignore_flags'],
                                              results_mid['gt_ignore_flags'],
                                              results_down['gt_ignore_flags']], 0)

        mosaic_tags = results_up['tags'] + results_mid['tags'] + results_down['tags']
        mosaic_captions = results_up['captions'] + results_mid['captions'] + results_down['captions']
        mosaic_image_ids = results_up['image_ids'] + results_mid['image_ids'] + results_down['image_ids']

        results = results_up

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        results['tags'] = mosaic_tags
        results['captions'] = mosaic_captions
        results['image_ids'] = mosaic_image_ids

        return results

    def mosaic_2xN(self, results_list, N, random_align=False):
        assert len(results_list) == 2 * N
        assert N in [3, 2, 4]

        mosaic_method = getattr(self, f'mosaic_1x{N}')

        if random_align:
            results_up = mosaic_method(*results_list[:N],
                                       align_up=random.choice([True, False]))
            results_down = mosaic_method(*results_list[N:2 * N],
                                         align_up=random.choice([True, False]))
        else:
            results_up = mosaic_method(*results_list[:N], align_up=False)
            results_down = mosaic_method(*results_list[N:2 * N], align_up=True)

        up_boxes = results_up['gt_bboxes']
        down_boxes = results_down['gt_bboxes']
        down_boxes.translate_([0.0, self.img_scale[1]])

        mosaic_img = np.concatenate([results_up['img'],
                                     results_down['img']], axis=0)
        mosaic_bboxes = up_boxes.cat([up_boxes, down_boxes], 0)
        mosaic_bboxes_labels = np.concatenate([results_up['gt_bboxes_labels'],
                                               results_down['gt_bboxes_labels']], 0)
        mosaic_ignore_flags = np.concatenate([results_up['gt_ignore_flags'],
                                              results_down['gt_ignore_flags']], 0)
        mosaic_tags = results_up['tags'] + results_down['tags']
        mosaic_captions = results_up['captions'] + results_down['captions']
        mosaic_image_ids = results_up['image_ids'] + results_down['image_ids']

        results = results_up

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        results['tags'] = mosaic_tags
        results['captions'] = mosaic_captions
        results['image_ids'] = mosaic_image_ids

        return results

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if random.uniform(0, 1) > self.prob:
            return results

        num_choices = len(self.choices)
        choice_idx = random.randint(0, num_choices)
        M, N = self.choices[choice_idx]
        if M * N <= len(self.results_cache):
            results = self._do_mosaic(results, M, N)
            if self.random_rescale is not None:
                results = self._random_rescale_results(results)
            post_process = self.post_process[choice_idx]
            for trans in post_process:
                results = trans(results)

        return results

    def _do_mosaic(self, results, M, N):
        indices = self.get_indexes(self.results_cache, M * N - 1)
        results_list = [results] + [copy.deepcopy(self.results_cache[idx]) for idx in indices]

        mosaic_method = getattr(self, f'mosaic_{M}xN')

        results = mosaic_method(results_list, N)

        # shift
        delta_x = random.uniform(*self.center_ratio_range) - 1.0
        delta_y = random.uniform(*self.center_ratio_range) - 1.0

        mosaic_img = results['img']
        img_shape = results['img_shape']
        h, w = img_shape[:2]
        delta_x = int(w / 2 * delta_x)
        delta_y = int(h / 2 * delta_y)

        shifted_img = np.full((h, w, 3), self.pad_val, dtype=mosaic_img.dtype)

        paste_x_0, paste_x_1 = max(0, -delta_x), min(w, w - delta_x)
        paste_y_0, paste_y_1 = max(0, -delta_y), min(h, h - delta_y)

        crop_x_0, crop_x_1 = max(0, delta_x), min(w, w + delta_x)
        crop_y_0, crop_y_1 = max(0, delta_y), min(h, h + delta_y)

        shifted_img[paste_y_0:paste_y_1, paste_x_0:paste_x_1] \
            = mosaic_img[crop_y_0:crop_y_1, crop_x_0:crop_x_1]

        mosaic_bboxes = results['gt_bboxes']

        mosaic_bboxes.translate_([-delta_x, -delta_y])

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([h, w])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside([h, w]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]

        mosaic_bboxes_labels = results['gt_bboxes_labels'][inside_inds]
        mosaic_ignore_flags = results['gt_ignore_flags'][inside_inds]

        results['img'] = shifted_img
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        image_boxes_inside_inds = np.where(inside_inds)[0]
        results['tags'] = [results['tags'][idx] for idx in image_boxes_inside_inds]
        results['captions'] = [results['captions'][idx] for idx in image_boxes_inside_inds]
        results['image_ids'] = [results['image_ids'][idx] for idx in image_boxes_inside_inds]

        return results

    @cache_randomness
    def get_indexes(self, cache, num) -> list:
        indexes = [random.randint(0, len(cache) - 1) for _ in range(num)]
        return indexes
