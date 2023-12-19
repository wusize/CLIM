import logging
import os
import random
from dataclasses import dataclass
from multiprocessing import Value
import numpy as np
from training.utils import mask2box
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from open_clip.transform import get_scale
from pycocotools.coco import COCO
from training.coco_api import COCOPanoptic
from panopticapi import utils
import json
import io
try:
    from petrel_client.client import Client
except:
    Client = None
from open_clip.transform import ResizeLongest


class COCOPanopticDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root, embed_path,
                 segm_root,
                 crop_size=224,
                 tokenizer=None,
                 downsample_factor=16,
                 min_size=8, max_size=1024):
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        self.coco = COCOPanoptic(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.embeddings = np.load(embed_path)
        self.image_ids = list(self.coco.imgs.keys())
        num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
        self.max_anns = min(max(num_annos), 100)
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self.min_size = 8  # fix for val
        self.max_size = 1024
        self.segm_root = segm_root
        self.downsample_factor = downsample_factor
        self.segm_transform = ResizeLongest(max_size=self.transforms[0].transforms[0].max_size // downsample_factor,
                                            fill=0)       # downsample to the output size of image encoder

        cat_ids = sorted([cat['id'] for cat in self.coco.cats.values()])

        self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _load_segm(segm_path):
        segmentation = np.array(
            Image.open(segm_path),
            dtype=np.uint8
        )
        # img_bytes = get(segm_path)
        # pan_png = mmcv.imfrombytes(
        #     img_bytes, flag='color', channel_order='rgb').squeeze()
        segm_map = utils.rgb2id(segmentation)

        return segm_map

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_name = image_info['file_name']
        segm_file = image_info['segm_file']
        image_path = os.path.join(self.image_root, image_name)
        segm_path = os.path.join(self.segm_root, segm_file)
        segm_map = self._load_segm(segm_path)

        old_image = Image.open(image_path)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)

        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 2 + 1 + 1)    # xyxy c valid size, isthing
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
        gt_masks = torch.zeros(self.max_anns, self.segm_transform.max_size,
                               self.segm_transform.max_size)
        masked_image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)

        for i, ann in enumerate(anns):
            if i == self.max_anns:
                break
            cat_id = ann['category_id']
            is_thing = self.coco.cats[cat_id]['isthing']
            if is_thing > 0:
                x, y, w, h = ann['bbox']
                cx, cy = x + w*0.5, y + h*0.5
                x0, y0, x1, y1 = \
                    max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            else:
                x0, y0, x1, y1 = mask2box(segm_map == ann['id'])
                x, y, w, h = x0, y0, x1 - x0, y1 - y0
            if w * h < (self.min_size ** 2) or w * h > (self.max_size ** 2):
                continue
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            # masked image crop
            np_old_image = np.asarray(old_image.copy())
            np_old_image[segm_map != ann['id']] = 114
            masked_old_image = Image.fromarray(np_old_image)
            masked_image_crops[i] = self.transforms[1](masked_old_image.crop((x0, y0, x1, y1)))   # image crops

            gt_mask = torch.from_numpy(segm_map == ann['id']).float()
            gt_mask = self.segm_transform(gt_mask[None]) > 0.0
            cls_label = self.cat_id2label[cat_id]
            box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0, w * h, is_thing])    # x, y, x + w, y + h
            boxes_template[i] = box_info
            gt_masks[i] = gt_mask[0]

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h

        return new_image, boxes_template, image_crops, gt_masks, masked_image_crops


class COCOCaptionDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root,
                 tokenizer=None, args=None):
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        with open(input_filename, 'r') as f:
            self.images = json.load(f)['images']
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.ceph_root = args.train_ceph_root
        self.use_ceph = (self.ceph_root != "")
        self.FILE_CLIENT = None

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None

        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None

        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        text = random.choice(image_info['captions'])
        image_name = image_info['file_name']
        image = self.read_image(image_name)
        if image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        image = self.transforms(image)
        text = self.tokenize([text])[0]

        return image, text


class COCODetectionDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root, embed_path,
                 crop_size=224,
                 tokenizer=None):
        logging.debug(f'Loading coco detection style data from {input_filename}.')
        self.coco = COCO(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.embeddings = np.load(embed_path)
        self.image_ids = list(self.coco.imgs.keys())
        num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
        self.max_anns = min(max(num_annos), 100)
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self.min_size = 8  # fix for val
        self.max_size = 1024

        cat_ids = sorted([cat['id'] for cat in self.coco.cats.values()])

        self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        if 'file_name' in image_info:
            image_name = image_info['file_name']
        else:
            assert 'coco_url' in image_info
            coco_url = image_info['coco_url'].split('/')
            image_name = os.path.join(coco_url[-2], coco_url[-1])
        image_path = os.path.join(self.image_root, image_name)
        old_image = Image.open(image_path)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)

        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 2 + 1)    # xyxy c valid size
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)

        for i, ann in enumerate(anns):
            if i == self.max_anns:
                break
            cat_id = ann['category_id']
            x, y, w, h = ann['bbox']
            if w*h < (self.min_size ** 2) or w*h > (self.max_size ** 2):
                continue
            cls_label = self.cat_id2label[cat_id]
            cx, cy = x + w*0.5, y + h*0.5
            x0, y0, x1, y1 = \
                max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0, w * h])    # x, y, x + w, y + h
            boxes_template[i] = box_info

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h

        return new_image, boxes_template, image_crops


def get_coco_panoptic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = COCOPanopticDataset(
        input_filename,
        preprocess_fn,
        segm_root=args.val_segm_root,
        image_root=args.val_image_root,
        embed_path=args.embed_path,
        tokenizer=tokenizer,
        crop_size=args.input_size,
        min_size=args.min_size,
        max_size=args.max_size,
        downsample_factor=args.downsample_factor
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    if is_train:
        batch_size = args.batch_size
    else:
        batch_size = min(args.batch_size, 1)     # only support bs = 1 for inference
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_coco_detection_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert not is_train
    input_filename = args.val_data
    assert input_filename
    dataset = COCODetectionDataset(
        input_filename,
        preprocess_fn,
        image_root=args.val_image_root,
        embed_path=args.embed_path,
        tokenizer=tokenizer,
        crop_size=args.input_size,
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    if is_train:
        batch_size = args.batch_size
    else:
        batch_size = min(args.batch_size, 1)     # only support bs = 1 for inference
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_coco_caption_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data
    assert input_filename
    dataset = COCOCaptionDataset(
        input_filename,
        preprocess_fn,
        image_root=args.train_image_root,
        tokenizer=tokenizer,
        args=args
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'coco_panoptic':
        return get_coco_panoptic_dataset
    elif dataset_type == 'coco_detection':
        return get_coco_detection_dataset
    elif dataset_type == 'coco_caption':
        return get_coco_caption_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type=args.test_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data
