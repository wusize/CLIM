# dataset settings
_base_ = 'mmdet::_base_/datasets/coco_detection.py'

data_root = 'data/coco/'
image_size = (640, 640)

# backend_args = None
backend_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/coco': 's3://openmmlab/datasets/detection/coco'
    }))
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args, to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(
        type="RandomResize",
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type="RandomCrop",
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type="Pad", size=image_size,
         pad_val=dict(img=(122.7709383, 116.7460125, 104.09373615), seg=255)),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs")
]

train_dataloader = dict(
    dataset=dict(
        ann_file='wusize/instances_train2017_base.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    )
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args, to_float32=True),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="Pad", size=image_size,
         pad_val=dict(img=(122.7709383, 116.7460125, 104.09373615), seg=255)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader


val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_base.json',
        metric='bbox',
        prefix='Base',
        format_only=False),
    dict(
        type='CocoMetric',
        ann_file=data_root + 'wusize/instances_val2017_novel.json',
        metric='bbox',
        prefix='Novel',
        format_only=False)
]
test_evaluator = val_evaluator
