# dataset settings
_base_ = 'mmdet::_base_/datasets/lvis_v1_instance.py'
image_size = (640, 640)

image_backend_args = None
# image_backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         'data/lvis_v1/train2017': 's3://openmmlab/datasets/detection/coco/train2017',
#         'data/lvis_v1/val2017': 's3://openmmlab/datasets/detection/coco/val2017'
#     })
# )

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=image_backend_args, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type="Pad", size=image_size,
         pad_val=dict(img=(122.7709383, 116.7460125, 104.09373615), seg=255)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            ann_file='wusize/lvis_v1_train_base.json',
            pipeline=train_pipeline,)
    )
)


test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=image_backend_args, to_float32=True),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="Pad", size=image_size,
         pad_val=dict(img=(122.7709383, 116.7460125, 104.09373615), seg=255)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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


val_evaluator = dict(metric=['segm'])
test_evaluator = val_evaluator
