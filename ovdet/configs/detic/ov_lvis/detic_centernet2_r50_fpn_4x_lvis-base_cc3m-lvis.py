_base_ = './detic_centernet2_r50_fpn_4x_lvis_boxsup.py'
branch_field = ['det_batch', 'cap_batch']

image_size_det = (640, 640)
image_size_cap = (320, 320)

# backend = 'pillow'
backend_args_det = dict(
    backend='petrel',
    path_mapping=dict({
        'data/lvis/': 's3://openmmlab/datasets/detection/lvis_v1/'
    }))
backend_args_cap = dict(
    backend='petrel',
    path_mapping=dict({
        'data/cc3m/images/': 'BJ16:s3://wusize/cc3m_original_size/cc3m/'
    }))
train_pipeline_det = [
    dict(type='LoadImageFromFile', backend_args=backend_args_det),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size_det,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size_det,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PackDetInputs')
    dict(type='MultiBranch',
         branch_field=branch_field,
         det_batch=dict(type='PackDetInputs'))
]

train_pipeline_cap = [
    dict(type='LoadImageFromFile', backend_args=backend_args_cap),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size_cap,
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size_cap,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PackDetInputs'),
    dict(type='MultiBranch',
         branch_field=branch_field,
         cap_batch=dict(type='PackDetInputs',
                        meta_keys=['img_id', 'img_path', 'ori_shape',
                                   'img_shape', 'scale_factor',
                                   'flip', 'flip_direction', 'captions',
                                   'tags', 'image_ids']
                        )
         )
]

dataset_det = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type='LVISV1Dataset',
        data_root='data/lvis/',
        ann_file='annotations/lvis_v1_train_norare.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline_det,
        backend_args=None))

dataset_cc3m = dict(
    type='CC3MLVISV1Dataset',
    data_root='data/cc3m',
    ann_file='annotations/cc3m_train_processed_lvis_v1.json',
    data_prefix=dict(img='images/'),
    pipeline=train_pipeline_cap,
    backend_args=None)

batch_split = [8, 32]
train_dataloader = dict(
    batch_size=sum(batch_split),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='CustomGroupMultiSourceSampler',
                 batch_size=sum(batch_split),
                 source_ratio=batch_split),
    batch_sampler=None,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[dataset_det, dataset_cc3m])
)

max_iter = 45000
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        by_epoch=False,
        T_max=max_iter,
    )
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iter, val_interval=15000)

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper'
)

# only keep latest 1 checkpoint
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, max_keep_ckpts=1))

load_from = 'checkpoints/detic_centernet2_r50_fpn_4x_lvis-base_boxsup_20230921_180638-c1685ee2.pth'
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)
find_unused_parameters = True

clip_cfg = dict(          # ViT-B/32
    type='CLIP',
    image_encoder=None,
    text_encoder=dict(
        type='CLIPTextEncoder',
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,    # also the word embedding dim
        transformer_heads=8,
        transformer_layers=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/clip_vitb32.pth')
    )
)

model = dict(
    type='Detic',
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        _delete_=True,
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32),
    ),
    backbone=dict(
        init_cfg=None),   # do not need init, we will load from pretrained model
    batch2ovd=dict(cap_batch=['detic_tags', 'detic_caption']),
    roi_head=dict(
        type='DeticRoIHead',
        clip_cfg=clip_cfg,
        ovd_cfg=dict(detic_tags=dict(type='DeticTags',
                                     tag_embeddings_path='data/metadata/lvis_v1_clip_a+cname.npy',
                                     sampling_cfg=dict(topk=128, iof_thr=0.3),
                                     base_batch_size=None,
                                     bce_bias=None, norm_temp=50.0, tag_weight=0.1,
                                     tag_neg_weight=1.0
                                     ),
                     detic_caption=dict(type='DeticCaption',
                                        base_batch_size=None,
                                        bce_bias=None, norm_temp=50.0, caption_weight=1.0,
                                        max_caps=1,
                                        queue_cfg=dict(lengths=[256], id_length=16,
                                                       names=['clip_caption_features']),
                                        cap_neg_weight=1/8),
                     ),
    ),
)
