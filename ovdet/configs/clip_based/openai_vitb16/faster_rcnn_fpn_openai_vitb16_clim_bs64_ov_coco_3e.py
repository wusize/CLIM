_base_ = ['mmdet::_base_/models/faster-rcnn_r50_fpn.py',
          'mmdet::_base_/default_runtime.py',
          '../../_base_/datasets/coco_ovd_base_lsj.py']
find_unused_parameters = True
class_weight = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                1, 0, 1, 1, 1, 1, 0, 0, 0, 1] + [0.6]
invalid_classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                   1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
                   1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                   1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 1, 1, 0] + [0]


norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='FVLM',
    data_preprocessor=dict(
        mean=[122.7709383, 116.7460125, 104.09373615],
        std=[68.5005327, 66.6321579, 70.32316305]),
    backbone=dict(
        _delete_=True,
        type='CLIPViT',
        model_name='ViT-B-16',
        cache_dir='checkpoints',
        pretrained='checkpoints/openai_vitb16_coco_clim.pt',
        out_indices=[3, 5, 7, 11],
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7,
                           sampling_ratio=0, use_torchvision=True),
            out_channels=512,
            featmap_strides=[16]),
        norm_cfg=norm_cfg
    ),
    neck=dict(
        in_channels=[768, 768, 768, 768],
        norm_cfg=norm_cfg
    ),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        type='FVLMStandardRoIHead',
        bbox_head=dict(
            type='FVLMConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=2,
            num_cls_fcs=1,
            num_reg_fcs=1,
            reg_class_agnostic=True,
            norm_cfg=norm_cfg,
            alpha=0.1,
            beta=0.8,
            clip_temp=75.0,
            cls_temp=50.0,
            invalid_classes=invalid_classes,
            learn_cls_temp=True,
            cls_embeddings_path="data/metadata/coco_openai_vitb16_hand_craft.npy",
            bg_embedding='learn',
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight
            ),
        )
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.4),
            max_per_img=100)
    )
)


# training schedule for 3e
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=3, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=3,
        by_epoch=True,
        milestones=[100, ],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.1),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)
train_dataloader = dict(
    batch_size=8,
    num_workers=4
)
