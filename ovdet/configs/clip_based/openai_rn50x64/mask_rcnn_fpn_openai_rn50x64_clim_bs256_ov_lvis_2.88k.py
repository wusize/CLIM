_base_ = ['mmdet::_base_/models/mask-rcnn_r50_fpn.py',
          '../../_base_/iter_based_runtime.py',
          '../../_base_/datasets/lvis_v1_ovd_base_lsj.py']
find_unused_parameters = True
class_weight = 'data/metadata/lvis_v1_train_cat_norare_info.json'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='FVLM',
    data_preprocessor=dict(
        mean=[122.7709383, 116.7460125, 104.09373615],
        std=[68.5005327, 66.6321579, 70.32316305]),
    backbone=dict(
        type='CLIPResNet',
        _delete_=True,
        model_name='RN50x64',
        cache_dir='checkpoints',
        pretrained='checkpoints/openai_rn50x64_cc3m_clim.pt',
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14,
                           sampling_ratio=0, use_torchvision=True),
            out_channels=4096,
            featmap_strides=[32]),
    ),
    neck=dict(
        in_channels=[512, 1024, 2048, 4096],
        norm_cfg=norm_cfg
    ),
    rpn_head=dict(
        type='CustomRPNHead',
        num_convs=2,
        norm_cfg=norm_cfg
    ),
    roi_head=dict(
        type='FVLMStandardRoIHead',
        bbox_head=dict(
            type='FVLMConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=2,
            num_cls_fcs=1,
            num_reg_fcs=1,
            reg_class_agnostic=True,
            num_classes=1203,
            norm_cfg=norm_cfg,
            alpha=0.35,
            beta=0.65,
            clip_temp=50.0,
            cls_temp=50.0,
            learn_cls_temp=True,
            cls_embeddings_path="data/metadata/lvis_openai_rn50x64_hand_craft.npy",
            bg_embedding='learn',
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight,
                bg_weight=0.9,
            ),
        ),
        mask_head=dict(
            norm_cfg=norm_cfg, class_agnostic=True, num_classes=1203)
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=2000),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300)
    )
)

default_hooks = dict(
    checkpoint=dict(interval=2880//2)
)

# training schedule for 2.88k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2880, val_interval=2880)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.009, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=2880,
        by_epoch=False,
        milestones=[2304, 2592, 2736],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.36, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=256)
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    sampler=dict(type='InfiniteSampler'),
)
