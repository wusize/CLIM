_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../../_base_/datasets/coco_ovd_detic_clim.py',
    '../../_base_/schedules/schedule_45k.py',
    '../../_base_/iter_based_runtime.py'
]
class_weight = [1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
                0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                1, 0, 0, 1, 0, 0, 0, 1, 0, 1,
                0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 0, 1, 1, 0, 1, 0, 0, 1,
                0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                1, 0, 1, 1, 1, 1, 0, 0, 0, 1] + [0]

reg_layer = [
    dict(type='Linear', in_features=2048, out_features=2048),
    dict(type='ReLU', inplace=True),
    dict(type='Linear', in_features=2048, out_features=4)
]

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
    type='OVDTwoStageDetector',
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        _delete_=True,
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32
        ),
    ),
    rpn_head=dict(
        type='CustomRPNHead',
        anchor_generator=dict(
            scale_major=False,      # align with detectron2
        )
    ),
    backbone=dict(init_cfg=None),
    batch2ovd=dict(caption_batch=['detic_tags', 'detic_caption'],
                   mosaic_batch=['detic_tags', 'detic_caption']),
    roi_head=dict(
        type='OVDStandardRoIHead',
        shared_head=dict(init_cfg=None),
        clip_cfg=clip_cfg,
        ovd_cfg=dict(detic_caption=dict(type='DeticCaptionWithComposition',
                                        base_batch_size=4,
                                        bce_bias=-20.0, norm_temp=25.0, caption_weight=0.1,
                                        max_caps=1,
                                        queue_cfg=dict(lengths=[256], id_length=16,
                                                       names=['clip_caption_features']),
                                        cap_neg_weight=0.125),
                     detic_tags=dict(type='DeticTagsWithComposition',
                                     tag_embeddings_path='data/metadata/coco_clip_hand_craft.npy',
                                     sampling_cfg=dict(topk=128, iof_thr=0.3),
                                     base_batch_size=None,
                                     bce_bias=-20.0, norm_temp=25.0, tag_weight=0.1 / 3,
                                     tag_neg_weight=1.0
                                     )
                     ),
        bbox_head=dict(
            type='DeticBBoxHead',
            reg_predictor_cfg=reg_layer,
            reg_class_agnostic=True,
            cls_bias=-20.0,
            cls_temp=25.0,
            cls_embeddings_path='data/metadata/coco_clip_hand_craft.npy',
            loss_cls=dict(
                type='CustomCrossEntropyLoss',
                use_sigmoid=True,
                class_weight=class_weight),
        ),
    ),
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',        # amp training
    clip_grad=dict(max_norm=35, norm_type=2),
)
# load_from = 'work_dirs/detic_base/iter_90000.pth'
load_from = 'checkpoints/detic_coco_base.pth'
