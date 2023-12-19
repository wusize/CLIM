torchrun --nproc_per_node 8 -m training.main --batch-size=32 --lr=1e-5 --wd=0.1 --epochs=3 --workers=4 \
--model RN50x64 --pretrained openai --warmup 1000  --zeroshot-frequency 1 --dataset-type coco_caption  \
--test-type coco_panoptic --train-data data/cc3m/cc3m_train_original_size_filtered.json \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_RN50x64.npy --train-image-root="" \
--train-ceph-root BJ16:s3://wusize/cc3m_original_size/cc3m \
--val-image-root data/coco/val2017  --cache-dir checkpoints --log-every-n-steps 50 \
--lock-image --save-frequency 3 --lock-image-unlocked-groups 1 --extract-type="v2" \
--name clim_cc3m_3_save3_test1_openai_rn50x64_1layer --downsample-factor 32 --det-image-size 1024 \
--alpha 0.5 --train-image-size 1024
