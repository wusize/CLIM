NAME=$1
CHECKPOINT=$2
torchrun --nproc_per_node 4 -m training.main --batch-size=1 \
--model ViT-B-16 --pretrained openai --test-type coco_panoptic --train-data="" \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_ViTB16.npy \
--val-image-root data/coco/val2017  --cache-dir $CHECKPOINT --extract-type="v2" \
--name $NAME --downsample-factor 16 --det-image-size 1024
