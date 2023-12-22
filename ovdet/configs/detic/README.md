## Installation
Please refer to this [README](ovdet/INSTALLATION.md).
## Data Preparation
Please refer to this [README](ovdet/DATA.md).

## Usage
### Obtain CLIP Checkpoints
We use CLIP's text encoder (ViT-B/32) for Detic. Obtain the state_dict 
of the model from [GoogleDrive](https://drive.google.com/file/d/1ilxBhjb3JXNDar8lKRQ9GA4hTmjxADfu/view?usp=sharing) and put it under `checkpoints`.

### Training
1. To pre-train the detector only on the detection data of base categories, run

```
cd CLIM/ovdet
bash tools/dist_train.sh \
     configs/detic/ov_coco/faster_rcnn_r50_caffe_c4_90k_ovcoco.py 8 \
     --work-dir your/output/directory/detic_coco_base
```
Rename the checkpoint of the trained model as `detic_coco_base.pth` and put it under `checkpoints`.
We also provide this checkpoint `detic_coco_base.pth` in [Google Drive]().

2.1 To fine-tune the detector with caption data (no tags), run 

```
cd CLIM/ovdet
bash tools/dist_train.sh \
     configs/detic/ov_coco/detic_no_tags_clim_faster_rcnn_r50_caffe_c4_45k.py 8 \
     --work-dir your/output/directory/detic_coco_cap_no_tags_clim
```
2.2 To fine-tune  the detector using caption loss and image tag loss, run

```
cd CLIM/ovdet
bash tools/dist_train.sh \
     configs/detic/ov_coco/detic_w_tags_clim_faster_rcnn_r50_caffe_c4_45k.py 8 \
     --work-dir your/output/directory/detic_coco_cap_w_tags_clim
```


### Testing
We have provided the following checkpoints in [Google Drive]().




| OV-COCO |    Losses     | Novel AP50 |                                Config                                | Download  |
|:-------:|:-------------:|:----------:|:--------------------------------------------------------------------:|:---------:|
|    1    |    Caption    |    32.3    | [config](ov_coco/detic_no_tags_clim_faster_rcnn_r50_caffe_c4_45k.py) | [model]()   |
|    2    | Caption & Tag |    35.4    | [config](ov_coco/detic_w_tags_clim_faster_rcnn_r50_caffe_c4_45k.py)  | [model]() |



For example, to evaluate the model trained with caption loss and tag loss, run

```
cd CLIM/ovdet
bash tools/dist_test.sh \
     configs/detic/ov_coco/detic_w_tags_clim_faster_rcnn_r50_caffe_c4_45k.py \
     checkpoints/detic_coco_cap_w_tags_clim.pth \
     8 --work-dir your/output/directory/detic_coco_cap_w_tags_clim
```
