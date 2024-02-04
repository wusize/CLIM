# Data preparation
## Open-Vocabulary COCO
Prepare data following [MMDetection](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download). 
Obtain the json files for OV-COCO from [GoogleDrive](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing) and put them
under `data/coco/wusize`
The data structure looks like:

```text
CLIM/ovdet/data
├── coco
    ├── annotations
        ├── instances_{train,val}2017.json
    ├── wusize
        ├── instances_train2017_base.json
        ├── instances_val2017_base.json
        ├── instances_val2017_novel.json
        ├── captions_train2017_tags_allcaps.json
    ├── train2017
    ├── val2017
    ├── test2017
```


## Open-Vocabulary LVIS
Prepare data following [MMDetection](https://mmdetection.readthedocs.io/en/latest/user_guides/useful_tools.html#dataset-download).
```text
CLIM/ovdet/data
├── lvis_v1
    ├── annotations
        ├── lvis_v1_val.json
        ├── lvis_v1_train.json
    ├── wusize
        ├── lvis_v1_train_base.json
    ├── train2017
    ├── val2017
├── cc3m
    ├── annotations
        ├── train_image_info_tags.json
    ├── images
```
We provide the json file `lvis_v1_train_base.json` than only contains annotations of base categories in 
[Google Drive](https://drive.google.com/file/d/1ahmCUXyFAQqnlMb-ZDDSQUMnIosYqhu5/view?usp=sharing). To obtain cc3m, please refer 
to [Detic](https://github.com/facebookresearch/Detic/blob/main/datasets/README.md).
