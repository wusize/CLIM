# Application to CLIP

## Installation
The code for applying CLIM to CLIP model is adapted from [OpenCLIP-v2.16.0](https://github.com/mlfoundations/open_clip/tree/v2.16.0). Run the
following command to install the package

```bash
cd CLIM/
pip install -e . -v
```

## Data Preparation
The main experiments are conducted using images from [COCO](https://cocodataset.org/#home) and 
[CC3M](https://ai.google.com/research/ConceptualCaptions/download)
Please prepare datasets and organize them like the following:

```text
CLIM/
├── data
    ├── coco
        ├── annotations
            ├── panoptic_val2017.json
            ├── panoptic_val2017     # panoptic masks
        ├── wusize
            ├── captions_train2017_tags_allcaps.json
        ├── train2017
        ├── val2017
    ├── cc3m
        ├── cc3m_captions_train.json
        ├── train
```
The json file `captions_train2017_tags_allcaps.json` for coco captions can be obtained from 
[GoogleDrive](https://drive.google.com/drive/folders/1O6rt6WN2ePPg6j-wVgF89T7ql2HiuRIG?usp=sharing).
For CC3M dataset, please download the image using the csv file from the official 
[website](https://ai.google.com/research/ConceptualCaptions/download), and then generate the json file
following the COCO format. The json file `cc3m_captions_train.json` might look like:

```json lines
{'images': 
  [
    {'id': 1, 'file_name': 'train/0/0.jpg', 'captions': ['a very typical bus station']},
    {'id': 4, 'file_name': 'train/3/3.jpg', 'captions': ['interior design of modern living room with fireplace in a new house']},
  ]
}
```

## Run
### Original Models 
To run CLIM, first obtain the original models using these 
[links](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L30), 
and put them under 
`checkpoints/` like the following:

```text
CLIM/
├── checkpoints
    ├── ViT-B-16.pt
    ├── RN50x64.pt
    
```

### Applying CLIM
We provide the scripts to run CLIM under this [directory](scripts).

For example, if we want to refine ViT-B/16 on the COCO dataset, simply run:
```bash
bash scripts/train_clim_coco_100e_openai_vitb16.sh
```
We also provide the checkpoints of the models trained by CLIM in
[Google Drive](https://drive.google.com/drive/folders/1v91n5SSXSOtgo2SlEESj_Gquwh9KMj3J?usp=sharing).

### Open-Vocabulary Object Detection

To build open-vocabulary detectors using the models trained by CLIM, 
please refer to the instructions in this [README](ovdet/configs/clip_based/README.md).
