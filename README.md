# CLIM: Contrastive Language-Image Mosaic for Region Representation
## Introduction

This is an official release of the paper **CLIM: Contrastive Language-Image Mosaic for Region Representation**.

> [**CLIM: Contrastive Language-Image Mosaic for Region Representation**](https://arxiv.org/abs/2312.11376),            
> Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Wentao Liu, Chen Change Loy            
> [Bibetex](https://github.com/wusize/CLIM#citation)


## Application to CLIP

The code for applying CLIM to CLIP model is adapted from [OpenCLIP-v2.16.0](https://github.com/mlfoundations/open_clip/tree/v2.16.0). Run the
following command to install the package

```bash
pip install -e . -v
```

## Data Preparation
The main experiments are conducted using images from [COCO](https://cocodataset.org/#home) 
and [LVIS](https://www.lvisdataset.org/) datasets. Please prepare datasets and organize them like the 
following:


```text
CLIM/
├── data
    ├── coco
        ├── annotations
            ├── instances_train2017.json  # the box annotations are not used
            ├── panoptic_val2017.json
            ├── panoptic_val2017     # panoptic masks
        ├── train2017
        ├── val2017
    ├── lvis_v1
        ├── annotations
            ├── lvis_v1_train.json  # the box annotations are not used
        ├── train2017    # the same with coco
        ├── val2017      # the same with coco
```

## Run
### Original Models 
To run CLIPSelf, first obtain the original models from 
[EVA-02-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP), and put them under 
`checkpoints/` like the following:

```text
CLIPSelf/
├── checkpoints
    ├── EVA02_CLIP_B_psz16_s8B.pt
    ├── EVA02_CLIP_L_336_psz14_s6B.pt
    
```

### Training and Testing 
We provide the scripts to train CLIPSelf and RegionCLIP under [scripts/](scripts), they are summarized as follows:

| # |       Model       | Method |  Training Data  |                              Script                               | Checkpoint |
|:-:|:-----------------:|:------:|:---------------:|:-----------------------------------------------------------------:|:----------:|
| 1 |     ViT-B/16      |  CLIM  |      COCO       | [script](scripts/train_clipself_coco_image_patches_eva_vitb16.sh) | [model](https://drive.google.com/file/d/1Nz1xH7cbR8HEW40rMtYUn3PE5ypLw5vb/view?usp=sharing)  |
| 2 |     ViT-B/16      |  CLIM  |      LVIS       |  [script](scripts/train_clipself_lvis_image_patches_eva_vitb16.sh)   | [model](https://drive.google.com/file/d/1-yfrMVaS4aN5uZSYCTalhJ_Pq3j_2aT4/view?usp=sharing)  |

For example, if we want to refine ViT-B/16 by CLIPSelf using only image patches on COCO, simply run:
```bash
bash scripts/train_clipself_coco_image_patches_eva_vitb16.sh    # 1
```
We also provide the checkpoints of the listed experiments above in [Drive](https://drive.google.com/drive/folders/1APWIE7M5zcymbjh5OONqXdBOxFy3Ghwm?usp=sharing). 
And they can be organized as follows:

```text
CLIM/
├── checkpoints
    ├── openai_vitb16_coco_clim.pt     # 1
    ├── openai_vitb16_lvis_clim.pt     # 2
```


## Application to Detic
Go to the folder `CLIM/ovdet` and follow the instructions in this [README](F-ViT/README.md).

## License
This project is licensed under [NTU S-Lab License 1.0](LICENSE).

## Citation

```bibtex
@article{wu2023clim,
    title={CLIM: Contrastive Language-Image Mosaic for Region Representation},
    author={Size Wu and Wenwei Zhang and Lumin Xu and Sheng Jin and Wentao Liu and Chen Change Loy},
    journal={arXiv preprint arXiv:2312.11376},
    year={2023}
}
```


## Acknowledgement

We thank [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/v2.16.0),
[MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.28.1) for their valuable code bases.
