## Installation

This code for open-vocabulary object detection is based on [MMDetection 3.x](https://github.com/open-mmlab/mmdetection/tree/3.x)

It requires the following OpenMMLab packages:

- MMEngine >= 0.6.0
- MMCV-full >= v2.0.0rc4
- MMDetection >= v3.0.0rc6
- lvisapi

```bash
pip install openmim mmengine
mim install "mmcv>=2.0.0rc4"
pip install git+https://github.com/lvis-dataset/lvis-api.git
mim install mmdet>=3.0.0rc6
```



## Usage
### Obtain CLIP Checkpoints
We use CLIP's ViT-B-32 model for the implementation of our method. Obtain the state_dict 
of the model from [GoogleDrive](https://drive.google.com/file/d/1ilxBhjb3JXNDar8lKRQ9GA4hTmjxADfu/view?usp=sharing) and 
put it under `checkpoints`. Otherwise, `pip install git+https://github.com/openai/CLIP.git` and
run 
```python
import clip
import torch
model, _ = clip.load("ViT-B/32")
torch.save(model.state_dict(), 'checkpoints/clip_vitb32.pth')
```

### Training and Testing

The training and testing on [OV-COCO](configs/baron/ov_coco/README.md) are supported now.


## Citation

```bibtex
@inproceedings{wu2023baron,
    title={Aligning Bag of Regions for Open-Vocabulary Object Detection},
    author={Size Wu and Wenwei Zhang and Sheng Jin and Wentao Liu and Chen Change Loy},
    year={2023},
    booktitle={CVPR},
}
```

