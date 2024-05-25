
# ___***Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control***___




[![arXiv](https://img.shields.io/badge/arXiv-2405.12970-b31b1b.svg)](https://arxiv.org/abs/2405.12970)
<a href='https://faceadapter.github.io/face-adapter.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://huggingface.co/FaceAdapter/FaceAdapter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/spaces/FaceAdapter/FaceAdapter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
[![GitHub](https://img.shields.io/github/stars/FaceAdapter/Face-Adapter?style=social)](https://github.com/FaceAdapter/Face-Adapter) 



<img src="__assets__/banner.gif">


## Introduction
Face-Adapter is an efficient and effective face editing adapter for pre-trained diffusion models, specifically targeting face reenactment and swapping tasks.

## Release
- [2024/5/25] 🔥 We release the [gradio demo](https://huggingface.co/spaces/FaceAdapter/FaceAdapter).
- [2024/5/24] 🔥 We release the code and models.


## Installation

```
# Torch >= 2.0 recommended for acceleration without xformers
pip install accelerate diffusers==0.26.0 insightface onnxruntime

```

## Download Models

You can download models of FaceAdapter directly from [here](https://huggingface.co/FaceAdapter/FaceAdapter/tree/main) or download using python script:
```python
# Download all files 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="FaceAdapter/FaceAdapter", local_dir="./checkpoints")

# If you want to download one specific file
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="FaceAdapter/FaceAdapter", filename="controlnet/config.json", local_dir="./checkpoints")
```


To run the demo, you should also download the pre-trained SD models below:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
<!-- - [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE) -->

## ⚡ Quick Inference

### SD_1.5
```python
python infer.py 
```

You can adjust the cropping size with the ```--crop_ratio``` （default:0.81）parameter. But be careful not to set the crop range too large, as this can decrease the quality of the generated images due to the limit of the training data size.


😊 FaceAdapter can be seamlessly plugged into community models:
```python
python infer.py --base_model "frankjoshua/toonyou_beta6"
```
<img src="__assets__/toonyou.gif">




## Disclaimer

This project strives to positively impact the domain of AI-driven image generation. Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and utilize it in a responsible manner. **The developers do not assume any responsibility for potential misuse by users.**

## Citation
If you find Face-Adapter useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{han2024face,
  title={Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control},
  author={Han, Yue and Zhu, Junwei and He, Keke and Chen, Xu and Ge, Yanhao and Li, Wei and Li, Xiangtai and Zhang, Jiangning and Wang, Chengjie and Liu, Yong},
  journal={arXiv preprint arXiv:2405.12970},
  year={2024}
}
```
