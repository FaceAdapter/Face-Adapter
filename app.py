import os
import cv2
import torch
import random
import numpy as np
import PIL
from PIL import Image
from typing import Tuple

# import spaces
import gradio as gr
import tqdm

import torch
import torch.nn.functional as F
import torchvision.utils as ttf
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

import data.datasets_faceswap as datasets_faceswap
import face_adapter.model_seg_unet as model_seg_unet
from face_adapter.model_to_token import Image2Token, ID2Token
from face_adapter_pipline import StableDiffusionFaceAdapterPipeline, draw_pts70_batch

from insightface.app import FaceAnalysis
from third_party import model_parsing
import third_party.model_resnet_d3dfr as model_resnet_d3dfr
import third_party.d3dfr.bfm as bfm
import third_party.insightface_backbone_conv as model_insightface_backbone


use_cache=False

# global variable
torch.set_grad_enabled(False)
test_image_size = 512
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# download checkpoints
from huggingface_hub import snapshot_download
snapshot_download(repo_id="FaceAdapter/FaceAdapter", local_dir="./checkpoints")


pil2tensor = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize(mean=0.5, std=0.5)])


def convert_batch_to_nprgb(batch, nrow):
    grid_tensor = ttf.make_grid(batch * 0.5 + 0.5, nrow=nrow)
    im_rgb = (255 * grid_tensor.permute(1, 2, 0).cpu().numpy()).astype('uint8')
    return im_rgb  


controlnet = ControlNetModel.from_pretrained('./checkpoints/controlnet', torch_dtype=weight_dtype).to(device)

pipe = StableDiffusionFaceAdapterPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=weight_dtype, cache_dir='./hub' if use_cache else None, local_files_only=use_cache, requires_safety_checker=False
).to(device)
# pretrained unet
pretrained_unet_path = './checkpoints/pretrained_unet'
if os.path.exists(pretrained_unet_path):
    pipe.unet = UNet2DConditionModel.from_pretrained(pretrained_unet_path, torch_dtype=weight_dtype).to(device)
    
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


vae_ft_mse = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir='./hub' if use_cache else None, torch_dtype=weight_dtype, local_files_only=use_cache).to(device)
pipe.vae = vae_ft_mse

net_d3dfr = model_resnet_d3dfr.getd3dfr_res50('./checkpoints/third_party/d3dfr_res50_nofc.pth').eval().to(device)
bfm_facemodel = bfm.BFM(focal=1015*256/224, image_size=256, bfm_model_path='./checkpoints/third_party/BFM_model_front.mat').to(device)

net_arcface = model_insightface_backbone.getarcface('./checkpoints/third_party/insightface_glint360k.pth').to(device)
clip_image_processor = CLIPImageProcessor()

# "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
net_vision_encoder = CLIPVisionModel.from_pretrained('./checkpoints/vision_encoder').to(device)
# net_vision_encoder.vision_model.post_layernorm.requires_grad_(False)
net_image2token = Image2Token(visual_hidden_size=net_vision_encoder.vision_model.config.hidden_size, text_hidden_size=768, max_length=77, num_layers=3).to(device)
net_image2token.load_state_dict(torch.load('./checkpoints/net_image2token.pth'))
    
net_id2token = ID2Token(id_dim=512, text_hidden_size=768, max_length=77, num_layers=3).to(device)
net_id2token.load_state_dict(torch.load('./checkpoints/net_id2token.pth'))

net_seg_res18 = model_seg_unet.UNet().eval().to(device)
net_seg_res18.load_state_dict(torch.load('./checkpoints/net_seg_res18.pth'))
app = FaceAnalysis(name='antelopev2', root='./checkpoints/third_party', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def remove_tips():
    return gr.update(visible=False)

def get_example():
    case = [
        [
            "./examples/altman9.png",
            "./examples/lecun.jpg",
        ],
        [
            "./examples/bengio.jpg",
            "./examples/sheeran.png",
        ],
        [
            "./examples/emma.jpeg",
            "./examples/beyonce.jpg",
        ],
        [
            "./examples/lecun3.jpg",
            "./examples/smith.jpeg",
        ],
    ]
    return case

def run_for_examples(face_file, pose_file):
    return generate_image(
        face_file,
        pose_file,
        num_steps,
        guidance_scale,
        seed,
    )


def generate_image(
    src_img_path,
    drive_img_path,
    num_steps,
    guidance_scale,
    seed,
    crop_ratio = 0.81,
    progress=gr.Progress(track_tqdm=True),
):

    
    src_im_pil = Image.open(src_img_path).convert("RGB")
    
    # ===== insightface crop and detect 5pts
    face_info = app.get(cv2.cvtColor(np.array(src_im_pil), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    dets = face_info['bbox']
    
    # scaled box 
    if crop_ratio>0:
        bbox = dets[0:4]
        bbox_size = max(bbox[2]-bbox[0], bbox[2]-bbox[0])
        bbox_x = 0.5*(bbox[2]+bbox[0])
        bbox_y = 0.5*(bbox[3]+bbox[1])
        x1 = bbox_x-bbox_size*crop_ratio
        x2 = bbox_x+bbox_size*crop_ratio
        y1 = bbox_y-bbox_size*crop_ratio
        y2 = bbox_y+bbox_size*crop_ratio
        bbox_pts4 = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]], dtype=np.float32)   
    else:
    # original box
        bbox = dets[0:4].reshape((2,2))
        bbox_pts4 = datasets_faceswap.get_box_lm4p(bbox)        

    warp_mat_crop = datasets_faceswap.transformation_from_points(bbox_pts4, datasets_faceswap.mean_box_lm4p_512)
    src_im_crop512 = cv2.warpAffine(np.array(src_im_pil), warp_mat_crop, (512, 512), flags=cv2.INTER_LINEAR)
    src_im_pil = Image.fromarray(src_im_crop512)
    face_info = app.get(cv2.cvtColor(np.array(src_im_pil), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    pts5 = face_info['kps']  
    warp_mat = datasets_faceswap.get_affine_transform(pts5, datasets_faceswap.mean_face_lm5p_256)
    src_im_crop256 = cv2.warpAffine(np.array(src_im_pil), warp_mat, (256, 256), flags=cv2.INTER_LINEAR)
    # ======

    src_im_crop256_pil = Image.fromarray(src_im_crop256)
    image_src_crop256 = pil2tensor(src_im_crop256_pil).view(1,3,256,256).to(device)
    images_src = pil2tensor(src_im_pil).view(1,3,test_image_size,test_image_size).to(device)
    clip_input_src_tensors = clip_image_processor(images=src_im_pil, return_tensors="pt").pixel_values.view(-1, 3, 224, 224).to(device)
    
    
    drive_im_pil = Image.open(drive_img_path).convert("RGB")   
    
    # ===== insightface crop and detect 5pts
    face_info = app.get(cv2.cvtColor(np.array(drive_im_pil), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    dets = face_info['bbox']
    
    # scaled box 
    if crop_ratio>0:
        bbox = dets[0:4]
        bbox_size = max(bbox[2]-bbox[0], bbox[2]-bbox[0])
        bbox_x = 0.5*(bbox[2]+bbox[0])
        bbox_y = 0.5*(bbox[3]+bbox[1])
        x1 = bbox_x-bbox_size*crop_ratio
        x2 = bbox_x+bbox_size*crop_ratio
        y1 = bbox_y-bbox_size*crop_ratio
        y2 = bbox_y+bbox_size*crop_ratio
        bbox_pts4 = np.array([[x1,y1],[x1,y2],[x2,y2],[x2,y1]], dtype=np.float32)   
    else:
    # original box
        bbox = dets[0:4].reshape((2,2))
        bbox_pts4 = datasets_faceswap.get_box_lm4p(bbox)

    warp_mat_crop = datasets_faceswap.transformation_from_points(bbox_pts4, datasets_faceswap.mean_box_lm4p_512)
    drive_im_crop512 = cv2.warpAffine(np.array(drive_im_pil), warp_mat_crop, (512, 512), flags=cv2.INTER_LINEAR)
    drive_im_pil = Image.fromarray(drive_im_crop512)
    
    face_info = app.get(cv2.cvtColor(np.array(drive_im_pil), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    pts5 = face_info['kps']        
    
    warp_mat = datasets_faceswap.get_affine_transform(pts5, datasets_faceswap.mean_face_lm5p_256)
    drive_im_crop256 = cv2.warpAffine(np.array(drive_im_pil), warp_mat, (256, 256), flags=cv2.INTER_LINEAR)            
    
    drive_im_crop256_pil = Image.fromarray(drive_im_crop256)
    image_tar_crop256 = pil2tensor(drive_im_crop256_pil).view(1,3,256,256).to(device)
    image_tar_warpmat256 = warp_mat.reshape((1,2,3))
    images_tar = pil2tensor(drive_im_pil).view(1,3,test_image_size,test_image_size).to(device)
    clip_input_tar_tensors = clip_image_processor(images=drive_im_pil, return_tensors="pt").pixel_values.view(-1, 3, 224, 224).to(device)
    
    src_d3d_coeff = net_d3dfr(image_src_crop256)
    gt_d3d_coeff = net_d3dfr(image_tar_crop256)
    gt_d3d_coeff[:, 0:80] = src_d3d_coeff[:, 0:80]
    gt_pts68 = bfm_facemodel.get_lm68(gt_d3d_coeff)
    
    im_pts70 = draw_pts70_batch(gt_pts68, gt_d3d_coeff[:, 257:], image_tar_warpmat256, test_image_size, return_pt=True)
    im_pts70 = im_pts70.to(images_tar)
    face_masks_tar = (net_seg_res18(torch.cat([images_src, im_pts70], dim=1))>0.5).float()
    controlnet_image = im_pts70*face_masks_tar + images_src*(1-face_masks_tar)  # tar for reconstruction
    controlnet_image = controlnet_image.to(dtype=weight_dtype)
    
    face_masks_tar_pad = F.pad(face_masks_tar, (16,16,16,16), "constant", 0)
    blend_mask = F.max_pool2d(face_masks_tar_pad, kernel_size=17, stride=1, padding=8)
    blend_mask = F.avg_pool2d(blend_mask, kernel_size=17, stride=1, padding=8)
    blend_mask = blend_mask[:,:,16:528,16:528]
    
    faceid = net_arcface(F.interpolate(image_src_crop256, [128,128], mode='bilinear'))
    encoder_hidden_states_src = net_id2token(faceid).to(dtype=weight_dtype)
                
    last_hidden_state = net_vision_encoder(clip_input_src_tensors).last_hidden_state
    controlnet_encoder_hidden_states_src = net_image2token(last_hidden_state).to(dtype=weight_dtype)
    
    last_hidden_state = net_vision_encoder(clip_input_tar_tensors).last_hidden_state
    controlnet_encoder_hidden_states_tar = net_image2token(last_hidden_state).to(dtype=weight_dtype)

    empty_prompt_token = torch.load('empty_prompt_embedding.pth').view(1, 77,768).to(dtype=weight_dtype).to(device)

    set_seed(seed)
    generator = torch.manual_seed(0)
    image = pipe(
        prompt_embeds=encoder_hidden_states_src, negative_prompt_embeds=empty_prompt_token,
        controlnet_prompt_embeds=controlnet_encoder_hidden_states_src, controlnet_negative_prompt_embeds=empty_prompt_token, 
        image=controlnet_image,
        num_inference_steps=num_steps, generator=generator, guidance_scale=guidance_scale
    ).images[0]
    
    res_tensor = pil2tensor(image).view(1,3,test_image_size,test_image_size).to(images_tar)
    res_tensor = res_tensor*blend_mask + images_src*(1-blend_mask)
    ##########  face reenactment  ##########
    
    ##########    face swapping   ########## 
    accurate_mask_path = './checkpoints/mask/' + drive_img_path.split('.')[0] + '.png'
    if os.path.isfile(accurate_mask_path):
        print('use precomputed mask')
        mask_image_loaded = Image.open(accurate_mask_path)
        mask_numpy_loaded = (np.array(mask_image_loaded).astype(np.float32) * (20.0 / 255))
        masks_tar = torch.round(torch.from_numpy(mask_numpy_loaded).unsqueeze(0).unsqueeze(0)).long().to(res_tensor)            
        face_masks_tar = ((masks_tar>1)*(masks_tar<12)).float()
        face_masks_tar_withear = ((masks_tar>1)*(masks_tar<14)).float()
        occ_mask = ((masks_tar>14)*(masks_tar<20)).float()
    else:
        #  (0, 'background'), (1, 'skin'),
        #  (2, 'l_brow'), (3, 'r_brow'), (4, 'l_eye'), (5, 'r_eye'),
        #  (6, 'eye_g (eye glasses)'), (7, 'l_ear'), (8, 'r_ear'), (9, 'ear_r (ear ring)'),
        #  (10, 'nose'), (11, 'mouth'), (12, 'u_lip'), (13, 'l_lip'),
        #  (14, 'neck'), (15, 'neck_l (necklace)'), (16, 'cloth'),
        #  (17, 'hair'), (18, 'hat')
        net_seg = model_parsing.get_face_parsing('./checkpoints/third_party/79999_iter.pth').eval().to(device)        
        seg_pred = net_seg(images_tar)[0]
        masks_tar = torch.argmax(F.interpolate(seg_pred, [test_image_size, test_image_size], mode='bilinear'), dim=1, keepdim=True) # torch.Size([1, 1, 512, 512]) 0-20
        face_masks_tar = torch.logical_or((masks_tar>0)*(masks_tar<7), (masks_tar>9)*(masks_tar<14)).float() - (masks_tar==6).float()
        face_masks_tar_withear = ((masks_tar>0)*(masks_tar<14)).float() - (masks_tar==9).float()- (masks_tar==6).float()
        occ_mask = ((masks_tar==6)+(masks_tar==9)+(masks_tar==15)+(masks_tar==18)).float() 

    face_masks_tar = torch.max(face_masks_tar_withear, F.max_pool2d(face_masks_tar, kernel_size=65, stride=1, padding=32))
    face_masks_tar = face_masks_tar*(1-occ_mask)
    face_masks_tar = F.max_pool2d(face_masks_tar, kernel_size=5, stride=1, padding=2)
    
    face_masks_tar_pad = F.pad(face_masks_tar, (16,16,16,16), "constant", 0)
    blend_mask = F.max_pool2d(face_masks_tar_pad, kernel_size=17, stride=1, padding=8)
    blend_mask = F.avg_pool2d(blend_mask, kernel_size=17, stride=1, padding=8)
    blend_mask = blend_mask[:,:,16:528,16:528]

    controlnet_image_swap = im_pts70*face_masks_tar + images_tar*(1-face_masks_tar)
    controlnet_image_swap = controlnet_image_swap.to(dtype=weight_dtype)
    
    generator = torch.manual_seed(0)
    image = pipe(
        prompt_embeds=encoder_hidden_states_src, negative_prompt_embeds=empty_prompt_token,
        controlnet_prompt_embeds=controlnet_encoder_hidden_states_tar, controlnet_negative_prompt_embeds=empty_prompt_token, 
        image=controlnet_image_swap,
        num_inference_steps=num_steps, generator=generator, guidance_scale=guidance_scale
    ).images[0]
                
    swap_res_tensor = pil2tensor(image).view(1,3,test_image_size,test_image_size).to(images_tar)
    swap_res_tensor = swap_res_tensor*blend_mask + images_tar*(1-blend_mask)
    ##########    face swapping   ##########


    # concated results
    im_rgb_pil = Image.fromarray(convert_batch_to_nprgb(torch.cat([res_tensor, swap_res_tensor]), 2))

    return im_rgb_pil, gr.update(visible=True)


# Description
title = r"""
<h1 align="center">            <span style="background: linear-gradient(to right,  indigo, skyblue,  indigo, violet, indigo); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Face Adapter
            </span> for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/FaceAdapter/Face-Adapter' target='_blank'><b>Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control</b></a>.<br>
"""

article = r"""
---
📝 **Citation**
<br>
If our work is helpful for your research or applications, please cite us via:
```bibtex
@article{han2024face,
  title={Face Adapter for Pre-Trained Diffusion Models with Fine-Grained ID and Attribute Control},
  author={Han, Yue and Zhu, Junwei and He, Keke and Chen, Xu and Ge, Yanhao and Li, Wei and Li, Xiangtai and Zhang, Jiangning and Wang, Chengjie and Liu, Yong},
  journal={arXiv preprint arXiv:2405.12970},
  year={2024}
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to open an issue or directly reach us out at <b>12432015@zju.edu.cn</b>.
"""


tips = r"""
## Usage tips 
1. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
"""

css = """
.gradio-container {width: 85% !important}
"""


with gr.Blocks(css=css) as demo:
    # description
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            with gr.Row(equal_height=True):
                # upload face image
                face_file = gr.Image(
                    label="Upload a source image", type="filepath"
                )
                # optional: upload a reference pose image
                pose_file = gr.Image(
                    label="Upload a reference/target image",
                    type="filepath",
                )

            # prompt
            # prompt = gr.Textbox(
            #     label="Prompt",
            #     info="Give simple prompt is enough to achieve good face fidelity",
            #     placeholder="A photo of a person",
            #     value="",
            # )

            submit = gr.Button("Submit", variant="primary")
            # enable_LCM = gr.Checkbox(
            #     label="Enable Fast Inference with LCM", value=enable_lcm_arg,
            #     info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
            # )
   


            with gr.Accordion(open=False, label="Advanced Options"):
                num_steps = gr.Slider(
                    label="Number of sample steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value= 30,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=20.0,
                    step=0.1,
                    value= 5.0,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )
                schedulers = [
                    "DEISMultistepScheduler",
                    "HeunDiscreteScheduler",
                    "EulerDiscreteScheduler",
                    "DPMSolverMultistepScheduler",
                    "DPMSolverMultistepScheduler-Karras",
                    "DPMSolverMultistepScheduler-Karras-SDE",
                ]
                scheduler = gr.Dropdown(
                    label="Schedulers",
                    choices=schedulers,
                    value="EulerDiscreteScheduler",
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)


        with gr.Column(scale=1):
            gallery = gr.Image(label="Generated Images")
            usage_tips = gr.Markdown(
                label="FaceAdapter Usage Tips", value=tips, visible=False
            )

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[
                face_file,
                pose_file,
                num_steps,
                guidance_scale,
                seed,
            ],
            outputs=[gallery, usage_tips],
        )

        # enable_LCM.input(
        #     fn=toggle_lcm_ui,
        #     inputs=[enable_LCM],
        #     outputs=[num_steps, guidance_scale],
        #     queue=False,
        # )

    gr.Examples(
        examples=get_example(),
        inputs=[face_file, pose_file],
        fn=run_for_examples,
        outputs=[gallery, usage_tips],
        cache_examples=True,
    )

    gr.Markdown(article)

demo.queue(api_open=False)
demo.launch()

