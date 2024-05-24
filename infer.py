import os
import sys
import cv2
import random
import argparse
import numpy as np
from PIL import Image

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

torch.set_grad_enabled(False)
test_image_size = 512
weight_dtype = torch.float16
device = 'cuda'

pil2tensor = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize(mean=0.5, std=0.5)])

def convert_batch_to_nprgb(batch, nrow):
    grid_tensor = ttf.make_grid(batch * 0.5 + 0.5, nrow=nrow)
    im_rgb = (255 * grid_tensor.permute(1, 2, 0).cpu().numpy()).astype('uint8')
    return im_rgb    

def infer(args):

    os.makedirs(args.output, exist_ok=True)
    save_drive_path = os.path.join(args.output, 'drive')
    os.makedirs(save_drive_path, exist_ok=True)
    save_swap_path = os.path.join(args.output, 'swap') 
    os.makedirs(save_swap_path, exist_ok=True)
    save_concat_path = os.path.join(args.output, 'concat')
    os.makedirs(save_concat_path, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(os.path.join(args.checkpoint, 'controlnet'), torch_dtype=weight_dtype).to(device)
    pipe = StableDiffusionFaceAdapterPipeline.from_pretrained(
        args.base_model, controlnet=controlnet, torch_dtype=weight_dtype, cache_dir=args.cache_dir if args.use_cache else None, local_files_only=args.use_cache, requires_safety_checker=False
    ).to(device)

    # pretrained unet
    pretrained_unet_path = os.path.join(args.checkpoint, 'pretrained_unet')
    if os.path.exists(pretrained_unet_path) and 'stable-diffusion' in args.base_model:
        pipe.unet = UNet2DConditionModel.from_pretrained(pretrained_unet_path, torch_dtype=weight_dtype).to(device)
        
    # ===speed up diffusion process with faster scheduler and memory optimization===
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # ===remove following line if xformers is not installed or when using Torch 2.0===
    # pipe.enable_xformers_memory_efficient_attention()
    # ===memory optimization===
    # pipe.enable_model_cpu_offload()
    
    vae_ft_mse = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir=args.cache_dir if args.use_cache else None, torch_dtype=weight_dtype, local_files_only=args.use_cache).to(device)
    pipe.vae = vae_ft_mse

    net_d3dfr = model_resnet_d3dfr.getd3dfr_res50(os.path.join(args.checkpoint, 'third_party/d3dfr_res50_nofc.pth')).eval().to(device)
    bfm_facemodel = bfm.BFM(focal=1015*256/224, image_size=256, bfm_model_path=os.path.join(args.checkpoint, 'third_party/BFM_model_front.mat')).to(device)

    net_arcface = model_insightface_backbone.getarcface(os.path.join(args.checkpoint, 'third_party/insightface_glint360k.pth')).to(device)
    clip_image_processor = CLIPImageProcessor()

    # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    net_vision_encoder = CLIPVisionModel.from_pretrained(os.path.join(args.checkpoint, 'vision_encoder')).to(device)
    # net_vision_encoder.vision_model.post_layernorm.requires_grad_(False)
    net_image2token = Image2Token(visual_hidden_size=net_vision_encoder.vision_model.config.hidden_size, text_hidden_size=768, max_length=77, num_layers=3).to(device)
    net_image2token.load_state_dict(torch.load(os.path.join(args.checkpoint, 'net_image2token.pth')))
        
    net_id2token = ID2Token(id_dim=512, text_hidden_size=768, max_length=77, num_layers=3).to(device)
    net_id2token.load_state_dict(torch.load(os.path.join(args.checkpoint, 'net_id2token.pth')))

    net_seg_res18 = model_seg_unet.UNet().eval().to(device)
    net_seg_res18.load_state_dict(torch.load(os.path.join(args.checkpoint, 'net_seg_res18.pth')))
    app = FaceAnalysis(name='antelopev2', root=os.path.join(args.checkpoint, 'third_party'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    src_img_list = [x for x in os.listdir(args.source) if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]
    src_img_list.sort()
    drive_img_list = [x for x in os.listdir(args.target) if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]

    for src_im_file in src_img_list:
        
        src_im_name = src_im_file.split('.')[0]
        src_img_path = os.path.join(args.source, src_im_file)
        src_im_pil = Image.open(src_img_path).convert("RGB")
        
        
        # ===== insightface detect 5pts
        face_info = app.get(cv2.cvtColor(np.array(src_im_pil), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
        dets = face_info['bbox']
        
        # scaled box 
        if args.crop_ratio>0:
            bbox = dets[0:4]
            bbox_size = max(bbox[2]-bbox[0], bbox[2]-bbox[0])
            bbox_x = 0.5*(bbox[2]+bbox[0])
            bbox_y = 0.5*(bbox[3]+bbox[1])
            x1 = bbox_x-bbox_size*args.crop_ratio
            x2 = bbox_x+bbox_size*args.crop_ratio
            y1 = bbox_y-bbox_size*args.crop_ratio
            y2 = bbox_y+bbox_size*args.crop_ratio
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
        
            
        for drive_im_file in drive_img_list:
                
            drive_im_name = drive_im_file.split('.')[0]
            drive_img_path = os.path.join(args.target, drive_im_file)
            drive_im_pil = Image.open(drive_img_path).convert("RGB")   
            
            # ===== insightface detect 5pts
            face_info = app.get(cv2.cvtColor(np.array(drive_im_pil), cv2.COLOR_RGB2BGR))
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
            dets = face_info['bbox']
            
            # scaled box 
            if args.crop_ratio>0:
                bbox = dets[0:4]
                bbox_size = max(bbox[2]-bbox[0], bbox[2]-bbox[0])
                bbox_x = 0.5*(bbox[2]+bbox[0])
                bbox_y = 0.5*(bbox[3]+bbox[1])
                x1 = bbox_x-bbox_size*args.crop_ratio
                x2 = bbox_x+bbox_size*args.crop_ratio
                y1 = bbox_y-bbox_size*args.crop_ratio
                y2 = bbox_y+bbox_size*args.crop_ratio
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

            set_seed(999)
            generator = torch.manual_seed(0)
            image = pipe(
                prompt_embeds=encoder_hidden_states_src, negative_prompt_embeds=empty_prompt_token,
                controlnet_prompt_embeds=controlnet_encoder_hidden_states_src, controlnet_negative_prompt_embeds=empty_prompt_token, 
                image=controlnet_image,
                num_inference_steps=25, generator=generator, guidance_scale=5.0
            ).images[0]
            
            res_tensor = pil2tensor(image).view(1,3,test_image_size,test_image_size).to(images_tar)
            res_tensor = res_tensor*blend_mask + images_src*(1-blend_mask)
            images_res_pil = Image.fromarray((res_tensor[0]*127.5+128).cpu().numpy().astype('uint8').transpose(1,2,0))
            images_res_pil.save(os.path.join(save_drive_path, src_im_file), quality=100)
            ##########  face reenactment  ##########
            
            ##########    face swapping   ########## 
            accurate_mask_path = os.path.join(args.target, 'mask', drive_im_name+'.png')
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
                net_seg = model_parsing.get_face_parsing(os.path.join(args.checkpoint, 'third_party/79999_iter.pth')).eval().to(device)        
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
                num_inference_steps=25, generator=generator, guidance_scale=5.0
            ).images[0]
                        
            swap_res_tensor = pil2tensor(image).view(1,3,test_image_size,test_image_size).to(images_tar)
            swap_res_tensor = swap_res_tensor*blend_mask + images_tar*(1-blend_mask)
            ##########    face swapping   ##########


            # concated results
            im_rgb_pil = Image.fromarray(convert_batch_to_nprgb(torch.cat([images_src, images_tar, res_tensor, swap_res_tensor]), 4))
            im_rgb_pil.save(os.path.join(save_concat_path, f'{src_im_name}_{drive_im_name}.jpg'), quality=100)
            images_res_pil = Image.fromarray((swap_res_tensor[0]*127.5+128).cpu().numpy().astype('uint8').transpose(1,2,0))
            images_res_pil.save(os.path.join(save_swap_path, src_im_file), quality=100)
        

        
        

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', type=str, default='./checkpoints')
    parser.add_argument('-o', '--output', type=str, default='./output')
    parser.add_argument('-s', '--source', type=str, default='./example/src')
    parser.add_argument('-t', '--target', type=str, default='./example/tgt')
    parser.add_argument('-r', '--crop_ratio', type=float, default=0.81)
    parser.add_argument('-d', '--cache_dir', type=str, default='./hub')
    parser.add_argument('-b', '--base_model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('-c', '--use_cache', action='store_true')
    args = parser.parse_args()
    infer(args)
   