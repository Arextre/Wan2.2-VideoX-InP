import os
import sys
import time

import cv2
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
import glob
from pathlib import Path

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, CLIPModel, WanT5EncoderModel,
                              Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunInpaintPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# GPU memory mode, which can be choosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "sequential_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold | Model Name          | threshold |
# | Wan2.1-T2V-1.3B     | 0.05~0.10 | Wan2.1-T2V-14B      | 0.10~0.15 | Wan2.1-I2V-14B-720P | 0.20~0.30 |
# | Wan2.1-I2V-14B-480P | 0.20~0.25 | Wan2.1-Fun-*-1.3B-* | 0.05~0.10 | Wan2.1-Fun-*-14B-*  | 0.20~0.30 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "./wan_civitai_5b.yaml"
# model path
model_name          = "/home/notebook/data/group/zhaoheng/pretrained_models/Wan2.2-Fun-5B-InP"
# transformer_path = "/home/notebook/data/group/zhaoheng/code/video/VideoX-Fun/videox_fun_work_dirs/train_pai_wan2.1_fun_v1.1_1.3b_inp/checkpoint-8000/transformer/diffusion_pytorch_model.safetensors"
# transformer_path = "/home/notebook/code/personal/S9060429/VideoX-Fun/train_output/wan2.2/no_person_cityscene_adapter_full/checkpoint-500/transformer/diffusion_pytorch_model*.safetensors"
transformer_path = None

# model_name          = "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-InP"
# transformer_path = "/home/notebook/data/group/zhaoheng/code/video/VideoX-Fun/videox_fun_work_dirs/train_pai_wan2.1_fun_v1.1_14b_inp/checkpoint-8000/transformer/diffusion_pytorch_model.safetensors"
fusionx_lora_path = None
# fusionx_lora_path = "/home/notebook/data/group/zhaoheng/pretrained_models/Wan2.1-14B-FusioniX/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc" # "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 3 

# Load pretrained model if need
# transformer_path    = None
transformer_path        = None
transformer_high_path   = None
vae_path            = None
lora_path           = None
lora_high_path          = None
# fusionx_lora_path  = None

# Other params
sample_size         = [512, 512] #[480, 832]
video_length        = 49 #81
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
# If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
validation_image_start  = "asset/1.png"
validation_image_end    = None

# prompts
# prompt              = "一只棕色的狗摇着头，坐在舒适房间里的浅色沙发上。在狗的后面，架子上有一幅镶框的画，周围是粉红色的花朵。房间里柔和温暖的灯光营造出舒适的氛围。"
# negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# prompt              = "High-quality inpainting of masked video regions, seamless integration with surroundings, maintaining strict temporal consistency and motion coherence. The inpainted area exhibits natural textures, lighting matching the original scene, and realistic dynamics. Preserve background structures and object interactions without artifacts. Professional video restoration, cinematic quality, 4K resolution."
# negative_prompt     = "blurry patches, flickering artifacts, inconsistent lighting, distorted geometry, floating objects, unnatural motion, abrupt transitions, visible seams, watermark, text overlay, low resolution, grainy noise, cartoonish style, surreal elements."
# prompt              = "Generate only static and clean background elements like scenery, architecture, sky, and ground. Do NOT include any humans, animals, moving vehicles, or objects—even if their shadows or reflections appear in the context. Remove all signs of humans, including shadows, reflections, or indirect cues. The result should look like no people or animals or moving vehicles were ever present, unpopulated scenery, empty landscape, no reflections, no shadows, no signs of life. The focus should be on the environment itself, with no distractions from living beings."
negative_prompt     = "shadow, silhouette, reflection, people, pedestrians, children, bicycles, cars, vehicles, body, limb, finger, skin, hair, hand, foot, leg, clothing, shoes, moving, movement, text subtitles, comics, ugly or broken visual elements, and corrupted text."
prompt              = ""

guidance_scale      = 0 #6.0
seed                = 43
num_inference_steps = 16 #50
lora_weight         = 0.55
lora_high_weight    = 0.55
# save_path           = "samples/wan-videos-fun-v1.1-1.3b-inpainting-8000-same-wo-prompt-union-mask-openvid-extend-prompt-nomain-50-wo-cfg"
save_path           = "samples/wan2.2-fun-InP"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
else:
    transformer_2 = None

if transformer_path is not None:
    print("-" * 70)
    print(f"From checkpoint: {transformer_path}")

    # Use glob to handle multiple safetensors files
    transformer_files = glob.glob(transformer_path)
    state_dict = {}

    for file in transformer_files:
        print(f"Loading weights from: {file}")
        if file.endswith("safetensors"):
            from safetensors.torch import load_file
            file_state_dict = load_file(file)
        else:
            file_state_dict = torch.load(file, map_location="cpu")

        # Merge weights into state_dict
        for key, value in file_state_dict.items():
            if key in state_dict:
                print(f"Warning: Duplicate key {key} found. Overwriting.")
            state_dict[key] = value

    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=True)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None:
    if transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = Wan2_2FunInpaintPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(transformer_2, ["modulation",], device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)
    if transformer_2 is not None:
        pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, sub_transformer_name="transformer_2")

# if fusionx_lora_path is not None:
#     merge_fusionX_lora(transformer, fusionx_lora_path, 1.0)
#     print('Load fusionx lora success')

# --------------------------------------------------------------------------- #
def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if input_video_path is not None:
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path

        input_video = torch.from_numpy(np.array(input_video))[:video_length]
        input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0)
        input_video = input_video / 255
        input_video_for_vis = input_video.clone()

        if validation_video_mask is not None:
            cap = cv2.VideoCapture(validation_video_mask)
            mask_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else max(1,int(original_fps // fps))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                    mask_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()

        input_video_mask = torch.from_numpy(np.array(mask_video))[:video_length]
        input_video_mask = input_video_mask.permute([3, 0, 1, 2]).unsqueeze(0)
        input_video_mask = input_video_mask[:,:1]

        # if validation_video_mask is not None:
        #     validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
        #     input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)
            
        #     input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
        #     input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
        #     input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
        # else:
        #     input_video_mask = torch.zeros_like(input_video[:, :1])
        #     input_video_mask[:, :, :] = 255
    else:
        input_video, input_video_mask = None, None


    # ------------------ for vis ---------------------
    masked_video_for_vis = input_video_path.replace('video.mp4', 'masked_video_nomain_vis.mp4')
    cap = cv2.VideoCapture(masked_video_for_vis)
    masked_video_for_vis = []

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 1 if fps is None else max(1,int(original_fps // fps))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
            masked_video_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
    cap.release()

    masked_video_for_vis = torch.from_numpy(np.array(masked_video_for_vis))[:video_length]
    masked_video_for_vis = masked_video_for_vis.permute([3, 0, 1, 2]).unsqueeze(0) / 255
    # ------------------ end ---------------------

    if ref_image is not None:
        if isinstance(ref_image, str):
            clip_image = Image.open(ref_image).convert("RGB")
        else:
            clip_image = Image.fromarray(np.array(ref_image, np.uint8))
    else:
        clip_image = None
    
    clip_image = input_video[0,:,0,:,:].permute(2, 1, 0) * 255

    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
    return input_video, input_video_mask, ref_image, clip_image, input_video_for_vis, masked_video_for_vis


# videos = glob.glob('/home/notebook/data/group/zhaoheng/code/video/VideoX-Fun/datasets/ai_remover_test_data/16_samples_49f_union_mask_nomain/*/video.mp4')
# videos = glob.glob('/home/notebook/data/group/zhaoheng/code/video/VideoX-Fun/datasets/ai_remover_test_data/50_samples_easy_49f_union_mask_nomain/*/video.mp4')
videos = glob.glob('./test_data/50_samples_easy_49f_union_mask_nomain/*/video.mp4')
for item in videos:
    print('*'*70)
    print('Processing: ', item)
    masked_video_path = item
    mask_path = masked_video_path.replace('video.mp4', 'mask_nomain.mp4')

    with torch.no_grad():
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

        if enable_riflex:
            pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)

        # input_video, input_video_mask, clip_image = get_image_to_video_latent(validation_image_start, validation_image_end, video_length=video_length, sample_size=sample_size)
        input_video, input_video_mask, clip_image, _,  input_video_for_vis, masked_video_for_vis = \
            get_video_to_video_latent(masked_video_path, video_length=video_length, sample_size=sample_size,
                                      validation_video_mask=mask_path)
        if input_video.shape[2] != 49:
            continue
        # print('input_video shape: ', input_video.shape)
        # print('input_video_mask shape: ', input_video_mask.shape)
        with open(masked_video_path.replace('video.mp4', 'qwen2vl_caption.txt'), 'r') as f:
            extend_prompt = f.readline()
        extend_prompt = extend_prompt.strip()

        sample = pipeline(
            # prompt + " " + extend_prompt, 
            extend_prompt, 
            
            num_frames = video_length,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            boundary = boundary,

            video      = input_video,
            mask_video   = input_video_mask,
            shift = shift,
        ).videos

    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

    def save_results():
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        index = len([path for path in os.listdir(save_path)]) + 1
        prefix = str(index).zfill(8)
        if video_length == 1:
            video_path = os.path.join(save_path, prefix + ".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(video_path)
        else:
            video_path = os.path.join(save_path, prefix + ".mp4")
            # concat raw video and masked video for vis
            tmp = torch.cat([input_video_for_vis, masked_video_for_vis, sample], dim=-1)
            save_videos_grid(tmp, video_path, fps=fps)

    if ulysses_degree * ring_degree > 1:
        import torch.distributed as dist
        if dist.get_rank() == 0:
            save_results()
    else:
        save_results()
