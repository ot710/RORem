from diffusers import AutoPipelineForInpainting
import numpy as np
from PIL import Image, ImageFilter
import torch
import gc
import os
from diffusers import UNet2DConditionModel
import argparse
from myutils.img_util import dilate_mask
from diffusers.utils import load_image
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--RORem_unet",
        type=str,
        default=None,
        required=True,
        help="Path to pretrain RORem Unet",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to the input images directory.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Path to the mask images directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the folder to save the removal results.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to the mask image.",
    )
    # This is used to restore vessel pixels inside disc (doesn't work well for now)
    # parser.add_argument(
    #     "--vessel_mask_path",
    #     type=str,
    #     default=None,
    #     help="Path to the vessel mask image.",
    # )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the removal result.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--resolution",
        default=512,
        type=int
    )
    parser.add_argument(
        "--dilate_size",
        default=20,
        type=int,
        help="dilate the mask"
    )
    parser.add_argument(
        "--blur_radius",
        default=0,
        type=int,
        help="apply box blur to the mask (radius of the square kernel)"
    )
    parser.add_argument(
        "--blur_sd",
        default=0,
        type=int,
        help="apply Gaussian blur to the mask (standard deviation of the Gaussian kernel)"
    )
    parser.add_argument(
        "--use_CFG",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="whether to enable CFG, can reduce the artifacts in the mask region in our final test"
    )
    args = parser.parse_args()

    return args

def load_pipeline(pretrain_path, checkpoint_path):
    # load pretrained SDXL-inpainting model
    pipe_edit = AutoPipelineForInpainting.from_pretrained(
        pretrain_path,
        torch_dtype=torch.float16,
    )

    # load RORem Unet
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path).to("cuda", dtype=torch.float16)
    print(f"Finish loading unet from {checkpoint_path}!!")

    pipe_edit.unet = unet
    pipe_edit.to("cuda")

    # disable the progress bar
    pipe_edit.set_progress_bar_config(disable=True)

    return pipe_edit

def process_single_image(args, pipe_edit):
    height = width = args.resolution
    image_name = args.image_path.split("/")[-1]
    if args.save_path is None:
        save_folder = "removal_result"
        os.makedirs(save_folder,exist_ok=True)
        args.save_path = f"{save_folder}/{image_name}"
    else:
        save_folder = os.path.dirname(args.save_path)
        os.makedirs(save_folder,exist_ok=True)
    input_image = load_image(args.image_path).resize((args.resolution,args.resolution))
    input_mask = load_image(args.mask_path).resize((args.resolution,args.resolution))
    # rescale mask image from 0-1 to 0-255
    mask_np = np.array(input_mask) # convert to np array
    mask_np = (mask_np > 0).astype(np.uint8) * 255 # scale
    input_mask = Image.fromarray(mask_np) # convert back to PIL
    if args.blur_radius != 0:
        input_mask = input_mask.filter(ImageFilter.BoxBlur(radius=args.blur_radius))
    elif args.blur_sd != 0:
        input_mask = input_mask.filter(ImageFilter.GaussianBlur(radius=args.blur_sd))
    if args.dilate_size != 0:
        input_mask = dilate_mask(input_mask,args.dilate_size)
    if not args.use_CFG:
        prompts = ""
        with torch.no_grad():
            Removal_result = pipe_edit(
                    prompt=prompts,
                    height=height,
                    width=width,
                    image=input_image,
                    mask_image=input_mask,
                    guidance_scale=1.,
                    num_inference_steps=50,  # steps between 15 and 30 also work well
                    strength=0.99,  # make sure to use `strength` below 1.0
                ).images[0]
    else:
        # we also find by adding these prompts, the model can work even better
        prompts = "smooth, 4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
        negative_prompts = "sharp edge, low quality, worst, bad proportions, extra finger, Deformed, disfigured, unclear background"
        with torch.no_grad():
            Removal_result = pipe_edit(
                    prompt=prompts,
                    negative_prompt=negative_prompts,
                    height=height,
                    width=width,
                    image=input_image,
                    mask_image=input_mask,
                    guidance_scale=1.,
                    num_inference_steps=30,  # steps between 15 and 30 also work well
                    strength=0.3, # make sure to use `strength` below 1.0
                ).images[0]

    Removal_result.save(args.save_path)
    # prevent memory accumulation
    del input_image, input_mask, Removal_result
    gc.collect()
    torch.cuda.empty_cache()
    
    # # Restore original vessel pixels (not used because of light contamination from disc)
    # gen = np.array(Removal_result)
    # orig = np.array(input_image)
    # vessel_mask = load_image(args.vessel_mask_path).resize((args.resolution,args.resolution))
    # vessel = np.array(vessel_mask.convert("L")) # ensure greyscale and not RGB
    # vessel = (vessel > 0).astype(np.uint8) # binarize vessel mask
    # final = orig * vessel[..., None] + gen * (1 - vessel[..., None]) # keep generated only where vessel==0
    # final_image = Image.fromarray(final.astype(np.uint8)) # convert to PIL
    # final_image.save(args.save_path)

def main(args, pipe_edit):
    if args.image_dir is not None:
        image_paths = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.endswith('.png')])

        for i, img_path in enumerate(image_paths):
            step = max(1, len(image_paths) // 10) # print progress every 10%
            if i % step == 0:
                print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                print(f'[{i}/{len(image_paths)}] ({(i/len(image_paths))*100:.1f}%)')
            filename = os.path.basename(img_path)
            mask_path = os.path.join(args.mask_dir, filename)

            if not os.path.exists(mask_path):
                print(f'Skipping: {filename} (no mask found)')
                continue

            args.image_path = img_path
            args.mask_path = mask_path
            args.save_path = os.path.join(args.output_dir, filename)

            if os.path.exists(args.save_path): # avoid rerunning already done images
                continue

            try:
                process_single_image(args, pipe_edit)
            except Exception as e:
                print(f'Error on {filename}: {e}')
                continue

        return
    
    process_single_image(args, pipe_edit)

if __name__ == "__main__":
    args = parse_args()
    pipe_edit = load_pipeline(args.pretrained_model, args.RORem_unet)
    main(args, pipe_edit)