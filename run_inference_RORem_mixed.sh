#!/bin/bash

# bash command: nice nohup bash run_inference_RORem_mixed.sh >> ../logs/run_inference_RORem_mixed.log 2>&1 &

# IMAGE_PATH="/HDD/data/vascx/models_versions/v2/fundus/rgb/1018948_21016_0_0.png"
# MASK_PATH="/NVME/decrypted/scratch/olga/github-BergmannLab/fundus-structure-removal/rorem-mixed-inpainted-macula/masks/1018948_21016_0_0.png"
# DEBUG_PATH="../debug-overlays/1018948_21016_0_0_macula.png"

# # --- Debug overlay step ---
# python - <<EOF
# import numpy as np
# from PIL import Image

# image_path = "$IMAGE_PATH"
# mask_path = "$MASK_PATH"
# save_path = "$DEBUG_PATH"

# # Load image (RGB)
# img = np.array(Image.open(image_path).convert("RGB"))

# # Load mask (grayscale)
# mask = np.array(Image.open(mask_path).convert("L"))

# # --- Debug prints ---
# print("Image shape:", img.shape)
# print("Mask shape:", mask.shape)
# print("Mask unique values:", np.unique(mask))

# # Create overlay
# overlay = img.copy()
# overlay[mask == 1] = [0, 255, 0]  # green

# # Blend original + overlay (so it's semi-transparent)
# alpha = 0.5
# blended = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)

# Image.fromarray(blended).save(save_path)

# print(f"Saved debug overlay to: {save_path}")
# EOF


# Path to custom masks
# --mask_path /HDD2/data/olga/masks/vascx/models_versions/v2/fundus/discs_minus_av_reco/1018804_21016_0_0.png \
# Path to VascX masks
# --mask_path /HDD/data/vascx/models_versions/v2/fundus/discs/1018948_21016_0_0.png \
# Path to vessel masks
# --vessel_mask_path /HDD/data/vascx/models_versions/v2/fundus/av_reco/1018804_21016_0_0.png \


# # Run one image (for tests)
# python inference_RORem.py \
#     --pretrained_model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
#     --RORem_unet /HDD2/data/olga/checkpoints/RORem-mixed \
#     --image_path /HDD/data/vascx/models_versions/v2/fundus/rgb/1018843_21016_1_0.png \
#     --mask_path /HDD/data/vascx/models_versions/v2/fundus/av_reco/1018843_21016_1_0.png \
#     --save_path ../tests/rorem-mixed-inpainted-vessels/1018843_21016_1_0_d10_1024_st30.png \
#     --resolution 1024 \
#     --use_CFG true \
#     --dilate_size 10 \
#     --blur_radius 0


# Run on all images
IMAGE_DIR="/HDD/data/vascx/models_versions/v2/fundus/rgb"
MASK_DIR="/HDD2/data/olga/masks/vascx/models_versions/v2/fundus/discs_minus_av_reco"
OUTPUT_DIR="/HDD2/data/olga/rorem-mixed-inpainted-disc"
CHECKPOINT="/HDD2/data/olga/checkpoints/RORem-mixed"

echo "Image directory: "$IMAGE_DIR
echo "Mask directory: "$MASK_DIR
echo "Output directory: "$OUTPUT_DIR
echo "Checkpoint directory: "$CHECKPOINT

mkdir -p "$OUTPUT_DIR"

set -x

python -u inference_RORem.py \
    --pretrained_model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --RORem_unet "$CHECKPOINT" \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution 512 \
    --use_CFG true \
    --dilate_size 0 \
    --blur_radius 0