#!/bin/bash

first_image=$1
last_image=$2

sam_model_path="./sam_vit_h_4b8939.pth"
sam2_model_path="./sam2_hiera_large.pt"
model_path="./CogVideo-X"

mkdir -p demo/rgb
cp $first_image demo/rgb/0001.png
cp $last_image demo/rgb/0002.png

# get segmentation maps
python auto-seg/auto-mask-align.py \
    --sam1_checkpoint $sam_model_path \
    --sam2_checkpoint $sam2_model_path \
    --video_path demo/rgb \
    --output_dir demo/seg \
    --level "default"

# get normal maps
python get_normal.py --base_path demo

RGB video interpolation
python video_inference.py \
    --model_path $model_path \
    --output_dir demo/video/rgb \
    --first_image demo/rgb/0001.png \
    --last_image demo/rgb/0002.png
# Segmentation map video interpolation
python video_inference.py \
    --model_path $model_path \
    --output_dir demo/video/seg \
    --first_image demo/seg/0001.png \
    --last_image demo/seg/0002.png
# Normal map video interpolation
python video_inference.py \
    --model_path $model_path \
    --output_dir demo/video/normal \
    --first_image demo/normal/0001.png \
    --last_image demo/normal/0002.png

# 3DGS field construction
mkdir -p demo/data
cp demo/seg/colors.npy demo/data

python entry_point.py \
    pipeline.rgb_video_path="demo/video/rgb/video_ckpt_800.mp4" \
    pipeline.normal_video_path="demo/video/normal/video_ckpt_800.mp4" \
    pipeline.seg_video_path="demo/video/seg/video_ckpt_800.mp4" \
    pipeline.data_path="demo/data" \
    gaussian.dataset.source_path="demo/data" \
    gaussian.dataset.model_path="demo/output" \
    pipeline.selection=False \
    gaussian.opt.max_geo_iter=1500 \
    gaussian.opt.normal_optim=True \
    gaussian.opt.optim_pose=False \
    pipeline.skip_video_process=False \
    pipeline.skip_pose_estimate=False \
    pipeline.skip_lang_feature_extraction=False