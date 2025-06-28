#!/bin/bash

scenes=("case1")
types=("rgb" "normal" "seg")

data_base="inputs"
output_base="outputs"
model_path="/home/lff/bigdata1/cjw/CogvideoX-Interpolation"
base_ckpt="/home/lff/bigdata1/cjw/checkpoints/cogvideox_ft"

for scene in "${scenes[@]}"; do
    for type in "${types[@]}"; do
        input_dir="${data_base}/${scene}/${type}"
        first_img=$(ls -1v "$input_dir"/*.jpg | head -n 1)
        last_img=$(ls -1v "$input_dir"/*.jpg | tail -n 1)
        output_dir="${output_base}/${scene}/${type}"
        mkdir -p "$output_dir"
        python video_inference.py \
            --model_path "$model_path" \
            --output_dir "$output_dir" \
            --first_image "$first_img" \
            --last_image "$last_img" \
            --base_ckpt_path "$base_ckpt" \
    done
done