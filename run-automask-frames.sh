#!bin/bash

data_base="/home/lff/data1/cjw/inputs"
scenes=("kitchen")

for scene in "${scenes[@]}"; do
    input_dir="${data_base}/${scene}/rgb"
    output_dir="${data_base}/${scene}"
    python auto-mask-align.py \
    --video_path $input_dir \
    --output_dir $output_dir \
    --level "default"
done
