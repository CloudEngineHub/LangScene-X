#!/bin/bash

scenes=("teatime")

data_base="/home/lff/data1/cjw/langscene/das/inputs"

for scene in "${scenes[@]}"; do
    scene_path="${data_base}/${scene}"
    echo "Processing scene: $scene"
    python get_normal.py --base_path $scene_path
done