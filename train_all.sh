#!bin/bash

scenes=("kitchen" "ramen" "teatime" "0085" "0114" "0616" "0617")

base_video_path="/home/lff/data1/cjw/langscene/outputs"
base_data_path="/home/lff/data1/cjw/langscene/field_construction/data"
base_out_path="/home/lff/data1/cjw/langscene/field_construction/outputs"

for scene in "${scenes[@]}"; do
    scene_video_path="${base_video_path}/${scene}"
    python entry_point.py \
    pipeline.rgb_video_path="${scene_video_path}/rgb/video_ckpt_800.mp4" \
    pipeline.normal_video_path="${scene_video_path}/normal/video_ckpt_800.mp4" \
    pipeline.seg_video_path="${scene_video_path}/seg/video_ckpt_800.mp4" \
    pipeline.data_path="${base_data_path}/${scene}" \
    gaussian.dataset.source_path="${base_data_path}/${scene}" \
    gaussian.dataset.model_path="${base_out_path}/${scene}" \
    pipeline.selection=False \
    gaussian.opt.max_geo_iter=1500 \
    gaussian.opt.normal_optim=False \
    gaussian.opt.optim_pose=True \
    pipeline.skip_video_process=True \
    pipeline.skip_lang_feature_extraction=True 
    # pipeline.mode="render"
done
