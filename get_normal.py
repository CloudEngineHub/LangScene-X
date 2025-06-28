import os
from argparse import ArgumentParser

import torch
import torchvision.io as io
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--video_process", action="store_true")
    return parser.parse_args()

def main():
    torch.manual_seed(42)
    args = parse_args()
    predictor = torch.hub.load(
        "Stable-X/StableNormal",
        "StableNormal",
        trust_repo=True,
        local_cache_dir="/home/lff/bigdata1/cjw/model_cache"
    )
    if not args.video_process:
        base_path = args.base_path
        img_names = os.listdir(os.path.join(base_path, "rgb"))
        for img_name in img_names:
            img = Image.open(os.path.join(base_path, "rgb", img_name))
            normal_img = predictor(img)
            normal_path = os.path.join(base_path, "normal")
            os.makedirs(normal_path, exist_ok=True)
            normal_img.save(os.path.join(normal_path, img_name))
    else:
        video_tensor, _, _ = io.read_video(args.base_path, pts_unit="sec")
        for frame_ind, frame in enumerate(tqdm(video_tensor)):
            normal_frame = predictor(Image.fromarray(frame.numpy()))
            normal_frame.save(os.path.join(args.normal_save_path, f"{frame_ind:04d}.png"))

if __name__ == "__main__":
    main()
