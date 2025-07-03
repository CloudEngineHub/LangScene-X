import argparse
import os

import safetensors
import torch
from diffusers.utils import export_to_video, load_image

from cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Video interpolation with different checkpoints')
    parser.add_argument('--model_path', type=str, help='Path to the base model')
    parser.add_argument('--base_ckpt_path', type=str, default=None, help='Base path for checkpoints')
    parser.add_argument('--output_dir', type=str, help='Directory for output videos')
    parser.add_argument('--first_image', type=str, help='Path to the first image')
    parser.add_argument('--last_image', type=str, help='Path to the last image')
    return parser.parse_args()

def main():
    args = parse_args()
    
    pipe = CogVideoXInterpolationPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    prompt = 'a 3D consistent video scene'

    # checkpoints = ['ori', 800] 
    checkpoints = [800]
    
    os.makedirs(args.output_dir, exist_ok=True)

    for ckpt_num in checkpoints:
        print(f"Processing checkpoint-{ckpt_num}")
        
        pipe = CogVideoXInterpolationPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16
        )
        
        if args.base_ckpt_path is not None:
            # ckpt_path = os.path.join(args.base_ckpt_path, f"checkpoint-{ckpt_num}")
            ckpt_path = args.base_ckpt_path
            
            state_dict = safetensors.torch.load_file(os.path.join(ckpt_path, "model.safetensors"))
            pipe.transformer.load_state_dict(state_dict)
        
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        first_image = load_image(args.first_image)
        last_image = load_image(args.last_image)
        
        videos = pipe(
            prompt=prompt,
            first_image=first_image,
            last_image=last_image,
            num_videos_per_prompt=50,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )
        video = videos[0]
        
        prefix = "ori" if ckpt_num == 'ori' else "video"
        output_path = os.path.join(args.output_dir, f"{prefix}_ckpt_{ckpt_num}.mp4")
        export_to_video(video[0], output_path, fps=8)
        print(f"{prefix}_ckpt_{ckpt_num}.mp4 saved")
        
        del pipe
        torch.cuda.empty_cache()


    print("All checkpoints processing completed!")

if __name__ == "__main__":
    main()