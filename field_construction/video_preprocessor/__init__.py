import logging
import os

import cv2
import ffmpeg
import numpy as np
import torch


class VideoPreprocessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def video_process(self):
        '''
        assuming the video is saved on somewhere. Need modified if video are given 
        as a obj directly.
        '''
        self.extract_frames(self.cfg.pipeline.rgb_video_path, "input")
        self.extract_frames(self.cfg.pipeline.normal_video_path, "normal")
        if self.cfg.feature_extractor.type == "open-seg":
            self.extract_masks("lang_features_dim3")
        elif self.cfg.feature_extractor.type == "lseg":
            self.extract_masks("lang_features_dim4")
        
    def extract_frames(self, video_path, file_name):
        img_save_path = os.path.join(self.cfg.pipeline.data_path, file_name)
        format = self.cfg.video_processor.img_format
        logging.info(f"Extracting frames from {video_path}...")
        os.makedirs(img_save_path, exist_ok=True)
        ffmpeg.input(video_path).output(os.path.join(img_save_path, f"%04d.{format}")).run(quiet=True)
            
    def extract_masks(self, save_dir_name):
        colors = np.load(os.path.join(self.cfg.pipeline.data_path, "colors.npy"))
        colors = torch.from_numpy(colors).to(dtype=torch.float32, device="cuda") # [n_masks, 3]
        colors /= 255 

        seg_video_path = self.cfg.pipeline.seg_video_path
        logging.info(f"Loading mask video from {seg_video_path}")
        seg_video = torch.from_numpy(self.load_from_video(seg_video_path)).to(dtype=torch.float32, device="cuda")
        seg_video /= 255

        for idx, frame in enumerate(seg_video):
            dist = ((frame.unsqueeze(-2) - colors[None, None, :, :]) ** 2).sum(dim=-1)
            mask = torch.argmin(dist, dim=-1) - 1 # -1 : background
            save_path = os.path.join(self.cfg.pipeline.data_path, save_dir_name)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, str(idx+1).zfill(4) + "_s.npy"), mask.cpu().numpy())
        
    def load_from_frames(self, frame_path):
        frame_list = os.listdir(frame_path)
        frame_list = sorted(filter(lambda x: x.split(".")[-1] in ["jpg", "png", "jpeg"], frame_list))
        all_frames = []
        for frame_name in frame_list:
            frame = cv2.imread(os.path.join(frame_path, frame_name))
            all_frames.append(frame)
        all_frames = np.array(all_frames)
        return all_frames
    
    def load_from_video(self, video_path):
        out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True)
        )
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])

        video_array = np.frombuffer(out, dtype=np.uint8)
        video_array = video_array.reshape((-1, height, width, 3))
        return video_array
