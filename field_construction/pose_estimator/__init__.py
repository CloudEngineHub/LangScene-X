import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from utils.sfm_utils import (compute_co_vis_masks, get_sorted_image_files,
                             load_images, save_extrinsic, save_intrinsics,
                             save_points3D)

from .utils import prepare_input, prepare_output, storePly


class BaseEstimator(ABC):
    @abstractmethod
    def get_poses():
        pass


class ColmapEstimator(BaseEstimator):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def get_poses(self, camera_model="OPENCV", use_gpu=True):
        save_path = self.cfg.pipeline.data_path
        database_path = os.path.join(save_path, "distorted", "database.db")
        raw_img_path = os.path.join(save_path, "input")
        sparse_path = os.path.join(save_path, "distorted", "sparse")
        os.makedirs(os.path.join(save_path, "distorted"), exist_ok=True)
        os.makedirs(sparse_path, exist_ok=True)

        feat_extraction_cmd = [
            "colmap", "feature_extractor", 
            "--database_path", database_path,
            "--image_path", raw_img_path,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", camera_model,
            "--SiftExtraction.use_gpu", str(int(use_gpu))
        ]
        feat_extraction_cmd = " ".join(feat_extraction_cmd)
        exit_code = os.system(feat_extraction_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        feat_matching_cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
            "--SiftMatching.use_gpu", str(int(use_gpu))
        ]
        feat_matching_cmd = " ".join(feat_matching_cmd)
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        mapper_cmd = [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", raw_img_path,
            "--output_path", sparse_path,
            "--Mapper.ba_global_function_tolerance=0.000001"
        ]
        mapper_cmd = " ".join(mapper_cmd)
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

        img_undist_cmd = [
            "colmap", "image_undistorter",
            "--image_path", raw_img_path,
            "--input_path", os.path.join(sparse_path, "0"),
            "--output_path", save_path,
            "--output_type", "COLMAP"
        ]
        img_undist_cmd = " ".join(img_undist_cmd)
        exit_code = os.system(img_undist_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)
        
        # move data:
        curr_path = os.path.join(save_path, "sparse")
        dest_path = os.path.join(curr_path, "0")  
        os.makedirs(dest_path, exist_ok=True)
        files = list(filter(lambda x: x != "0", os.listdir(curr_path)))
        for file in files:
            src_file = os.path.join(curr_path, file)
            dest_file = os.path.join(dest_path, file)
            shutil.move(src_file, dest_file)


class MASt3REstimator(BaseEstimator):
    def __init__(self, cfg):
        from mast3r.model import AsymmetricMASt3R
        self.cfg = cfg
        self.device = cfg.pose_estimator.device
        self.model = AsymmetricMASt3R.from_pretrained(cfg.pose_estimator.model_path).to(self.device)

    def get_poses(self):
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.device import to_numpy
        from dust3r.utils.geometry import inv

        save_path = self.cfg.pipeline.data_path
        co_vis_dsp = self.cfg.pose_estimator.co_vis_dsp
        sparse_path = os.path.join(save_path, "sparse", "0")
        os.makedirs(sparse_path, exist_ok=True)
        image_dir = Path(save_path) / "input"
        image_files, image_suffix = get_sorted_image_files(image_dir)
        n_views = len(image_files)
        images, org_imgs_shape = load_images(image_files, size=512)

        logging.info(">> Making pairs...")
        pairs = make_pairs(images)
        logging.info(">> Inference...")
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=True)
        logging.info(f'>> Global alignment...')
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)

        extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
        intrinsics = to_numpy(scene.get_intrinsics())
        focals = to_numpy(scene.get_focals())
        imgs = np.array(scene.imgs)
        pts3d = to_numpy(scene.get_pts3d())
        pts3d = np.array(pts3d)
        depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
        values = [param.detach().cpu().numpy() for param in scene.im_conf]
        confs = np.array(values)
        
        logging.info(f'>> Confiden-aware Ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        sorted_conf_avg_conf_scores = avg_conf_scores[sorted_conf_indices]
        logging.info("Sorted indices:", str(sorted_conf_indices))
        logging.info("Sorted average confidence scores:", str(sorted_conf_avg_conf_scores))

        logging.info(f'>> Calculate the co-visibility mask...')
        depth_thre = self.cfg.pose_estimator.depth_thre
        if  depth_thre > 0:
            overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
            overlapping_masks = ~overlapping_masks
        else:
            co_vis_dsp = False
            overlapping_masks = None

        focals = np.repeat(focals[0], n_views)
        logging.info(f'>> Saving results...')
        save_extrinsic(Path(sparse_path), extrinsics_w2c, image_files, image_suffix)
        save_intrinsics(Path(sparse_path), focals, org_imgs_shape, imgs.shape, save_focals=True)
        pts_num = save_points3D(Path(sparse_path), imgs, pts3d, confs.reshape(pts3d.shape[0], -1), overlapping_masks, use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=save_path, depth_threshold=depth_thre)
        # save_images_and_masks(Path(sparse_path), n_views, imgs, overlapping_masks, image_files, image_suffix)
        logging.info(f'MASt3R Reconstruction is successfully converted to COLMAP files in: {sparse_path}')
        logging.info(f'Number of points: {pts3d.reshape(-1, 3).shape[0]}')    
        logging.info(f'Number of points after downsampling: {pts_num}')
            
    
class CUT3REstimator(BaseEstimator):
    def __init__(self, cfg):
        self.cfg = cfg  
        self.device = cfg.pose_estimator.device

    def get_poses(self):
        cfg = self.cfg
        if self.device == "cuda" and not torch.cuda.is_available():
            print("cuda not available. switching to cpu.")
            self.device = "cpu"
        
        from cut3r.dust3r.inference import inference
        from cut3r.dust3r.model import ARCroco3DStereo
        
        save_path = self.cfg.pipeline.data_path
        img_folder_path = os.path.join(save_path, "input")
        img_paths = [os.path.join(img_folder_path, img_name) for img_name in os.listdir(img_folder_path)]
        img_mask = [True] * len(img_paths)
        views, orig_shape = prepare_input(
            img_paths=img_paths,
            img_mask=img_mask,
            size=512,
            revisit=1,
            update=True,
        )
        model = ARCroco3DStereo.from_pretrained(cfg.pose_estimator.model_path).to(self.device)
        model.eval()

        logging.info("Running inference...")
        start_time = time.time()
        outputs, state_args = inference(views, model, self.device)
        total_time = time.time() - start_time
        per_frame_time = total_time / len(views)
        print(
            f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
        )

        pts3ds_other, colors, conf, cam_dict = prepare_output(
            outputs, orig_shape, save_path, 1, True
        )
        conf = torch.cat(conf, dim=0)
        if self.cfg.pipeline.selection:
            conf_score = conf.mean(dim=(1, 2))
            chunk_num = self.cfg.pipeline.chunk_num
            keep_num_per_chunk = self.cfg.pipeline.keep_num_per_chunk
            conf_scores_tuple = conf_score.chunk(chunk_num)
            selected_idxs = []
            total_conf_len = 0
            for conf_scores_chunk in conf_scores_tuple:
                _, idxs = conf_scores_chunk.sort(descending=True)
                idxs = idxs[:keep_num_per_chunk]
                selected_idxs += [(idx + total_conf_len).item() for idx in idxs]
                total_conf_len += len(conf_scores_chunk)
            self.cfg.pipeline.selected_idxs = sorted(selected_idxs)
        
        pts3ds_to_save = [pts3ds_other[idx].cpu().numpy() for idx in self.cfg.pipeline.selected_idxs]
        colors_to_save = [colors[idx].cpu().numpy() for idx in self.cfg.pipeline.selected_idxs]
        all_pts3ds = np.stack(pts3ds_to_save).reshape(-1, 3)
        all_colors = np.stack(colors_to_save).reshape(-1, 3)
        storePly(os.path.join(save_path, "points3D.ply"), all_pts3ds, all_colors)

class VGGTEstimator(BaseEstimator):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.pose_estimator.device

    def get_poses(self):
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        cfg = self.cfg
        if self.device == "cuda" and not torch.cuda.is_available():
            print("cuda not available. switching to cpu.")
            self.device = "cpu"
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        logging.info("Loading vggt...")
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        save_path = self.cfg.pipeline.data_path
        img_folder_path = os.path.join(save_path, "input")
        img_paths = [os.path.join(img_folder_path, img_name) for img_name in os.listdir(img_folder_path)]
        images = load_and_preprocess_images(img_paths).to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            point_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0), 
                extrinsic.squeeze(0), 
                intrinsic.squeeze(0)
            )
            extrinsic, intrinsic = extrinsic.squeeze(), intrinsic.squeeze()
            extrinsics_w2c = torch.eye(4)[None].repeat(len(extrinsic), 1, 1)
            extrinsics_w2c[:, :3, :4] = extrinsic.cpu()
            extrinsics_w2c = extrinsics_w2c.cpu().numpy()
            intrinsics = intrinsic.cpu().numpy()

            scaled_y, scaled_x = images.shape[-2:]
            intrinsics[:, 0, 0] *= 720 / scaled_x
            intrinsics[:, 1, 1] *= 480 / scaled_y
            intrinsics[:, 0, 2] *= 720 / scaled_x
            intrinsics[:, 1, 2] *= 480 / scaled_y

            images = torch.stack([images[:, 0], images[:, -1]], dim=1)
            point_map = np.stack([point_map[0], point_map[-1]], axis=0)
            colors = images.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
            colors = colors.reshape(-1, 3)
            point_map = point_map.reshape(-1, 3).astype(np.float32)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_map)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(save_path, "points3D.ply"), pcd)
        camera_dir = os.path.join(save_path, "camera")
        os.makedirs(camera_dir, exist_ok=True)
        for i, (w2c, intrinsic) in enumerate(zip(extrinsics_w2c, intrinsics)):
            c2w = np.eye(4)
            c2w[:3, :3] = w2c[:3, :3].T
            c2w[:3, 3] = - w2c[:3, :3].T @ w2c[:3, 3]
            np.savez(
                os.path.join(camera_dir, f"{i+1:04d}.npz"), 
                pose=c2w,
                intrinsics=intrinsic
            )

def get_pose_estimator(cfg):
    POSE_ESTIMATOR = {
        "colmap": ColmapEstimator, 
        "mast3r": MASt3REstimator,
        "cut3r": CUT3REstimator,
        "vggt": VGGTEstimator,
    }
    return POSE_ESTIMATOR[cfg.pose_estimator.type](cfg)
