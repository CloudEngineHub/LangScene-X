#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy
import logging
import os
import random
from random import randint

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from cogvideox_interpolation.utils.colormaps import apply_pca_colormap

from .gaussian_renderer import render
from .scene import GaussianModel, Scene
from .scene.app_model import AppModel
from .scene.cameras import Camera
from .utils.camera_utils import gen_virtul_cam
from .utils.general_utils import safe_state
from .utils.graphics_utils import patch_offsets, patch_warp
from .utils.image_utils import psnr
from .utils.loss_utils import (get_img_grad_weight, get_loss_instance_group,
                               get_loss_semantic_group, l1_loss, lncc,
                               loss_cls_3d, ranking_loss, ssim)
from .utils.pose_utils import (get_camera_from_tensor, get_tensor_from_camera,
                               post_pose_process, quad2rotation)


def post_process_mesh(mesh, cluster_to_keep=3):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def permuted_pca(image):
    return apply_pca_colormap(image.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

def save_pose(path, quat_pose, train_cams):
    # Get camera IDs and convert quaternion poses to camera matrices
    camera_ids = [cam.colmap_id for cam in train_cams]
    world_to_camera = [get_camera_from_tensor(quat) for quat in quat_pose]
    
    # Reorder poses according to colmap IDs
    colmap_poses = []
    for i in range(len(camera_ids)):
        idx = camera_ids.index(i + 1)  # Find position of camera i+1
        pose = world_to_camera[idx]
        colmap_poses.append(pose)
    
    # Convert to numpy array and save
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def load_and_prepare_confidence(confidence_path, device='cuda', scale=(0.1, 1.0)):
    """
    Loads, normalizes, inverts, and scales confidence values to obtain learning rate modifiers.
    
    Args:
        confidence_path (str): Path to the .npy confidence file.
        device (str): Device to load the tensor onto.
        scale (tuple): Desired range for the learning rate modifiers.
    
    Returns:
        torch.Tensor: Learning rate modifiers.
    """
    # Load and normalize
    confidence_np = np.load(confidence_path)
    confidence_tensor = torch.from_numpy(confidence_np).float().to(device)
    normalized_confidence = torch.sigmoid(confidence_tensor)

    # Invert confidence and scale to desired range
    inverted_confidence = 1.0 - normalized_confidence
    min_scale, max_scale = scale
    lr_modifiers = inverted_confidence * (max_scale - min_scale) + min_scale
    
    return lr_modifiers

class GaussianField():
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        cfg = self.cfg
        dataset = cfg.gaussian.dataset
        opt = cfg.gaussian.opt
        pipe = cfg.gaussian.pipe
        device = cfg.gaussian.dataset.data_device

        self.gaussians = GaussianModel(cfg.gaussian.dataset.sh_degree)
        self.scene = Scene(cfg.gaussian.dataset, self.gaussians)
        self.app_model = AppModel()
        self.app_model.train().cuda()

        logging.info("Optimizing " + dataset.model_path)
        safe_state(cfg.gaussian.quiet)
        
        if opt.pp_optimizer:
            confidence_path = os.path.join(dataset.source_path, f"sparse/0", "confidence_dsp.npy")
            try:
                confidence_lr = load_and_prepare_confidence(confidence_path, device='cuda', scale=(2, 100))
                self.gaussians.training_setup_pp(opt, confidence_lr, device)                          
            except:
                logging.warning("can not load confidence. ")
                cfg.opt.pp_optimizer = False
                self.gaussians.training_setup(opt, device)
        else:
            self.gaussians.training_setup(opt, device)
        
        train_cams_init = self.scene.getTrainCameras().copy()
        for save_iter in cfg.gaussian.save_iterations:
            os.makedirs(self.scene.model_path + f'/pose/iter_{save_iter}', exist_ok=True)
            save_pose(self.scene.model_path + f'/pose/iter_{save_iter}/pose_org.npy', self.gaussians.P, train_cams_init)
        
        first_iter = 0
        if cfg.gaussian.start_checkpoint != "None":
            model_params, first_iter = torch.load(cfg.gaussian.start_checkpoint)
            self.gaussians.restore(model_params, opt)
            self.app_model.load_weights(self.scene.model_path)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        viewpoint_stack = None
        ema_loss_for_log = 0.0
        ema_single_view_for_log = 0.0
        ema_multi_view_geo_for_log = 0.0
        ema_multi_view_pho_for_log = 0.0
        ema_language_loss_for_log = 0.0
        ema_grouping_loss = 0.0
        ema_loss_obj_3d = 0.0
        ema_ins_grouping_loss = 0.0
        ema_ins_obj_3d_loss = 0.0
        normal_loss, geo_loss, ncc_loss = None, None, None
        language_loss = None
        grouping_loss = None
        include_feature = True
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
        first_iter += 1
        debug_path = os.path.join(self.scene.model_path, "debug")
        os.makedirs(debug_path, exist_ok=True)

        camera_list = self.scene.getTrainCameras().copy()
        last_cam_id = -1

        self.gaussians.change_reqiures_grad("semantic", iteration=first_iter, quiet=False)

        if not opt.optim_pose:
            self.gaussians.P.requires_grad_(False)

        for iteration in range(first_iter, opt.iterations + 1):
            iter_start.record()

            self.gaussians.update_learning_rate(iteration)

            if iteration % 100 == 0:
                self.gaussians.oneupSHdegree()
            
            if not viewpoint_stack:
                viewpoint_stack = camera_list.copy()

            # update camera lists: 
            for cam_idx, cam in enumerate(camera_list):
                if cam.uid == last_cam_id:
                    updated_pose = self.gaussians.get_RT(self.gaussians.index_mapping[last_cam_id]).clone().detach()
                    extrinsics = get_camera_from_tensor(updated_pose)
                    camera_list[cam_idx].R = extrinsics[:3, :3].T
                    camera_list[cam_idx].T = extrinsics[:3, 3]
                    break
            
            viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            last_cam_id = viewpoint_cam.uid
            pose = self.gaussians.get_RT(self.gaussians.index_mapping[last_cam_id]) # quad t

            if (iteration - 1) == cfg.gaussian.debug_from:
                pipe.debug = True
            
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            if not opt.optim_pose:
                render_pkg = render(viewpoint_cam, self.gaussians, pipe, bg, app_model=self.app_model,
                                    return_depth_normal=iteration > opt.single_view_weight_from_iter,
                                    include_feature=include_feature)
            else:
                render_pkg = render(viewpoint_cam, self.gaussians, pipe, bg, app_model=self.app_model,
                                    return_depth_normal=iteration > opt.single_view_weight_from_iter,
                                    include_feature=include_feature, camera_pose=pose)
            
            image, viewspace_point_tensor, visibility_filter, radii, language_feature, instance_feature = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], \
                    render_pkg["language_feature"], render_pkg["instance_feature"]
                

            overall_loss = 0
            image_loss = None
            obj_3d_loss = None
            grouping_loss = None
            ins_obj_3d_loss = None
            ins_grouping_loss = None

            if iteration == opt.max_geo_iter:
                self.gaussians.change_reqiures_grad("semantic_only", iteration=iteration, quiet=False)

            if iteration < opt.max_geo_iter:
                gt_image, gt_image_gray = viewpoint_cam.get_image()
                ssim_loss = (1.0 - ssim(image, gt_image))
                if 'app_image' in render_pkg and ssim_loss < 0.5:
                    app_image = render_pkg['app_image']
                    Ll1 = l1_loss(app_image, gt_image)
                else:
                    Ll1 = l1_loss(image, gt_image)
            
                image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

                overall_loss = overall_loss + image_loss

                # scale loss
                if visibility_filter.sum() > 0:
                    scale = self.gaussians.get_scaling[visibility_filter]
                    sorted_scale, _ = torch.sort(scale, dim=-1)
                    min_scale_loss = sorted_scale[..., 0]
                    overall_loss = overall_loss + opt.scale_loss_weight * min_scale_loss.mean()
                
                # single view loss:
                if opt.single_view_weight_from_iter < iteration < opt.single_view_weight_end_iter:
                    weight = opt.single_view_weight
                    normal = render_pkg["rendered_normal"]
                    depth_normal = render_pkg["depth_normal"]

                    image_weight = (1.0 - get_img_grad_weight(gt_image))
                    image_weight = (image_weight).clamp(0, 1).detach() ** 2

                    if opt.normal_optim:
                        render_normal = (normal.permute(1, 2, 0) @ (viewpoint_cam.world_view_transform[:3, :3].T)).permute(2, 0, 1)
                        rendered_depth_normal = (depth_normal.permute(1, 2, 0) @ (viewpoint_cam.world_view_transform[:3, :3].T)).permute(2, 0, 1)
                        normal_gt, normal_mask = viewpoint_cam.get_normal()
                        prior_normal = normal_gt
                        prior_normal_mask = normal_mask[0]
                        normal_prior_error = (1 - F.cosine_similarity(prior_normal, render_normal, dim=0)) + \
                            (1 - F.cosine_similarity(prior_normal, rendered_depth_normal, dim=0))
                        normal_prior_error = ranking_loss(normal_prior_error[prior_normal_mask], 
                                                        penalize_ratio=1.0, type="mean")
                        normal_loss = weight * normal_prior_error
                    else:
                        if not opt.wo_image_weight:
                            normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
                        else:
                            normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                    overall_loss = overall_loss + normal_loss
                
                # multi-view loss
                if opt.multi_view_weight_from_iter < iteration < opt.multi_view_weight_end_iter:
                    nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else camera_list[
                        random.sample(viewpoint_cam.nearest_id, 1)[0]]
                    use_virtul_cam = False
                    if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                        nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis,
                                                    deg_noise=dataset.multi_view_max_angle, device=device)
                        use_virtul_cam = True
                    if nearest_cam is not None:
                        patch_size = opt.multi_view_patch_size
                        sample_num = opt.multi_view_sample_num
                        pixel_noise_th = opt.multi_view_pixel_noise_th
                        total_patch_size = (patch_size * 2 + 1) ** 2
                        ncc_weight = opt.multi_view_ncc_weight
                        geo_weight = opt.multi_view_geo_weight
                        H, W = render_pkg['plane_depth'].squeeze().shape
                        ix, iy = torch.meshgrid(
                            torch.arange(W), torch.arange(H), indexing='xy')
                        pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)
                        if not use_virtul_cam:
                            nearest_pose = self.gaussians.get_RT(self.gaussians.index_mapping[nearest_cam.uid]) # quad t
                            if not opt.optim_pose:
                                nearest_render_pkg = render(nearest_cam, self.gaussians, pipe, bg, app_model=self.app_model,
                                                            return_plane=True, return_depth_normal=False)
                            else:
                                nearest_render_pkg = render(nearest_cam, self.gaussians, pipe, bg, app_model=self.app_model,
                                                            return_plane=True, return_depth_normal=False, camera_pose=nearest_pose.clone().detach())
                        else:
                            nearest_render_pkg = render(nearest_cam, self.gaussians, pipe, bg, app_model=self.app_model,
                                                        return_plane=True, return_depth_normal=False)
                        pts = self.gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                        pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,
                                                :3] + nearest_cam.world_view_transform[3, :3]
                        map_z, d_mask = self.gaussians.get_points_depth_in_depth_map(nearest_cam,
                                                                                nearest_render_pkg['plane_depth'],
                                                                                pts_in_nearest_cam)

                        pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                        pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                        R = torch.tensor(nearest_cam.R).float().cuda()
                        T = torch.tensor(nearest_cam.T).float().cuda()
                        pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
                        pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,
                                                :3] + viewpoint_cam.world_view_transform[3, :3]
                        pts_projections = torch.stack(
                            [pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy],
                            -1).float()
                        pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                        if not opt.wo_use_geo_occ_aware:
                            d_mask = d_mask & (pixel_noise < pixel_noise_th)
                            weights = (1.0 / torch.exp(pixel_noise)).detach()
                            weights[~d_mask] = 0
                        else:
                            d_mask = d_mask
                            weights = torch.ones_like(pixel_noise)
                            weights[~d_mask] = 0
                        if iteration % 200 == 0:
                            gt_img_show = ((gt_image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                        [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                            if 'app_image' in render_pkg:
                                img_show = ((render_pkg['app_image']).permute(1, 2, 0).clamp(0, 1)[:, :,
                                            [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                            else:
                                img_show = ((image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                            [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                            normal_show = (((normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,1) * 255).detach().cpu().numpy().astype(np.uint8)
                            depth_normal_show = (((depth_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,1) * 255).detach().cpu().numpy().astype(np.uint8)

                            if not opt.normal_optim:
                                normal_gt = torch.zeros_like(normal)
                
                            normal_gt_show = (normal_gt.permute(1, 2, 0) @ (viewpoint_cam.world_view_transform[:3, :3])).permute(2, 0, 1)
                            normal_gt_show = (((normal_gt_show + 1.0) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
                            d_mask_show = (weights.float() * 255).detach().cpu().numpy().astype(np.uint8).reshape(H, W)
                            d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                            depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                            depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                            depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                            depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                            distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                            distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                            distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                            distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                            image_weight = image_weight.detach().cpu().numpy()
                            image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                            image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                            row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                            row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, normal_gt_show],
                                                axis=1)
                            image_to_show = np.concatenate([row0, row1], axis=0)
                            cv2.imwrite(
                                os.path.join(debug_path, "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg"),
                                image_to_show)

                        if d_mask.sum() > 0:
                            geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                            overall_loss += geo_loss
                            if use_virtul_cam is False:
                                with torch.no_grad():
                                    # sample mask
                                    d_mask = d_mask.reshape(-1)
                                    valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                                    if d_mask.sum() > sample_num:
                                        index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                        valid_indices = valid_indices[index]

                                    weights = weights.reshape(-1)[valid_indices]
                                    # sample ref frame patch
                                    pixels = pixels.reshape(-1, 2)[valid_indices]
                                    offsets = patch_offsets(patch_size, pixels.device)
                                    ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                                    H, W = gt_image_gray.squeeze().shape
                                    pixels_patch = ori_pixels_patch.clone()
                                    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                                    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                                    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                                                align_corners=True)
                                    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                                    ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1,
                                                                                                        -2) @ viewpoint_cam.world_view_transform[
                                                                                                                :3, :3]
                                    ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,
                                                                            :3] + nearest_cam.world_view_transform[3, :3]

                                # compute Homography
                                ref_local_n = render_pkg["rendered_normal"].permute(1, 2, 0)
                                ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]
                                ref_local_d = render_pkg['rendered_distance'].squeeze()
                                ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                                H_ref_to_neareast = ref_to_neareast_r[None] - \
                                                    torch.matmul(
                                                        ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                                        ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(
                                                            0, 2, 1)) / ref_local_d[..., None, None]
                                H_ref_to_neareast = torch.matmul(
                                    nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
                                    H_ref_to_neareast)
                                H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                                # compute neareast frame patch
                                grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
                                grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                                grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                                _, nearest_image_gray = nearest_cam.get_image()
                                sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2),
                                                                align_corners=True)
                                sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                                # compute loss
                                ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                                mask = ncc_mask.reshape(-1)
                                ncc = ncc.reshape(-1) * weights
                                ncc = ncc[mask].squeeze()

                                if mask.sum() > 0:
                                    ncc_loss = ncc_weight * ncc.mean()
                                    overall_loss = overall_loss + ncc_loss

            if opt.lang_loss_start_iter <= iteration < opt.instance_supervision_from_iter:
                # language feature loss
                lf_path = os.path.join(dataset.source_path, dataset.language_features_name)
                gt_language_feature, language_feature_mask, gt_seg = viewpoint_cam.get_language_feature(lf_path)
                language_loss = l1_loss(language_feature * language_feature_mask,
                                        gt_language_feature * language_feature_mask)
                
                overall_loss = overall_loss + language_loss

                language_feature_mask = language_feature_mask.reshape(-1)
                if opt.grouping_loss:
                    grouping_loss = get_loss_semantic_group(gt_seg.reshape(-1)[language_feature_mask],
                                                            language_feature.permute(1, 2, 0).reshape(-1, 3)[
                                                            language_feature_mask])
                    overall_loss = overall_loss + grouping_loss
                if opt.loss_obj_3d:
                    obj_3d_loss = loss_cls_3d(self.gaussians._xyz.detach().squeeze(),
                                            self.gaussians._language_feature.squeeze(), opt.reg3d_k,
                                            opt.reg3d_lambda_val, 2000000, 800)
                    overall_loss += obj_3d_loss

            elif iteration >= opt.instance_supervision_from_iter:
                # change the grad mode and copy the semantic featuers into instance-level
                if iteration == opt.instance_supervision_from_iter:
                    self.gaussians._instance_feature.data.copy_(self.gaussians._language_feature.detach().clone())
                    self.gaussians.change_reqiures_grad("instance", iteration=iteration, quiet=False)
                _, language_feature_mask, gt_seg = viewpoint_cam.get_language_feature(lf_path)
                language_feature_mask = language_feature_mask.reshape(-1)
                # supervise the instance features
                if opt.grouping_loss:
                    ins_grouping_loss = get_loss_instance_group(gt_seg.reshape(-1)[language_feature_mask],
                                                                instance_feature.permute(1, 2, 0).reshape(-1, 3)[
                                                                    language_feature_mask],
                                                                language_feature.permute(1, 2, 0).reshape(-1, 3)[
                                                                    language_feature_mask])
                    overall_loss = overall_loss + ins_grouping_loss
                if opt.loss_obj_3d:
                    ins_obj_3d_loss = loss_cls_3d(self.gaussians._xyz.detach().squeeze(), self.gaussians._instance_feature.squeeze(),
                                                opt.reg3d_k, opt.reg3d_lambda_val, 2000000, 800)
                    overall_loss += ins_obj_3d_loss

            overall_loss.backward()
            iter_end.record()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log if image_loss is not None else 0.0 + 0.6 * ema_loss_for_log
                ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
                ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
                ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
                ema_language_loss_for_log = 0.4 * language_loss.item() if language_loss is not None else 0.0 + 0.6 * ema_language_loss_for_log
                ema_grouping_loss = 0.4 * grouping_loss.item() if grouping_loss is not None else 0.0 + 0.6 * ema_grouping_loss
                ema_loss_obj_3d = 0.4 * obj_3d_loss.item() if obj_3d_loss is not None else 0.0 + 0.6 * ema_loss_obj_3d

                ema_ins_obj_3d_loss = 0.4 * ins_obj_3d_loss.item() if ins_obj_3d_loss is not None else 0.0 + 0.6 * ema_ins_obj_3d_loss
                ema_ins_grouping_loss = 0.4 * ins_grouping_loss.item() if ins_grouping_loss is not None else 0.0 + 0.6 * ema_ins_grouping_loss
                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "Lang": f"{ema_language_loss_for_log:.{5}f}",
                        "Points": f"{len(self.gaussians.get_xyz)}",
                        "gp": f"{ema_grouping_loss:.{5}f}",
                        "3d": f"{ema_loss_obj_3d:.{5}f}",
                        "Ins": f"{ema_ins_grouping_loss:.{5}f}",
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                self.training_report(iteration, camera_list, l1_loss, render, (pipe, background))
                if (iteration in cfg.gaussian.save_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.scene.save(iteration, include_feature=include_feature)
                    save_pose(self.scene.model_path + f'/pose/iter_{iteration}/pose_optimized.npy', self.gaussians.P, train_cams_init)

                # Densification
                if iteration < min(opt.max_geo_iter, opt.densify_until_iter):
                    # Keep track of max radii in image-space for pruning
                    mask = (render_pkg["out_observe"] > 0) & visibility_filter
                    self.gaussians.max_radii2D[mask] = torch.max(self.gaussians.max_radii2D[mask], radii[mask])
                    viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                    self.gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if opt.densify_from_iter < iteration < min(opt.max_geo_iter, opt.densify_until_iter) and iteration % opt.densification_interval == 0:
                    logging.info("densifying and pruning...")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,
                                                opt.opacity_cull_threshold, self.scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    self.gaussians.reset_opacity()            

                if iteration < opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.cam_optimizer.step()
                    self.app_model.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.cam_optimizer.zero_grad(set_to_none=True)
                    self.app_model.optimizer.zero_grad(set_to_none=True)
                    
                if (iteration in cfg.gaussian.checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((self.gaussians.capture(include_feature=include_feature), iteration),
                            self.scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    self.app_model.save_weights(self.scene.model_path, iteration)

        self.app_model.save_weights(self.scene.model_path, opt.iterations)
        torch.cuda.empty_cache()
        # move camera poses to target path.       
        max_save_iter = max(cfg.gaussian.save_iterations)
        orig_path = self.scene.model_path + f'/pose/iter_{max_save_iter}/pose_optimized.npy'
        camera_path = os.path.join(cfg.pipeline.data_path, "camera")
        eg_file = os.listdir(camera_path)[0]
        logging.info("Post processing pose & move to data path...")
        post_pose_process(orig_path, os.path.join(camera_path, eg_file), os.path.join(cfg.pipeline.data_path, "render_camera"))


    def training_report(self, iteration, camera_list, l1_loss, renderFunc, renderArgs):
        # Report test and samples of training set
        # do not use the optimized poses. 
        if iteration in self.cfg.gaussian.test_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras': camera_list},
                                {'name': 'train',
                                'cameras': [self.scene.getTrainCameras()[idx % len(self.scene.getTrainCameras())] for idx in
                                            range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0.0
                    psnr_test = 0.0
                    for idx, viewpoint in enumerate(config['cameras']):
                        if self.cfg.gaussian.opt.optim_pose:
                            camera_pose = get_tensor_from_camera(viewpoint.world_view_transform.transpose(0, 1))
                            out = renderFunc(viewpoint, self.scene.gaussians, *renderArgs, app_model=self.app_model, camera_pose=camera_pose)
                        else:
                            out = renderFunc(viewpoint, self.scene.gaussians, *renderArgs, app_model=self.app_model)
                        image = out["render"]
                        if 'app_image' in out:
                            image = out['app_image']
                        image = torch.clamp(image, 0.0, 1.0)
                        gt_image, _ = viewpoint.get_image()
                        gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        img_show = ((image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                    [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                        img_gt_show = ((gt_image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                    [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                        img_tosave = np.concatenate([img_show, img_gt_show], axis=1)
                        valid_path = os.path.join(self.cfg.gaussian.dataset.model_path, "valid")
                        os.makedirs(valid_path, exist_ok=True)
                        cv2.imwrite(os.path.join(valid_path, f"{iteration}_{viewpoint.uid}.png"), img_tosave)

                    psnr_test /= len(config['cameras'])
                    l1_test /= len(config['cameras'])
                    logging.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            torch.cuda.empty_cache()
    
    
    def render(self):
        cfg = self.cfg
        dataset = cfg.gaussian.dataset
        pipe = cfg.gaussian.pipe
        device = cfg.gaussian.dataset.data_device
        render_cfg = cfg.gaussian.render

        logging.info("Rendering " + dataset.model_path)
        safe_state(cfg.gaussian.quiet)

        voxel_size = 0.01
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        volume_feature = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        with torch.no_grad():
            self.gaussians = GaussianModel(cfg.gaussian.dataset.sh_degree)
            self.scene = Scene(cfg.gaussian.dataset, self.gaussians, load_iteration=cfg.pipeline.load_iteration, shuffle=False)
            self.app_model = AppModel()

            self.scene.loaded_iter = None
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device=device)
            
            render_path = os.path.join(dataset.model_path, "test", "renders_rgb")
            render_depth_path = os.path.join(dataset.model_path, "test", "renders_depth")
            render_depth_npy_path = os.path.join(dataset.model_path, "test", "renders_depth_npy")
            render_normal_path = os.path.join(dataset.model_path, "test", "renders_normal")

            os.makedirs(render_path, exist_ok=True)
            os.makedirs(render_depth_path, exist_ok=True)
            os.makedirs(render_depth_npy_path, exist_ok=True)
            os.makedirs(render_normal_path, exist_ok=True)
            depths_tsdf_fusion = []

            all_language_feature = []
            all_gt_language_feature = []
            all_instance_feature = []
            for idx, view in enumerate(tqdm(self.scene.getTrainCameras(), desc="Rendering progress")):
                camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
                gt, _ = view.get_image()
                out = render(view, self.gaussians, pipe, background, app_model=None, camera_pose=camera_pose)
                rendering = out["render"].clamp(0.0, 1.0)
                _, H, W = rendering.shape

                depth = out["plane_depth"].squeeze()
                depth_tsdf = depth.clone()

                depth = depth.detach().cpu().numpy()
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

                normal = out["rendered_normal"].permute(1, 2, 0)
                normal = normal @ view.world_view_transform[:3, :3]
                normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)

                # normal = normal.detach().cpu().numpy()
                # normal = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)
                normal = normal.detach().cpu().numpy()[:, :, ::-1]
                normal = ((1-normal) * 127.5).astype(np.uint8).clip(0, 255)

                language_feature = out["language_feature"]
                instance_feature = out["instance_feature"]
                all_language_feature.append(language_feature)
                all_instance_feature.append(instance_feature)

                lf_path = os.path.join(dataset.source_path, dataset.language_features_name)
                if os.path.exists(lf_path):
                    gt_language, _, _ = view.get_language_feature(lf_path)
                    all_gt_language_feature.append(gt_language)

                gts_path = os.path.join(dataset.model_path, "test", "gt_rgb")
                os.makedirs(gts_path, exist_ok=True)
                torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
                torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

                cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
                np.save(os.path.join(render_depth_npy_path, view.image_name + ".npy"), depth)
                cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

                view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
                depth_normal = out["depth_normal"].permute(1, 2, 0)
                depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
                dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
                angle = torch.acos(dot)
                mask = angle > (80.0 / 180 * 3.14159)
                depth_tsdf[mask] = 0
                depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())

            depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
            max_depth = 5.0
            for idx, view in enumerate(tqdm(self.scene.getTrainCameras(), desc="TSDF Fusion progress")):
                ref_depth = depths_tsdf_fusion[idx].cuda()

                if view.mask is not None:
                    ref_depth[view.mask.squeeze() < 0.5] = 0
                ref_depth[ref_depth > max_depth] = 0
                ref_depth = ref_depth.detach().cpu().numpy()
                pose = np.identity(4)
                pose[:3, :3] = view.R.transpose(-1, -2)
                pose[:3, 3] = view.T
                color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".png"))

                depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
                
                volume.integrate(
                    rgbd,
                    o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                    pose
                )
            num_cluster = 3
            path = os.path.join(dataset.model_path, "mesh")
            os.makedirs(path, exist_ok=True)
            
            mesh = volume.extract_triangle_mesh()
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh,
                                        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

            mesh = post_process_mesh(mesh, num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh,
                                        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        
        # perform pca among all lang/instance features
        render_language_path = os.path.join(dataset.model_path, "test", "renders_language")
        render_instance_path = os.path.join(dataset.model_path, "test", "renders_instance")
        gts_language_path = os.path.join(dataset.model_path, "test", "gt_language")
        render_language_npy_path = os.path.join(dataset.model_path, "test", "renders_language_npy")
        render_instance_npy_path = os.path.join(dataset.model_path, "test", "renders_instance_npy")
        gts_language_npy_path = os.path.join(dataset.model_path, "test", "gt_language_npy")
        os.makedirs(render_language_path, exist_ok=True)
        os.makedirs(gts_language_path, exist_ok=True)
        os.makedirs(render_language_npy_path, exist_ok=True)
        os.makedirs(gts_language_npy_path, exist_ok=True)
        os.makedirs(render_instance_path, exist_ok=True)
        os.makedirs(render_instance_npy_path, exist_ok=True)

        all_language_feature = torch.stack(all_language_feature)
        all_instance_feature = torch.stack(all_instance_feature)
        if len(all_gt_language_feature):
            all_gt_language_feature = torch.stack(all_gt_language_feature)
        if render_cfg.normalized:
            all_language_feature = torch.clamp(all_language_feature, min=-1, max=2)
            min_value = torch.min(all_language_feature)
            max_value = torch.max(all_language_feature)
            normalized_language_feature = (all_language_feature - min_value) / (max_value - min_value)
            pca_language_feature = permuted_pca(normalized_language_feature)
            for idx, view in enumerate(self.scene.getTrainCameras()):
                torchvision.utils.save_image(normalized_language_feature[idx], os.path.join(render_language_path, view.image_name + ".png"))

            all_instance_feature = torch.clamp(all_instance_feature, min=-1, max=2)
            min_value = torch.min(all_instance_feature)
            max_value = torch.max(all_instance_feature)
            normalized_instance_feature = (all_instance_feature - min_value) / (max_value - min_value)
            pca_instance_feature = permuted_pca(normalized_instance_feature)
            for idx, view in enumerate(self.scene.getTrainCameras()):
                torchvision.utils.save_image(
                    # pca_instance_feature[idx],
                    normalized_instance_feature[idx],
                    os.path.join(render_instance_path, view.image_name + ".png")
                )

            if os.path.exists(lf_path):
                all_gt_language_feature = torch.clamp(all_gt_language_feature, min=-1, max=2)
                min_value = torch.min(all_gt_language_feature)
                max_value = torch.max(all_gt_language_feature)
                normalized_gt_language = (all_gt_language_feature - min_value) / (max_value - min_value)
                pca_gt_language = permuted_pca(normalized_gt_language)
                for idx, view in enumerate(self.scene.getTrainCameras()):
                    torchvision.utils.save_image(
                        pca_gt_language[idx],
                        os.path.join(gts_language_path, view.image_name + ".png")
                    )
        else:
            breakpoint()
            all_language_feature = torch.clamp(all_language_feature, min=-1, max=2)
            pca_language_feature = permuted_pca(all_language_feature)
            for idx, view in enumerate(self.scene.getTrainCameras()):
                torchvision.utils.save_image(
                    pca_language_feature[idx],
                    os.path.join(render_language_path, view.image_name + ".png")
                )

            all_instance_feature = torch.clamp(all_instance_feature, min=-1, max=2)
            pca_instance_feature = permuted_pca(all_instance_feature)
            for idx, view in enumerate(self.scene.getTrainCameras()):
                torchvision.utils.save_image(
                    pca_instance_feature[idx],
                    os.path.join(render_instance_path, view.image_name + ".png")
                )

            if os.path.exists(lf_path):
                all_gt_language_feature = torch.clamp(all_gt_language_feature, min=-1, max=2)
                pca_gt_language = permuted_pca(all_gt_language_feature)
                for idx, view in enumerate(self.scene.getTrainCameras()):
                    torchvision.utils.save_image(
                        pca_gt_language[idx],
                        os.path.join(gts_language_path, view.image_name + ".png")
                    )


        for idx, view in enumerate(self.scene.getTrainCameras()):
            np.save(
                os.path.join(render_language_npy_path, view.image_name + ".npy"),
                all_language_feature[idx].permute(1, 2, 0).cpu().numpy()
            )
            np.save(
                os.path.join(render_instance_npy_path, view.image_name + ".npy"),
                all_instance_feature[idx].permute(1, 2, 0).cpu().numpy()
            )
            if os.path.exists(lf_path):
                np.save(
                    os.path.join(gts_language_npy_path, view.image_name + ".npy"),
                    all_gt_language_feature[idx].permute(1, 2, 0).cpu().numpy()
                )

        for idx, view in enumerate(tqdm(self.scene.getTrainCameras(), desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth > max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T
            color_feature = o3d.io.read_image(os.path.join(render_language_path, view.image_name + ".png"))
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))
            rgbd_feature = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_feature, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False
            )
            volume_feature.integrate(
                rgbd_feature,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose
            )
        
        num_cluster = 3
        mesh_feature = volume_feature.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(os.path.join(path, "feature_tsdf_fusion.ply"), mesh_feature,
                                    write_triangle_uvs=True, write_vertex_colors=True,
                                    write_vertex_normals=True)
        mesh_feature = post_process_mesh(mesh_feature, num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(path, "feature_tsdf_fusion_post.ply"), mesh_feature,
                                    write_triangle_uvs=True, write_vertex_colors=True,
                                    write_vertex_normals=True)


    
    
    def eval(self):
        cfg = self.cfg
        dataset = cfg.gaussian.dataset
        opt = cfg.gaussian.opt
        pipe = cfg.gaussian.pipe
        device = cfg.gaussian.dataset.data_device
        
        dataset.source_path = cfg.gaussian.eval.eval_data_path

        logging.info("Evaling " + dataset.model_path)
        safe_state(cfg.gaussian.quiet)
        # optimizing poses:
        self.gaussians = GaussianModel(cfg.gaussian.dataset.sh_degree)
        self.scene = Scene(cfg.gaussian.dataset, self.gaussians, load_iteration=cfg.pipeline.load_iteration, shuffle=False)

        self.gaussians.training_setup(opt, device)

        self.scene.loaded_iter = None
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        render_path = os.path.join(dataset.model_path, "eval", "renders_rgb")
        render_depth_path = os.path.join(dataset.model_path, "eval", "renders_depth")
        render_depth_npy_path = os.path.join(dataset.model_path, "eval", "renders_depth_npy")
        render_normal_path = os.path.join(dataset.model_path, "eval", "renders_normal")

        render_lang_path = os.path.join(dataset.model_path, "eval", "renders_lang")
        render_instance_path = os.path.join(dataset.model_path, "eval", "renders_instance")
        render_lang_npy_path = os.path.join(dataset.model_path, "eval", "renders_lang_npy")
        render_instance_npy_path = os.path.join(dataset.model_path, "eval", "renders_instance_npy")

        os.makedirs(render_path, exist_ok=True)
        os.makedirs(render_depth_path, exist_ok=True)
        os.makedirs(render_depth_npy_path, exist_ok=True)
        os.makedirs(render_normal_path, exist_ok=True)
        os.makedirs(render_lang_path, exist_ok=True)
        os.makedirs(render_instance_path, exist_ok=True)  
        os.makedirs(render_lang_npy_path, exist_ok=True)
        os.makedirs(render_instance_npy_path, exist_ok=True)

        self.gaussians.change_reqiures_grad("pose_only", iteration=0, quiet=False)

        for cam_idx, cam in enumerate(self.scene.getTrainCameras().copy()):
            # optim pose iter:
            first_iter = 1
            ema_loss_for_log = 0.0
            include_feature = True
            progress_bar = tqdm(range(first_iter, cfg.gaussian.eval.pose_optim_iter + 1))

            logging.info(f"Optimizing camera {cam_idx}")
            iter_start = torch.cuda.Event(enable_timing=True)
            iter_end = torch.cuda.Event(enable_timing=True)
            for iteration in progress_bar:
                iter_start.record()
                self.gaussians.update_learning_rate(iteration)
                pose = self.gaussians.get_RT(self.gaussians.index_mapping[cam.uid])
                bg = torch.rand((3), device="cuda") if opt.random_background else background
                render_pkg = render(cam, self.gaussians, pipe, bg, app_model=None, 
                                    return_plane=False, return_depth_normal=False,
                                    include_feature=include_feature, camera_pose=pose)
                
                image = render_pkg["render"]
                gt_image, _ = cam.get_image()
                ssim_loss = (1.0 - ssim(image, gt_image))
                Ll1 = l1_loss(image, gt_image)
                image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
                image_loss.backward()
                iter_end.record()
                with torch.no_grad():
                    ema_loss_for_log = 0.4 * image_loss + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        loss_dict = {
                            "Loss": f"{ema_loss_for_log:.5f}"
                        }
                        progress_bar.set_postfix(loss_dict)
                        progress_bar.update(10)
                    if iteration < cfg.gaussian.eval.pose_optim_iter:
                        self.gaussians.cam_optimizer.step()
                        self.gaussians.cam_optimizer.zero_grad(set_to_none=True)

                    if iteration == cfg.gaussian.eval.pose_optim_iter:
                        # saving results:
                        progress_bar.close()
                        logging.info("Saving results...")
                        language_feature, instance_feature = render_pkg["language_feature"], render_pkg["instance_feature"]
                        image_tosave = torch.cat([image, gt_image], dim=2).clamp(0, 1)
                        torchvision.utils.save_image(image_tosave, os.path.join(render_path, cam.image_name + ".png"))
                        min_value = torch.min(language_feature)
                        max_value = torch.max(language_feature)
                        normalized_language_feature = (language_feature - min_value) / (max_value - min_value)
                        torchvision.utils.save_image(permuted_pca(normalized_language_feature), 
                                os.path.join(render_lang_path, cam.image_name + ".png"))
                        np.save(os.path.join(render_lang_npy_path, cam.image_name + ".npy"),
                                language_feature.permute(1, 2, 0).cpu().numpy())

                        min_value = torch.min(instance_feature)
                        max_value = torch.max(instance_feature)
                        normalized_instance_feature = (instance_feature - min_value) / (max_value - min_value)
                        torchvision.utils.save_image(permuted_pca(normalized_instance_feature), 
                                os.path.join(render_instance_path, cam.image_name + ".png"))
                        np.save(os.path.join(render_instance_npy_path, cam.image_name + ".npy"),
                                instance_feature.permute(1, 2, 0).cpu().numpy())

            torch.cuda.empty_cache()

                
