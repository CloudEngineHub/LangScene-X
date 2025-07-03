import glob
import logging
import os
import shutil
import subprocess

import cv2
import numpy as np
import torch
from diffusers.models.autoencoders.vq_model import VQModel
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .auto_encoder import Autoencoder, Autoencoder_dataset
from .pose_estimator import get_pose_estimator
from .utils.loss_utils import cos_loss, l2_loss
from .video_preprocessor import VideoPreprocessor


def extract_with_openseg(cfg):
    import tensorflow as tf2
    import tensorflow._api.v2.compat.v1 as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    openseg = tf2.saved_model.load(
        cfg.feature_extractor.model_path, 
        tags=[tf.saved_model.tag_constants.SERVING]
    )
    imgs_path = os.path.join(cfg.pipeline.data_path, "input")
    img_names = list(
        filter(
            lambda x: x.endswith("png") or x.endswith("jpg"), sorted(os.listdir(imgs_path))
        )
    )
    img_list = []
    np_image_string_list = []
    for img_name in img_names:
        img_path = os.path.join(imgs_path, img_name)
        image = cv2.imread(img_path)
        with tf.gfile.GFile(img_path, 'rb') as f:
            np_image_string = np.array([f.read()])

        image = torch.from_numpy(image)
        img_list.append(image)
        np_image_string_list.append(np_image_string)

    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)
    save_path = os.path.join(cfg.pipeline.data_path, "lang_features")
    os.makedirs(save_path, exist_ok=True)
    embed_size = 768
    for i, (img, np_image_string) in enumerate(tqdm((zip(imgs, np_image_string_list)), desc="Extracting lang features", total=(len(imgs)))):
        text_emb = tf.zeros([1, 1, embed_size])
        results = openseg.signatures["serving_default"](
            inp_image_bytes=tf.convert_to_tensor(np_image_string[0]),
            inp_text_emb=text_emb
        )
        img_info = results['image_info']
        crop_sz = [
            int(img_info[0, 0] * img_info[2, 0]),
            int(img_info[0, 1] * img_info[2, 1])
        ]
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
        img_size = (img.shape[1], img.shape[2])
        feat_2d = tf.cast(
            tf.image.resize_nearest_neighbor(
                image_embedding_feat, img_size, align_corners=True
            )[0], dtype=tf.float32
        ).numpy()
        # perform mask-pooling over feat2d
        feat_2d = np.transpose(feat_2d, axes=(2, 0, 1))
        pooled_feats2d = []
        curr_mask = np.load(os.path.join(cfg.pipeline.data_path, "lang_features_dim3", str(i+1).zfill(4)+"_s.npy"))
        for color_id in range(-1, curr_mask.max() + 1):
            if not feat_2d[:, curr_mask == color_id].shape[-1]:
                continue
            pooled = feat_2d[:, curr_mask == color_id].mean(axis=-1)
            pooled /= np.linalg.norm(pooled)
            pooled_feats2d.append(pooled)

        pooled_feats2d = np.stack(pooled_feats2d)
        np.save(os.path.join(save_path, str(i+1).zfill(4)+".npy"), pooled_feats2d)

class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.pipeline.skip_video_process:
            self.video_processor = VideoPreprocessor(cfg)
        else:
            self.video_processor = None

        if not cfg.pipeline.skip_pose_estimate:
            self.pose_estimator = get_pose_estimator(cfg)
        else:
            self.pose_estimator = None

        if not cfg.pipeline.skip_lang_feature_extraction:
            # load feature extractor
            if cfg.feature_extractor.type == "open-seg":
                self.lseg = None
                self.sem_ae = Autoencoder()
                self.sem_ae.cuda()

            elif cfg.feature_extractor.type == "lseg":
                self.lseg = LSegFeatureExtractor.from_pretrained(cfg.lseg.model_path)
                self.lseg.to(cfg.lseg.device, dtype=torch.float32).eval()

                self.sem_ae = VQModel(
                    in_channels=512,
                    out_channels=512,
                    latent_channels=4,
                    norm_num_groups=2,
                    block_out_channels=[256, 64, 16],
                    down_block_types=["DownEncoderBlock2D"] * 3,
                    up_block_types=["UpDecoderBlock2D"] * 3,
                    layers_per_block=1,
                    norm_type="spatial",
                    num_vq_embeddings=1024,
                )

                self.sem_ae.load_state_dict(load_file(cfg.ae.model_path))
                self.sem_ae.to(cfg.ae.device, dtype=torch.float32).eval()
                self.img_transform = transforms.Compose(
                    [
                        transforms.Lambda(lambda x: x / 255),
                        transforms.Normalize(
                            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                        ),
                    ]
                )

        else:
            self.lseg = None
            self.sem_ae = None
            self.img_transform = None

        
    def generate_lang_features_with_openseg(self):
        extract_with_openseg(self.cfg)
        logging.info("Done feature extraction.")

        num_epochs = 400
        os.makedirs(os.path.join(self.cfg.pipeline.data_path, "ckpt"), exist_ok=True)
        save_path = os.path.join(self.cfg.pipeline.data_path, "lang_features")
        train_dataset = Autoencoder_dataset(save_path)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=32,
            drop_last=False
        )
        test_loader = DataLoader(
            dataset=train_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=32,
            drop_last=False
        )
        optimizer = torch.optim.Adam(self.sem_ae.parameters(), lr=1e-4)
        pbar = tqdm(range(num_epochs))
        best_eval_loss = 100.0
        best_epoch = 0

        for epoch in pbar:
            self.sem_ae.train()
            for idx, feature in enumerate(train_loader):
                data = feature.to("cuda")
                outputs_dim3 = self.sem_ae.encode(data)
                outputs = self.sem_ae.decode(outputs_dim3)

                l2loss = l2_loss(outputs, data)
                cosloss = cos_loss(outputs, data)
                loss = l2loss + cosloss * 0.001

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch > 300:
                eval_loss = 0.0
                self.sem_ae.eval()
                for idx, feature in enumerate(test_loader):
                    data = feature.to("cuda")
                    with torch.no_grad():
                        outputs = self.sem_ae(data)
                    loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                    eval_loss += loss * len(feature)
                eval_loss = eval_loss / len(train_dataset)
                print("eval_loss:{:.8f}".format(eval_loss))
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_epoch = epoch
                    torch.save(self.sem_ae.state_dict(), os.path.join(self.cfg.pipeline.data_path, "ckpt", "best_ckpt.pth"))
            pbar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
            pbar.update(1)

        print(f"best_epoch: {best_epoch}")
        print("best_loss: {:.8f}".format(best_eval_loss))
        # compress lang_feats with ae
        logging.info("Compresing language features with best ckpt...")
        best_state_dict = torch.load(os.path.join(self.cfg.pipeline.data_path, "ckpt", "best_ckpt.pth"), weights_only=False)
        self.sem_ae.load_state_dict(best_state_dict)
        # check device
        orig_lang_feat_names = sorted(glob.glob(os.path.join(save_path, "*.npy")))
        dim3_save_path = os.path.join(self.cfg.pipeline.data_path, "lang_features_dim3")
        with torch.no_grad():
            for idx, orig_lang_feat_name in enumerate(orig_lang_feat_names):
                orig_lang_feat = torch.from_numpy(np.load(orig_lang_feat_name)).cuda()
                mask = np.load(os.path.join(dim3_save_path, str(idx+1).zfill(4)+"_s.npy"))
                # check dtype
                lang_feat = self.sem_ae.encode(orig_lang_feat).detach().cpu().numpy()
                full_lang_feat = np.zeros((3, mask.shape[0], mask.shape[1]))
                curr_id = 0
                for color_id in range(-1, mask.max() + 1):
                    if not mask[mask == color_id].shape[-1]:
                        continue
                    full_lang_feat[:, mask == color_id] = lang_feat[curr_id][:, None]
                    curr_id += 1
                np.save(os.path.join(dim3_save_path, str(idx+1).zfill(4)+"_f.npy"), full_lang_feat)

    def generate_lang_features_with_lseg(self):
        from cogvideox_interpolation.lseg import LSegFeatureExtractor
        imgs_path = os.path.join(self.cfg.pipeline.data_path, "input")
        img_names = list(
            filter(
                lambda x: x.endswith("png") or x.endswith("jpg"), os.listdir(imgs_path)
            )
        )
        save_path = os.path.join(self.cfg.pipeline.data_path, "lang_features_dim4")
        os.makedirs(save_path, exist_ok=True)
        for img_name in tqdm(img_names):
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            resolution = (640, 480)
            img = cv2.resize(img, resolution)
            frame_embed = self.img_transform(torch.from_numpy(img).permute(2, 0, 1)).to(
                self.cfg.lseg.device, dtype=torch.float32
            )[None, ...]
            lseg_features = self.lseg.extract_features(frame_embed)
            if lseg_features.device != self.sem_ae.device:
                lseg_features = lseg_features.to("cpu").to(self.sem_ae.device)

            z = self.sem_ae.encode(lseg_features).latents  # [1, 4, 240, 320]
            np.save(
                os.path.join(save_path, f"{img_name.split('.')[0]}_f.npy"),
                z.detach().cpu().numpy(),
            )

    def select_valid_data(self):
        cfg = self.cfg
        curr_data_path = cfg.pipeline.data_path
        raw_data_path = os.path.join(curr_data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        dirs_to_move = ["camera", "input", "lang_features_dim3", "normal"]
        
        orig_view_nums = len(os.listdir(os.path.join(curr_data_path, "camera")))
        indexs = np.linspace(0, orig_view_nums-1, cfg.pipeline.chunk_num * cfg.pipeline.keep_num_per_chunk)
        indexs = indexs.astype(np.int32).tolist()
        cfg.pipeline.selected_idxs = indexs

        for dir_to_move in dirs_to_move:
            shutil.move(os.path.join(curr_data_path, dir_to_move), raw_data_path)
            src_dir = os.path.join(raw_data_path, dir_to_move)
            tar_dir = os.path.join(curr_data_path, dir_to_move)
            os.makedirs(tar_dir, exist_ok=True)
            file_lst = sorted(os.listdir(src_dir))
            file_suffix = file_lst[0].split(".")[-1]
            if dir_to_move == "lang_features_dim3":
                f_file_lst = [file_lst[2 * idx] for idx in cfg.pipeline.selected_idxs]
                s_file_lst = [file_lst[2 * idx + 1] for idx in cfg.pipeline.selected_idxs]
                for file_idx in range(len(f_file_lst)):
                    shutil.copy(
                        os.path.join(src_dir, f_file_lst[file_idx]),
                        os.path.join(tar_dir, f"{file_idx+1:04d}_f.{file_suffix}"),
                    )
                    shutil.copy(
                        os.path.join(src_dir, s_file_lst[file_idx]),
                        os.path.join(tar_dir, f"{file_idx+1:04d}_s.{file_suffix}"),
                    )
            else:
                file_lst = [file_lst[idx] for idx in cfg.pipeline.selected_idxs]
                for file_idx, file_name in enumerate(file_lst):
                    shutil.copy(
                        os.path.join(src_dir, file_name),
                        os.path.join(tar_dir, f"{file_idx+1:04d}.{file_suffix}"),
                    )

    def preprocess(self):
        if not self.cfg.pipeline.skip_video_process:
            logging.info("Processing input videos...")
            self.video_processor.video_process()

        if not self.cfg.pipeline.skip_pose_estimate:
            logging.info("Estimating poses...")
            self.pose_estimator.get_poses()

        if not self.cfg.pipeline.skip_lang_feature_extraction:
            logging.info("Generating language features...")
            if self.cfg.feature_extractor.type == "lseg":
                self.generate_lang_features_with_lseg()
            elif self.cfg.feature_extractor.type == "open-seg":
                self.generate_lang_features_with_openseg()

        if self.cfg.pipeline.selection:
            logging.info("Selecting views with higher confidence...")
            self.select_valid_data()

        logging.info("Done all preprocessing!")
