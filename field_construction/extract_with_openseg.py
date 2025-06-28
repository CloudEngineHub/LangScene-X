import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from tqdm import tqdm


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
            lambda x: x.endswith("png") or x.endswith("jpg"), os.listdir(imgs_path)
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
    for i, (img, np_image_string) in enumerate(tqdm((zip(imgs, np_image_string_list)), desc="Extracting lang features")):
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
            )[0], dtype=tf.float16
        ).numpy()
        # save feat_2d
        np.save(os.path.join(save_path, str(i+1).zfill(4)+".npy"), feat_2d)
    
if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--cfg")
    args = arg_parser.parse_args()
    extract_with_openseg(args.cfg)