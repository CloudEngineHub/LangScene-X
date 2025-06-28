# validation 
val_steps=10
wandb=False
record_time = True

# autoencoder config
# exp_name = "AE-gpu8-bs1-channel-16"
# autoencoder_name = "ae-channel-16"

exp_name = "test"
autoencoder_name = "ae-channel-3"

dataset_name = "lvis"

lseg_weights="model_zoo/lseg/demo_e200.ckpt"

# loss weights
perceptual_loss_weight = 0.1  # use vgg is not None and more than 0
kl_loss_weight = 1e-6

mixed_strategy = "mixed_video_image"
mixed_image_ratio = 0.2
use_real_rec_loss = False
use_z_rec_loss = True
use_image_identity_loss = True