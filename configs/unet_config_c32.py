# == datasets == 
dataset_path = "/mnt/juicefs/datasets/lvis"
json_path = "/mnt/juicefs/datasets/lvis/annotations/image_info_unlabeled2017.json"

#== train ==
mixed_precision = "no"
num_train_epochs = 5
train_batch_size = 4
wandb=True
exp_name = "train-acc-unet-c16"
record_time = False

pretrained_model_ae = None

# validation 
val_steps=100
checkpointing_steps = 50000

#== model ==
in_channels = 512
out_channels = 512
latent_channels = 32
encoder_block_out_channels=[256, 64, 16]
decoder_block_out_channels=[16, 64, 256]
num_encoder_blocks=(1, 1, 1)
num_decoder_blocks=(1, 1, 1)