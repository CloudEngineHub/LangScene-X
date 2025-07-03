import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        data = []
        for i in range(len(data_names)):
            features = torch.from_numpy(np.load(data_names[i]))
            data.append(features)

        self.data = torch.cat(data, dim=0).float()

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return self.data.shape[0] 


class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims=None, decoder_hidden_dims=None):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        if not encoder_hidden_dims:
            encoder_hidden_dims = [512, 256, 128, 64, 32, 16, 3]
        if not decoder_hidden_dims:
            decoder_hidden_dims = [16, 32, 64, 128, 256, 512, 768]
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(768, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)
        print(self.encoder, self.decoder)
    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def decode(self, x):
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x