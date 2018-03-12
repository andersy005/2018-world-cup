# Script to load, process/augment data from GTA5 and CityScape datasets.abs
# License: MIT
# Author: Anderson Banihirwe

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage import io
import pathlib
import os


class Config:
    """Configuration class that
    contains needed configuration settings."""
    def __init__(self, gta='/home/abanihirwe/datasets/gta/images/',
                 city='/home/abanihirwe/datasets/city_real/',
                 image_size=32, g_conv_dim=64, d_conv_dim=64,
                 use_reconst_loss=True, use_labels=None, num_classes=None,
                 train_iters=4000, batch_size=64, num_workers=2, lr=0.0002,
                 beta1=0.5, beta2=0.999, mode='train', model_path='models',
                 sample_path=None, log_step=10, sample_step=100):
        self.gta_path = gta
        self.city_path = city
        self.image_size = image_size
        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.use_reconst_loss = use_reconst_loss
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.beta1 = beta1
        self.beta2 = beta2
        self.mode = mode
        self.model_path = model_path
        self.sample_path = sample_path
        self.log_step = log_step
        self.sample_step = sample_step


class CustomDataset(Dataset):
    """Create a custom dataset object."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        img_name = self.root_dir/self.files[idx]
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image


def get_loader(config):
    """Builds and returns Dataloader for GTA5 and CityScape dataset."""
    gta_path = pathlib.Path(config.gta_path)
    city_path = pathlib.Path(config.city_path)

    gta_dataset = CustomDataset(gta_path)
    city_dataset = CustomDataset(city_path)

    return gta_dataset, city_dataset
