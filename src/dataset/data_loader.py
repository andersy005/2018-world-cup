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
                 city='/home/abanihirwe/datasets/city_real/'):
        self.gta_path = gta
        self.city_path = city


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
