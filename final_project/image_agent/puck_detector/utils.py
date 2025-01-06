from typing import List
import numpy as np
import pystk

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

DATASET_PATH = "hockey_training_data"

def load_all(dataset_path=DATASET_PATH) -> List:
    from PIL import Image
    from glob import glob
    from os import path
    data = []

    for f in glob(dataset_path + '/td_00*/im_*.png'):
        image = Image.open(f)
        mask = Image.open(f.replace('im', 'mask'))
        
        image.load()
        mask.load()
        data.append((image, mask))


    print('data loaded')
    return data

class SuperTuxDataset(Dataset):
    def __init__(self, data, transform=dense_transforms.ToTensor()):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)

        return data

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)