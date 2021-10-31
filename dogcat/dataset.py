import zipfile
import gzip
import os
import pickle
from dataclasses import dataclass

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


def prepare_data(create_pickle=False, quick=False):
    """
    Download and extract the data
    :return: images: list[PIL.Image] and labels : list[int]
    """
    pkl_file = 'train.pkl.zip'
    if create_pickle or not os.path.exists(pkl_file):
        train_zip = zipfile.ZipFile('train.zip')
        file_names = [name for name in train_zip.namelist() if 'dog' in name or 'cat' in name]
        if quick:
            file_names = file_names[:1000]
        print('Loading images metadata...')
        images = [Image.open(train_zip.open(name)) for name in file_names]
        print('Reading pixels...')
        for i, img in enumerate(images):
            img.load()
            if i % 1000 == 0:
                print(i)
        [img.load() for img in images]
        print('Done', len(images))
        targets = [1 if 'dog' in name else 0 for name in file_names]

        if create_pickle:
            with gzip.open(pkl_file, 'w') as pkl_zip:
                print('Creating pickle file...')
                pickle.dump((images, targets), pkl_zip)
    else:
        print('Loading pickle file...')
        with gzip.open(pkl_file, 'r') as pkl_zip:
            images, targets = pickle.load(pkl_zip)
            print('Done', len(images))

    return images, targets


class MemoryImageDataset(Dataset):
    """
    Dataset for memory images
    """

    def __init__(self, images, targets, transform=None):
        super().__init__()
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        y = self.targets[idx]
        if self.transform is not None:
            img = self.transform(image=np.array(img))["image"]

        return img, y


@dataclass
class EasyTransforms:
    train = A.Compose([
        A.LongestMaxSize(max_size=256, always_apply=True),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomCrop(height=224, width=224, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    val = A.Compose([
        A.LongestMaxSize(max_size=256, always_apply=True),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
        A.CenterCrop(height=224, width=224, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(always_apply=True)
    ])

    test = A.Compose([
        A.LongestMaxSize(max_size=256, always_apply=True),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
        A.CenterCrop(height=224, width=224, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(always_apply=True)
    ])
