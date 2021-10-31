import zipfile
import gzip
from collections import namedtuple
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def prepare_data(create_pickle=False):
    """
    Download and extract the data
    :return: images: list[PIL.Image] and labels : list[int]
    """
    pkl_file = 'train.pkl.zip'
    if create_pickle or not os.path.exists(pkl_file):
        train_zip = zipfile.ZipFile('train.zip')
        file_names = [name for name in train_zip.namelist() if 'dog' in name or 'cat' in name]
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
            img = self.transform(img)

        return img, y


TrainValTest = namedtuple('TrainValTest', ['train', 'val', 'test'])

easy_transforms = TrainValTest(
    train=T.Compose([
        T.Resize(256), T.RandomHorizontalFlip(), T.RandomRotation(30),
        T.RandomCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    val=T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    test=T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
