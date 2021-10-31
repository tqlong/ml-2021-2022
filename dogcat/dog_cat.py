import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import MemoryImageDataset, prepare_data, easy_transforms
from model import DogCatModel


class MyDataModule(pl.LightningDataModule):
    def __init__(self, images, targets, args: argparse.Namespace):
        super().__init__()
        self.images = images
        self.targets = targets
        self.args = args
        self.train_images, self.val_images, self.train_targets, self.val_targets = None, None, None, None
        self.train, self.val = None, None

    def prepare_data(self):
        # called only on 1 GPU
        self.train_images, self.val_images, self.train_targets, self.val_targets = \
            train_test_split(self.images, self.targets, test_size=self.args.test_size)

    # noinspection PyUnusedLocal
    def setup(self, stage=None):
        train_transform = easy_transforms.train
        val_transform = easy_transforms.val
        self.train = MemoryImageDataset(self.train_images, self.train_targets, transform=train_transform)
        self.val = MemoryImageDataset(self.val_images, self.val_targets, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.batch_size, num_workers=self.args.num_workers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MyDataModule")
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--test_size', type=float, default=0.3)
        parser.add_argument('--num_workers', type=int, default=16)
        return parent_parser


def main():
    parser = argparse.ArgumentParser()

    parser = MyDataModule.add_model_specific_args(parser)
    parser = DogCatModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    net = DogCatModel(args)

    images, targets = prepare_data()
    dm = MyDataModule(images, targets, args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="lightning_logs/ckpt1",
        filename="dog-cat-resnet18-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(net, dm)
    print('best checkpoint', checkpoint_callback.best_model_path)
    return net


if __name__ == '__main__':
    main()
