import argparse

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchvision import models


class DogCatModel(pl.LightningModule):
    def __init__(self, args: argparse.Namespace = None, n_classes: int = 2):
        """
        Initialize the model
        :param args: argparse arguments (lr: learning_rate)
        :param n_classes: number of classes
        """
        super().__init__()
        self.features = models.resnet18(pretrained=True)
        for param in self.features.parameters():
            param.requires_grad = False
        # change the last layer
        self.features.fc = torch.nn.Linear(512, n_classes)
        self.args = args
        self.lr = args.lr if args is not None else 0.001

    def forward(self, x):
        out = self.features(x)
        # out = F.log_softmax(out, dim=1)
        return out

    # noinspection PyCallingNonCallable,PyUnusedLocal
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    # noinspection PyCallingNonCallable,PyUnusedLocal
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        _, prediction = torch.max(logits, dim=1)
        self.log("val_loss", loss)
        self.log("val_acc", (prediction == y).sum() / len(y))
        return prediction

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DogCatModel")
        parser.add_argument('--lr', type=float, default=0.001)
        return parent_parser
