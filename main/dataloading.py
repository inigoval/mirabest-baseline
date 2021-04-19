import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.trainer.supporters import CombinedLoader

from paths import Path_Handler
from utilities import load_config
from datasets import MB_nohybrids

paths = Path_Handler()
path_dict = paths._dict()
config = load_config()

split = config["data"]["split"]
fraction = config["data"]["fraction"]


class Circle_Crop(torch.nn.Module):
    """
    Set all values outside largest possible circle that fits inside image to 0
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        !!! Support for multiple channels not implemented yet !!!
        """
        H, W, C = img.shape[-1], img.shape[-2], img.shape[-3]
        assert H == W
        x = torch.arange(W, dtype=torch.float).repeat(H, 1)
        x = (x - 74.5) / 74.5
        y = torch.transpose(x, 0, 1)
        r = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))
        r = r / torch.max(r)
        r[r < 0.5] = -1
        r[r == 0.5] = -1
        r[r != -1] = 0
        r = torch.pow(r, 2).view(C, H, W)
        assert r.shape == img.shape
        img = torch.mul(r, img)
        return img


transform = torchvision.transforms.Compose(
    [T.RandomRotation(180), T.ToTensor(), Circle_Crop()]
)

test_transform = torchvision.transforms.Compose([T.ToTensor(), Circle_Crop()])


class MB_nohybridsDataModule(pl.LightningDataModule):
    def __init__(
        self, fraction=fraction, split=split, batch_size=50, path=path_dict["data"]
    ):
        super().__init__()
        self.fraction = fraction
        self.split = split
        self.batch_size = batch_size
        self.transform = transform
        self.path = path

    def prepare_data(self):
        MB_nohybrids(self.path, train=True, download=True)
        MB_nohybrids(self.path, train=False, download=True)

    def setup(self, stage=None):
        mb_train = MB_nohybrids(self.path, train=True, transform=self.transform)
        mb_test = MB_nohybrids(self.path, train=False, transform=self.transform)
        self.test_dataset = mb_test

        if self.fraction != 1:
            n = len(mb_train)
            n_train = int(n * self.fraction)
            mb_train, _ = random_split(mb_train, [n_train, n - n_train])

        n = len(mb_train)
        n_l = int(n * self.split)
        mb_l, mb_u = random_split(mb_train, [n_l, n - n_l])
        self.train_dataset_u = mb_u
        self.train_dataset_l = mb_l

    def train_dataloader(self):
        loader_l = DataLoader(self.train_dataset_l, self.batch_size)
        return loader_l

    def val_dataloader(self):
        loader_test = DataLoader(self.test_dataset, len(self.test_dataset))
        loader_u = DataLoader(self.train_dataset_u, len(self.train_dataset_u))
        loaders = {"u": loader_u, "test": loader_test}
        combined_loaders = CombinedLoader(loaders)
        return combined_loaders
