import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utilities import load_config, accuracy
from networks.layers import conv_block, convT_block, linear_block


config = load_config()
n_f = config["model"]["n_f"]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Conv2D(in_channels, out_channels, kernel size, stride, padding)

        # conv1 (1, 28, 28)  ->  (32, 14, 14)
        # conv2 (32, 14, 14) ->  (64, 7, 7)
        # conv3 (64, 7, 7) ->  (128, 4, 4)
        # conv4 (128, 4, 4)   ->  (1, 1, 1 )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 11, 1, 1), nn.ReLU(), nn.MaxPool2d(2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1), nn.BatchNorm2d(24), nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.BatchNorm2d(
                16,
            ),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=4),
        )

        # 8192 -> 2048
        # 2048 -> 512
        # 512  -> 512
        # 512  -> 3
        self.linear1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.linear2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 256)
        x = self.linear1(x)
        x = self.linear2(x)
        y = self.softmax(x)
        return y


class clf(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CNN()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.cross_entropy(y_pred, y)
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        for key, data in batch.items():
            x, y = data
            y_pred = self.forward(x)

            loss = F.cross_entropy(y_pred, y)
            self.log(f"{key} loss", loss)

            acc = accuracy(y_pred, y)
            self.log(f"{key} accuracy", acc)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=config["train"]["lr"])
        return opt
