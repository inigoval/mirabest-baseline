import yaml

from paths import Path_Handler
import torch

# Define paths
paths = Path_Handler()
path_dict = paths._dict()


def load_config():
    """
    Helper function to load config file
    """
    path = path_dict["root"] / "config.yaml"
    with open(path, "r") as ymlconfig:
        config = yaml.load(ymlconfig, Loader=yaml.FullLoader)
    return config


def accuracy(y_hat, y):
    _, y_pred = torch.max(y_hat, 1)
    n_test = y.size(0)
    n_correct = (y_pred == y).sum().item()
    accuracy = n_correct / n_test
    return accuracy
