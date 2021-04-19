import torch
import torch.nn as nn


def conv_block(
    C_in,
    C_out,
    K=3,
    S=1,
    P=1,
    activation="lrelu",
    batchnorm=True,
    bias=False,
):

    activations = nn.ModuleDict(
        [
            ["lrelu", nn.LeakyReLU(0.2, inplace=True)],
            ["relu", nn.ReLU(inplace=True)],
            ["tanh", nn.Tanh()],
            ["sig", nn.Sigmoid()],
        ]
    )

    layers = []

    # Convolution layer
    layers.append(nn.Conv2d(C_in, C_out, K, S, P, bias=bias))

    # Batchnorm layer
    if batchnorm:
        layers.append(nn.BatchNorm2d(C_out))

    # Activation layer
    if activation:
        layers.append(activations[activation])

    # Unpack list of layers into nn.Sequential
    block = nn.Sequential(*layers)
    return block


def convT_block(
    C_in,
    C_out,
    K=3,
    S=1,
    P=1,
    activation="lrelu",
    batchnorm=True,
    bias=False,
):
    activations = nn.ModuleDict(
        [
            ["lrelu", nn.LeakyReLU(0.2, inplace=True)],
            ["relu", nn.ReLU(inplace=True)],
            ["tanh", nn.Tanh()],
        ]
    )

    layers = []

    # Convolution layer
    layers.append(nn.ConvTranspose2d(C_in, C_out, K, S, P, bias=bias))

    # Batchnorm layer
    if batchnorm:
        layers.append(nn.BatchNorm2d(C_out))

    # Activation layer
    layers.append(activations[activation])

    # Unpack list of layers into nn.Sequential
    block = nn.Sequential(*layers)
    return block


def linear_block(C_in, C_out, activation="lrelu", dropout=True):

    activations = nn.ModuleDict(
        [
            ["lrelu", nn.LeakyReLU(0.2, inplace=True)],
            ["relu", nn.ReLU(inplace=True)],
            ["tanh", nn.Tanh()],
            ["sig", nn.Sigmoid()],
            ["softmax", nn.Softmax(dim=1)],
        ]
    )

    layers = []
    layers.append(nn.Linear(C_in, C_out))
    if activation:
        layers.append(activations[activation])
    if dropout:
        layers.append(nn.Dropout())

    # Unpack list of layers into nn.Sequential
    block = nn.Sequential(*layers)
    return block
