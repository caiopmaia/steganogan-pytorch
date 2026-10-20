import torch
from torch import nn


def bloco_conv(entrada, saida):
    """Convolucao 3x3 + LeakyReLU + BatchNorm (bloco basico das tres redes)."""
    return nn.Sequential(
        nn.Conv2d(entrada, saida, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(saida),
    )
