import torch
from torch import nn


def bloco_conv(entrada, saida):
    """Convolucao 3x3 + LeakyReLU + BatchNorm (bloco basico das tres redes)."""
    return nn.Sequential(
        nn.Conv2d(entrada, saida, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(saida),
    )

class DenseEncoder(nn.Module):

    def __init__(self, data_depth, hidden_size=32):
        super().__init__()
        self.conv1 = bloco_conv(3, hidden_size)
        self.conv2 = bloco_conv(hidden_size + data_depth, hidden_size)
        self.conv3 = bloco_conv(2 * hidden_size + data_depth, hidden_size)
        self.conv4 = nn.Conv2d(3 * hidden_size + data_depth, 3, kernel_size=3, padding=1)

    def forward(self, imagem, bits):
        x1 = self.conv1(imagem)
        x2 = self.conv2(torch.cat([x1, bits], dim=1))
        x3 = self.conv3(torch.cat([x1, x2, bits], dim=1))
        residuo = self.conv4(torch.cat([x1, x2, x3, bits], dim=1))
        return imagem + residuo

class DenseDecoder(nn.Module):

    def __init__(self, data_depth, hidden_size=32):
        super().__init__()
        self.conv1 = bloco_conv(3, hidden_size)
        self.conv2 = bloco_conv(hidden_size, hidden_size)
        self.conv3 = bloco_conv(2 * hidden_size, hidden_size)
        self.conv4 = nn.Conv2d(3 * hidden_size, data_depth, kernel_size=3, padding=1)

    def forward(self, stego):
        x1 = self.conv1(stego)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        return self.conv4(torch.cat([x1, x2, x3], dim=1))

class BasicCritic(nn.Module):
     def __init__(self, hidden_size=32):
        super().__init__()
        self.layers = nn.Sequential(
            bloco_conv(3, hidden_size),
            bloco_conv(hidden_size, hidden_size),
            bloco_conv(hidden_size, hidden_size),
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
        )

        def forward(self, imagem):
        x = self.layers(imagem)
        return x.mean(dim=(1, 2, 3))
