"""Basic network components for the VTA world model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LatentDistribution(nn.Module):
    """Gaussian distribution parameterizer for latent variables z or s."""

    def __init__(self, input_size, latent_size, feat_size=None):
        super().__init__()
        if feat_size is None:
            self.feat = nn.Identity()
            feat_size = input_size
        else:
            self.feat = nn.Sequential(nn.Linear(input_size, feat_size), nn.ELU(inplace=True))
        self.mean = nn.Linear(feat_size, latent_size)
        self.std = nn.Sequential(nn.Linear(feat_size, latent_size), nn.Sigmoid())

    def forward(self, input_data):
        feat = self.feat(input_data)
        return Normal(loc=self.mean(feat), scale=self.std(feat))


class Encoder(nn.Module):
    """CNN encoder that maps images to feature vectors."""

    def __init__(self, output_size=None, feat_size=64):
        super().__init__()
        network_list = []
        num_layers = 4
        for l in range(num_layers):
            input_size = 3 if l == 0 else feat_size
            is_final_layer = l == num_layers - 1
            network_list.append(
                nn.Conv2d(
                    input_size,
                    feat_size,
                    kernel_size=4,
                    stride=1 if is_final_layer else 2,
                    padding=0 if is_final_layer else 1,
                )
            )
            network_list.append(nn.BatchNorm2d(feat_size))
            network_list.append(nn.ELU(inplace=True))
        network_list.append(nn.Flatten())
        if output_size is not None:
            network_list.append(nn.Linear(feat_size, output_size))
            network_list.append(nn.ELU(inplace=True))
            self.output_size = output_size
        else:
            self.output_size = feat_size
        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        return self.network(input_data)


class Decoder(nn.Module):
    """Deconvolutional decoder that reconstructs images from features."""

    def __init__(self, input_size, feat_size=64, activation="sigmoid"):
        super().__init__()
        if input_size == feat_size:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(input_size, feat_size)

        network_list = []
        num_layers = 4
        for l in range(num_layers):
            is_final_layer = l == num_layers - 1
            network_list.append(
                nn.ConvTranspose2d(
                    feat_size,
                    3 if is_final_layer else feat_size,
                    kernel_size=4,
                    stride=1 if l == 0 else 2,
                    padding=0 if l == 0 else 1,
                )
            )
            if is_final_layer:
                if activation == "sigmoid":
                    network_list.append(nn.Sigmoid())
                elif activation == "tanh":
                    network_list.append(nn.Tanh())
            else:
                network_list.append(nn.BatchNorm2d(feat_size))
                network_list.append(nn.ELU(inplace=True))
        self.network = nn.Sequential(*network_list)

    def forward(self, input_data):
        return self.network(self.linear(input_data).unsqueeze(-1).unsqueeze(-1))


class PriorBoundaryDetector(nn.Module):
    """Predicts boundary logits p(m_t|s_t) from observation state features."""

    def __init__(self, input_size, output_size=2):
        super().__init__()
        self.network = nn.Linear(input_size, output_size)

    def forward(self, input_data):
        return self.network(input_data)


class PostBoundaryDetector(nn.Module):
    """Predicts boundary logits q(m|x) from encoded observations."""

    def __init__(self, input_size, output_size=2, num_layers=1):
        super().__init__()
        network = []
        for _ in range(num_layers):
            network.append(nn.Conv1d(input_size, input_size, 3, stride=1, padding=1, bias=False))
            network.append(nn.BatchNorm1d(input_size))
            network.append(nn.ELU(inplace=True))
        network.append(nn.Conv1d(input_size, output_size, 3, stride=1, padding=1))
        self.network = nn.Sequential(*network)

    def forward(self, input_data_list):
        input_data = input_data_list.permute(0, 2, 1)
        return self.network(input_data).permute(0, 2, 1)


__all__ = [
    "LatentDistribution",
    "Encoder",
    "Decoder",
    "PriorBoundaryDetector",
    "PostBoundaryDetector",
]
