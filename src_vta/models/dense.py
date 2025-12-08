import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class DenseModel(nn.Module):
    """
    全結合層による予測モデル (Reward, Value, Discount用)
    """
    def __init__(
        self, feature_size, output_shape, layers=3, units=400, dist="normal", act=nn.ELU
    ):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act

        # ネットワーク構築
        model = []
        for index in range(self._layers):
            model.append(nn.Linear(feature_size if index == 0 else self._units, self._units))
            model.append(self._act())
        model.append(nn.Linear(self._units, int(np.prod(self._output_shape))))
        self.model = nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        
        # 出力形状の整形
        reshaped_inputs = dist_inputs.view(
            *dist_inputs.shape[:-1], *self._output_shape
        )
        
        if self._dist == "normal":
            return Normal(reshaped_inputs, 1.0)
        elif self._dist == "binary":
            return torch.distributions.Bernoulli(logits=reshaped_inputs)
        elif self._dist == "none":
            return reshaped_inputs
        else:
            raise NotImplementedError(self._dist)