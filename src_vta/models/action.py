import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform, OneHotCategorical

class OneHotDist(OneHotCategorical):
    """
    rsample()に対応したOneHotCategorical分布。
    学習時はStraight-Through Gumbel-Softmaxを使用して勾配を通す。
    """
    def rsample(self, sample_shape=torch.Size()):
        # Gumbel-Softmax (Hard=TrueでOneHot化、勾配はSoftmaxを通るStraight-Through)
        return F.gumbel_softmax(self.logits, tau=1.0, hard=True, dim=-1)

class ActionModel(nn.Module):
    """
    行動モデル (Actor)
    Continuous (tanh_normal) と Discrete (onehot) の両方に対応
    """
    def __init__(
        self,
        action_size,
        feature_size,
        hidden_units=400,
        act=nn.ELU,
        dist="tanh_normal", # "tanh_normal" or "onehot"
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
    ):
        super().__init__()
        self._action_size = action_size
        self._layers = 4
        self._units = hidden_units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        model = []
        for index in range(self._layers):
            model.append(nn.Linear(feature_size if index == 0 else self._units, self._units))
            model.append(self._act())
        
        # 出力層の分岐
        if self._dist == "onehot":
            # 離散行動: ロジットを直接出力 (サイズはaction_size)
            model.append(nn.Linear(self._units, self._action_size))
        else:
            # 連続行動: Mean と Std を出力 (サイズは2 * action_size)
            model.append(nn.Linear(self._units, 2 * self._action_size))
            
        self.model = nn.Sequential(*model)

    def forward(self, features):
        x = self.model(features)
        
        if self._dist == "onehot":
            # 離散行動用の分布を返す
            return OneHotDist(logits=x)
            
        elif self._dist == "tanh_normal":
            # 連続行動用の分布を返す
            mean, std = torch.chunk(x, 2, -1)
            
            # パラメータの正規化と制限
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + self._init_std) + self._min_std
            
            dist = Normal(mean, std)
            # Tanh変換を適用して (-1, 1) の範囲に収める
            return TransformedDistribution(dist, TanhTransform())
        
        else:
            raise NotImplementedError(f"Unknown distribution: {self._dist}")