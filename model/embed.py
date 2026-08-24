import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, sigma=0.0):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        scale = torch.exp(-0.5 * (div_term * float(sigma)) ** 2)

        pe[:, 0::2] = torch.sin(position * div_term) * scale
        pe[:, 1::2] = torch.cos(position * div_term) * scale

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, token_embedding='conv3_circular'):
        super(TokenEmbedding, self).__init__()
        self.norm = nn.LayerNorm(d_model) if token_embedding == 'linear_norm' else None
        self.residual_mlp = None
        self.residual_scale = None
        if token_embedding in ('linear', 'linear_norm', 'linear_residual_mlp'):
            self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                       kernel_size=1, bias=False)
            if token_embedding == 'linear_residual_mlp':
                self.residual_mlp = nn.Sequential(
                    nn.Linear(c_in, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model)
                )
                self.residual_scale = nn.Parameter(torch.tensor(0.1))
        elif token_embedding == 'conv3_circular':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                       kernel_size=3, padding=padding,
                                       padding_mode='circular', bias=False)
        else:
            raise ValueError("Unknown token_embedding: {}".format(token_embedding))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        base = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        if self.residual_mlp is not None:
            base = base + self.residual_scale * self.residual_mlp(x)
        if self.norm is not None:
            base = self.norm(base)
        return base


class GroupedTokenEmbedding(nn.Module):
    GROUPS = (
        ("return", ("log_return_1d", "return_5d", "return_20d", "abs_return", "squared_return")),
        ("volume", ("volume_z",)),
        ("gap", ("gap", "high_low_range")),
        ("volatility", ("rolling_vol_5", "rolling_vol_20", "vol_ratio_5_20")),
    )

    def __init__(self, d_model, feature_names):
        super(GroupedTokenEmbedding, self).__init__()
        if not feature_names:
            raise ValueError("grouped token embedding requires feature_names")
        feature_names = list(feature_names)
        missing = [name for _, names in self.GROUPS for name in names if name not in feature_names]
        if missing:
            raise ValueError("grouped token embedding missing features: {}".format(",".join(missing)))

        group_dim = d_model // len(self.GROUPS)
        if group_dim < 1:
            raise ValueError("d_model too small for grouped token embedding")
        self.group_indices = [
            [feature_names.index(name) for name in names]
            for _, names in self.GROUPS
        ]
        self.group_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(len(indices), group_dim), nn.GELU())
            for indices in self.group_indices
        ])
        self.fusion = nn.Linear(group_dim * len(self.GROUPS), d_model)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        parts = [
            layer(x[:, :, indices])
            for indices, layer in zip(self.group_indices, self.group_layers)
        ]
        return self.fusion(torch.cat(parts, dim=-1))


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0, token_embedding='conv3_circular',
                 feature_names=None, use_positional_embedding=True, position_sigma=0.0):
        super(DataEmbedding, self).__init__()

        if token_embedding == 'grouped':
            self.value_embedding = GroupedTokenEmbedding(d_model=d_model, feature_names=feature_names)
        else:
            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model,
                                                  token_embedding=token_embedding)
        self.position_embedding = (
            PositionalEmbedding(d_model=d_model, sigma=position_sigma)
            if use_positional_embedding else None
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)
        return self.dropout(x)
