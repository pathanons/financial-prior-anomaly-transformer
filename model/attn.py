import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0,
                 output_attention=False, prior_type='time', z_state_indices=None, eps=1e-8):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.prior_type = prior_type
        self.z_state_indices = z_state_indices
        self.eps = eps
        self.dropout = nn.Dropout(attention_dropout)
        distances = torch.arange(win_size).view(-1, 1) - torch.arange(win_size).view(1, -1)
        self.register_buffer('distances', distances.abs().float())

    def _reshape_param(self, param):
        return param.transpose(1, 2).unsqueeze(-1)

    def _time_prior(self, sigma, window_size):
        sigma = self._reshape_param(sigma)
        d_time = self.distances[:window_size, :window_size].view(1, 1, window_size, window_size)
        logits = -(d_time ** 2) / (2.0 * sigma ** 2)
        return torch.softmax(logits, dim=-1), {'sigma': sigma, 'tau': None, 'lambda': None}

    def _state_distance(self, x_state):
        if x_state is None:
            raise ValueError("{} prior requires x_state".format(self.prior_type))
        state = x_state[:, :, self.z_state_indices] if self.z_state_indices else x_state
        return torch.cdist(state, state, p=2).pow(2).unsqueeze(1)

    def _state_prior(self, tau, x_state):
        tau = self._reshape_param(tau)
        logits = -self._state_distance(x_state) / (2.0 * tau ** 2)
        return torch.softmax(logits, dim=-1), {'sigma': None, 'tau': tau, 'lambda': None}

    def _finance_prior(self, sigma, tau, lam, x_state, window_size):
        sigma = self._reshape_param(sigma)
        tau = self._reshape_param(tau)
        lam = self._reshape_param(lam)
        d_state = self._state_distance(x_state)
        d_time = self.distances[:window_size, :window_size].view(1, 1, window_size, window_size)
        time_term = -(d_time ** 2) / (2.0 * sigma ** 2)
        state_term = -lam * d_state / (2.0 * tau ** 2)
        prior = torch.softmax(time_term + state_term, dim=-1)
        return prior, {'sigma': sigma, 'tau': tau, 'lambda': lam}

    def forward(self, queries, keys, values, sigma, tau, lam, attn_mask, x_state=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        window_size = attn.shape[-1]
        if self.prior_type == 'time':
            prior, prior_params = self._time_prior(sigma, window_size)
        elif self.prior_type == 'time_state':
            prior, prior_params = self._finance_prior(sigma, tau, lam, x_state, window_size)
        elif self.prior_type == 'state':
            prior, prior_params = self._state_prior(tau, x_state)
        else:
            raise ValueError("Unknown prior_type: {}".format(self.prior_type))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, prior_params)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, eps=1e-8):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.tau_projection = nn.Linear(d_model,
                                        n_heads)
        self.lambda_projection = nn.Linear(d_model,
                                           n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask, x_state=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = F.softplus(self.sigma_projection(x).view(B, L, H)) + self.eps
        tau = F.softplus(self.tau_projection(x).view(B, L, H)) + self.eps
        lam = torch.sigmoid(self.lambda_projection(x).view(B, L, H))

        out, series, prior, prior_params = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            tau,
            lam,
            attn_mask
            ,
            x_state=x_state
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, prior_params
