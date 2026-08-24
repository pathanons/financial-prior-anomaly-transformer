import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, x_state=None):
        new_x, series, prior, prior_params = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            x_state=x_state
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), series, prior, prior_params


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, x_state=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        prior_param_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, prior_params = attn_layer(x, attn_mask=attn_mask, x_state=x_state)
            series_list.append(series)
            prior_list.append(prior)
            prior_param_list.append(prior_params)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, prior_param_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, prior_type='time',
                 z_state_indices=None, state_projection_dim=0, token_embedding='conv3_circular',
                 feature_names=None, use_positional_embedding=True, position_sigma=0.0,
                 use_return_nll=False):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.use_return_nll = use_return_nll
        self.z_state_indices = z_state_indices
        state_projection_dim = int(state_projection_dim or 0)
        state_input_dim = len(z_state_indices) if z_state_indices else enc_in
        self.state_projection = None
        if state_projection_dim > 0:
            self.state_projection = nn.Sequential(
                nn.Linear(state_input_dim, state_projection_dim),
                nn.LayerNorm(state_projection_dim)
            )

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout,
                                       token_embedding=token_embedding,
                                       feature_names=feature_names,
                                       use_positional_embedding=use_positional_embedding,
                                       position_sigma=position_sigma)
        attention_state_indices = None if self.state_projection is not None else z_state_indices

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout,
                                         output_attention=output_attention, prior_type=prior_type,
                                         z_state_indices=attention_state_indices),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        if use_return_nll:
            self.return_mu_projection = nn.Linear(d_model, 1)
            self.return_sigma_projection = nn.Linear(d_model, 1)

    def forward(self, x):
        enc_out = self.embedding(x)
        x_state = x
        if self.state_projection is not None:
            x_state = x[:, :, self.z_state_indices] if self.z_state_indices else x
            x_state = self.state_projection(x_state)
        hidden, series, prior, prior_params = self.encoder(enc_out, x_state=x_state)
        enc_out = self.projection(hidden)
        return_params = None
        if self.use_return_nll:
            sigma = F.softplus(self.return_sigma_projection(hidden)).squeeze(-1) + 1e-8
            return_params = {
                'mu': self.return_mu_projection(hidden).squeeze(-1),
                'sigma': sigma
            }

        if self.output_attention:
            return enc_out, series, prior, prior_params, return_params
        else:
            return enc_out  # [B, L, D]
