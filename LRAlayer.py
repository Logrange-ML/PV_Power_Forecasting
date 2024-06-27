import torch
import torch.nn as nn
from math import sqrt
from typing import Optional, Literal, Sequence

class LowRankAttention(nn.Module):
    def __init__(self, seq_len: int, d_model: int, n_heads: int, d_r: int, qk_scale: Optional[float] = None,
                 attn_dropout: float = 0.05, proj_dropout: float = 0.05, kv_bias: bool = False):
        super(LowRankAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.routers = nn.Parameter(torch.randn(n_heads, d_r, d_model), requires_grad=True)
        self.w_kv = nn.Linear(seq_len, n_heads * d_model, bias=kv_bias)
        # self.w_k = nn.Linear(seq_len, n_heads * d_model, bias=kv_bias)
        # self.w_v = nn.Linear(seq_len, n_heads * d_model, bias=kv_bias)
        self.qk_scale = qk_scale if qk_scale else 1. / sqrt(d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        # self.proj_dropout = nn.Dropout(proj_dropout)
        self.proj = nn.Linear(n_heads * d_model, seq_len)

    def forward(self, x: torch.Tensor):
        bs, ns, ds = x.shape

        kv = self.w_kv(x).reshape(bs, ns, self.n_heads, self.d_model).transpose(1, 2)
        # keys = self.w_k(x).reshape(bs, ns, self.n_heads, self.d_model).transpose(1, 2)
        # values = self.w_v(x).reshape(bs, ns, self.n_heads, self.d_model).transpose(1, 2)
        keys, values = kv, kv
        attn_score = (self.routers @ keys.transpose(-2, -1)) * self.qk_scale
        attn_weight = attn_score.softmax(dim=-1)
        attn_weight = self.attn_dropout(attn_weight)

        output = (attn_weight @ values).transpose(1, 2).reshape(bs, -1, self.n_heads * self.d_model)
        output = self.proj(output)  # self.proj_dropout(self.proj(output))

        return output, attn_weight


class EncoderLayer(nn.Module):
    def __init__(self, attention, c_in: int, d_r: int, d_model: int, seq_len: int, d_ff: int, dropout: float = 0.05,
                 activation: Literal['gelu', 'relu'] = 'gelu'):
        super(EncoderLayer, self).__init__()

        self.attention = attention
        self.activation = activation

        if activation.lower() == 'gelu':
            self.activation_fn = nn.GELU()
        elif activation.lower() == 'relu':
            self.activation_fn = nn.ReLU()

        self.w_e = nn.Linear(d_r, c_in)
        # self.w_p = nn.Linear(d_model, seq_len)
        self.norm1 = nn.LayerNorm(seq_len)
        self.norm2 = nn.LayerNorm(seq_len)

        self.ffn = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            self.activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len),
            nn.Dropout(dropout)
        )
        self.add_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        new_x, attn = self.attention(x)
        new_x = self.w_e(new_x.transpose(-2, -1)).transpose(-2, -1)
        # new_x = self.w_p(new_x)
        x = x + self.add_dropout(new_x)
        y = x = self.norm1(x)
        y = self.ffn(y)
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, enc_layers: Sequence[EncoderLayer], norm=None):
        super(Encoder, self).__init__()
        self.__enc_layers = nn.ModuleList(enc_layers)
        self.__norm = norm

    def forward(self, x: torch.Tensor):
        attn = []
        for layer in self.__enc_layers:
            x, a = layer(x)
            attn.append(a)

        if self.__norm:
            x = self.__norm(x)
        return x, attn
