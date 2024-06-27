import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class MaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, sampling: int, lin_bias: bool = True, dropout: float = 0.2,
                 init_scaler: float = 1., act_func=nn.GELU()):
        super(MaskMLP, self).__init__()
        self.input_dim = input_dim
        self.lin_bias = lin_bias
        self.hidden_dim = hidden_dim
        self.sampling = sampling
        h_dim = sampling * hidden_dim

        self.wes1 = nn.Parameter(init_scaler * torch.randn(hidden_dim, input_dim))
        if lin_bias:
            self.bias1 = nn.Parameter(init_scaler * torch.randn(h_dim))
            self.bias2 = nn.Parameter(init_scaler * torch.randn(input_dim))
            self.bias3 = nn.Parameter(init_scaler * torch.randn(input_dim // sampling))
        self.act_func = act_func
        self.dropout = nn.Dropout(dropout)
        self.wes2 = nn.Parameter(init_scaler * torch.randn(input_dim, h_dim))
        self.wes3 = nn.Parameter(init_scaler * torch.randn(input_dim // sampling, input_dim))

        *vec_mask, wes_mask1, wes_msk2 = self.__generate_mask(sampling)
        self.v_mask, self.w_mask1, self.w_mask2 = vec_mask, wes_mask1, wes_msk2

    def __generate_mask(self, sampling: int):
        # 生成输入掩码与权重掩码
        msks = [None] * (sampling + 2)
        seg = self.input_dim // sampling
        wes_msk1 = torch.zeros(self.wes2.shape)
        wes_msk2 = torch.zeros(self.wes3.shape)

        for i in range(sampling):
            vec_msk = torch.zeros(self.input_dim)
            vec_msk[i::sampling] = 1.
            msks[i] = vec_msk
            rs = (i * seg, (i + 1) * seg)
            cs = (i * self.hidden_dim, (i + 1) * self.hidden_dim)
            wes_msk1[rs[0]: rs[-1], cs[0]: cs[-1]] = 1.

        msks[-2] = wes_msk1

        for i in range(seg):
            wes_msk2[i, i * sampling: (i + 1) * sampling] = 1.

        msks[-1] = wes_msk2

        return msks

    def forward(self, x: torch.Tensor):
        bs, cs, ls = x.shape
        # B, C, L
        out_list = [None] * self.sampling
        for i in range(self.sampling):
            out_list[i] = (x * self.v_mask[i].to(x.device)) @ self.wes1.T

        mid = torch.cat(out_list, dim=-1)  # B, C, h_dim
        mid = mid + self.bias1
        mid = self.dropout(self.act_func(mid))

        mid = mid @ (self.wes2 * self.w_mask1.to(x.device)).T + self.bias2  # B, C, L

        res = x @ (self.wes3 * self.w_mask2.to(x.device)).T + self.bias3  # B, C, L/sampling

        out = self.dropout(mid.reshape(bs, cs, self.sampling, -1) + res.unsqueeze(dim=-2))
        out = out.transpose(-2, -1)
        return out.reshape(bs, cs, ls)


class ConvLayer1D(nn.Module):
    def __init__(self, hidden: int, kernel_size: int = 3, dropout: float = 0.):
        super(ConvLayer1D, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=(kernel_size,), stride=(1,),
                      padding=kernel_size // 2))
        self.activation1 = nn.GELU()

        self.conv2 = weight_norm(nn.Conv1d(in_channels=hidden, out_channels=1, kernel_size=(1,), stride=(1,)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        y = x
        x = self.dropout(self.activation1(self.conv1(x)))
        x = self.conv2(x)  # self.dropout(self.conv2(x))
        return x + y


class ConvBlock(nn.Module):
    def __init__(self, hidden: int, kernel_size: int = 3, dropout: float = 0., conv_layer: int = 1):
        super(ConvBlock, self).__init__()
        layers = [ConvLayer1D(hidden, kernel_size=kernel_size, dropout=dropout) for _ in range(conv_layer)]
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.conv_layers(x)


class DecomposedBlock(nn.Module):
    def __init__(self, seq_len: int, hidden, kernel_size: int = 3, dropout: float = 0., pyramid: int = 3):
        super(DecomposedBlock, self).__init__()
        self.seq_len = seq_len
        self.pyramid = pyramid
        self.patch_len = seq_len // 2 ** pyramid

        blocks = [ConvBlock(hidden, kernel_size=kernel_size, dropout=dropout, conv_layer=i + 1) for i in
                  range(pyramid + 1)]

        self.backbone = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        bs, cs, ls = x.shape
        x = x.reshape(bs * cs, 1, ls)
        enc_input_list = [None] * (self.pyramid + 1)
        enc_out_list = enc_input_list.copy()
        enc_input_list[0] = x[:, :, -self.seq_len // (2 ** self.pyramid):]

        for i in range(self.pyramid):
            enc_input_list[i + 1] = x[:, :, -self.seq_len // (2 ** (self.pyramid - i - 1)): -self.seq_len // (
                    2 ** (self.pyramid - i))]

        for curr_input, block, i in zip(enc_input_list, self.backbone, range(self.pyramid, -1, -1)):
            enc_out_list[i] = block(curr_input)

        enc_out = torch.cat(enc_out_list, dim=-1)
        # print(enc_out.shape)
        enc_out = enc_out.reshape(bs, cs, ls)

        return enc_out


class LocalGlobalBlock(nn.Module):
    def __init__(self, seq_len: int, mlp_dim: int, sampling: int, d_hidden: int, kernel_size: int = 3,
                 pyramid: int = 3, dropout: float = 0.2, norm=None):
        super(LocalGlobalBlock, self).__init__()
        # self.tokens_mixing = FactorizedGlobalTemporalMixing(sampling, seq_len, mlp_dim, dropout=dropout)
        self.tokens_mixing = MaskMLP(seq_len, mlp_dim, sampling, dropout=dropout, init_scaler=0.3)
        self.dec = DecomposedBlock(seq_len, d_hidden, kernel_size=kernel_size, dropout=dropout, pyramid=pyramid)

        self.norm = norm

        # self.mlp_branch = ShallowMLPBlock(seq_len, mlp_dim, dropout)

    def forward(self, x: torch.Tensor, f_weight: torch.Tensor):
        # return self.mlp_branch(x)

        l_out = self.dec(x)
        mid = f_weight * x
        g_out = self.tokens_mixing(mid)
        out = l_out + g_out
        if self.norm:
            out = self.norm(out)
        return out
