import torch
import torch.nn as nn
import torch_dct
from LGmodule import *
from LRAlayer import *


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        print('new__m1')

        self.output_attention = configs.output_attention
        print(self.output_attention)
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in

        self.k = configs.kf
        self.last_steps = configs.last_steps
        jft = configs.d_model = self.k + self.last_steps
        fs = configs.feature_scaler

        dct_hidden = int(fs * jft)
        self.lin_dropout = configs.lin_dropout

        self.dct_norm_2 = nn.LayerNorm(self.seq_len, eps=1e-5)
        self.dct_norm_1 = nn.LayerNorm(jft, eps=1e-5)
        self.dct = torch_dct.dct

        # -----------

        self.dct_ffn = nn.Sequential(
            nn.Linear(jft, dct_hidden, bias=False),
            nn.GELU(),
            nn.Dropout(self.lin_dropout),
            nn.Linear(dct_hidden, self.seq_len, bias=False),
            nn.Sigmoid(),
        )

        self.local_global = LocalGlobalBlock(self.seq_len, int(24 * fs), sampling=8
                                             # site-1A/site35: 24， 8, site-1B: 32, 6
                                             , d_hidden=24, kernel_size=3,
                                             # 新版的Site-1A kernel = 3
                                             # kernel_size: Site-1A都是7， site-35都是3, site-1B也是3
                                             dropout=self.lin_dropout,
                                             pyramid=3)  # , norm=nn.LayerNorm(self.seq_len)) # norm不要加
        # Site-1A input->288 : sampling = 8, 36 * fs , kernel_size=7

        '''
        self.local_global = LocalGlobalBlock(self.seq_len, 256, sampling=8
                                             # site-1A/site35: 24， 8, site-1B: 32, 6
                                             , d_hidden=24, kernel_size=configs.kernel_size,
                                             # kernel_size: Site-1A都是7， site-35都是3, site-1B也是3
                                             dropout=self.lin_dropout,
                                             pyramid=2)  # , norm=nn.LayerNorm(self.seq_len)) # norm不要加
        '''

        # 线性部分：
        self.lin_proj = nn.Linear(self.seq_len, self.pred_len)

        '''
        # configs.d_model = configs.seq_len
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.proj = nn.Linear(configs.d_model, self.pred_len, bias=True)

        '''

        # LowRankAttention
        self.encoder_input_dim = configs.d_model
        self.output_attention = configs.output_attention
        # self.attn_dropout = configs.attn_dropout
        # self.proj_dropout = configs.proj_dropout

        self.d_r = configs.d_r  # 路由数
        self.encoder_hidden_dim = configs.enc_hidden_dim  # 路由维度

        self.enc_layers = [
            EncoderLayer(
                attention=LowRankAttention(
                    seq_len=self.encoder_input_dim,
                    d_model=self.encoder_hidden_dim,
                    n_heads=configs.n_heads,
                    d_r=self.d_r,
                    attn_dropout=configs.dropout,  # self.attn_dropout,
                    # proj_dropout=self.proj_dropout
                ),
                c_in=self.c_in,
                d_r=self.d_r,
                d_model=self.encoder_hidden_dim,
                seq_len=self.encoder_input_dim,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            )
            for _ in range(configs.e_layers)]

        self.tr_proj = nn.Linear(self.encoder_input_dim, self.pred_len, bias=True)
        self.encoder = Encoder(self.enc_layers, norm=nn.LayerNorm(self.encoder_input_dim))

        self.w_dec_1 = torch.nn.Parameter(torch.FloatTensor([configs.w_lin] * configs.enc_in), requires_grad=True)
        self.w_dec = torch.nn.Parameter(torch.FloatTensor([configs.w_lin] * configs.enc_in), requires_grad=True)
        # self.revIn = RevIN(self.c_in)
        self.dct = torch_dct.dct

    def __dct(self, x: torch.Tensor):
        """
        bs, cs, ls = x.shape
        loop_num = cs
        if cs > bs:
            loop_num = bs
        dct_components = []
        for i in range(loop_num):
            if cs > bs:
                freq = self.dct(x[i])
            else:
                freq = self.dct(x[:, i, :])
            dct_components.append(freq)

        if cs > bs:
            freq = torch.stack(dct_components, dim=0).detach()
        else:
            freq = torch.stack(dct_components, dim=1).detach()
        """
        freq = self.dct(x)

        # mf = abs(freq).mean(dim=0).mean(dim=0)
        # _, indices = torch.topk(mf, self.k)
        # indices = indices.detach()

        # k_freq = freq[:, :, indices]  # .detach()
        k_freq = freq[:, :, :self.k]
        new_x = torch.cat([k_freq, x[:, :, -self.last_steps:]], dim=-1)

        freq = self.dct_norm_1(new_x)
        f_weight = self.dct_ffn(freq)
        f_weight = self.dct_norm_2(f_weight)
        return f_weight, new_x

    def forward(self, x: torch.Tensor, a, b, c):
        x_enc = x.transpose(-2, -1)
        # dct
        f_weight, enc_in = self.__dct(x_enc)

        enc_out, attn = self.encoder(enc_in)
        enc_out = self.tr_proj(enc_out).transpose(-2, -1)
        # enc_out = self.proj(enc_out).transpose(-2, -1)

        g_out = self.lin_proj(self.local_global(x_enc, f_weight)).transpose(-2, -1)

        output = g_out * self.w_dec_1 + enc_out * self.w_dec
        if self.output_attention:
            return output, attn
        return output
