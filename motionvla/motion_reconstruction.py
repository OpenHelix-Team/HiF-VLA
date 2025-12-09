import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

try:
    from apex.normalization import FusedLayerNorm
except:
    FusedLayerNorm = nn.LayerNorm
    print("Please 'pip install apex'")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) (B, H, N, C)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_qkv:
            return x, qkv

        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv, kernel_size=None, stride=None, pad=None, permute=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=0)

        self.pad = pad
        self.permute = permute

    def forward(self, x):
        if self.with_conv:
            if self.permute:
                x = x.permute(0, 4, 1, 2, 3)
            x = torch.nn.functional.pad(x, self.pad, mode="constant", value=0)
            x = self.conv(x)
            if self.permute:
                x = x.permute(0, 2, 3, 4, 1)
        else:
            raise NotImplementedError("Not implemented")

        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv=True, scale_factor=None):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        ori_dtype = x.dtype
        x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=self.scale_factor, mode="nearest")
        x = x.to(ori_dtype)
        if self.with_conv:
            x = self.conv(x)
        return x


class SpatioTemporalBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
            attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-6)):
        super().__init__()

        self.norm0 = norm_layer(dim)

        self.spatial_attention = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
        )

        self.norm1 = norm_layer(dim)
        self.temporal_attention = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = nn.Linear(dim, 2048)
        # self.mlp1 = nn.Linear(2048, dim)

    def forward(self, x):
        bs, t, seq_len, c = x.shape
        x = x.reshape(bs * t, seq_len, c).contiguous()
        x = x + self.spatial_attention(self.norm0(x))
        x = x.reshape(bs, t, seq_len, c).permute(0, 2, 1, 3).contiguous()
        x = x.reshape(bs * seq_len, t, c).contiguous()
        x = x + self.temporal_attention(self.norm1(x))
        x1 = x + self.mlp(self.norm2(x)).contiguous()
        x1 = x1.reshape(bs, seq_len, t, c).permute(0, 2, 1, 3).contiguous()
        return x1

class TokenizerDecoder(nn.Module):

    def __init__(self, in_channel=32, dim=512, num_heads=8, img_size=(36, 20), depth=2, mlp_ratio=4., qkv_bias=True, 
            qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5), num_frames=24):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(dim)
        self.in_proj = nn.Linear(in_channel, dim)
        self.dim = dim
        self.stage = [2]


    def forward(self, x):
        # x: bs, h, w, dim
        bs, t, h, w, dim = x.shape
        x = self.in_proj(x)


        i_block = 0

        for i_s in range(len(self.stage)):
            layer_num = self.stage[i_s]
            
            # upsample
            # x = x.permute(0, 4, 1, 2, 3)
            # x = self.up_sample_layers[i_s](x)   # x: bs, dim, t, h, w
            bs, t, h, w, dim = x.shape
            # x = x.permute(0, 2, 3, 4, 1)  # bs, t, h, w, c
            x = x.reshape(bs, t, h * w, dim)

            for i_b in range(i_block, i_block + layer_num):
                x = self.blocks[i_b](x)

            x = x.reshape(bs, t, h, w, dim)

            i_block = i_block + layer_num

        x = self.norm(x)    # the post norm, for next stage use,    # bs, t, h, w, c

        return x



class MotionTransformerTokenizer(nn.Module):
    def __init__(self,
                 decoder_config,
                 decoder_out_dim,
                #  n_embed=1024, 
                #  embed_dim=32,
                #  decay=0.99,
                #  quantize_kmeans_init=True,
                #  rec_loss_type='l2',
                #  **kwargs
                 ):
        """
        The motion tokenizer
        """
        super().__init__()

        self.decoder = TokenizerDecoder(**decoder_config)

        self.decoder_out_dim = decoder_out_dim

        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['dim'], decoder_config['dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['dim'], self.decoder_out_dim),
        )
    
    def decode(self, pred_motion, **kwargs):
        # input quantize: [bs, t, h, w, 32]
        decoder_features = self.decoder(pred_motion)
        rec = self.decode_task_layer(decoder_features) # [bs, t, h, w, 2]
        rec = rec.permute(0,1, 4, 2, 3).contiguous()  # [bs, t, 2, h, w]
        return rec


def get_motion_trans_model_params():
    return dict(in_channel=2, dim=256, num_heads=8, img_size=(36, 20), depth=2, mlp_ratio=4, qkv_bias=True, 
            qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=partial(FusedLayerNorm, eps=1e-5), num_frames=8)


def build_motion_tokenizer(pretrained_weight=None, n_code=1024, code_dim=32, **kwargs):

    encoder_config, decoder_config = get_motion_trans_model_params(), get_motion_trans_model_params()

    # decoder settings
    decoder_config['in_channel'] = code_dim
    decoder_out_dim = 2
    
    model = MotionTransformerTokenizer(encoder_config, decoder_config, decoder_out_dim, n_code, code_dim, **kwargs)

    return model


class MotionDecoder(nn.Module):
    def __init__(self, code_dim=32, out_dim=2, **kwargs):
        super().__init__()
        self.decoder_config = get_motion_trans_model_params()
        self.decoder_config['in_channel'] = code_dim
        self.out_dim = out_dim
        self.motion_tokenizer = MotionTransformerTokenizer(self.decoder_config, self.out_dim)

    def forward(self, x):
        # x: [bs, t, h, w, 2]
        return self.motion_tokenizer.decode(x)


class HisMotionEncoder(nn.Module):
    def __init__(self, in_channels=32, hidden_dim=1024, out_dim=2, num_frames=4, num_patches=64, **kwargs):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1), # (2,8,16,16) → (hidden_dim,4,8,8)
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.space_pos = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, num_frames, hidden_dim) * 0.02)
         # CLS token
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        B, T, C,H, W = x.shape
        x = x.permute(0,2,1,3,4)  # (B,C,T,H,W) for Conv3D


        feat = self.conv3d(x)  # (B, token_dim, T, H', W')
        B, D, T1, H1, W1 = feat.shape
        

        feat = feat.permute(0,2,3,4,1).reshape(B, T1*H1*W1, D)  # (B, N_tokens, D)
        
        space_pos = self.space_pos.unsqueeze(1).repeat(1, T1, 1, 1).reshape(1, T1*H1*W1, D)

        time_pos = self.time_pos.unsqueeze(2).repeat(1,1,H1*W1,1).reshape(1, T1*H1*W1, D)
        
        feat = feat + space_pos + time_pos  # (B, N_tokens, D)

        cls_tok = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        tokens = torch.cat([cls_tok, feat], dim=1)  # (B,1+N_tokens,D)
        

        tokens = self.transformer(tokens)  # (B,1+N_tokens,D)
        
        cls_out = tokens[:,0]  # (B,D)

        out = self.fc_out(cls_out)  # (B, out_dim)
        return out
