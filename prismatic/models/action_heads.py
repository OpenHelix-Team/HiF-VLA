"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
from sympy import im
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

from prismatic.models.create_diff import create_diffusion
from . import gaussian_diffusion as gd
from .models import DiT
from torch.nn import Module, ModuleList

from torch import Tensor
from motionvla.motion_reconstruction import MotionDecoder
from functools import partial
import torch.nn.functional as F
FusedLayerNorm = nn.LayerNorm

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action


class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps_train=50,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.num_diffusion_steps_train = num_diffusion_steps_train
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps_train, beta_schedule="squaredcos_cap_v2")
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred




# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

# Create ActionModel
class Cond_actionModel(nn.Module):
    def __init__(self, 
                 token_size, 
                 model_type, 
                 in_channels, 
                 future_action_window_size, 
                 past_action_window_size,
                 diffusion_steps = 50,
                 noise_schedule = 'squaredcos_cap_v2'
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_B(
                                        token_size = token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob = 0.1, 
                                        learn_sigma = learn_sigma, 
                                        future_action_window_size = future_action_window_size, 
                                        past_action_window_size = past_action_window_size
                                        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z, y):
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z, y)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(timestep_respacing = "ddim"+str(ddim_step), 
                                               noise_schedule = self.noise_schedule,
                                               diffusion_steps = self.diffusion_steps, 
                                               sigma_small = True, 
                                               learn_sigma = False
                                               )
        return self.ddim_diffusion
    


class MotionTokenManager(nn.Module):
    def __init__(self, llm_dim):
        super().__init__()
        self.motion_pred_token = nn.Parameter(torch.randn(1, NUM_ACTIONS_CHUNK, llm_dim) * 0.01)
        self.pos_embed = nn.Parameter(torch.randn(1, NUM_ACTIONS_CHUNK, llm_dim) * .02)
    
    def get_motion_token(self, batch_size):
        motion_token = self.motion_pred_token.repeat(batch_size, 1, 1) + self.pos_embed
        return motion_token


class MotionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4,
        hidden_dim=512,
        out_dim=2,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.model = MLPResNet(
            num_blocks=2, input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=out_dim
        )

    def forward(self, motion_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = motion_states.shape[0]
        rearranged_hidden_states = self.in_proj(motion_states)
        res = self.model(rearranged_hidden_states)
        res = res.permute(0,1, 4, 2, 3).contiguous()
        return res
    
    

class DoubleStream_Expert(nn.Module):

    def __init__(self, dim=768, n_heads=8, mlp_ratio=4.0, use_qknorm=True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_head = dim // n_heads
        mlp_dim = int(dim * mlp_ratio)

        self.rotary_emb = RotaryEmbedding(dim=int(dim/n_heads))


        self.norm1_1 = RMSNorm(dim)
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.attention = Attention(dim, n_heads, self.dim_head, use_qknorm)

        self.norm1_2 = RMSNorm(dim)
        self.qkv_linear_2 = nn.Linear(dim, dim * 3)
        
        self.norm2_1 = RMSNorm(dim)
        self.mlp_1 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

        self.norm2_2 = RMSNorm(dim)
        self.mlp_2 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

        self.modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, dim * 12)
        )

    def forward(self, x_stream1, x_stream2, p_emb):
        # import pdb; pdb.set_trace()

        mod_params = self.modulation_mlp(p_emb.squeeze(dim=1))

        mod1_scale, mod1_shift, mod1_gate, mod2_scale, mod2_shift, mod2_gate, mod3_scale, mod3_shift, mod3_gate, mod4_scale, mod4_shift, mod4_gate = mod_params.chunk(12, dim=1)
        
        # import pdb; pdb.set_trace()

        x_stream1_norm = self.norm1_1(x_stream1)#应该是layernorm
        x_modulated_1 = x_stream1_norm * (1 + mod1_scale.unsqueeze(1)) + mod1_shift.unsqueeze(1)

        x_stream2_norm = self.norm1_2(x_stream2)
        x_modulated_2 = x_stream2_norm * (1 + mod2_scale.unsqueeze(1)) + mod2_shift.unsqueeze(1)


        B, T, D = x_stream1.size()

        qkv = self.qkv_linear(x_modulated_1)
        q, k, v = qkv.chunk(3, dim=-1)

        qkv_2 = self.qkv_linear_2(x_modulated_2)
        q2, k2, v2= qkv_2.chunk(3, dim=-1)

        combined_q = torch.cat([q, q2], dim=1)
        combined_k = torch.cat([k, k2], dim=1)
        combined_v = torch.cat([v, v2], dim=1)
        
        attn_out = self.attention(combined_q, combined_k, combined_v, self.rotary_emb)
        
        x_1 = x_stream1 + attn_out[:,:T,:] * mod1_gate.unsqueeze(1)
        x_2 = x_stream2 + attn_out[:,T:,:] * mod2_gate.unsqueeze(1)
        
        x_norm_1 = self.norm2_1(x_1)
        x_modulated_norm_1= x_norm_1 * (1 + mod3_scale.unsqueeze(1)) + mod3_shift.unsqueeze(1)
        mlp_out_1 = self.mlp_1(x_modulated_norm_1)

        x_norm_2 = self.norm2_2(x_2)
        x_modulated_norm_2= x_norm_2 * (1 + mod4_scale.unsqueeze(1)) + mod4_shift.unsqueeze(1)
        mlp_out_2 = self.mlp_2(x_modulated_norm_2)
        
        x_1 = mlp_out_1 * mod3_gate.unsqueeze(1) + x_1 
        x_2 = mlp_out_2 * mod4_gate.unsqueeze(1) + x_2 
        
        return x_1, x_2
    

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding - RoPE
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)).bfloat16()

    def forward(self, x, seq_len: int):
        self.inv_freq = self.inv_freq.to(x.device)
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :] # [1, seq_len, dim]

    def apply_rotary_emb(self, q, k, freqs_cis):
        q_ = (q * freqs_cis.cos()) + (self._rotate_half(q) * freqs_cis.sin())
        k_ = (k * freqs_cis.cos()) + (self._rotate_half(k) * freqs_cis.sin())
        return q_, k_

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class QKNorm(nn.Module):
    """
    Query-Key norm
    """
    def __init__(self, dim_head: int):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, 1, 1, dim_head), 10.0))

    def forward(self, q, k):

        q = nn.functional.normalize(q, p=2, dim=-1)
        k = nn.functional.normalize(k, p=2, dim=-1)

        return q * self.scale, k * self.scale



class Attention(nn.Module):
    """
    集成了 RoPE 和 QK-Norm 的多头自注意力模块
    """
    def __init__(self, dim, n_heads, dim_head, use_qknorm=True):
        super().__init__()
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.use_qknorm = use_qknorm

        if self.use_qknorm:
            self.qknorm = QKNorm(dim_head)

        self.to_out = nn.Linear(dim, dim)

    def forward(self, q, k, v, rotary_emb):
        # import pdb; pdb.set_trace()
        B, N, _ = q.shape
        
        q = q.view(B, N, self.n_heads, self.dim_head).transpose(1, 2).contiguous()
        k = k.view(B, N, self.n_heads, self.dim_head).transpose(1, 2).contiguous()
        v = v.view(B, N, self.n_heads, self.dim_head).transpose(1, 2).contiguous()

        seq_len = q.shape[-2]
        freqs_cis = rotary_emb(q, seq_len=seq_len)
        q, k = rotary_emb.apply_rotary_emb(q, k, freqs_cis)


        if self.use_qknorm:
            q, k = self.qknorm(q, k)

        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)
                
        output = torch.einsum('bhij,bhjd->bhid', attn_probs, v)
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        # import pdb; pdb.set_trace()

        
        return self.to_out(output)
    

class JointExpert(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_dim=4096,
        hidden_dim=1024,
        action_dim=7,
    ):
        super().__init__()

        self.proj_act = nn.Linear(input_dim, hidden_dim)
        self.proj_mv = nn.Linear(input_dim, hidden_dim)
       
        blocks = [DoubleStream_Expert(dim=hidden_dim) for _ in range(depth)]
        self.blocks = ModuleList(blocks)
        dim_list=[hidden_dim,hidden_dim]
        norms = [RMSNorm(dims) for dims in dim_list]
        self.norms = ModuleList(norms)
        self.model = MLPResNet(
            num_blocks=2, input_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

        self.motion_decoder = MotionDecoder(code_dim=4,out_dim=2)
        # self.motion_decoder = nn.Sequential(nn.Linear(4, 8),nn.Tanh(),
        #                                     nn.Linear(8, 2))


    def forward(
        self,
        modality_tokens1,
        modality_tokens2,
        time_cond = None
    ):
        act = self.proj_act(modality_tokens1)
        mv = self.proj_mv(modality_tokens2)
        # modal_tokens=[act,mv]
        for block in self.blocks:
            act, mv = block(
                x_stream1 = act,
                x_stream2 = mv,
                p_emb = time_cond,
            )

        act = self.norms[0](act)
        mv = self.norms[1](mv)

        batch_size = act.shape[0]
        device = act.device
        rearranged_actions_hidden_states = act.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)

        motion = self.motion_decoder(mv.reshape(batch_size, NUM_ACTIONS_CHUNK, 16,16,4))  # [B, T, H, W, 2]

        return action,motion.contiguous()  # [B, T, 2, H, W]


