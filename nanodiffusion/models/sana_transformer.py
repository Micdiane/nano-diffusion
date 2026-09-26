# Copyright 2026 The HuggingFace Team. All rights reserved.
# Copyright 2026 Micdiane.
# SPDX-License-Identifier: Apache-2.0
"""Readable SANA DiT, adapted from Diffusers' Apache-2.0 implementation.

This is the actual pretrained architecture: ReLU linear self-attention, text
cross-attention, spatial GLUMBConv and adaptive layer normalization. Parameter
names match the checkpoint; loading is strict, with no randomly initialized gaps.
Only the original SANA configuration (patch size 1, no embedded guidance) is in scope.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, inputs, width, activation):
        super().__init__()
        self.linear_1 = nn.Linear(inputs, width)
        self.linear_2 = nn.Linear(width, width)
        self.activation = activation

    def forward(self, x):
        return self.linear_2(self.activation(self.linear_1(x)))


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Preserve the original checkpoint hierarchy without importing model code.
        self.emb = nn.Module()
        self.emb.timestep_embedder = MLP(256, width, F.silu)
        self.linear = nn.Linear(width, 6 * width)

    def forward(self, timestep, dtype):
        exponent = -math.log(10000) * torch.arange(128, device=timestep.device, dtype=torch.float32) / 128
        phase = timestep[:, None].float() * exponent.exp()[None]
        sinusoid = torch.cat([phase.cos(), phase.sin()], dim=-1).to(dtype)
        embedded = self.emb.timestep_embedder(sinusoid)
        return self.linear(F.silu(embedded)), embedded


class CaptionNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        variance = x.float().square().mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + 1e-5)).to(self.weight.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, width, heads, head_dim, *, cross=False, bias=False):
        super().__init__()
        self.heads, self.head_dim, self.cross = heads, head_dim, cross
        inner = heads * head_dim
        self.to_q = nn.Linear(width, inner, bias=bias)
        self.to_k = nn.Linear(width, inner, bias=bias)
        self.to_v = nn.Linear(width, inner, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(inner, width))

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        if self.cross:
            # Text cross-attention remains ordinary softmax attention.
            q, k, v = [t.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2) for t in (q, k, v)]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out = out.transpose(1, 2).flatten(2)
        else:
            # Associate as (V K^T) Q: never construct the spatial N x N matrix.
            q = q.transpose(1, 2).unflatten(1, (self.heads, self.head_dim)).relu().float()
            k = k.transpose(1, 2).unflatten(1, (self.heads, self.head_dim)).transpose(2, 3).relu().float()
            v = v.transpose(1, 2).unflatten(1, (self.heads, self.head_dim)).float()
            v = F.pad(v, (0, 0, 0, 1), value=1.0)
            out = (v @ k) @ q
            out = out[:, :, :-1] / (out[:, :, -1:] + 1e-15)
            out = out.flatten(1, 2).transpose(1, 2).to(x.dtype)
        out = self.to_out(out)
        return out.clamp(-65504, 65504) if x.dtype == torch.float16 and not self.cross else out


class GLUMBConv(nn.Module):
    def __init__(self, width, ratio):
        super().__init__()
        hidden = int(width * ratio)
        self.conv_inverted = nn.Conv2d(width, 2 * hidden, 1)
        self.conv_depth = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden)
        self.conv_point = nn.Conv2d(hidden, width, 1, bias=False)

    def forward(self, x):
        value, gate = self.conv_depth(F.silu(self.conv_inverted(x))).chunk(2, dim=1)
        return self.conv_point(value * F.silu(gate))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config['num_attention_heads'] * config['attention_head_dim']
        self.norm1 = nn.LayerNorm(width, eps=config['norm_eps'], elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, eps=config['norm_eps'], elementwise_affine=False)
        self.attn1 = Attention(width, config['num_attention_heads'], config['attention_head_dim'], bias=config['attention_bias'])
        self.attn2 = Attention(width, config['num_cross_attention_heads'], config['cross_attention_head_dim'], cross=True, bias=True)
        self.ff = GLUMBConv(width, config['mlp_ratio'])
        self.scale_shift_table = nn.Parameter(torch.empty(6, width))

    def forward(self, x, caption, mask, modulation, height, width):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.scale_shift_table[None] + modulation.reshape(x.shape[0], 6, -1)
        ).chunk(6, dim=1)
        h = (self.norm1(x) * (1 + scale_a) + shift_a).to(x.dtype)
        x = x + gate_a * self.attn1(h)
        # Keep the reference operand order: it also determines the output layout,
        # which can change the GEMM selected by subsequent bf16 projections.
        x = self.attn2(x, caption, mask) + x
        h = self.norm2(x) * (1 + scale_m) + shift_m
        h = h.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        return x + gate_m * self.ff(h).flatten(2).transpose(1, 2)


class SanaTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        unsupported = {'patch_size': 1, 'norm_elementwise_affine': False, 'dropout': 0.0,
                       'guidance_embeds': False, 'qk_norm': None, 'interpolation_scale': None}
        for name, supported in unsupported.items():
            if config.get(name, supported) != supported:
                raise ValueError(f'Unsupported SANA configuration: {name}={config[name]}')
        self.config = config
        width = config['num_attention_heads'] * config['attention_head_dim']
        if config['cross_attention_dim'] != width:
            raise ValueError('cross_attention_dim must equal model width')
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(config['in_channels'], width, 1)
        self.time_embed = TimeEmbedding(width)
        self.caption_projection = MLP(config['caption_channels'], width, lambda x: F.gelu(x, approximate='tanh'))
        self.caption_norm = CaptionNorm(width)
        self.transformer_blocks = nn.ModuleList([Block(config) for _ in range(config['num_layers'])])
        self.scale_shift_table = nn.Parameter(torch.empty(2, width))
        self.proj_out = nn.Linear(width, config['out_channels'])

    @classmethod
    def from_pretrained(cls, directory, *, device='cuda', dtype=torch.bfloat16):
        from safetensors.torch import load_file
        directory = Path(directory)
        config = json.loads((directory / 'config.json').read_text())
        with torch.device('meta'):
            model = cls(config)
        state = load_file(str(directory / 'diffusion_pytorch_model.safetensors'), device='cpu')
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def forward(self, latents, timestep, caption, mask):
        batch, _, height, width = latents.shape
        x = self.patch_embed.proj(latents).flatten(2).transpose(1, 2)
        modulation, embedded = self.time_embed(timestep, x.dtype)
        caption = self.caption_norm(self.caption_projection(caption))
        bias = ((1 - mask.to(x.dtype)) * -10000)[:, None, None, :]
        for block in self.transformer_blocks:
            x = block(x, caption, bias, modulation, height, width)
        shift, scale = (self.scale_shift_table[None] + embedded[:, None]).chunk(2, dim=1)
        x = F.layer_norm(x, (x.shape[-1],), eps=1e-6) * (1 + scale) + shift
        return self.proj_out(x).transpose(1, 2).reshape(batch, -1, height, width)
