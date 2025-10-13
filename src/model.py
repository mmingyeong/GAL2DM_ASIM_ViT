"""
model.py (VoxelViT3D: Custom 3D Vision Transformer for voxel-wise regression)

Description:
    Implementation of a custom 3D Vision Transformer (VoxelViT3D) model
    designed for dense voxel-to-voxel regression tasks in volumetric data.

    This architecture divides the 3D volume into patches along depth, height, and width,
    encodes them using transformer blocks, and decodes the output embeddings back into
    full-resolution 3D volumes.

    Applications:
    - Cosmic density field reconstruction
    - Medical 3D image translation
    - Scientific volume modeling and super-resolution

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-07-30
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# ----------------------------
# Helper blocks
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if (heads > 1 or dim_head != dim)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # [B, N, H*Dh] -> [B, H, N, Dh]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, N, N]
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                 # [B, H, N, Dh]
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads, dim_head, dropout),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ----------------------------
# VoxelViT3D
# ----------------------------
class ViT3D(nn.Module):
    """
    Voxel-wise 3D Vision Transformer (ViT) with dynamic shape checks and positional embedding sync.

    Inputs:
        x: FloatTensor of shape [B, C_in, D, H, W]
    Outputs:
        y: FloatTensor of shape [B, C_out, D, H, W]
    """

    def __init__(
        self,
        image_size: int = 128,          # H == W
        frames: int = 128,              # D
        image_patch_size: int = 16,     # must divide image_size
        frame_patch_size: int = 16,     # must divide frames
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        in_channels: int = 2,
        out_channels: int = 1,
        dim_head: int = 64,
    ):
        super().__init__()

        # ---- Cache hyperparams / basic checks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.frames = int(frames)            # D
        self.image_size = int(image_size)    # H == W
        self.frame_patch = int(frame_patch_size)
        self.image_patch = int(image_patch_size)
        self.dim = int(dim)

        # Divisibility checks
        assert self.frames % self.frame_patch == 0, \
            f"D({self.frames}) must be divisible by frame_patch_size({self.frame_patch})."
        assert self.image_size % self.image_patch == 0, \
            f"H/W({self.image_size}) must be divisible by image_patch_size({self.image_patch})."

        # Number of patches per axis
        self.nf = self.frames // self.frame_patch  # along depth
        self.nh = self.image_size // self.image_patch
        self.nw = self.image_size // self.image_patch
        self.num_patches = self.nf * self.nh * self.nw

        # ---- Patch embedding
        # Flattened patch volume size (per channel)
        patch_voxels = self.frame_patch * self.image_patch * self.image_patch
        patch_dim = self.in_channels * patch_voxels  # input features per patch

        # [B, C, (D pf), (H ps), (W ps)] -> [B, (D/pf * H/ps * W/ps), (C*pf*ps*ps)]
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (d pf) (h ph) (w pw) -> b (d h w) (c pf ph pw)",
                pf=self.frame_patch, ph=self.image_patch, pw=self.image_patch
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.dim),
            nn.LayerNorm(self.dim),
        )

        # ---- Positional embedding (dynamic length = num_patches)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # ---- Transformer encoder
        self.transformer = Transformer(dim=self.dim, depth=depth, heads=heads, dim_head=dim_head,
                                       mlp_dim=mlp_dim, dropout=dropout)

        # ---- Patch-to-voxel projection: [B, N, dim] -> [B, N, C_out * pf * ph * pw]
        self.to_voxel_patch = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.out_channels * patch_voxels),
        )

    # --------------- Forward ---------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, D, H, W]
        return: [B, C_out, D, H, W]
        """
        # 1) Shape & channel checks
        assert x.ndim == 5, f"Expected x as [B, C, D, H, W], got {tuple(x.shape)}"
        B, C, D, H, W = x.shape
        assert C == self.in_channels, f"in_channels mismatch: expected {self.in_channels}, got {C}"
        assert D == self.frames and H == self.image_size and W == self.image_size, \
            f"Input spatial mismatch: got (D,H,W)=({D},{H},{W}) vs expected " \
            f"({self.frames},{self.image_size},{self.image_size})"

        # 2) Patch embedding  -> [B, N, dim], with N = nf*nh*nw
        x = self.to_patch_embedding(x)

        # 3) Positional embedding (dynamic length sync)
        assert x.shape[1] == self.num_patches, \
            f"Mismatch in number of patches: x has {x.shape[1]}, expected {self.num_patches} " \
            f"(= {self.frames}//{self.frame_patch} * {self.image_size}//{self.image_patch} * {self.image_size}//{self.image_patch})"
        x = x + self.pos_embedding
        x = self.emb_dropout(x)

        # 4) Transformer encoder
        x = self.transformer(x)  # [B, N, dim]

        # 5) Project each patch back to voxel block
        x = self.to_voxel_patch(x)  # [B, N, C_out * pf * ph * pw]

        # 6) Fold patches back to volume
        # Recover axes: (nf, nh, nw) patches; each patch has (pf, ph, pw) voxels
        x = x.view(
            B,
            self.nf, self.nh, self.nw,
            self.out_channels,
            self.frame_patch, self.image_patch, self.image_patch
        )  # [B, nf, nh, nw, C_out, pf, ph, pw]

        # Permute to [B, C_out, nf, pf, nh, ph, nw, pw] and merge
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(
            B,
            self.out_channels,
            self.nf * self.frame_patch,
            self.nh * self.image_patch,
            self.nw * self.image_patch
        )  # [B, C_out, D, H, W]

        return x
