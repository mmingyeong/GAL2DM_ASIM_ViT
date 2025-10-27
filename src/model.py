# -*- coding: utf-8 -*-
"""
VoxelViTUNet3D: Hybrid CNN + Vision Transformer + U-Net style decoder for 3D voxel-wise regression.

This implementation is adapted from (and inspired by) ViT-V-Net for 3D Image Registration:
- Code: https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch/blob/main/ViT-V-Net/models.py
- Paper: Chen et al., "ViT-V-Net: Vision Transformer for Unsupervised Volumetric Medical Image Registration"
         (arXiv:2104.06468) https://arxiv.org/abs/2104.06468

Differences from the reference:
- Registration-specific components (deformation fields, SpatialTransformer, VecInt) are removed.
- Final head produces scalar field(s) for voxel-wise regression (e.g., cosmic density map).
- Patch embedding via Conv3d with configurable stride allows overlapping patches (reduces block artifacts).
- U-Net style decoder with skip connections from the CNN encoder to recover fine details.

Author: Mingyeong Yang (UST-KASI), 2025-07-30 (modified for voxel regression in 2025-10-14)
"""

from __future__ import annotations
import math
import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)  # 프로젝트 표준 로거로 교체 가능 (예: get_logger(__name__))


# ----------------------------
# Utility / Init
# ----------------------------
def _triple(x: int | Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    x = tuple(x)
    assert len(x) == 3, f"patch/stride tuple must be length 3, got {x}"
    return x  # type: ignore


def init_weights(module: nn.Module) -> None:
    """Xavier for Linear, Kaiming for Conv3d; zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if module.weight is not None:
            nn.init.ones_(module.weight)


# ----------------------------
# Blocks
# ----------------------------
class Conv3dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm
        )
        bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        relu = nn.ReLU(inplace=True)
        super().__init__(conv, bn, relu)


class DoubleConv(nn.Module):
    """(Conv3d -> ReLU) x 2 (BN optional inside Conv3dReLU)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            Conv3dReLU(in_channels, out_channels, 3, 1),
            Conv3dReLU(out_channels, out_channels, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool3d(2) + DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    Lightweight 3-level encoder producing multi-scale features for skip connections.
    Output features are returned in high->low order for convenience in the decoder.
    """
    def __init__(self, in_channels: int, encoder_channels: Sequence[int] = (32, 64, 128), extra_pools: int = 0):
        super().__init__()
        assert len(encoder_channels) >= 3, "encoder_channels must have at least 3 levels"
        self.inc = DoubleConv(in_channels, encoder_channels[0])
        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        self.down2 = Down(encoder_channels[1], encoder_channels[2])
        self.extra_pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(extra_pools)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        f1 = self.inc(x)       # highest resolution
        f2 = self.down1(f1)
        f3 = self.down2(f2)    # lowest (encoder) resolution
        feats: List[torch.Tensor] = [f1, f2, f3]
        cur = f3
        # optional deeper pools if needed by patch embedding down_factor
        for pool in self.extra_pools:
            cur = pool(cur)
            feats.append(cur)
        # return lowest feature for patch embedding + reversed skip list (low->high for simple indexing)
        return feats[-1], feats[::-1]


# ----------------------------
# Transformer
# ----------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        inner = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        B, N, _ = x.shape
        H = self.heads
        q = q.view(B, N, H, -1).transpose(1, 2)  # [B,H,N,Dh]
        k = k.view(B, N, H, -1).transpose(1, 2)
        v = v.view(B, N, H, -1).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [B,H,N,Dh]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, drop: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleList([Attention(dim, heads, dim_head, drop, drop), FeedForward(dim, mlp_dim, drop)]) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.norm(x)


# ----------------------------
# Embedding (Conv3d patch, stride<=patch for overlap)
# ----------------------------
class ConvPatchEmbedding3D(nn.Module):
    """
    Conv3d-based patch embedding:
      - kernel_size = patch_size, stride = patch_stride (<= patch_size for overlap)
      - input feature map is typically the CNN encoder's lowest-resolution feature.
    """
    def __init__(self, in_channels: int, dim: int, patch_size: Sequence[int], patch_stride: Optional[Sequence[int]] = None):
        super().__init__()
        pf, ph, pw = _triple(patch_size)
        if patch_stride is None:
            sf, sh, sw = pf, ph, pw
        else:
            sf, sh, sw = _triple(patch_stride)
        self.patch_size = (pf, ph, pw)
        self.patch_stride = (sf, sh, sw)
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=(pf, ph, pw), stride=(sf, sh, sw), bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x = self.proj(x)  # [B, dim, D', H', W']
        B, C, Dp, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, N, dim], N = D'*H'*W'
        return tokens, (Dp, Hp, Wp)


# ----------------------------
# Decoder (U-Net style)
# ----------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int = 0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # spatial size align (just in case)
            if x.shape[-3:] != skip.shape[-3:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder3D(nn.Module):
    """
    Expect input as [B, C, D', H', W'] coming from reshaped Transformer tokens.
    Produces [B, out_channels, D, H, W].
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        skip_channels: Sequence[int] = (128, 64, 32),
        decoder_channels: Sequence[int] = (256, 128, 64),
    ):
        super().__init__()
        assert len(decoder_channels) == len(skip_channels), "decoder_channels and skip_channels must match"
        self.in_proj = Conv3dReLU(in_ch, decoder_channels[0], 3, 1)
        blocks = []
        for i in range(len(decoder_channels) - 1):
            blocks.append(DecoderBlock(decoder_channels[i], decoder_channels[i + 1], skip_channels[i + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.out_head = nn.Conv3d(decoder_channels[-1], out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """
        skips: list of encoder features in low->high order, e.g., [f_lowest, f_mid, f_highest]
               we align indices so that later blocks see higher-res skips.
        """
        x = self.in_proj(x)
        cur = x
        for i, block in enumerate(self.blocks):
            skip = skips[i + 1] if (i + 1) < len(skips) else None
            cur = block(cur, skip)
        y = self.out_head(cur)
        return y


# ----------------------------
# Main model
# ----------------------------
class VoxelViTUNet3D(nn.Module):
    """
    CNN stem -> Conv3d patch embedding (stride<=patch) -> Transformer -> U-Net decoder -> voxel-wise map.

    Args
    ----
    image_size: Tuple[int,int,int] = (D,H,W)
    in_channels: int = 2
    out_channels: int = 1
    patch_size: int|Tuple[int,int,int] = 8
        Kernel size for Conv3d patch embedding.
    patch_stride: Optional[int|Tuple[int,int,int]] = None
        Stride for Conv3d patch embedding. If None, equals patch_size (non-overlapping).
        Use smaller stride (e.g., patch_size//2) for overlapping patches.
    dim: int = 256
    depth: int = 3
    heads: int = 8
    mlp_dim: int = 512
    dim_head: int = 64
    encoder_channels: Sequence[int] = (32, 64, 128)
    decoder_channels: Sequence[int] = (256, 128, 64)
    """

    def __init__(
        self,
        image_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 2,
        out_channels: int = 1,
        patch_size: int | Sequence[int] = 8,
        patch_stride: Optional[int | Sequence[int]] = None,
        dim: int = 256,
        depth: int = 3,
        heads: int = 8,
        mlp_dim: int = 512,
        dim_head: int = 64,
        encoder_channels: Sequence[int] = (32, 64, 128),
        decoder_channels: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = tuple(image_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        # 1) CNN encoder (stem)
        self.encoder = CNNEncoder(in_channels=in_channels, encoder_channels=encoder_channels, extra_pools=0)

        # 2) Conv3d patch embedding (from the lowest encoder feature)
        enc_low_ch = encoder_channels[-1]
        self.patch_embed = ConvPatchEmbedding3D(
            in_channels=enc_low_ch, dim=dim, patch_size=patch_size, patch_stride=patch_stride
        )

        # 3) Positional embedding (learned; size determined at runtime after patching)
        #    (원한다면 sin-cos 3D로 교체 가능)
        self.pos_embedding: Optional[nn.Parameter] = None  # lazy-create on first forward
        self.emb_drop = nn.Dropout(dropout)

        # 4) Transformer encoder
        self.transformer = TransformerEncoder(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, drop=dropout)

        # 5) Token grid -> 3D feature grid -> U-Net decoder -> output
        #    decoder skip_channels는 encoder의 feature 채널과 정렬되도록 구성
        skip_channels = [f for f in encoder_channels[::-1]]  # low->high order
        self.decoder = UNetDecoder3D(in_ch=dim, out_ch=out_channels, skip_channels=skip_channels, decoder_channels=decoder_channels)

        # init
        self.apply(init_weights)

    # --------- helpers ---------
    def _ensure_pos_embed(self, N: int, device: torch.device) -> None:
        """Ensure learned PE exists with correct length and lives on the same device as tokens."""
        if (self.pos_embedding is None) or (self.pos_embedding.shape[1] != N):
            logger.info(f"[VoxelViTUNet3D] Init/resize positional embedding: N={N}, dim={self.dim}, device={device}")
            pe = torch.zeros(1, N, self.dim, device=device)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embedding = nn.Parameter(pe)
        elif self.pos_embedding.device != device:
            # 이동만 필요한 경우
            self.pos_embedding = nn.Parameter(self.pos_embedding.to(device), requires_grad=True)

    # --------- forward ---------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, D, H, W]  ->  y: [B, C_out, D, H, W]
        """
        assert x.ndim == 5, f"Expected [B,C,D,H,W], got {tuple(x.shape)}"
        B, C, D, H, W = x.shape
        assert C == self.in_channels, f"in_channels mismatch: expected {self.in_channels}, got {C}"
        assert (D, H, W) == tuple(self.image_size), f"Input spatial mismatch: got {(D,H,W)} vs expected {self.image_size}"

        # 1) CNN encoder
        enc_low, skips = self.encoder(x)  # enc_low = lowest; skips = [low, mid, high]

        # 2) Patch embedding -> tokens
        tokens, grid_shape = self.patch_embed(enc_low)   # tokens: [B, N, dim], grid_shape=(Dp,Hp,Wp)
        N = tokens.shape[1]
        self._ensure_pos_embed(N, tokens.device)
        tokens = self.emb_drop(tokens + self.pos_embedding)  # [B, N, dim]

        # 3) Transformer
        tokens = self.transformer(tokens)  # [B, N, dim]

        # 4) Tokens -> 3D feature grid
        Dp, Hp, Wp = grid_shape
        feat3d = tokens.transpose(1, 2).contiguous().view(B, self.dim, Dp, Hp, Wp)  # [B, dim, D',H',W']

        # 5) Decoder (+ skip connections)
        y = self.decoder(feat3d, skips)  # [B, out_channels, D, H, W] (decoder 내에서 업샘플하여 원해상도 도달)

        # 보정(드문 케이스): 디코더 업샘플/패딩으로 1-2 voxel 차이 발생 시 interpolate로 정합
        if y.shape[-3:] != (D, H, W):
            logger.warning(f"[VoxelViTUNet3D] Output shape {y.shape[-3:]} != input {(D,H,W)}; resizing by trilinear.")
            y = F.interpolate(y, size=(D, H, W), mode="trilinear", align_corners=False)

        return y
