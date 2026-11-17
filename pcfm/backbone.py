"""Frozen RadioDINO-S (ViT-S/16). Weights come from the path in configs/paths.yaml."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_DTYPE = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float32,
    "I64": np.int64,
    "I32": np.int32,
    "BOOL": np.bool_,
}


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        blob = handle.read()
    tensors = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        start, end = info["data_offsets"]
        array = np.frombuffer(blob[start:end], dtype=_DTYPE[info["dtype"]])
        array = array.reshape(info["shape"]).copy()
        tensors[key] = torch.from_numpy(array)
    return tensors


def resolve_weight_file(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin"):
            candidate = path / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"RadioDINO weights not found at {path}. "
        "Set radiodino_weights in configs/paths.yaml. "
        "The checkpoint is the ViT-S/16 file from hf.co/Snarcy/RadioDino-s16."
    )


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        mixed = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(mixed)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(image).flatten(2).transpose(1, 2)


class RadioDINO(nn.Module):
    """ViT-S/16 whose parameter names match the RadioDINO-S checkpoint."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 384, depth: int = 12, num_heads: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patches = tokens.shape[1] - 1
        stored = self.pos_embed.shape[1] - 1
        if patches == stored and height == width:
            return self.pos_embed
        class_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = tokens.shape[-1]
        grid_h = height // self.patch_size + 0.1
        grid_w = width // self.patch_size + 0.1
        side = math.sqrt(stored)
        # The 0.1 offset is the DINO positional-interpolation convention.
        patch_pos = patch_pos.reshape(1, int(side), int(side), dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos,
            scale_factor=(grid_h / side, grid_w / side),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos, patch_pos), dim=1)

    def prepare_tokens(self, image: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        tokens = self.patch_embed(image)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, height, width)
        return tokens


def freeze_backbone(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def load_radio_dino(weight_path: Path, img_size: int = 384) -> RadioDINO:
    weight_file = resolve_weight_file(Path(weight_path))
    if weight_file.suffix != ".safetensors":
        raise ValueError(f"expected a .safetensors checkpoint, got {weight_file}")
    model = RadioDINO(img_size=224)
    state = _load_safetensors(weight_file)
    model.load_state_dict(state)
    if img_size % model.patch_size != 0:
        raise ValueError(f"img_size {img_size} must be divisible by {model.patch_size}")
    freeze_backbone(model)
    return model
