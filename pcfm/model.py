

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import freeze_backbone, load_radio_dino
from .physics import data_residual_image

ADAPTER_BLOCKS = (2, 5, 8, 11)
PATCH = 16


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    nn.init.trunc_normal_(tensor, std=std)


def unpatchify(patch_values: torch.Tensor, patch_size: int = PATCH) -> torch.Tensor:
    """(B, N, 2*p*p) -> (B, 2, H, W)."""
    batch, count, channels = patch_values.shape
    if channels != 2 * patch_size * patch_size:
        raise ValueError(f"expected {2 * patch_size * patch_size} channels, got {channels}")
    grid = int(count ** 0.5)
    if grid * grid != count:
        raise ValueError(f"{count} tokens is not a square grid")
    image = patch_values.reshape(batch, grid, grid, 2, patch_size, patch_size)
    image = image.permute(0, 3, 1, 4, 2, 5).contiguous()
    return image.reshape(batch, 2, grid * patch_size, grid * patch_size)


def patchify(image_2ch: torch.Tensor, patch_size: int = PATCH) -> torch.Tensor:
    """(B, 2, H, W) -> (B, N, 2*p*p)."""
    batch, channels, height, width = image_2ch.shape
    if channels != 2:
        raise ValueError(f"expected 2 image channels, got {channels}")
    if height % patch_size or width % patch_size:
        raise ValueError(f"image {(height, width)} is not divisible by {patch_size}")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = image_2ch.reshape(batch, 2, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.reshape(batch, grid_h * grid_w, 2 * patch_size * patch_size)


class TokenImageMap(nn.Module):
    """Stage-specific map between patch tokens and a 2-channel image.

    A bottleneck keeps four of these maps, together with the shared correction
    network, under 1% of the foundation-model parameter count.
    """

    def __init__(self, embed_dim: int, bottleneck: int, to_image: bool):
        super().__init__()
        patch_dim = 2 * PATCH * PATCH
        in_dim, out_dim = (embed_dim, patch_dim) if to_image else (patch_dim, embed_dim)
        self.to_image = to_image
        self.fc1 = nn.Linear(in_dim, bottleneck)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(bottleneck, out_dim)
        _trunc_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        _trunc_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, 1:, :]
        return unpatchify(self.fc2(self.act(self.fc1(patches))))

    def forward_image(self, delta: torch.Tensor, token_count: int) -> torch.Tensor:
        batch = delta.shape[0]
        update = delta.new_zeros(batch, token_count, self.fc2.out_features)
        update[:, 1:, :] = self.fc2(self.act(self.fc1(patchify(delta))))
        return update


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        _trunc_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(self.act(x))))


    def train(self, mode: bool = True):
        super().train(mode)
        freeze_backbone(self.backbone)
        return self

    def _report_parameters(self) -> None:
        foundation = sum(parameter.numel() for parameter in self.backbone.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        fraction = 100.0 * trainable / foundation
        print(
            f"RadioDINO frozen parameters: {foundation:,}\n"
            f"trainable parameters: {trainable:,} ({fraction:.2f}% of the foundation model)\n"
            f"adapters after blocks {list(self.adapter_blocks)}",
            flush=True,
        )

    def _check_inputs(self, image, measurement, mask, coilmap) -> None:
        expected = (self.img_size, self.img_size)
        for name, tensor in (("image", image), ("measurement", measurement), ("mask", mask), ("coilmap", coilmap)):
            if tuple(tensor.shape[-2:]) != expected:
                raise ValueError(f"{name} spatial size {tuple(tensor.shape[-2:])} != {expected}")
        if image.shape[1] != 2:
            raise ValueError(f"image must have 2 channels, got {tuple(image.shape)}")

    def forward(self, image, measurement, mask, coilmap) -> torch.Tensor:
        self._check_inputs(image, measurement, mask, coilmap)
        ones = torch.ones_like(image[:, :1])
        tokens = self.backbone.prepare_tokens(torch.cat([image, ones], dim=1))
        for index, block in enumerate(self.backbone.blocks):
            tokens = block(tokens)
            key = str(index)
            if key not in self.projections:
                continue
            projected = self.projections[key].forward_tokens(tokens)
            residual = data_residual_image(projected, measurement, mask, coilmap)
            updated = projected + self.lam[key] * self.correction(projected, residual)
            lifted = self.lifts[key].forward_image(updated - projected, tokens.shape[1])
            tokens = tokens + self.alpha[key] * lifted
        tokens = self.backbone.norm(tokens)
        return self.head.forward_tokens(tokens)
