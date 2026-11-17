"""Multi-coil Cartesian operator A = M F S, and the zero-filled input."""

from __future__ import annotations

import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2, -1))
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2, -1))
    return torch.fft.fftshift(x, dim=(-2, -1))


def to_complex(image_2ch: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) real/imag -> (B, 1, H, W) complex."""
    return torch.complex(image_2ch[:, :1], image_2ch[:, 1:2])


def as_mask(mask: torch.Tensor) -> torch.Tensor:
    """(B, H, W) or (B, 1, H, W) -> float (B, 1, H, W)."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (B, 1, H, W) or (B, H, W), got {tuple(mask.shape)}")
    return mask.to(dtype=torch.float32)


def forward_operator(image_2ch: torch.Tensor, mask: torch.Tensor, coilmap: torch.Tensor) -> torch.Tensor:
    """A(x) = M F S x."""
    mask = as_mask(mask)
    return mask * fft2c(to_complex(image_2ch) * coilmap)


def adjoint_operator(kspace: torch.Tensor, mask: torch.Tensor, coilmap: torch.Tensor) -> torch.Tensor:
    """A^H(k) = S^H F^H M k, as a 2-channel image."""
    mask = as_mask(mask)
    image = ifft2c(mask * kspace)
    combined = (torch.conj(coilmap) * image).sum(dim=1, keepdim=True)
    return torch.cat([combined.real, combined.imag], dim=1)


def measured_kspace(measurement: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """y = M F(measurement). measurement is multi-coil image-domain data."""
    return as_mask(mask) * fft2c(measurement)


def data_residual_image(
    image_2ch: torch.Tensor,
    measurement: torch.Tensor,
    mask: torch.Tensor,
    coilmap: torch.Tensor,
) -> torch.Tensor:
    """A^H(y - A x), the k-space residual brought back to the image."""
    y = measured_kspace(measurement, mask)
    residual = y - forward_operator(image_2ch, mask, coilmap)
    return adjoint_operator(residual, mask, coilmap)


def zero_filled(measurement: torch.Tensor, mask: torch.Tensor, coilmap: torch.Tensor):
    """Coil-combined zero-filled image and its 2-channel real/imag view."""
    filled = ifft2c(measured_kspace(measurement, mask))
    combined = (torch.conj(coilmap) * filled).sum(dim=1, keepdim=True)
    image_2ch = torch.cat([combined.real, combined.imag], dim=1)
    return combined, image_2ch


def coil_support(coilmap: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) complex coil maps -> (B, 1, H, W) float support."""
    return (coilmap.abs().sum(dim=1, keepdim=True) > 0).to(dtype=coilmap.real.dtype)
