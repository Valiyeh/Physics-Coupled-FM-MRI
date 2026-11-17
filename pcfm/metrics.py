"""Subject-wise PSNR (dB) and SSIM (%), inside the coil-support mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _mean_normalize(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    image = np.abs(image)
    if mask is not None:
        image = image * mask
    mean = image.mean(axis=(-2, -1), keepdims=True)
    return image / np.maximum(mean, 1e-12)


def score_volumes(target: np.ndarray, prediction: np.ndarray, subject_ids: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Mean and standard deviation of per-subject means."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
        prediction = prediction[:, 0]
    if mask is not None:
        mask = np.asarray(mask)
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
    target = _mean_normalize(target, mask)
    prediction = _mean_normalize(prediction, mask)

    slice_psnr = []
    slice_ssim = []
    for gt, pred in zip(target, prediction):
        data_range = float(gt.max())
        slice_psnr.append(peak_signal_noise_ratio(gt, pred, data_range=data_range))
        slice_ssim.append(100.0 * structural_similarity(gt, pred, data_range=data_range))
    slice_psnr = np.asarray(slice_psnr, dtype=np.float64)
    slice_ssim = np.asarray(slice_ssim, dtype=np.float64)

    subject_psnr = []
    subject_ssim = []
    for subject in np.unique(subject_ids):
        chosen = subject_ids == subject
        subject_psnr.append(np.nanmean(slice_psnr[chosen]))
        subject_ssim.append(np.nanmean(slice_ssim[chosen]))
    subject_psnr = np.asarray(subject_psnr, dtype=np.float64)
    subject_ssim = np.asarray(subject_ssim, dtype=np.float64)
    return {
        "psnr_mean": float(np.nanmean(subject_psnr)),
        "psnr_std": float(np.nanstd(subject_psnr)),
        "ssim_mean": float(np.nanmean(subject_ssim)),
        "ssim_std": float(np.nanstd(subject_ssim)),
        "n_subjects": int(subject_psnr.size),
    }


def format_scores(scores: dict) -> str:
    return (
        f"PSNR {scores['psnr_mean']:.2f} ± {scores['psnr_std']:.2f} dB\n"
        f"SSIM {scores['ssim_mean']:.2f} ± {scores['ssim_std']:.2f} %\n"
        f"subjects {scores['n_subjects']}"
    )


def write_report(path: Path, scores: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_scores(scores) + "\n", encoding="utf-8")
