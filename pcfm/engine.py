"""Supervised training and evaluation for the physics-coupled model."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import format_scores, score_volumes, write_report
from .physics import coil_support, to_complex, zero_filled


def reconstruction_loss(model, batch, device, use_support: bool) -> torch.Tensor:
    image, measurement, mask, coilmap, _index = batch
    image = image.to(device, non_blocking=True)
    measurement = measurement.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    coilmap = coilmap.to(device, non_blocking=True)
    _combined, network_input = zero_filled(measurement, mask, coilmap)
    prediction = model(network_input, measurement, mask, coilmap)
    target = torch.cat([image.real, image.imag], dim=1)
    if use_support:
        support = coil_support(coilmap).to(dtype=prediction.dtype)
        prediction = prediction * support
        target = target * support
    return F.l1_loss(prediction, target)


@torch.no_grad()
def evaluate_loader(model, loader, device, use_support: bool) -> dict:
    model.eval()
    magnitudes = []
    targets = []
    supports = []
    indices = []
    for batch in loader:
        image, measurement, mask, coilmap, index = batch
        image = image.to(device, non_blocking=True)
        measurement = measurement.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        coilmap = coilmap.to(device, non_blocking=True)
        _combined, network_input = zero_filled(measurement, mask, coilmap)
        prediction = model(network_input, measurement, mask, coilmap)
        predicted = to_complex(prediction).abs()
        reference = image.abs()
        support = coil_support(coilmap)
        if use_support:
            predicted = predicted * support
            reference = reference * support
        magnitudes.append(predicted.squeeze(1).cpu().numpy())
        targets.append(reference.squeeze(1).cpu().numpy())
        supports.append(support.squeeze(1).cpu().numpy())
        indices.append(index.numpy() if torch.is_tensor(index) else np.asarray(index))
    order = np.concatenate(indices)
    sorted_at = np.argsort(order)
    dataset_index = order[sorted_at]
    prediction = np.concatenate(magnitudes)[sorted_at]
    target = np.concatenate(targets)[sorted_at]
    mask = np.concatenate(supports)[sorted_at] if use_support else None
    subject_ids = np.asarray(loader.dataset.subject_ids)[dataset_index]
    return score_volumes(target, prediction, subject_ids, mask)


def save_checkpoint(path: Path, model, config: dict, epoch: int, scores: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "scores": scores,
        },
        path,
    )


def train_model(model, loaders, config: dict, output_dir: Path, device: torch.device) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(config["lr"]), betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["max_epochs"]),
        eta_min=float(config["lr_min"]),
    )
    use_support = bool(config.get("use_support_mask", True))
    grad_clip = float(config.get("grad_clip", 0.0) or 0.0)
    log_path = output_dir / "metrics.csv"
    best_path = output_dir / "best.pt"
    best_psnr = -float("inf")

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_psnr", "val_psnr_std", "val_ssim", "val_ssim_std", "lr"],
        )
        writer.writeheader()
        for epoch in range(int(config["max_epochs"])):
            model.train()
            total = 0.0
            steps = 0
            for batch in loaders["train"]:
                loss = reconstruction_loss(model, batch, device, use_support)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                total += float(loss.detach())
                steps += 1
            scheduler.step()
            scores = evaluate_loader(model, loaders["val"], device, use_support)
            row = {
                "epoch": epoch,
                "train_loss": total / max(steps, 1),
                "val_psnr": scores["psnr_mean"],
                "val_psnr_std": scores["psnr_std"],
                "val_ssim": scores["ssim_mean"],
                "val_ssim_std": scores["ssim_std"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"epoch {epoch:03d}  loss {row['train_loss']:.4f}  {format_scores(scores).splitlines()[0]}",
                flush=True,
            )
            if scores["psnr_mean"] > best_psnr:
                best_psnr = scores["psnr_mean"]
                save_checkpoint(best_path, model, config, epoch, scores)
    return best_path


def load_model(checkpoint_path: Path, weight_path: Path, device: torch.device):
    from .model import PhysicsCoupledFM

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = PhysicsCoupledFM(
        weight_path=weight_path,
        img_size=int(config.get("img_size", 384)),
        adapter_blocks=config.get("adapter_blocks", (2, 5, 8, 11)),
        correction_width=int(config.get("correction_width", 16)),
        projection_bottleneck=int(config.get("projection_bottleneck", 16)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def test_model(model, loader, device, use_support: bool, report_path: Path) -> dict:
    scores = evaluate_loader(model, loader, device, use_support)
    write_report(report_path, scores)
    print(format_scores(scores), flush=True)
    return scores
