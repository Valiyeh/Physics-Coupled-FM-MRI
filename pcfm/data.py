"""FastMRI brain and SKM-TEA knee slices from the directories in configs/paths.yaml.

Expected layout, for acceleration R:

    <fastmri_root>/us{R}x/{train,val,test}/{T1,T2,FLAIR}.npz_mmap/
    <skmtea_root>/us{R}x/{train,val,test}/echo1.npz_mmap/

Each mmap directory holds image_fs.npy, image_us.npy, coilmaps.npy,
us_masks.npy, subject_ids.npy, and us_factor.npy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_MMAP_FILES = ("image_fs.npy", "image_us.npy", "coilmaps.npy", "us_masks.npy", "subject_ids.npy", "us_factor.npy")


def _open_mmap(directory: Path) -> dict[str, np.ndarray]:
    missing = [name for name in _MMAP_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} is missing {missing}")
    opened = {name[:-4]: np.load(directory / name, mmap_mode="r") for name in _MMAP_FILES}
    return opened


def _subject_blocks(subject_ids: np.ndarray):
    blocks = []
    start = 0
    count = len(subject_ids)
    while start < count:
        subject = int(subject_ids[start])
        stop = start + 1
        while stop < count and int(subject_ids[stop]) == subject:
            stop += 1
        blocks.append((start, stop))
        start = stop
    return blocks


def _subject_scales(image_fs: np.ndarray, subject_ids: np.ndarray, cache_file: Path) -> np.ndarray:
    if cache_file.is_file():
        cached = np.load(cache_file)
        if cached.shape == (len(subject_ids),) and np.isfinite(cached).all() and float(cached.min()) > 0:
            return cached.astype(np.float64)
    scales = np.empty(len(subject_ids), dtype=np.float64)
    for start, stop in _subject_blocks(subject_ids):
        peak = float(np.abs(np.asarray(image_fs[start:stop])).max()) + 1e-12
        scales[start:stop] = peak
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".tmp.npy")
    np.save(temporary, scales)
    temporary.replace(cache_file)
    return scales


class VolumePart:
    def __init__(self, directory: Path, acceleration: int, scale_mode: str, cache_file: Path | None):
        self.arrays = _open_mmap(directory)
        stored = int(np.asarray(self.arrays["us_factor"]).reshape(-1)[0])
        if stored != int(acceleration):
            raise ValueError(f"{directory} stores acceleration {stored}, requested {acceleration}")
        self.subject_ids = np.asarray(self.arrays["subject_ids"]).reshape(-1)
        self.count = int(self.subject_ids.shape[0])
        if scale_mode == "subject":
            if cache_file is None:
                raise ValueError("SKM-TEA subject scaling needs cache_dir from the paths file")
            self.scales = _subject_scales(self.arrays["image_fs"], self.subject_ids, cache_file)
        elif scale_mode == "slice":
            self.scales = None
        else:
            raise ValueError(scale_mode)

    def item(self, index: int):
        image = np.asarray(self.arrays["image_fs"][index])
        measurement = np.asarray(self.arrays["image_us"][index])
        coilmap = np.asarray(self.arrays["coilmaps"][index])
        mask = np.asarray(self.arrays["us_masks"][index], dtype=np.float32)
        if self.scales is None:
            scale = float(np.abs(image).max()) + 1e-12
        else:
            scale = float(self.scales[index])
        image = np.array(image / scale, copy=True)
        measurement = np.array(measurement / scale, copy=True)
        mask = np.array(mask, dtype=np.float32, copy=True)
        coilmap = np.array(coilmap, copy=True)
        return image, measurement, mask, coilmap


class ReconstructionDataset(Dataset):
    def __init__(self, parts: list[VolumePart]):
        self.parts = parts
        self._edges = [0]
        subject_ids = []
        offset = 0
        for part in parts:
            shifted = part.subject_ids.astype(np.int64) + offset
            offset = int(shifted.max()) + 1 if len(shifted) else offset
            subject_ids.append(shifted)
            self._edges.append(self._edges[-1] + part.count)
        self.subject_ids = np.concatenate(subject_ids) if subject_ids else np.zeros((0,), dtype=np.int64)

    def __len__(self) -> int:
        return int(self._edges[-1])

    def _locate(self, index: int):
        for part, start, stop in zip(self.parts, self._edges, self._edges[1:]):
            if start <= index < stop:
                return part, index - start
        raise IndexError(index)

    def __getitem__(self, index: int):
        part, local = self._locate(int(index))
        image, measurement, mask, coilmap = part.item(local)
        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        measurement = torch.from_numpy(np.ascontiguousarray(measurement))
        coilmap = torch.from_numpy(np.ascontiguousarray(coilmap))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0)
        return image, measurement, mask, coilmap, int(index)


def _split_directory(paths: dict, dataset: str, acceleration: int, split: str) -> Path:
    if dataset == "fastmri":
        return paths["fastmri_root"] / f"us{acceleration}x" / split
    if dataset == "skmtea":
        return paths["skmtea_root"] / f"us{acceleration}x" / split
    raise ValueError(f"dataset must be 'fastmri' or 'skmtea', got {dataset!r}")


def build_dataset(paths: dict, config: dict, split: str) -> ReconstructionDataset:
    dataset = str(config["dataset"]).lower()
    acceleration = int(config["acceleration"])
    root = _split_directory(paths, dataset, acceleration, split)
    if dataset == "fastmri":
        names = list(config.get("contrasts") or ["T1", "T2", "FLAIR"])
        scale_mode = "slice"
    else:
        names = ["echo1"]
        scale_mode = "subject"
    parts = []
    for name in names:
        directory = root / f"{name}.npz_mmap"
        cache = None
        if scale_mode == "subject":
            cache = paths["cache_dir"] / f"skmtea_r{acceleration}_{split}_scales.npy"
        parts.append(VolumePart(directory, acceleration, scale_mode, cache))
        print(f"{split}/{name}: {parts[-1].count} slices from {directory}", flush=True)
    return ReconstructionDataset(parts)


def make_loader(dataset: ReconstructionDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=int(num_workers) > 0,
    )
