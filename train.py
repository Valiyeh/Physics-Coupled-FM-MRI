"""Train the physics-coupled reconstruction model.

    python train.py --paths configs/paths.yaml --config configs/fastmri_r4.yaml
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from pcfm.data import build_dataset, make_loader
from pcfm.engine import train_model
from pcfm.model import PhysicsCoupledFM
from pcfm.paths import load_paths, load_yaml, run_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physics-coupled RadioDINO reconstruction")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_dataset(paths, config, "train")
    val_set = build_dataset(paths, config, "val")
    batch_size = int(config["batch_size"])
    workers = int(config["num_workers"])
    loaders = {
        "train": make_loader(train_set, batch_size, workers, shuffle=True),
        "val": make_loader(val_set, batch_size, workers, shuffle=False),
    }
    model = PhysicsCoupledFM(
        weight_path=paths["radiodino_weights"],
        img_size=int(config.get("img_size", 384)),
        adapter_blocks=config.get("adapter_blocks", (2, 5, 8, 11)),
        correction_width=int(config.get("correction_width", 16)),
        projection_bottleneck=int(config.get("projection_bottleneck", 16)),
    ).to(device)

    output_dir = paths["output_dir"] / run_name(config)
    best = train_model(model, loaders, config, output_dir, device)
    print(f"best checkpoint: {best}", flush=True)


if __name__ == "__main__":
    main()
