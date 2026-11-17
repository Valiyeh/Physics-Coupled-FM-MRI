"""Evaluate a trained checkpoint on the test split.

    python evaluate.py --paths configs/paths.yaml --config configs/fastmri_r4.yaml
"""

from __future__ import annotations

import argparse

import torch

from pcfm.data import build_dataset, make_loader
from pcfm.engine import load_model, test_model
from pcfm.paths import load_paths, load_yaml, run_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Test physics-coupled RadioDINO reconstruction")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    config = load_yaml(args.config)
    checkpoint = args.checkpoint or (paths["output_dir"] / run_name(config) / "best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _saved, _checkpoint = load_model(checkpoint, paths["radiodino_weights"], device)

    test_set = build_dataset(paths, config, "test")
    loader = make_loader(test_set, int(config["batch_size"]), int(config["num_workers"]), shuffle=False)
    report = paths["output_dir"] / run_name(config) / "test_report.txt"
    test_model(model, loader, device, bool(config.get("use_support_mask", True)), report)
    print(f"report: {report}", flush=True)


if __name__ == "__main__":
    main()
