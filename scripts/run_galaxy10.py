"""Hydra entry point for Galaxy10 DECaLS runs.

Defaults: ``data=galaxy10``, ``model=resnet50``, ``experiment=galaxy10_hidden_priors``.
Override on the command line as usual:

    python scripts/run_galaxy10.py seed=7
    python scripts/run_galaxy10.py seed=7 data=galaxy10_m12
    python scripts/run_galaxy10.py seed=7 model=resnet50_linear
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiclass_cwola.experiments.pipeline import run_experiment


@hydra.main(version_base=None, config_path="../configs", config_name="default_galaxy10")
def main(config: DictConfig) -> None:
    run_experiment(config)


if __name__ == "__main__":
    main()
