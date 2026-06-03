from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multiclass_cwola.experiments.pipeline import run_experiment


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(config: DictConfig) -> None:
    config.task = "real"
    run_experiment(config)


if __name__ == "__main__":
    main()
