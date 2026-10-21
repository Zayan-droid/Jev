"""``decider-train experiment=<name> [overrides...]``: train, calibrate, evaluate, bundle.

Resume an interrupted run by pointing at its directory: ``run_dir=runs/<name>/<timestamp>``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from decider.cli._common import CONFIG_DIR, prepare_environment
from decider.training.trainer import RunConfig, Trainer


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    prepare_environment()
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    Trainer(RunConfig.from_dict(container), container).fit()  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
