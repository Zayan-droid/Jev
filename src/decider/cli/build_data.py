"""``decider-build-data data=<name>``: build a versioned dataset (see configs/data/)."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from decider.cli._common import CONFIG_DIR, prepare_environment
from decider.data.build import BuildConfig, build_dataset

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="build_data")
def main(cfg: DictConfig) -> None:
    prepare_environment()
    data = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data, dict)
    out_dir = build_dataset(BuildConfig.from_dict(data))  # type: ignore[arg-type]
    log.info("dataset written to %s", out_dir)


if __name__ == "__main__":
    main()
