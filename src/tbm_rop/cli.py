"""Command-line interface.

Usage:
    tbm-rop prepare [--config CONFIG] [key=value ...]
    tbm-rop denoise [--config CONFIG] [key=value ...]
    tbm-rop tune    [--config CONFIG] [key=value ...]
    tbm-rop stack   [--config CONFIG] [key=value ...]
    tbm-rop all     [--config CONFIG] [key=value ...]   # prepare+denoise+stack (tune is explicit)

Overrides use OmegaConf dotlist syntax, e.g.::

    tbm-rop all kalman.mode=causal kalman.process_noise=0.1
"""

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from .pipeline import DEFAULT_STAGES, STAGES

DEFAULT_CONFIG = "run/conf/config.yaml"

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tbm-rop", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"], help="Pipeline stage to run")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides (key=value)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return 1

    cfg = OmegaConf.load(config_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    stages = list(DEFAULT_STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        logger.info(">>> Running stage: %s", stage)
        STAGES[stage](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
