"""Checkpointing utils."""

from os import PathLike
from pathlib import Path

import orbax.checkpoint as ocp
import pyrallis

from facet.config import MainConfig


def run_config(run_dir: PathLike):
    with open(Path(run_dir) / 'config.toml') as conf_file:
        config = pyrallis.cfgparsing.load(MainConfig, conf_file)
    return config


def best_ckpt(run_dir: PathLike):
    mngr = ocp.CheckpointManager(
        Path(run_dir).absolute() / 'final_ckpt' / 'ckpts',
        ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            enable_async_checkpointing=False,
            read_only=True,
            save_interval_steps=0,
            create=False,
        ),
    )

    model = mngr.restore(mngr.best_step())
    return model
