"""Checkpointing utils."""

from os import PathLike
from pathlib import Path

import orbax.checkpoint as ocp
import pyrallis

from avid.config import MainConfig
from avid.utils import debug_structure


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


if __name__ == '__main__':
    from avid.vit import TrainingRun

    run_dir = Path('logs') / 'e_form_mae_481'

    with open(run_dir / 'config.toml') as conf_file:
        config = pyrallis.cfgparsing.load(MainConfig, conf_file)

    mngr = ocp.CheckpointManager(
        run_dir.absolute() / 'final_ckpt' / 'ckpts',
        ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            enable_async_checkpointing=False,
            read_only=True,
            save_interval_steps=0,
            create=False,
        ),
    )

    run = TrainingRun(config)

    model = mngr.restore(
        mngr.best_step(),
        # args=ocp.args.StandardRestore(Checkpoint(run.state, run.seed, run.metrics_history, 0)),
    )
    debug_structure(model=model)
