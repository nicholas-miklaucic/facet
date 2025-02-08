"""Evaluates a checkpoint on the train/test/valid data."""
from functools import cache
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from tqdm import tqdm
from facet.data.databatch import CrystalGraphs
from facet.data.dataset import dataloader
from facet.layers import Context
import pyrallis
from facet.config import MainConfig
import orbax.checkpoint as ocp

from facet.training_state import TrainingRun
from facet.checkpointing import best_ckpt

from pathlib import Path
import pyrallis
from facet.config import MainConfig
import orbax.checkpoint as ocp

from facet.training_state import TrainingRun
from facet.checkpointing import best_ckpt


@cache
def load_model_and_params(run_dir: Path):
    """Given path to folder, returns (config, model, params)"""
    conf_file = run_dir / 'config.toml'

    with open(conf_file) as f:
        config = pyrallis.cfgparsing.load(MainConfig, f)

    ckpt = best_ckpt(run_dir)
    ema_params = ckpt['state']['opt_state'][-1]['ema']

    model = config.build_regressor()
    return (config, model, ema_params)


def eval_model(
    run_dir: Path, split: Literal['train', 'valid', 'test'], batch_size: int | None = None
):
    config, model, params = load_model_and_params(run_dir)

    if batch_size is not None:
        config.batch_size = batch_size

    num_data, dl = dataloader(config, split)

    predict = jax.pmap(lambda cg: model.apply(params, cg=cg, ctx=Context(training=False)))

    yhats = []
    ys = []
    ds_ids = []
    for batch in tqdm(dl, total=num_data // 3):
        yhats.append(np.array(predict(batch)[batch.padding_mask].tolist()).reshape(-1))
        ys.append(np.array(batch.e_form[batch.padding_mask].tolist()).reshape(-1))
        ds_ids.append(
            np.array(batch.graph_data.dataset_id[batch.padding_mask].tolist()).reshape(-1)
        )

    yhat = np.concatenate(yhats)
    y = np.concatenate(ys)
    ds_id = np.concatenate(ds_ids)

    df = pd.DataFrame({'dataset_id': ds_id, 'target': y, 'prediction': yhat})
    df.to_feather(Path('data') / f'{run_dir.stem}_{split}.feather')

    print('MAE: {:.2f} meV'.format(jnp.abs(y - yhat).mean() * 1e3))


if __name__ == '__main__':
    for split in ('train', 'test', 'valid'):
        eval_model(Path('logs') / 'enb-198', split)
    # eval_model(Path('logs') / 'enb-159', 'test')
