from dataclasses import field
import functools as ft
import random
from re import T
import time
from collections import defaultdict
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from shutil import copytree
from typing import Any, Mapping, Optional, Sequence

import chex
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyrallis
from clu import metrics as cmetrics
from flax import struct
from flax import linen as nn
from flax.training import train_state

from avid.checkpointing import best_ckpt
from avid.config import LossConfig, MainConfig
from avid.dataset import DataBatch, dataloader
from avid.utils import item_if_arr

@struct.dataclass
class Metrics:
    totals: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs):
        totals = self.totals.copy()
        counts = self.counts.copy()
        for k, v in kwargs.items():
            totals[k] = totals.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1

        return Metrics(totals, counts)

    def items(self):
        return {k: item_if_arr(self.totals[k] / self.counts[k]) for k in self.totals}.items()

class TrainState(train_state.TrainState):
    metrics: Metrics
    last_grad_norm: float


def create_train_state(module: nn.Module, optimizer, rng, batch: DataBatch):
    loss, params = module.init_with_output(rng, batch, training=False)
    tx = optimizer
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=Metrics(),
        last_grad_norm=0,
    )


@struct.dataclass
class Checkpoint:
    state: TrainState
    seed: int
    metrics_history: Mapping[str, Sequence[float]]
    curr_epoch: float


class TrainingRun:
    def __init__(self, config: MainConfig):
        self.seed = random.randint(100, 1000)
        self.rng = {'params': jax.random.key(self.seed)}
        if config.task == 'diled':
            self.rng['params'], self.rng['noise'], self.rng['time'] = jax.random.split(
                self.rng['params'], 3
            )
        print(f'Seed: {self.seed}')
        self.config = config

        self.metrics_history = defaultdict(list)
        self.num_epochs = config.num_epochs
        self.steps_in_epoch, self.dl = dataloader(config, split='train', infinite=True)
        self.steps_in_test_epoch, self.test_dl = dataloader(config, split='valid', infinite=True)
        self.num_steps = self.steps_in_epoch * self.num_epochs
        self.steps = range(self.num_steps)
        self.curr_step = 0
        self.start_time = time.monotonic()
        self.scheduler = self.config.train.lr_schedule(
            num_epochs=self.num_epochs, steps_in_epoch=self.steps_in_epoch
        )
        self.optimizer = self.config.train.optimizer(self.scheduler)
        self.model = self.make_model()
        self.metrics = Metrics()

        if config.restart_from is not None:
            self.state = self.state.replace(
                params=best_ckpt(config.restart_from)['state']['params']
            )

        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            best_fn=lambda metrics: metrics['te_loss'],
            best_mode='min',
            max_to_keep=4,
            enable_async_checkpointing=False,
            keep_time_interval=timedelta(minutes=30),
        )
        self.mngr = ocp.CheckpointManager(
            ocp.test_utils.create_empty('/tmp/jax_ckpt'),
            options=opts,
        )

        self.test_loss = 1000

    @staticmethod
    @ft.partial(jax.jit, static_argnames=('task', 'config'))
    @chex.assert_max_traces(5)
    def compute_metrics(*, task: str, config: LossConfig, state: TrainState, batch: DataBatch, rng):
        if task == 'e_form':
            preds = state.apply_fn(state.params, batch, training=False, rngs=rng).squeeze()
            loss = config.regression_loss(preds, batch.e_form)
            mae = jnp.abs(preds - batch.e_form).mean()
            rmse = jnp.sqrt(optax.losses.squared_error(preds, batch.e_form).mean())
            metric_updates = dict(
                mae=mae, loss=loss, rmse=rmse, grad_norm=state.last_grad_norm
            )
        elif task == 'diled':
            losses = state.apply_fn(state.params, batch, training=False, rngs=rng)
            losses = {k: jnp.mean(v) for k, v in losses.items()}
            losses['grad_norm'] = state.last_grad_norm
            metric_updates = dict(**losses)
        else:
            msg = f'Unknown task {task}'
            raise ValueError(msg)

        state = state.replace(metrics=state.metrics.update(**metric_updates))
        return state


    @staticmethod
    @ft.partial(jax.jit, static_argnames=('task', 'config'))
    @chex.assert_max_traces(5)
    def train_step(task: str, config: LossConfig, state: TrainState, batch: DataBatch, rng):
        """Train for a single step."""
        rngs = {k: v for k, v in rng.items()} if isinstance(rng, dict) else {'params': rng}
        rngs['dropout'] = jax.random.fold_in(key=rng['params'], data=state.step)
        if task == 'diled':
            rngs['dropout'], rngs['noise'], rngs['time'] = jax.random.split(rngs['dropout'], 3)

        def loss_fn(params):
            if task == 'e_form':
                preds = state.apply_fn(params, batch, training=True, rngs=rngs).squeeze()
                loss = config.regression_loss(preds, batch.e_form)
                return loss
            else:
                losses = state.apply_fn(params, batch, training=True, rngs=rngs)
                return losses['loss'].mean()

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads, last_grad_norm=grad_norm)
        return state

    def make_model(self):
        if self.config.task == 'diled':
            return self.config.build_diled()
        elif self.config.task == 'e_form':
            return self.config.build_regressor()
        else:
            raise ValueError


    def next_step(self):
        return self.step(self.curr_step + 1, next(self.dl))

    def step(self, step: int, batch: DataBatch):
        self.curr_step = step
        if step == 0:
            # initialize model
            self.state = create_train_state(
                module=self.model,
                optimizer=self.optimizer,
                rng=self.rng,
                batch=batch
            )
        elif step >= self.num_steps:
            return None

        kwargs = dict(
            task=self.config.task,
            config=self.config.train.loss,
            batch=batch,
            rng=self.rng,
        )

        self.state = self.train_step(state=self.state, **kwargs)
        self.state = self.compute_metrics(state=self.state, **kwargs)

        if self.should_log or self.should_ckpt:  # one training epoch has passed
            for metric, value in self.state.metrics.items():  # compute metrics
                if metric == 'grad_norm':
                    self.metrics_history['grad_norm'].append(value)  # record metrics
                    continue
                self.metrics_history[f'tr_{metric}'].append(value)  # record metrics

                if not self.should_ckpt:
                    if f'te_{metric}' in self.metrics_history:
                        self.metrics_history[f'te_{metric}'].append(
                            self.metrics_history[f'te_{metric}'][-1]
                        )
                    else:
                        self.metrics_history[f'te_{metric}'].append(0)

            self.state = self.state.replace(
                metrics=Metrics()
            )  # reset train_metrics for next training epoch

            self.metrics_history['lr'].append(self.lr)
            self.metrics_history['step'].append(self.curr_step)
            self.metrics_history['epoch'].append(self.curr_step / self.steps_in_epoch)
            self.metrics_history['rel_mins'].append((time.monotonic() - self.start_time) / 60)
            self.metrics_history['throughput'].append(
                self.curr_step * self.config.batch_size / self.metrics_history['rel_mins'][-1]
            )

        # print(self.metrics_history)

        if self.should_ckpt:
            # Compute metrics on the test set after each training epoch
            self.test_state = self.state.replace(metrics=Metrics())
            for _i, test_batch in zip(range(self.steps_in_test_epoch), self.test_dl):
                self.test_state = self.compute_metrics(
                    task=self.config.task,
                    config=self.config.train.loss,
                    state=self.test_state,
                    batch=test_batch,
                    rng=self.rng,
                )

            for metric, value in self.test_state.metrics.items():
                if metric == 'grad_norm':
                    continue
                self.metrics_history[f'te_{metric}'].append(value)
                if f'{metric}' == 'loss':
                    self.test_loss = value
            self.mngr.save(
                self.curr_step,
                args=ocp.args.StandardSave(self.ckpt()),
                metrics={'te_loss': self.test_loss},
            )
            self.mngr.wait_until_finished()

        return self

    def step_until_done(self):
        for step, batch in zip(self.steps, self.dl):
            yield self.step(step, batch)

    def run_to_completion(self):
        for step, batch in zip(self.steps, self.dl):
            self.step(step, batch)

    @property
    def should_log(self):
        return (self.curr_step + 1) % (self.steps_in_epoch // self.config.log.logs_per_epoch) == 0

    @property
    def should_ckpt(self):
        return (self.curr_step + 1) % (2 * self.steps_in_epoch) == 0

    @property
    def lr(self):
        return self.scheduler(self.curr_step).item()

    def ckpt(self):
        """Checkpoint PyTree."""
        return Checkpoint(
            self.state, self.seed, dict(self.metrics_history), self.curr_step / self.steps_in_epoch
        )

    def save_final(self, out_dir: str | PathLike):
        """Save final model to directory."""
        self.mngr.wait_until_finished()
        copytree(self.mngr.directory, Path(out_dir) / 'ckpts/')

    def finish(self):
        now = datetime.now()
        if self.config.log.exp_name is None:
            exp_name = now.strftime('%m-%d-%H')
        else:
            exp_name = self.config.log.exp_name

        folder = Path('logs/') / f'{exp_name}_{self.seed}'
        folder.mkdir(exist_ok=True)

        pd.DataFrame(self.metrics_history).to_feather(folder / 'metrics.feather')

        with open(folder / 'time_stopped.txt', 'w') as f:
            f.write(now.isoformat())

        with open(folder / 'config.toml', 'w') as outfile:
            pyrallis.cfgparsing.dump(self.config, outfile)

        self.save_final(folder / 'final_ckpt')

        return folder
