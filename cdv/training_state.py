from dataclasses import field
import functools as ft
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from shutil import copytree
from typing import Any, Mapping, Sequence

import chex
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyrallis
from flax import struct
from flax import linen as nn
from flax.training import train_state

from cdv.checkpointing import best_ckpt
from cdv.config import LossConfig, MainConfig
from cdv.dataset import CrystalGraphs, dataloader
from cdv.layers import Context
from cdv.utils import item_if_arr


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

    def replicate_params(self, devices: set[jax.Device]):
        return jax.device_put_replicated(self.params, devices)


def create_train_state(module: nn.Module, optimizer, rng, batch: CrystalGraphs):
    b1 = jax.tree_map(lambda x: x[0], batch)
    # debug_structure(b1=b1)
    # jax.debug.visualize_array_sharding(b1.e_form)
    loss, params = module.init_with_output(rng, b1, ctx=Context(training=False))
    # jax.debug.visualize_array_sharding(loss)
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

        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=1,
            best_fn=lambda metrics: metrics['te_loss'],
            best_mode='min',
            max_to_keep=4,
            enable_async_checkpointing=True,
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
    def compute_metrics(
        *,
        task: str,
        config: LossConfig,
        state: TrainState,
        batch: CrystalGraphs,
        preds,
        rng=None,
    ):
        if task == 'e_form':
            preds = preds
            e_form = batch.e_form[..., None]
            mask = batch.padding_mask[..., None]
            loss = config.regression_loss(preds, e_form, mask)
            mae = jnp.abs(preds - e_form).mean(where=mask)
            rmse = jnp.sqrt(optax.losses.squared_error(preds, e_form).mean(where=mask))
            metric_updates = dict(mae=mae, loss=loss, rmse=rmse, grad_norm=state.last_grad_norm)
        elif task == 'diled' or task == 'vae':
            losses = {k: jnp.mean(v) for k, v in preds.items()}
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
    def train_grads(
        task: str, config: LossConfig, state: TrainState, params, batch: CrystalGraphs, rng
    ):
        """Train for a single step."""
        rngs = {k: v for k, v in rng.items()} if isinstance(rng, dict) else {'params': rng}
        rng = jax.random.fold_in(rngs['params'], state.step)

        from jax.experimental import mesh_utils
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh, PartitionSpec as P

        devs = jax.local_devices()
        mesh = Mesh(mesh_utils.create_device_mesh(len(devs), devices=devs), 'batch')

        def loss_fn(params, batch):
            preds = state.apply_fn(params, batch, ctx=Context(training=True), rngs=rng)
            if task == 'e_form':
                loss = config.regression_loss(preds.squeeze(), batch.e_form, batch.padding_mask)
                return loss, preds
            else:
                return preds['loss'].mean(axis=-1), preds

        @ft.partial(jax.vmap, in_axes=(None, 0))
        def vgrad_fn(params, batch):
            grad_loss_fn = jax.grad(loss_fn, has_aux=True)
            grads, preds = grad_loss_fn(params, batch)
            return grads, preds

        def agg_grads(grads):
            return jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)

        @ft.partial(
            shard_map,
            mesh=mesh,
            in_specs=(P(None), P('batch')),
            out_specs=(P(), P('batch')),
        )
        def pgrad_fn(params, batch):
            grads, preds = vgrad_fn(params, batch)
            grads = agg_grads(grads)
            grads = jax.lax.pmean(grads, axis_name='batch')
            return grads, preds

        grads, preds = pgrad_fn(params, batch)
        return grads, preds

    @staticmethod
    @ft.partial(jax.jit)
    @chex.assert_max_traces(5)
    def update_train_state(state: TrainState, grads) -> TrainState:
        # grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads, last_grad_norm=grad_norm)
        return state

    @staticmethod
    def train_step(task: str, config: LossConfig, state: TrainState, batch: CrystalGraphs, rng):
        """Train for a single step."""
        # debug_structure(state.params)
        # debug_structure(batch)
        grads, preds = TrainingRun.train_grads(task, config, state, state.params, batch, rng)
        # debug_structure(grads=grads, preds=preds)
        # print('params')
        # jax.debug.visualize_array_sharding(jax.tree_leaves(state.params)[0].reshape(-1))
        # print('preds')
        # jax.debug.visualize_array_sharding(preds[..., 0])
        state = TrainingRun.update_train_state(state, grads)
        return state, preds

    def make_model(self):
        if self.config.task == 'diled':
            return self.config.build_diled()
        elif self.config.task == 'e_form':
            return self.config.build_regressor()
        elif self.config.task == 'vae':
            return self.config.build_vae()
        else:
            raise ValueError

    def next_step(self):
        return self.step(self.curr_step + 1, next(self.dl))

    @property
    def eval_state(self) -> TrainState:
        if self.config.train.schedule_free:
            params = optax.contrib.schedule_free_eval_params(
                self.state.opt_state, self.state.params
            )
            return self.state.replace(params=params)
        else:
            return self.state

    def step(self, step: int, batch: CrystalGraphs):
        self.curr_step = step
        if step == 0:
            # initialize model
            self.state = create_train_state(
                module=self.model,
                optimizer=self.optimizer,
                rng=self.rng,
                batch=batch,
            )

            if self.config.restart_from is not None:
                self.state = self.state.replace(
                    params=best_ckpt(self.config.restart_from)['state']['params']
                )
        elif step >= self.num_steps:
            return None

        kwargs = dict(
            task=self.config.task,
            config=self.config.train.loss,
            batch=batch,
            rng=self.rng,
        )

        self.state, preds = self.train_step(state=self.state, **kwargs)
        # jax.debug.visualize_array_sharding(preds[..., 0])
        self.state = self.compute_metrics(preds=preds, state=self.eval_state, **kwargs)

        if (
            self.should_log or self.should_ckpt or self.should_validate
        ):  # one training epoch has passed
            for metric, value in self.state.metrics.items():  # compute metrics
                if metric == 'grad_norm':
                    self.metrics_history['grad_norm'].append(value)  # record metrics
                    continue
                self.metrics_history[f'tr_{metric}'].append(value)  # record metrics

                if not self.should_validate:
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
            if max(self.metrics_history['epoch'], default=0) < 1:
                self.metrics_history['throughput'].append(
                    self.curr_step * self.config.batch_size / self.metrics_history['rel_mins'][-1]
                )
            else:
                prev, curr = self.metrics_history['rel_mins'][-2:]
                min_delta = curr - prev

                prev, curr = self.metrics_history['step'][-2:]
                size = (curr - prev) * self.config.batch_size
                self.metrics_history['throughput'].append(size / min_delta)

        # debug_structure(self.state)
        # print(self.metrics_history)

        if self.should_validate:
            # Compute metrics on the test set after each training epoch
            self.test_state = self.eval_state.replace(metrics=Metrics())
            for _i, test_batch in zip(range(self.steps_in_test_epoch), self.test_dl):
                test_preds = jax.vmap(
                    lambda p, b: self.test_state.apply_fn(
                        p, b, ctx=Context(training=False), rngs=self.rng
                    ),
                    in_axes=(None, 0),
                )(self.test_state.params, test_batch)

                self.test_state = self.compute_metrics(
                    task=self.config.task,
                    config=self.config.train.loss,
                    state=self.test_state,
                    batch=test_batch,
                    preds=test_preds,
                )

            for metric, value in self.test_state.metrics.items():
                if metric == 'grad_norm':
                    continue
                self.metrics_history[f'te_{metric}'].append(value)
                if f'{metric}' == 'loss':
                    self.test_loss = value

        if self.should_ckpt:
            # print(self.test_loss)
            self.mngr.save(
                self.curr_step,
                args=ocp.args.StandardSave(self.ckpt()),
                metrics={'te_loss': self.test_loss},
            )

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
        return (self.curr_step + 1) % (self.config.log.epochs_per_ckpt * self.steps_in_epoch) == 0

    @property
    def should_validate(self):
        return (self.curr_step + 1) % (self.config.log.epochs_per_valid * self.steps_in_epoch) == 0

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
