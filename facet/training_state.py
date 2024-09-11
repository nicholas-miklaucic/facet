from dataclasses import field
import functools as ft
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from shutil import copytree, make_archive
from typing import Any, Literal, Mapping, Sequence

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyrallis
from flax import struct
from flax import linen as nn
from flax.training import train_state

from facet.checkpointing import best_ckpt
from facet.config import LossConfig, MainConfig
from facet.dataset import CrystalGraphs, dataloader
from facet.layers import Context
from facet.model_summary import model_summary
from facet.regression import EFSOutput, EFSWrapper
from facet.schedule_free import schedule_free_eval_params
from facet.utils import item_if_arr

import neptune
from neptune.types import File


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

        self.run = self.init_run()

        self.test_loss = 1000

    def init_run(self) -> neptune.Run:
        run = neptune.init_run(
            mode='debug' if self.config.debug_mode else 'async',
            tags=[self.config.task] + self.config.log.tags,
            capture_stderr=False,
            capture_stdout=False,
            source_files='cdv/**/*.py',
        )

        run['parameters'] = self.config.as_dict()
        run['full_config'] = File.from_content(
            pyrallis.cfgparsing.dump(self.config, omit_defaults=False), extension='toml'
        )

        if not self.config.debug_mode:
            summary = model_summary(self.config)
            run['model_summary'] = File.from_content(summary['html'], extension='html')
            run['gflops'] = summary['gflops']

        run['dataset_metadata'].track_files(str(self.config.data.dataset_folder / 'metadata.json'))
        run['seed'] = self.seed
        return run

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
        if task == 'e_form' or task == 'diled' or task == 'vae':
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
    def test_preds(
        task: str, config: LossConfig, state: TrainState, params, batch: CrystalGraphs, rng
    ):
        """Evaluate metrics for a single batch."""
        rngs = {k: v for k, v in rng.items()} if isinstance(rng, dict) else {'params': rng}
        rng = jax.random.fold_in(rngs['params'], state.step)

        from jax.experimental import mesh_utils
        from jax.experimental.shard_map import shard_map
        from jax.sharding import Mesh, PartitionSpec as P

        devs = jax.local_devices()
        mesh = Mesh(mesh_utils.create_device_mesh(len(devs), devices=devs), 'batch')

        @ft.partial(jax.vmap, in_axes=(None, 0))
        def loss_fn(params, batch):
            if task == 'e_form':
                preds = config.efs_wrapper(
                    state.apply_fn, params, batch, ctx=Context(training=True), rngs=rng
                )
                loss = config.efs_loss(batch, preds)
                return loss
            else:
                preds = state.apply_fn(params, batch, ctx=Context(training=True), rngs=rng)
                return preds['loss'].mean(axis=-1), preds

        @ft.partial(
            shard_map,
            mesh=mesh,
            in_specs=(P(None), P('batch')),
            out_specs=P('batch'),
        )
        def ploss_fn(params, batch):
            loss = loss_fn(params, batch)
            loss = jax.lax.pmean(loss, axis_name='batch')
            return loss

        loss = ploss_fn(params, batch)
        return loss

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
            if task == 'e_form':
                preds = config.efs_wrapper(
                    state.apply_fn, params, batch, ctx=Context(training=True), rngs=rng
                )
                loss = config.efs_loss(batch, preds)
                return loss['loss'].mean(), loss
            else:
                preds = state.apply_fn(params, batch, ctx=Context(training=True), rngs=rng)
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
        if self.config.task == 'e_form':
            return self.config.build_regressor()
        elif self.config.task == 'vae':
            return self.config.build_vae()
        else:
            raise ValueError

    def next_step(self):
        return self.step(self.curr_step + 1, next(self.dl))

    def log_metric(self, metric_name: str, metric_value, split: Literal['train', 'valid', None]):
        if split == 'train':
            self.metrics_history[f'tr_{metric_name}'].append(metric_value)
            self.run[f'train/{metric_name}'].append(value=metric_value, step=self.curr_epoch)
        elif split == 'valid':
            self.metrics_history[f'te_{metric_name}'].append(metric_value)
            self.run[f'valid/{metric_name}'].append(value=metric_value, step=self.curr_epoch)
        elif split is None:
            self.metrics_history[f'{metric_name}'].append(metric_value)
            self.run[f'{metric_name}'].append(value=metric_value, step=self.curr_epoch)
        else:
            raise ValueError(f'Split invalid: {split}')

    @property
    def eval_state(self) -> TrainState:
        if self.config.train.schedule_free:
            params = schedule_free_eval_params(self.state.opt_state, self.state.params)
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

            # log number of parameters
            self.run['params'] = int(
                jax.tree.reduce(
                    lambda x, y: x + y, jax.tree.map(lambda x: x.size, self.state.params)
                )
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

        if self.should_log or self.should_ckpt or self.should_validate:
            for metric, value in self.state.metrics.items():  # compute metrics
                if metric == 'grad_norm':
                    self.log_metric('grad_norm', value, None)
                    continue
                self.log_metric(metric, value, 'train')

                if not self.should_validate:
                    if f'te_{metric}' in self.metrics_history:
                        test_value = self.metrics_history[f'te_{metric}'][-1]
                    else:
                        test_value = 0
                    self.log_metric(metric, test_value, 'valid')

            self.state = self.state.replace(
                metrics=Metrics()
            )  # reset train_metrics for next training epoch

            self.log_metric('lr', self.lr, None)
            self.log_metric('step', self.curr_step, None)
            self.log_metric('epoch', self.curr_epoch, None)
            self.log_metric('rel_mins', (time.monotonic() - self.start_time) / 60, None)
            if max(self.metrics_history['epoch'], default=0) < 1:
                self.log_metric(
                    'throughput',
                    self.curr_step
                    * self.config.batch_size
                    * self.config.stack_size
                    / self.metrics_history['rel_mins'][-1],
                    None,
                )
            else:
                prev, curr = self.metrics_history['rel_mins'][-2:]
                min_delta = curr - prev

                prev, curr = self.metrics_history['step'][-2:]
                size = (curr - prev) * self.config.batch_size * self.config.stack_size
                self.log_metric('throughput', size / min_delta, None)

        # debug_structure(self.state)
        # print(self.metrics_history)

        if self.should_validate:
            # Compute metrics on the test set after each training epoch
            self.test_state = self.eval_state.replace(metrics=Metrics())
            for _i, test_batch in zip(range(self.steps_in_test_epoch), self.test_dl):
                test_preds = TrainingRun.test_preds(
                    self.config.task,
                    self.config.train.loss,
                    self.test_state,
                    self.test_state.params,
                    test_batch,
                    self.rng,
                )

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
                self.log_metric(metric, value, 'valid')
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
    def curr_epoch(self) -> float:
        return self.curr_step / self.steps_in_epoch

    @property
    def is_last_step(self):
        return self.curr_step + 1 == self.num_steps

    @property
    def i_in_epoch(self):
        return (self.curr_step + 1) % self.steps_in_epoch

    def equispaced_steps(self, step, epoch_frac):
        """Selects steps that are equispaced and
        appear the correct number of times per epoch.

        For example, epoch_frac=4 selects every 4th epoch's last step, and epoch_frac=1/16 selects
        the steps after 1/16, 2/16, 3/16, ..., 16/16 of an epoch has completed.
        """
        repeat_after = int(np.ceil(epoch_frac)) * self.steps_in_epoch
        num_steps = round(repeat_after / (epoch_frac * self.steps_in_epoch))
        valid_steps = [round(x) for x in np.linspace(0, repeat_after, num_steps, endpoint=False)]
        return ((step + 1) % repeat_after) in valid_steps

    @property
    def should_log(self):
        return (
            self.equispaced_steps(self.curr_step, 1 / self.config.log.logs_per_epoch)
            or self.is_last_step
        ) or self.is_last_step

    @property
    def should_ckpt(self):
        return (
            self.equispaced_steps(self.curr_step, self.config.log.epochs_per_ckpt)
            or self.is_last_step
        ) and not self.config.debug_mode

    @property
    def should_validate(self):
        return (
            self.equispaced_steps(self.curr_step, self.config.log.epochs_per_valid)
            or self.is_last_step
        )

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
        if not self.config.debug_mode:
            self.mngr.wait_until_finished()
            copytree(self.mngr.directory, Path(out_dir) / 'ckpts/')

    def finish(self):
        now = datetime.now()
        if self.config.log.exp_name is None:
            exp_name = now.strftime('%m-%d-%H-%M')
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

        self.run['exp_name'] = exp_name

        zipname = make_archive(exp_name, 'zip', root_dir=folder)
        new_path = Path('logs') / f'{exp_name}.zip'
        shutil.move(zipname, new_path)
        self.run['checkpoint'].upload(str(new_path))

        self.run.stop()

        return folder


# if __name__ == '__main__':
#     import pyrallis

#     # avoid neptune logging on startup
#     with open('configs/test.toml', 'r') as f:
#         cfg = pyrallis.cfgparsing.load(MainConfig, f)
#         cfg = pyrallis.encode(cfg)
#         cfg['debug_mode'] = True
#         cfg = pyrallis.decode(MainConfig, cfg)

#     run = TrainingRun(cfg)

#     cfg = pyrallis.encode(cfg)
#     cfg['debug_mode'] = False
#     cfg = pyrallis.decode(MainConfig, cfg)
#     run.config = cfg

#     print(run.steps_in_epoch)

#     print('i', 'ckpt', 'val', 'log', sep='\t')
#     for i in range(run.steps_in_epoch * cfg.log.epochs_per_ckpt + 2):
#         run.curr_step = i
#         flags = (run.should_ckpt, run.should_validate, run.should_log)
#         if any(flags):
#             print(i, *flags, sep='\t')