import chex
from facet.train_e_form import MainConfig, run_using_dashboard
from pyrallis import cfgparsing
from time import monotonic
import rich

from facet.training_runner import run_using_progress
from facet.training_state import TrainingRun

import jax.profiler

if __name__ == '__main__':
    with open('configs/sevennet-plus.toml', 'r') as cfg:
        conf = cfgparsing.load(MainConfig, cfg)
        conf.debug_mode = True
        conf.display = 'progress'

        conf.data.batches_per_group = 4
        conf.log.epochs_per_valid = 0.25

    # run = None
    run = TrainingRun(conf)

    # jit compile and warm up
    for run_state, i in zip(run.step_until_done(), range(3)):
        continue

    # memory profiling
    # for run_state, i in zip(run.step_until_done(), range(50)):
    #     if (i + 1) % 5 == 0:
    #         jax.block_until_ready(run_state.state.params)
    #         jax.profiler.save_device_memory_profile(f'/tmp/jax_profile/memory{(i // 5):02}.prof')

    # tensorboard profiling
    max_step = run.steps_in_epoch // 4
    for i, batch in zip(range(max_step + 1), run.dl):
        if i == max_step:
            jax.profiler.start_trace('/tmp/tensorboard')

        chex.clear_trace_counter()
        run_state = run_state.step(i, batch)

    jax.block_until_ready(run_state.state.params)
    jax.profiler.stop_trace()

    # simple wall-clock profiling
    # start = monotonic()
    # for run_state in run.step_until_done():
    #     continue
    # jax.block_until_ready(run_state.state.params)
    # end = monotonic()
    # print(
    #     f'Run took {end - start:.3f} seconds',
    # )

    print(run_state.curr_step)
    rich.print({k: v[-1] for k, v in run_state.metrics_history.items()})
