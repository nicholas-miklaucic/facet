from facet.train_e_form import MainConfig, run_using_dashboard
from pyrallis import cfgparsing
from time import monotonic
import rich

from facet.training_runner import run_using_progress
from facet.training_state import TrainingRun

if __name__ == '__main__':
    with open('configs/profiling.toml', 'r') as cfg:
        conf = cfgparsing.load(MainConfig, cfg)

    # run = None
    run = TrainingRun(conf)

    # jit compile and warm up
    for _run_state, i in zip(run.step_until_done(), range(2)):
        continue

    start = monotonic()
    for run_state in run.step_until_done():
        continue
    end = monotonic()

    rich.print({k: v[-1] for k, v in run_state.metrics_history.items()})

    print(f'Run took {end - start:.3f} seconds', )