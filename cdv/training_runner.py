"""Training state runner interface."""

from time import sleep
from cdv.config import MainConfig
from cdv.dashboard import Dashboard
from cdv.training_state import TrainingRun


def run_using_dashboard(config: MainConfig):
    run = TrainingRun(config)
    app = Dashboard(run, config=config)
    app.run()
    return run


def run_quietly(config: MainConfig):
    run = TrainingRun(config)

    for _run_state in run.step_until_done():
        continue

    print('Saved to:')
    print(run.finish().absolute())
    return run


def run_using_progress(config: MainConfig):
    import rich.progress as prog

    run = TrainingRun(config)

    update_every = 1
    with prog.Progress(
        prog.TextColumn('[progress.description]{task.description}'),
        prog.BarColumn(80, 'light_pink3', 'deep_sky_blue4', 'green'),
        prog.MofNCompleteColumn(),
        prog.TimeElapsedColumn(),
        prog.TimeRemainingColumn(),
        prog.SpinnerColumn(),
        refresh_per_second=3,
        disable=not config.cli.show_progress,
    ) as progress:
        task = progress.add_task(
            '[bold] [deep_pink3] Training [/deep_pink3] [/bold]',
            total=run.num_steps // update_every,
        )
        for run_state in run.step_until_done():
            if run_state.curr_step % update_every == 0:
                progress.advance(task)

            if run_state.should_log:
                status = []
                for k, v in run_state.metrics_history.items():
                    if k.endswith('loss'):
                        status.append(f'{k}={v[-1]:4.02f}')

                progress.update(task, description=' '.join(status))

        progress.advance(task, advance=0)
    print('Saved to:')
    print(run.finish().absolute())
    sleep(60)
    return run
