"""Training run file."""

from pyrallis import wrap

from cdv.config import MainConfig
from cdv.training_runner import run_quietly, run_using_dashboard, run_using_progress


@wrap()
def train_e_form(config: MainConfig):
    """Trains the encoder/ViT to predict formation energy."""
    if config.do_profile:
        import jax

        jax.profiler.start_trace('/tmp/tensorboard', create_perfetto_trace=True)

    display = config.display
    if display == 'progress':
        run_using_progress(config)
    elif display == 'dashboard':
        run_using_dashboard(config)
    elif display == 'quiet':
        run_quietly(config)

    if config.do_profile:
        jax.profiler.stop_trace()


if __name__ == '__main__':
    train_e_form()
