"""Training run file."""

from pyrallis import wrap

from avid.config import MainConfig
from avid.training_runner import run_using_dashboard


@wrap()
def train_e_form(config: MainConfig):
    """Trains the encoder/ViT to predict formation energy."""
    run_using_dashboard(config)


if __name__ == '__main__':
    # with jax.log_compiles():
    train_e_form()
