"""Training run file."""

from pyrallis import wrap

from cdv.config import MainConfig
from cdv.training_runner import run_using_dashboard


@wrap()
def train_e_form(config: MainConfig):
    """Trains the encoder/ViT to predict formation energy."""
    run_using_dashboard(config)


if __name__ == '__main__':
    train_e_form()
