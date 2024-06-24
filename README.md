# AVID

## Acronym Very Incomplete: Diffusion?

Work on applying diffusion and vision transformers to materials. Right now, I'm
just doing cubic perovskites. Cool stuff TBD.

## Installation

[Install JAX](https://jax.readthedocs.io/en/latest/installation.html), and then
install the other necessary libraries:

```bash
pip install toml pyrallis rho-plus equinox diffrax optax plotext rich einops pandas plotly matplotlib seaborn pymatgen tfp-nightly[jax] textual textual-dev textual-plotext flax orbax beartype humanize clu pyarrow
```

TODO:

- [ ] Perhaps configure DiLED in addition to normal diffusion and encoder losses
- [ ] How to train the encoder? As part of the normal process or separately?
- [ ] EMA
- [ ] Refactor diffusion trainer into TrainingRun so we can get dashboards again
