# Facet
Exploring the design surface of ACE for crystal deep learning

⚠️ Facet is a cutting-edge research repository. It works well, but the code within has not been
thoroughly tested for correctness at the level of more mature models. Copy and use at your own risk.
⚠️

## Overview
Facet is an E(3) equivariant graph neural network framework for crystal property prediction.
Specifically, current work is on predicting energy, forces, and stress in materials. 

It uses a message passing architecture with steerable representations of SO(3), following the broad
structure of [MACE](https://arxiv.org/abs/2206.07697). By using these steerable representations, we
can achieve the same effect as many-body interactions in conventional GNN architectures like GemNet
with a smaller computational and architectural burden.

### Code
This repository implements a JAX/Flax neural network architecture based on
[MACE](https://github.com/ACEsuit/mace) and [SevenNet](https://github.com/MDIL-SNU/SevenNet). It is
built from the ground up, starting from [`e3nn`](https://github.com/e3nn/e3nn-jax) and Flax. 

Using JAX, we can safely use irreps and compute equivariant tensor products without any runtime
cost. JAX also enables efficient parallelism. Testing on 3 RTX 3090 GPUs, I achieve 95+% GPU
utilization throughout training with only a few lines of code.

### Algorithms
This repository also contains my exploration of the design surface of MACE-family architectures.
Early work shows significant gains over SevenNet and MACE in early training, although of course any
pronouncements must wait for models trained at scale. 

## Installation

When the library is ready for production use, it will be containerized and packaged. Until then,
installation is a DIY process. You will need to first install JAX and GPU support. The `reqs.txt`
file should contain the necessary libraries, along with 

```bash
pip install --editable .
```

### Data

Data is processed using the scripts in `scripts/`, but that's not an exact science yet. Due to
JAX's requirement that code have static shapes to avoid recompilation, packing the graphs into
batches currently requires some manual intervention. File an issue or contact me directly at
[nmiklaucic@sc.edu](mailto:nmiklaucic@sc.edu) if you would like guidance or preprocessed data.
Credit to [CHGNet](https://chgnet.lbl.gov/) for compiling the MPTrj dataset.

I am currently working on a pipeline that will download and preprocess everything without any user
intervention.

## Usage
For the best experience, it's recommended to use [Neptune](https://neptune.ai/) logging. The
`env.sh` and `secrets_template.sh` files indicate what environment variables you need. Fill in your
API key in `secrets_template.sh` and save that to `secrets.sh`. Then, run `source env.sh` to set up
the environment variables.

Everything is run from a configuration file. Check `configs/default.toml` to see the default
options and `facet/configs/` to see the different options with some documentation. To check that a
configuration works, you can run

```bash
python facet/show_model.py --config_path=configs/your_config.toml
```

This should print out a very detailed description of the model, an example of which can be found in
`reports/model.html`. 

When that looks good, you can run 

```bash
python facet/train_e_form.py --config_path=configs/your_config.toml
```

For local use, I have a dashboard that works entirely within the terminal! You can set the config
options within your config file, but you can also pass overrides in directly. So, if you want to run
a config file without logging to Neptune and with a dashboard showing the metrics, you can run

```bash
python facet/train_e_form.py --config_path=configs/your_config.toml --display=dashboard --debug_mode=true
```

You'll get a terminal that looks something like this: 

[![image.png](https://i.postimg.cc/QCzLX3x4/image.png)](https://postimg.cc/BP27gRFH)

Neptune really is nicer, so I wouldn't recommend this for continual testing, but it will let you
debug coding issues without polluting your experiment tracking repository.

## Project Structure
The code in `facet/` is where the stuff I'm running repeatedly goes: the model, layer, trainer API,
dashboard, etc. Code in that folder that does not work is considered a bug.

The code in `notebooks/` has some stuff that may be useful to an observer: `embedding_visual.ipynb`
has a useful interface for understanding what's going on internally, and `model_visualization.ipynb`
is a WIP use of `treescope` to view models in more depth. However, many of those notebooks are for
exploratory purposes. Do not assume that code in there is correct, up to date, or what the code in
`facet/` does.

The `scripts/` directory contains one-off scripts that don't need to be run multiple times, mostly
for data preprocessing. They are not thoroughly tested, because I plan to replace them with a more
rigorous pipeline, but you will need them for now.

## Questions? Comments? Concerns?
File an issue or contact me: either through this GitHub account or at
[nmiklaucic@sc.edu](mailto:nmiklaucic@sc.edu).

I work in the [Machine Learning and Evolution Lab at the University of South
Carolina](https://github.com/usccolumbia), to which I am greatly indebted for support and counsel.
Any and all mistakes within this code are purely my own.