# Facet
Exploring the design surface of ACE for crystal deep learning

⚠️ **Facet is a cutting-edge research repository. It works well, but the code within has not been thoroughly tested for correctness at the level of more mature models. Copy at your own risk.** ⚠️

## Overview
Facet is a graph neural network architecture for crystal property prediction. Specifically, current work is on predicting energy, forces, and stress in materials. 

It uses a message-passing architecture with steerable representations of SO(3), following the broad structure of [MACE](https://arxiv.org/abs/2206.07697). By using these steerable representations, we can achieve the same effect as many-body interactions in conventional GNN architectures like GemNet with a smaller computational and architectural burden.
### Code
This repository implements a JAX/Flax neural network architecture based on [MACE](https://github.com/ACEsuit/mace) and [SevenNet](https://github.com/MDIL-SNU/SevenNet). It is built from the ground up, starting from [`e3nn`](https://github.com/e3nn/e3nn-jax) and Flax. 

Using JAX, we can safely use irreps and compute equivariant tensor products without any runtime cost. JAX also enables efficient parallelism. Testing on 3 RTX 3090 GPUs, I achieve 95+% GPU utilization throughout training with only a few lines of code.

### Algorithms
This repository also contains my exploration of the design surface of MACE-family architectures. Early work shows significant gains over SevenNet and MACE in early training, although of course any pronouncements must wait for models trained at scale. 
