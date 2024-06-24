"""Standard U-Net Diffusion. Adapted to 3D."""

# https://github.com/HMUNACHI/nanodl/blob/main/nanodl/__src/models/diffusion.py

import abc
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import dataclass
from eins import EinsOp

from avid.layers import LazyInMLP
from avid.vit import Encoder


@dataclass
class DiffusionInput:
    """Input to a diffusion step."""

    # Noised latent.
    x_t: jax.Array
    # Time step.
    t: jax.Array
    # Class/label, as an int.
    y: jax.Array


# https://github.com/facebookresearch/DiT/blob/main/models.py
class TimestepEmbed(nn.Module):
    frequency_embed_dim: int
    mlp: nn.Module

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concat([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, t, training: bool):
        t_freq = self.timestep_embedding(t, self.frequency_embed_dim)
        t_emb = self.mlp(t_freq, training=training)
        return t_emb


class LabelEmbed(nn.Module):
    """Diffusion class label embedding: reserves an embedding for a Ø class."""

    # Number of actual classes: one is added for the null class.
    num_classes: int

    # Embed dimension.
    embed_dim: int

    def setup(self):
        self.embeddings = nn.Embed(self.num_classes + 1, self.embed_dim)

    def __call__(self, y: jax.Array) -> jax.Array:
        return self.embeddings(y + 1)


class DiffusionNoiseSchedule(metaclass=abc.ABCMeta):
    """Diffusion noise schedule interface."""

    @property
    def max_t(self) -> int:
        raise NotImplementedError

    def alpha(self, times):
        """Gets α_t schedule, for floats between 0 and 1. Should be alpha_bar."""
        raise NotImplementedError

    def alpha_beta(self, times):
        """Gets α_t and β_t schedule, for times with T = max_t."""
        alpha = self.alpha(times)
        return {'alpha': alpha, 'beta': 1 - (alpha / self.alpha(times - 1))}

    def add_noise(self, x, t, noise):
        alpha = self.alpha(t / self.max_t)
        return x * jnp.sqrt(alpha) + jnp.sqrt(1 - alpha) * noise


@dataclass
class KumaraswamySchedule(DiffusionNoiseSchedule):
    """
    Flexible noise schedule parameterized with Kumaraswamy CDF.
    a, b = (1.7, 1.9) is close to the standard cosine schedule.
    """

    a: float = 1.7
    b: float = 1.9
    timesteps: int = 100

    @property
    def max_t(self) -> int:
        return self.timesteps

    def alpha(self, times) -> jax.Array:
        return (1 - (times / self.max_t) ** self.a) ** self.b


class DiffusionBackbone:
    """Abstract model class. Returns denoised images given noisy inputs and noise scale."""

    def eps_sigma(self, data: DiffusionInput, schedule: DiffusionNoiseSchedule, training: bool):
        """Returns the predicted noise from the input and times."""
        raise NotImplementedError

    def __call__(self, data: DiffusionInput, schedule: DiffusionNoiseSchedule, training: bool):
        return self.eps_sigma(data, schedule, training=training)


class DiffusionModel(nn.Module):
    """
    Implements a diffusion model for image generation using JAX.
    """

    model: DiffusionBackbone
    schedule: DiffusionNoiseSchedule

    def setup(self):
        self.backbone = self.model.copy()

    def mu_eps_sigma(self, data: DiffusionInput, training: bool):
        """Gets predicted images and covariance given noisy inputs and time."""
        out = self.backbone(data, self.schedule, training=training)
        eps = out['eps']
        sigma = out['sigma']
        sched = self.schedule.alpha_beta(data.t)
        pred_images = 1 / jnp.sqrt(1 - sched['beta']) * (data.x_t - jnp.sqrt(1 - sched['alpha']) * eps)
        return {'mu': pred_images, 'eps': eps, 'sigma': sigma}

    def q_sample(self, x_t, t, noise):
        return self.schedule.add_noise(x_t, t, noise)

    def __call__(self, data: DiffusionInput, training: bool):
        """Gets μ, ε, and Σ."""
        return self.mu_eps_sigma(data, training=training)

    def generation_loop(self, data: DiffusionInput, rng):
        """Simple sampler that generates images from the given state. Use a real sampler if you need to."""

        def step(
            i: int, state_rng: Tuple[DiffusionInput, jax.Array]
        ) -> Tuple[DiffusionInput, jax.Array]:
            state, rng = state_rng
            curr_rng, next_rng = jax.random.split(rng)
            out = self.mu_eps_sigma(state, training=False)
            noise = jax.random.normal(curr_rng, shape=out['mu'].shape)
            x_tm1 = out['mu'] * out['sigma'] * noise
            next_state = DiffusionInput(x_tm1, state.t - 1, state.y)
            return (next_state, next_rng)

        last_state, last_rng = jax.lax.fori_loop(0, data.t - 1, step, (data, rng))
        return last_state


@dataclass
class DiT(DiffusionBackbone, nn.Module):
    condition_mlp_dims: Sequence[int]
    time_dim: int
    time_mlp: nn.Module
    num_classes: int
    label_dim: int
    encoder: Encoder
    hidden_dim: int
    condition_dropout: float = 0.0

    def setup(self):
        self.time_embed = TimestepEmbed(self.time_dim, mlp=self.time_mlp)
        self.label_embed = LabelEmbed(num_classes=self.num_classes, embed_dim=self.label_dim)
        self.condition_mlp = LazyInMLP(
            self.condition_mlp_dims,
            out_dim=6 * self.hidden_dim,
            dropout_rate=self.condition_dropout,
        )
        self.aby_scale = self.param(
            'aby_scale', lambda key: jnp.array([0, 1, 1, 0, 1, 1], dtype=jnp.float32)
        )

    def eps_sigma(self, data: DiffusionInput, schedule: DiffusionNoiseSchedule, training: bool):
        t_emb = self.time_embed(jnp.broadcast_to(data.t, (data.x_t.shape[0],)), training=training)
        y_emb = self.label_embed(data.y)
        cond_emb = jnp.concatenate([t_emb, y_emb], axis=-1)
        abys = self.condition_mlp(cond_emb, training=training)

        abys = EinsOp('batch (dim aby), aby -> aby batch 1 dim', combine='multiply')(
            abys, self.aby_scale
        ).astype(jnp.bfloat16)

        x_t_reshaped = EinsOp('batch n n n dim -> batch (n n n) dim')(data.x_t).astype(jnp.bfloat16)

        unflatten = EinsOp('batch (n=8 n n) dim -> batch n n n dim')

        return {
            'eps': unflatten(self.encoder(x_t_reshaped, abys, training=training)).astype(
                jnp.bfloat16
            ),
            'sigma': schedule.alpha_beta(data.t)['beta'],
        }
