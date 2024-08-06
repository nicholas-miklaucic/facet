"""Variational autoencoder for materials."""

from typing import Callable, Mapping
from eins import EinsOp, Reductions as R
from flax import linen as nn
from flax import struct
import jax
import optax

from cdv.databatch import CrystalGraphs
from cdv.gnn import GN, InputEncoder, NodeAggReadout, ProcessingBlock, Readout
from cdv.layers import Context, E3Irreps, E3IrrepsArray, LazyInMLP
from jaxtyping import Float, Array
import jax.numpy as jnp
import jax.random as jr

from cdv.mace import MaceModel
from cdv.utils import debug_structure, debug_stat


class LatentOutput(struct.PyTreeNode):
    z_e: Float[Array, ' batch latent']
    output: Float[Array, 'batch latent']
    loss: Float[Array, ' batch']
    losses: Mapping[str, Float[Array, ' batch']]


class LatentSpace(nn.Module):
    """Latent space: abstracts different priors and quantizations."""

    def __call__(self, z_e: Float[Array, 'batch latent'], mask, ctx: Context) -> LatentOutput:
        raise NotImplementedError


class LatticeVAE(LatentSpace):
    """Lattice VAE (https://arxiv.org/pdf/2310.09382)."""

    desired_codebook_size: float = 128
    commitment_cost: float = 0.25
    sparsity_penalty: float = 1

    @nn.compact
    def __call__(self, z_e: Float[Array, 'batch latent'], mask, ctx: Context) -> LatentOutput:
        def lattice_init(key, shape, dtype):
            # scale = 1 / (self.desired_codebook_size ** (1 / shape[-1]) - 1)
            # return jax.random.uniform(key, shape, dtype, -scale, scale)
            return jax.random.normal(key, shape, dtype)

        B = self.param('lattice', lattice_init, (z_e.shape[-1],), z_e.dtype)
        B = jax.nn.sigmoid(B) * 0.3 + 0.05

        sg = jax.lax.stop_gradient

        # round each dimension to the nearest multiple of the corresponding scale in B
        # straight-through estimator: otherwise output has no gradient
        quantized = z_e + sg(jnp.rint(z_e / B) * B - z_e)

        sparsity_loss = jnp.mean(jnp.abs(B))
        commitment_loss = jnp.mean(R.mean((z_e - sg(quantized)) ** 2, axis=-1), where=mask)
        embedding_loss = jnp.mean(R.mean((sg(z_e) - quantized) ** 2, axis=-1), where=mask)

        losses = {
            'sparsity': -self.sparsity_penalty * sparsity_loss,
            'β': self.commitment_cost * commitment_loss,
            'embed': embedding_loss,
            'K': jnp.mean(jnp.ptp(jnp.rint(z_e / B), axis=1), where=mask),
        }

        total_loss = losses['sparsity'] + losses['β'] + losses['embed']

        return LatentOutput(z_e, quantized, total_loss, losses)


class PropertyOutput(struct.PyTreeNode):
    e_form: Float[Array, 'graphs 1']
    abc_aby: Float[Array, 'graphs 6']
    crystal_system: Float[Array, 'graphs 7']


def prop_loss(regression_loss, cg: CrystalGraphs, preds: PropertyOutput):
    lat_params = jnp.concat([cg.graph_data.abc, cg.graph_data.angles_rad], axis=-1)
    lat_loss = regression_loss(preds.abc_aby, lat_params, cg.padding_mask)
    sym_loss = optax.softmax_cross_entropy_with_integer_labels(
        preds.crystal_system, cg.crystal_system_code
    )
    sym_loss = jnp.mean(sym_loss, where=cg.padding_mask)
    losses = {
        'e_form': regression_loss(preds.e_form.squeeze(-1), cg.e_form, cg.padding_mask),
        'lat': lat_loss * 0.3,
        'sym': sym_loss * 0.3,
    }

    losses['loss'] = jnp.sum(jnp.stack(list(losses.values())), axis=0)
    return losses


class PropertyPredictor(nn.Module):
    """Property predictor using encoded latents."""

    output_mlp: LazyInMLP

    @nn.compact
    def __call__(self, z: Float[Array, 'graphs latent'], ctx: Context) -> PropertyOutput:
        scalar_splits = (1, 6, 7)
        head = self.output_mlp.copy(out_dim=sum(scalar_splits))

        head_output = head(z, ctx)

        splits = []
        for split in scalar_splits:
            splits.append(head_output[..., :split])
            head_output = head_output[..., split:]

        assert head_output.size == 0, 'Leftover outputs'

        return PropertyOutput(*splits)


class Encoder(nn.Module):
    encoder_model: MaceModel
    latent_space: LatentSpace
    latent_dim: int = 128

    def setup(self):
        self.encoder = self.encoder_model.copy(
            output_graph_irreps=f'{self.latent_dim}x0e', output_node_irreps=None
        )

        self.norm = nn.LayerNorm(use_scale=False)

    def __call__(self, cg: CrystalGraphs, ctx: Context) -> LatentOutput:
        graph_out, node_out = self.encoder(cg, ctx)
        # in future, maybe this could have other irreps, but for now just scalars
        assert node_out is None
        assert graph_out.irreps.is_scalar()
        z_e = graph_out.array
        z_e = jnp.tanh(z_e)
        latent_out = self.latent_space(z_e, mask=cg.padding_mask, ctx=ctx)
        return latent_out


class Decoder(nn.Module):
    decoder_model: MaceModel

    def setup(self):
        self.decoder = self.decoder_model.copy(output_node_irreps='1x1o', output_graph_irreps=None)

    def __call__(
        self, cg: CrystalGraphs, latent: LatentOutput, ctx: Context
    ) -> Float[Array, 'nodes 3']:
        graph_out, node_out = self.decoder(cg, ctx, global_feats=latent.output)
        return node_out.array


class VAE(nn.Module):
    encoder: Encoder
    prop_head: PropertyPredictor
    decoder: Decoder
    prop_reg_loss: Callable

    coord_noise_scale: float = 0.2

    def setup(self):
        pass

    def add_coord_noise(
        self, cg: CrystalGraphs, ctx: Context
    ) -> tuple[Float[Array, 'nodes 3'], CrystalGraphs]:
        """
        Adds noise to coordinates. Does not recompute edges: changes are assumed to be local.
        """
        key = self.make_rng('noise')

        noise = jr.normal(key, cg.nodes.cart.shape, cg.nodes.cart.dtype) * self.coord_noise_scale

        new_carts = cg.nodes.cart + noise

        cg = cg.replace(nodes=cg.nodes.replace(cart=new_carts, frac=None))
        return noise, cg

    def __call__(self, cg: CrystalGraphs, ctx: Context):
        z = self.encoder(cg, ctx)
        eps, noisy_cg = self.add_coord_noise(cg, ctx)
        eps_rec = self.decoder(noisy_cg, z, ctx)

        rec_err = EinsOp('nodes 3 -> nodes', reduce='l2_norm')(
            # flip eps_rec: decoder predicts denoising, eps is noise
            (-eps_rec - eps) / self.coord_noise_scale
        )
        node_mask = cg.padding_mask[cg.nodes.graph_i]
        rec_err = jnp.mean(rec_err, where=node_mask)

        # scale rec_loss by this or not?
        # nodes_per_graph = jnp.sum(node_mask.astype(jnp.float_)) / jnp.sum(
        #     cg.padding_mask.astype(jnp.float_)
        # )

        losses = z.losses

        # losses['z_s'] = jnp.mean(jnp.std(z.output, axis=1), where=cg.padding_mask)

        # losses['sparsity'] = losses['sparsity']
        losses['rec'] = rec_err
        losses['enc'] = 0.2 * z.loss

        prop_pred = self.prop_head(z.output, ctx)
        prop_losses = prop_loss(self.prop_reg_loss, cg, prop_pred)

        losses['prop'] = 1 * prop_losses.pop('loss')
        losses.update(**prop_losses)

        losses['loss'] = losses['enc'] + losses['rec'] + losses['prop']
        return losses
