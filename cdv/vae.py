"""Variational autoencoder for materials."""

from eins import EinsOp
from flax import linen as nn
from flax import struct
import jax
import optax

from cdv.databatch import CrystalGraphs
from cdv.gnn import GN, InputEncoder, NodeAggReadout, ProcessingBlock, Readout
from cdv.layers import Context, E3Irreps, E3IrrepsArray, LazyInMLP
from jaxtyping import Float, Array
import jax.numpy as jnp

from cdv.mace import MaceModel
from cdv.utils import debug_structure


class EncoderOutput(struct.PyTreeNode):
    e_form: Float[Array, 'graphs 1']
    abc_aby: Float[Array, 'graphs 6']
    crystal_system: Float[Array, 'graphs 7']


class Encoder(nn.Module):
    """Encoder."""

    model: MaceModel
    output_mlp: LazyInMLP
    output_dim: int = 256

    def setup(self):
        e_form = '1x0e'
        lattice = '6x0e'
        crystal_system = '7x0e'
        self.mace = self.model.copy(output_irreps=f'{e_form} + {lattice} + {crystal_system}')
        self.head = self.output_mlp.copy(out_dim=1)

    def __call__(self, cg: CrystalGraphs, ctx: Context) -> EncoderOutput:
        output_irreps = self.mace(cg, ctx)

        return EncoderOutput(
            output_irreps.slice_by_chunk[:1].array,
            output_irreps.slice_by_chunk[1:2].array,
            output_irreps.slice_by_chunk[2:].array,
        )


def vae_loss(config: ' LossConfig', cg: CrystalGraphs, preds: EncoderOutput):
    lat_params = jnp.concat([cg.graph_data.abc, cg.graph_data.angles_rad], axis=-1)
    lat_loss = config.regression_loss(preds.abc_aby, lat_params, cg.padding_mask)
    sym_loss = optax.softmax_cross_entropy_with_integer_labels(
        preds.crystal_system, cg.crystal_system_code
    )
    sym_loss = jnp.mean(sym_loss, where=cg.padding_mask)
    losses = {
        'e_form': config.regression_loss(preds.e_form.squeeze(-1), cg.e_form, cg.padding_mask),
        'lat': lat_loss * 0.3,
        'sym': sym_loss * 0.3,
    }

    losses['loss'] = jnp.sum(jnp.stack(list(losses.values())), axis=0)
    return losses
