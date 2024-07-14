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
    # 1x0e
    e_form: E3IrrepsArray
    # 1x0e + 1x2e
    metric_tensor: E3IrrepsArray


class Encoder(nn.Module):
    """Encoder."""

    model: MaceModel

    def setup(self):
        e_form = '1x0e'
        metric_tensor = '1x0e + 1x2e'
        self.mace = self.model.copy(output_irreps=f'{e_form} + {metric_tensor}', name='mace')

    def __call__(self, cg: CrystalGraphs, ctx: Context) -> EncoderOutput:
        output_irreps = self.mace(cg, ctx)

        return EncoderOutput(output_irreps.slice_by_chunk[:1], output_irreps.slice_by_chunk[1:])


def vae_loss(config: ' LossConfig', cg: CrystalGraphs, preds: EncoderOutput):
    lat_loss = EinsOp('batch basis -> batch', reduce=('sqrt', 'mean', 'square'))(
        cg.metric_tensor_irreps.array / 10 - preds.metric_tensor.array
    )
    lat_loss = jnp.mean(lat_loss, where=cg.padding_mask)
    losses = {
        'e_form': config.regression_loss(preds.e_form.array, cg.e_form, cg.padding_mask),
        'lat': lat_loss * 0.02,
    }

    losses['loss'] = jnp.mean(jnp.stack(list(losses.values())), axis=0)
    return losses
