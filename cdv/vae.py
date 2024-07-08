"""Variational autoencoder for materials."""
from flax import linen as nn
import jax
import optax

from cdv.databatch import CrystalGraphs
from cdv.gnn import GN, InputEncoder, NodeAggReadout, ProcessingBlock, Readout
from cdv.layers import Context, LazyInMLP
from jaxtyping import Float, Array
import jax.numpy as jnp

from cdv.utils import debug_structure

class Encoder(nn.Module):    
    """Encoder."""
    input_enc: InputEncoder
    num_blocks: int
    block_templ: ProcessingBlock
    latent_dim: int
    head: LazyInMLP

    @nn.compact    
    def __call__(self, cg: CrystalGraphs, ctx: Context) -> Float[Array, 'graphs latent_dim']:
        g = self.input_enc(cg, ctx)
        for _i in range(self.num_blocks):
            block = self.block_templ.copy()
            g = block(g, ctx)

        readout = NodeAggReadout(self.head.copy(out_dim=self.latent_dim), name='out_head')
        return readout(g, ctx)
    

class AggMLP(nn.Module):
    """Predicts global graph properties from z."""
    mlp: LazyInMLP

    @nn.compact
    def __call__(self, z: Float[Array, 'graphs latent_dim'], ctx: Context) -> tuple[Float[Array, 'graphs 3'], Float[Array, 'graphs 3'], Float[Array, 'graphs 1'], Float[Array, 'graphs 1']]:
        """Predicts lattice abc, αβγ, and number of atoms from latent dimension."""
        mlp = self.mlp.copy(out_dim=30, name='head')

        out = mlp(z, ctx)

        raw_abc = nn.Dense(1, name='abc_norm')(out[..., :3][..., None])[..., 0]
        raw_aby = nn.Dense(1, name='aby_norm')(out[..., 3:9][..., None])[..., 0]        
        raw_e_f = nn.Dense(1, name='e_f_norm')(out[..., [9]])
        raw_n_a = out[..., 10:]

        abc = jax.nn.elu(raw_abc * 2 + 5)
        aby = jnp.atan2(jax.nn.softplus(raw_aby[..., :3]), raw_aby[..., 3:])
        n_a = raw_n_a
        e_f = raw_e_f

        return (abc, aby, n_a, e_f)
    

class VAE(nn.Module):
    encoder: Encoder
    agg_mlp: AggMLP

    @nn.compact
    def __call__(self, cg: CrystalGraphs, ctx: Context) -> tuple[Float[Array, 'graphs 3'], Float[Array, 'graphs 3'], Float[Array, 'graphs 1'], Float[Array, 'graphs 1']]:
        z = self.encoder(cg, ctx)
        return self.agg_mlp(z, ctx)
    

def vae_loss(config, cg: CrystalGraphs, abc, aby, n_a, e_f):
    loss = config.regression_loss

    mask3 = jnp.tile(cg.padding_mask, (3, 1)).T

    l_abc = loss(cg.globals.abc, abc, mask3)
    l_aby = loss(cg.globals.angles_rad, aby, mask3)
    l_n_a = optax.softmax_cross_entropy_with_integer_labels(n_a, cg.n_node - 1).mean(where=cg.padding_mask)

    # debug_structure(n_a=n_a, l_n_a=l_n_a)
    l_e_f = loss(cg.e_form, e_f, cg.padding_mask)

    total = l_abc + l_aby + l_n_a + l_e_f

    return {
        'abc': l_abc,
        'aby': l_aby,
        'n_a': l_n_a,
        'e_f': l_e_f,
        'loss': total
    }