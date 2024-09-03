"""
MACE network code. Adapted from https://github.com/ACEsuit/mace-jax.
"""

from collections.abc import Sequence
import functools
import math
from typing import Callable, Literal, Optional
from typing import Set, Union
from flax import linen as nn
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int
import json
from eins import EinsOp

from cdv.databatch import CrystalGraphs
from cdv.mace import edge_embedding
from cdv.mace.e3_layers import IrrepsModule, Linear, LinearReadoutBlock, NonlinearReadoutBlock
from cdv.layers import SegmentReduction, SegmentReductionKind
from cdv.layers import Context, E3NormNorm, LazyInMLP, E3Irreps, E3IrrepsArray, edge_vecs
from cdv.mace.edge_embedding import BesselBasis, ExpCutoff, RadialEmbeddingBlock, GaussBasis
from cdv.mace.message_passing import SimpleInteraction
from cdv.mace.node_embedding import NodeEmbedding, SevenNetEmbedding
from cdv.utils import debug_stat, debug_structure
from cdv.mace.self_connection import (
    EquivariantProductBasisBlock,
    LinearSelfConnection,
    MLPSelfGate,
    SelfConnectionBlock,
)


def safe_norm(x: jnp.ndarray, axis: int | None = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


class SpeciesWiseRescale(nn.Module):
    def setup(self):
        with open('data/sevennet_stats.json', 'r') as stats_file:
            stats = json.load(stats_file)

        with jax.ensure_compile_time_eval():
            self.values = jnp.zeros((max(stats['atomic_numbers']) + 1,), dtype=jnp.float32)
            self.values = self.values.at[jnp.array(stats['atomic_numbers'], dtype=jnp.uint32)].set(
                jnp.array(stats['atomic_energies'])
            )

    def __call__(self, energies, node_species):
        return energies + self.values[node_species]


class MACELayer(nn.Module):
    interaction: SimpleInteraction
    self_connection: SelfConnectionBlock
    readout: IrrepsModule | None
    residual: bool

    @nn.compact
    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_edges, 3]
        node_feats: E3IrrepsArray,  # [n_nodes, irreps]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        receivers: jnp.ndarray,  # [n_edges]
        species_embed: Float[Array, 'num_species num_embed'] | None,
        ctx: Context,
    ):
        """-> (n_nodes output_irreps, n_nodes features*hidden_irreps)"""
        x = self.interaction(vectors, node_feats, radial_embedding, receivers, ctx=ctx)
        if self.residual:
            resid = Linear(x.irreps, force_irreps_out=True)(node_feats)
            # debug_stat(resid=resid, x=x)
            x = x + resid
        x = self.self_connection(x, node_species, species_embed, ctx)

        if self.readout is not None:
            readout = self.readout(x, ctx=ctx)
            return x, readout
        else:
            return x, None


class MACE(IrrepsModule):
    hidden_irreps: Sequence[E3Irreps]
    node_embedding: NodeEmbedding
    radial_embedding: RadialEmbeddingBlock
    interaction_templ: SimpleInteraction
    self_connection_templ: SelfConnectionBlock
    readout_templ: IrrepsModule
    only_last_readout: bool
    share_species_embed: bool
    residual: bool

    def setup(self):
        layers = []
        for i, ir_out in enumerate(list(self.hidden_irreps) + [self.ir_out]):
            ir_out = E3Irreps(ir_out)
            last = i == len(self.hidden_irreps)

            self_connection = self.self_connection_templ.copy(irreps_out=ir_out)

            # jax.debug.print(
            #     'irreps = {}, true_irreps = {}',
            #     self_connection.irreps_in(),
            #     self_connection.ir_out,
            # )
            interaction = self.interaction_templ.copy(irreps_out=self_connection.irreps_in())

            if last or not self.only_last_readout:
                readout = self.readout_templ.copy(irreps_out=self.ir_out)
            else:
                readout = None

            layers.append(
                MACELayer(
                    interaction=interaction,
                    self_connection=self_connection,
                    readout=readout,
                    residual=self.residual,
                    name=f'layer_{i}',
                )
            )

        self.layers = layers

    def __call__(
        self,
        vectors: E3IrrepsArray,  # [n_nodes, k, 3]
        node_species: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        receivers: jnp.ndarray,  # [n_nodes, k]
        ctx: Context,
    ) -> E3IrrepsArray:
        # Embeddings
        node_feats = self.node_embedding(node_species, ctx=ctx).astype(
            vectors.dtype
        )  # [n_nodes, feature * irreps]
        species_embs = node_feats

        if not (hasattr(vectors, 'irreps') and hasattr(vectors, 'array')):
            vectors = E3IrrepsArray('1e', vectors)

        radial_embedding = self.radial_embedding(safe_norm(vectors.array, axis=-1), ctx=ctx)
        radial_embedding = radial_embedding.astype(vectors.dtype)
        # debug_structure(radial_embedding=radial_embedding)

        # vector_shs = e3nn.spherical_harmonics(self.interaction_templ.conv.max_ell, vectors, True)
        vector_shs = e3nn.spherical_harmonics(
            e3nn.Irreps.spherical_harmonics(self.interaction_templ.conv.max_ell, p=1),
            vectors,
            normalize=True,
        )

        outputs = []
        for i, layer in enumerate(self.layers):
            node_feats, readout = layer(
                vector_shs,
                node_feats,
                node_species,
                radial_embedding,
                receivers,
                species_embed=species_embs.array if self.share_species_embed else None,
                ctx=ctx,
            )
            if readout is not None:
                outputs.append(readout)

        # print([k.shape for k in outputs])
        return e3nn.stack(outputs, axis=-2)  # [n_nodes, num_interactions, output_irreps]


class MaceModel(nn.Module):
    """Graph network that wraps MACE for invariant regression."""

    hidden_irreps: Sequence[str]

    node_embedding: NodeEmbedding
    edge_embedding: RadialEmbeddingBlock
    interaction: SimpleInteraction
    self_connection: SelfConnectionBlock
    readout: IrrepsModule
    head_templ: LazyInMLP
    residual: bool

    precision: Literal['f32', 'bf16']
    outs_per_node: int
    share_species_embed: bool = True

    # How to combine the outputs of different interaction blocks.
    # 'last' is special: it means the last block.
    interaction_reduction: str = 'last'

    def setup(self):
        self.mace = MACE(
            irreps_out=f'{self.outs_per_node}x0e',
            hidden_irreps=self.hidden_irreps,
            node_embedding=self.node_embedding,
            radial_embedding=self.edge_embedding,
            interaction_templ=self.interaction,
            self_connection_templ=self.self_connection,
            readout_templ=self.readout,
            only_last_readout=self.interaction_reduction == 'last',
            share_species_embed=self.share_species_embed,
            residual=self.residual,
        )

        self.rescale = SpeciesWiseRescale()
        self.norm = nn.LayerNorm()
        self.node2graph_reduction = SegmentReduction('mean')
        self.head = self.head_templ.copy(out_dim=1, name='head')
        self.dtype = jnp.float32 if self.precision == 'f32' else jnp.bfloat16

    def __call__(
        self,
        cg: CrystalGraphs,
        ctx: Context,
    ) -> Float[Array, ' graphs 1']:
        vecs = edge_vecs(cg).astype(self.dtype)

        # shape [n_nodes, n_interactions, output_irreps]
        mace_out = self.mace(
            vecs,
            cg.nodes.species,
            cg.receivers,
            ctx=ctx,
        ).array

        if self.interaction_reduction == 'last':
            out = mace_out[..., -1, :]
        else:
            out = EinsOp('nodes blocks mul -> nodes mul', reduce=self.interaction_reduction)(
                mace_out
            )

        out = self.norm(out)

        mlp_out = self.head(out, ctx=ctx)[..., 0]
        scaled_out = self.rescale(mlp_out, cg.nodes.species)
        graph_out = self.node2graph_reduction(
            scaled_out, cg.nodes.graph_i, cg.n_total_graphs, ctx=ctx
        )[..., None]

        return graph_out
