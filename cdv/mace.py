"""
MACE network code. Adapted from https://github.com/ACEsuit/mace-jax.
"""

import functools
import math
from typing import Callable, Optional
from typing import Set, Union
from flax import linen as nn
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int

from cdv.layers import Context, Identity
from cdv.utils import debug_stat, debug_structure, flax_summary

class LinearNodeEmbedding(nn.Module):
    num_species: int
    irreps_out: e3nn.Irreps

    def setup(self):
        self.irreps_out_calc = e3nn.Irreps(self.irreps_out).filter("0e").regroup()
        self.embed = nn.Embed(self.num_species, e3nn.Irreps(self.irreps_out_calc).dim)

    def __call__(self, node_species: Int[Array, 'batch']) -> e3nn.IrrepsArray:
        return e3nn.IrrepsArray(self.irreps_out_calc, self.embed(node_species))
    

class LinearReadoutBlock(nn.Module):
    output_irreps: e3nn.Irreps

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        return e3nn.flax.Linear(self.output_irreps)(x)
    

class NonLinearReadoutBlock(nn.Module):
    hidden_irreps: e3nn.Irreps
    output_irreps: e3nn.Irreps
    # activation: Optional[Callable] = None
    # gate: Optional[Callable] = None

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        """batch irreps -> batch irreps_out"""
        # x = [n_nodes, irreps]
        num_vectors = self.hidden_irreps.filter(
            drop=["0e", "0o"]
        ).num_irreps  # Multiplicity of (l > 0) irreps
        x = e3nn.flax.Linear(
            (self.hidden_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify()
        )(x)
        x = e3nn.gate(x, even_act=self.activation, even_gate_act=self.gate)
        return e3nn.flax.Linear(self.output_irreps)(x)  # [n_nodes, output_irreps]
    

class RadialEmbeddingBlock(nn.Module):
    r_max: float
    basis_functions: Callable[[jnp.ndarray], jnp.ndarray]
    envelope_function: Callable[[jnp.ndarray], jnp.ndarray]
    avg_r_min: Optional[float] = None

    def __call__(self, edge_lengths: jnp.ndarray) -> e3nn.IrrepsArray:
        """batch -> batch num_basis"""

        def func(lengths):
            basis = self.basis_functions(lengths, self.r_max)  # [n_edges, num_basis]
            cutoff = self.envelope_function(lengths, self.r_max)  # [n_edges]
            return basis * cutoff[:, None]  # [n_edges, num_basis]

        with jax.ensure_compile_time_eval():
            if self.avg_r_min is None:
                factor = 1.0
            else:
                samples = jnp.linspace(
                    self.avg_r_min, self.r_max, 1000, dtype=jnp.float64
                )
                factor = jnp.mean(func(samples) ** 2).item() ** -0.5

        embedding = factor * jnp.where(
            (edge_lengths == 0.0)[:, None], 0.0, func(edge_lengths)
        )  # [n_edges, num_basis]
        return e3nn.IrrepsArray(f"{embedding.shape[-1]}x0e", embedding)
    




A025582 = [0, 1, 3, 7, 12, 20, 30, 44, 65, 80, 96, 122, 147, 181, 203, 251, 289]

import functools as ft

reduced_symmetric_tensor_product_basis = ft.cache(e3nn.reduced_symmetric_tensor_product_basis)
reduced_tensor_product_basis = ft.cache(e3nn.reduced_tensor_product_basis)
# e3nn.reduced_symmetric_tensor_product_basis

class SymmetricContraction(nn.Module):
    correlation: int
    keep_irrep_out: Union[Set[e3nn.Irrep], str]
    num_species: int
    gradient_normalization: Union[str, float, None] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    @nn.compact
    def __call__(self, input: e3nn.IrrepsArray, index: jnp.ndarray) -> e3nn.IrrepsArray:
        gradient_normalization = self.gradient_normalization
        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[
                gradient_normalization
            ]        
        
        if isinstance(self.keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(self.keep_irrep_out)
            assert all(mul == 1 for mul, _ in keep_irrep_out)
        else:
            keep_irrep_out = self.keep_irrep_out


        keep_irrep_out_set = {e3nn.Irrep(ir) for ir in keep_irrep_out}
        def fn(self: nn.Module, input: e3nn.IrrepsArray, index: jnp.ndarray):
            # - This operation is parallel on the feature dimension (but each feature has its own parameters)
            # This operation is an efficient implementation of
            # vmap(lambda w, x: FunctionalLinear(irreps_out)(w, concatenate([x, tensor_product(x, x), tensor_product(x, x, x), ...])))(w, x)
            # up to x power self.correlation
            assert input.ndim == 2  # [num_features, irreps_x.dim]
            assert index.ndim == 0  # int

            out = dict()

            for order in range(self.correlation, 0, -1):  # correlation, ..., 1
                if self.off_diagonal:
                    x_ = jnp.roll(input.array, A025582[order - 1])
                else:
                    x_ = input.array

                if self.symmetric_tensor_product_basis:
                    U = e3nn.reduced_symmetric_tensor_product_basis(
                        input.irreps, order, keep_ir=keep_irrep_out_set
                    )
                else:
                    U = e3nn.reduced_tensor_product_basis(
                        [input.irreps] * order, keep_ir=keep_irrep_out_set
                    )
                # U = U / order  # normalization TODO(mario): put back after testing
                # NOTE(mario): The normalization constants (/order and /mul**0.5)
                # has been numerically checked to be correct.

                # TODO(mario) implement norm_p

                # ((w3 x + w2) x + w1) x
                #  \-----------/
                #       out

                for (mul, ir_out), u in zip(U.irreps, U.list):
                    u = u.astype(x_.dtype)
                    # u: ndarray [(irreps_x.dim)^order, multiplicity, ir_out.dim]
                    print(self)
                    w = self.param(
                        f'w{order}_{ir_out}',
                        nn.initializers.normal(stddev=(mul**-0.5) ** (1.0 - gradient_normalization)),
                        (self.num_species, mul, input.shape[0]))[index]  # [multiplicity, num_features]
                    
                    w = (
                        w * (mul**-0.5) ** gradient_normalization
                    )  # normalize weights

                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x_),
                        )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]
                    else:
                        out[ir_out] += jnp.einsum(
                            "...ki,kc->c...i", u, w
                        )  # [num_features, (irreps_x.dim)^order, ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \----------------/
                #         out (in the normal case)

                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                        continue  # already done (special case optimization above)

                    out[ir_out] = jnp.einsum(
                        "c...ji,cj->c...i", out[ir_out], x_
                    )  # [num_features, (irreps_x.dim)^(oder-1), ir_out.dim]

                # ((w3 x + w2) x + w1) x
                #  \-------------------/
                #           out

            # out[irrep_out] : [num_features, ir_out.dim]
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.IrrepsArray.from_list(
                irreps_out,
                [out[ir][:, None, :] for (_, ir) in irreps_out],
                (input.shape[0],),
            )

        # Treat batch indices using vmap
        shape = jnp.broadcast_shapes(input.shape[:-2], index.shape)
        input = input.broadcast_to(shape + input.shape[-2:])
        index = jnp.broadcast_to(index, shape)

        fn_mapped = fn
        for _ in range(input.ndim - 2):
            fn_mapped = nn.vmap(fn_mapped, split_rngs={'params': False})

        return fn_mapped(self, input, index)
    

class EquivariantProductBasisBlock(nn.Module):
    target_irreps: e3nn.Irreps
    correlation: int
    num_species: int
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False

    def setup(self):
        self.symmetric_contractions = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=self.correlation,
            num_species=self.num_species,
            gradient_normalization="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )

        self.linear = e3nn.flax.Linear(self.target_irreps)

    @nn.compact
    def __call__(
        self,
        node_feats: e3nn.IrrepsArray,  # [n_nodes, feature * irreps]
        node_specie: jnp.ndarray,  # [n_nodes, ] int
    ) -> e3nn.IrrepsArray:

        node_feats = node_feats.mul_to_axis().remove_nones()
        node_feats = SymmetricContraction(
            keep_irrep_out={ir for _, ir in self.target_irreps},
            correlation=self.correlation,
            num_species=self.num_species,
            gradient_normalization="element",  # NOTE: This is to copy mace-torch
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats, node_specie)
        node_feats = node_feats.axis_to_mul()
        return e3nn.flax.Linear(self.target_irreps)(node_feats)
    


class MessagePassingConvolution(nn.Module):
    avg_num_neighbors: float
    target_irreps: e3nn.Irreps
    max_ell: int
    activation: Callable

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> e3nn.IrrepsArray:
        """-> n_nodes irreps"""
        assert node_feats.ndim == 2

        messages = node_feats[senders]

        messages = e3nn.concatenate(
            [
                messages.filter(self.target_irreps),
                e3nn.tensor_product(
                    messages,
                    e3nn.spherical_harmonics(range(1, self.max_ell + 1), vectors, True),
                    filter_ir_out=self.target_irreps,
                ),
                # e3nn.tensor_product_with_spherical_harmonics(
                #     messages, vectors, self.max_ell
                # ).filter(self.target_irreps),
            ]
        ).regroup()  # [n_edges, irreps]

        # one = e3nn.IrrepsArray.ones("0e", edge_attrs.shape[:-1])
        # messages = e3nn.tensor_product(
        #     messages, e3nn.concatenate([one, edge_attrs.filter(drop="0e")])
        # ).filter(self.target_irreps)

        mix = e3nn.flax.MultiLayerPerceptron(
            3 * [64] + [messages.irreps.num_irreps],
            self.activation,
            output_activation=False,
        )(
            radial_embedding
        )  # [n_edges, num_irreps]

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, node_feats.shape[:1], messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)
    

class InteractionBlock(nn.Module):
    conv: MessagePassingConvolution

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges, ]
        receivers: jnp.ndarray,  # [n_edges, ]
    ) -> tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        """-> n_nodes irreps"""
        assert node_feats.ndim == 2
        assert vectors.ndim == 2
        assert radial_embedding.ndim == 2

        node_feats = e3nn.flax.Linear(node_feats.irreps, name="linear_up")(node_feats)

        node_feats = self.conv(vectors, node_feats, radial_embedding, senders, receivers)

        node_feats = e3nn.flax.Linear(self.conv.target_irreps, name="linear_down")(
            node_feats
        )

        assert node_feats.ndim == 2
        return node_feats  # [n_nodes, target_irreps]
    

def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm."""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)
    

class MACE(nn.Module):
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray]
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray]
    output_irreps: e3nn.Irreps  # Irreps of the output, default 1x0e    
    r_max: float
    avg_r_min: float
    num_interactions: int  # Number of interactions (layers), default 2
    hidden_irreps: e3nn.Irreps  # 256x0e or 128x0e + 128x1o
    readout_mlp_irreps: e3nn.Irreps  # Hidden irreps of the MLP in last readout, default 16x0e
    avg_num_neighbors: float
    num_species: int
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray]
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray]
    # Number of zero derivatives at small and large distances, default 4 and 2
    # If both are None, it uses a smooth C^inf envelope function
    max_ell: int = 3  # Max spherical harmonic degree, default 3
    epsilon: Optional[float] = None
    correlation: int = 3  # Correlation order at each layer (~ node_features^correlation), default 3
    gate: Callable = jax.nn.silu  # activation function
    soft_normalization: Optional[float] = None
    symmetric_tensor_product_basis: bool = True
    off_diagonal: bool = False
    interaction_irreps: Union[str, e3nn.Irreps] = "o3_restricted"  # or o3_full
    node_embedding_type: type[nn.Module] = LinearNodeEmbedding
    skip_connection_first_layer: bool = False
    num_features: Optional[int] = None  # Number of features per node, default gcd of hidden_irreps multiplicities
   

    def setup(self):
        self.output_irreps_calc = e3nn.Irreps(self.output_irreps)
        self.hidden_irreps_calc = e3nn.Irreps(self.hidden_irreps)
        self.readout_mlp_irreps_calc = e3nn.Irreps(self.readout_mlp_irreps)

        if self.num_features is None:
            self.num_features_calc = functools.reduce(
                math.gcd, (mul for mul, _ in self.hidden_irreps_calc)
            )
            self.hidden_irreps_calc = e3nn.Irreps(
                [(mul // self.num_features_calc, ir) for mul, ir in self.hidden_irreps_calc]
            )
        else:
            self.num_features_calc = self.num_features            

        if self.interaction_irreps == "o3_restricted":
            self.interaction_irreps_calc = e3nn.Irreps.spherical_harmonics(self.max_ell)
        elif self.interaction_irreps == "o3_full":
            self.interaction_irreps_calc = e3nn.Irreps(e3nn.Irrep.iterator(self.max_ell))
        else:
            self.interaction_irreps_calc = e3nn.Irreps(self.interaction_irreps)

        # Embeddings
        self.node_embedding = self.node_embedding_type(
            self.num_species, self.num_features_calc * self.hidden_irreps_calc
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=self.r_max,
            avg_r_min=self.avg_r_min,
            basis_functions=self.radial_basis,
            envelope_function=self.radial_envelope,
        )

        layers = []
        for i in range(self.num_interactions):
            first = i == 0
            last = i == self.num_interactions - 1            
            hidden_irreps = (
                e3nn.Irreps(self.hidden_irreps_calc)
                if not last
                else e3nn.Irreps(self.hidden_irreps_calc).filter(self.output_irreps_calc)
            )
            layers.append(MACELayer(
                first=first,
                last=last,
                num_features=self.num_features_calc,
                interaction_irreps=self.interaction_irreps_calc,
                hidden_irreps=hidden_irreps,
                max_ell=self.max_ell,
                avg_num_neighbors=self.avg_num_neighbors,
                activation=self.gate,
                num_species=self.num_species,
                epsilon=self.epsilon,
                correlation=self.correlation,
                output_irreps=self.output_irreps_calc,
                readout_mlp_irreps=self.readout_mlp_irreps_calc,
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                off_diagonal=self.off_diagonal,
                soft_normalization=self.soft_normalization,
                skip_connection_first_layer=self.skip_connection_first_layer,
                name=f"layer_{i}",
            ))

        self.layers = layers

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ) -> e3nn.IrrepsArray:
        """
        -> n_nodes num_interactions output_irreps
        """
        assert vectors.ndim == 2 and vectors.shape[1] == 3
        assert node_specie.ndim == 1
        assert senders.ndim == 1 and receivers.ndim == 1
        assert vectors.shape[0] == senders.shape[0] == receivers.shape[0]

        if node_mask is None:
            node_mask = jnp.ones(node_specie.shape[0], dtype=jnp.bool_)

        # Embeddings
        node_feats = self.node_embedding(node_specie).astype(
            vectors.dtype
        )  # [n_nodes, feature * irreps]
        node_feats = profile("embedding: node_feats", node_feats, node_mask[:, None])

        if not (hasattr(vectors, "irreps") and hasattr(vectors, "array")):
            vectors = e3nn.IrrepsArray("1o", vectors)

        radial_embedding = self.radial_embedding(safe_norm(vectors.array, axis=-1))

        # Interactions
        outputs = []
        for i in range(self.num_interactions):
            node_outputs, node_feats = self.layers[i](
                vectors,
                node_feats,
                node_specie,
                radial_embedding,
                senders,
                receivers,
                node_mask,
            )
            outputs += [node_outputs]  # list of [n_nodes, output_irreps]

        return e3nn.stack(outputs, axis=1)  # [n_nodes, num_interactions, output_irreps]


try:
    from profile_nn_jax import profile
except ImportError:
    def profile(_, x, __=None):
        return x

class MACELayer(nn.Module):
    first: bool
    last: bool
    num_features: int
    interaction_irreps: e3nn.Irreps
    hidden_irreps: e3nn.Irreps
    activation: Callable
    num_species: int
    epsilon: Optional[float]
    name: Optional[str]
    # InteractionBlock:
    max_ell: int
    avg_num_neighbors: float
    # EquivariantProductBasisBlock:
    correlation: int
    symmetric_tensor_product_basis: bool
    off_diagonal: bool
    soft_normalization: Optional[float]
    # ReadoutBlock:
    output_irreps: e3nn.Irreps
    readout_mlp_irreps: e3nn.Irreps
    skip_connection_first_layer: bool = False

    @nn.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        radial_embedding: jnp.ndarray,  # [n_edges, radial_embedding_dim]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes] only used for profiling
    ):
        """-> (n_nodes output_irreps, n_nodes features*hidden_irreps) """
        if node_mask is None:
            node_mask = jnp.ones(node_specie.shape[0], dtype=jnp.bool_)

        node_feats = profile(f"{self.name}: input", node_feats, node_mask[:, None])

        sc = None
        if not self.first or self.skip_connection_first_layer:
            sc = e3nn.flax.Linear(
                self.num_features * self.hidden_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp",
            )(
                node_specie, node_feats
            )  # [n_nodes, feature * hidden_irreps]
            sc = profile(f"{self.name}: self-connexion", sc, node_mask[:, None])

        node_feats = InteractionBlock(MessagePassingConvolution(
            target_irreps=self.num_features * self.interaction_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            max_ell=self.max_ell,
            activation=self.activation,
        ))(
            vectors=vectors,
            node_feats=node_feats,
            radial_embedding=radial_embedding,
            receivers=receivers,
            senders=senders,
        )

        if self.epsilon is not None:
            node_feats *= self.epsilon
        else:
            node_feats /= jnp.sqrt(self.avg_num_neighbors)

        node_feats = profile(
            f"{self.name}: interaction", node_feats, node_mask[:, None]
        )

        if self.first:
            # Selector TensorProduct
            node_feats = e3nn.flax.Linear(
                self.num_features * self.interaction_irreps,
                num_indexed_weights=self.num_species,
                name="skip_tp_first",
            )(node_specie, node_feats)
            node_feats = profile(
                f"{self.name}: skip_tp_first", node_feats, node_mask[:, None]
            )

        node_feats = EquivariantProductBasisBlock(
            target_irreps=self.num_features * self.hidden_irreps,
            correlation=self.correlation,
            num_species=self.num_species,
            symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
            off_diagonal=self.off_diagonal,
        )(node_feats=node_feats, node_specie=node_specie)

        node_feats = profile(
            f"{self.name}: tensor power", node_feats, node_mask[:, None]
        )

        if self.soft_normalization is not None:

            def phi(n):
                n = n / self.soft_normalization
                return 1.0 / (1.0 + n * e3nn.sus(n))

            node_feats = e3nn.norm_activation(
                node_feats, [phi] * len(node_feats.irreps)
            )

            node_feats = profile(
                f"{self.name}: soft normalization", node_feats, node_mask[:, None]
            )

        if sc is not None:
            node_feats = node_feats + sc  # [n_nodes, feature * hidden_irreps]

        if not self.last:
            node_outputs = LinearReadoutBlock(self.output_irreps)(
                node_feats
            )  # [n_nodes, output_irreps]
        else:  # Non linear readout for last layer
            node_outputs = NonLinearReadoutBlock(
                self.readout_mlp_irreps,
                self.output_irreps,
                activation=self.activation,
            )(
                node_feats
            )  # [n_nodes, output_irreps]

        node_outputs = profile(f"{self.name}: output", node_outputs, node_mask[:, None])
        return node_outputs, node_feats
    
if __name__ == '__main__':
    from cdv.config import MainConfig
    import pyrallis
    from cdv.dataset import load_file, dataloader
    from eins import EinsOp

    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()

    r_max: float = 7
    avg_num_neighbors: float = 20
    mean: float = 0
    std: float = 1
    avg_r_min: float = 1.5

    num_bessel: int = 8
    envelope_arg_multiplicator: float = 2.0
    envelope_value_at_origin: float = 1.2

    symmetric_tensor_product_basis = True  # symmetric is slightly worse but slightly faster
    off_diagonal = False
    max_ell = 1
    num_interactions = 1
    # hidden_irreps = "256x0e + 256x1o"
    hidden_irreps = "16x0e + 16x1o"
    interaction_irreps = "o3_restricted"  # "o3_restricted" seems to be better than "o3_full"
    epsilon = 0.4  # set to None to use the default value of MACE, 1/sqrt(avg_num_neighbors)
    correlation = 2  # 4 is better but 5x slower
    gate = nn.silu
    readout_mlp_irreps = "16x0e"
    num_species = config.data.num_species

    def bessel_basis(length, max_length):
        return e3nn.bessel(length, num_bessel, max_length)
    
    def soft_envelope(length, max_length):
        return e3nn.soft_envelope(
            length,
            max_length,
            arg_multiplicator=envelope_arg_multiplicator,
            value_at_origin=envelope_value_at_origin,
        )
    
    def polynomial_envelope(length, max_length, degree0: int, degree1: int):
        return e3nn.poly_envelope(degree0, degree1, max_length)(length)
    
    def u_envelope(length, max_length, p: int):
        return e3nn.poly_envelope(p - 1, 2, max_length)(length)
    
    mace = MACE(output_irreps='0e', r_max=r_max, num_interactions=num_interactions, avg_r_min=avg_r_min, hidden_irreps=hidden_irreps, 
                readout_mlp_irreps=readout_mlp_irreps, avg_num_neighbors=avg_num_neighbors, num_species=num_species, num_features=None, max_ell=max_ell, epsilon=epsilon, 
                correlation=correlation, gate=gate, soft_normalization=None, symmetric_tensor_product_basis=symmetric_tensor_product_basis, off_diagonal=off_diagonal, 
                interaction_irreps=interaction_irreps, node_embedding_type=LinearNodeEmbedding, skip_connection_first_layer=False,
                radial_basis=bessel_basis, radial_envelope=soft_envelope)
    
    cg = load_file(config, 20)

    send_pos = cg.nodes.cart[cg.senders]        
    offsets = EinsOp('e abc xyz, e abc -> e xyz')(cg.graph_data.lat[cg.edges.graph_i], cg.edges.to_jimage)
    recv_pos = cg.nodes.cart[cg.receivers] + offsets

    vecs = recv_pos - send_pos

    key = jax.random.key(12345)        
    ctx = Context(training=True)

    with jax.debug_nans(True):
        with jax.log_compiles(True):
            contributions, params = mace.init_with_output(key, vecs, cg.nodes.species, cg.edges.sender, cg.edges.receiver)
            # [n_nodes, num_interactions, 0e]

            contributions = contributions.array[:, :, 0] # [n_nodes, num_interactions]
    
    debug_structure(contributions)
    debug_stat(contributions)
    flax_summary(mace, vectors=vecs, node_species=cg.nodes.species, sender=cg.edges.sender, receiver=cg.edges.receiver)
    # steps_per_epoch, dl = dataloader(config, split='train', infinite=True)
        
    # @jax.jit
    # def loss(params, batch):
    #     cg = batch
    #     send_pos = cg.nodes.cart[cg.senders]        
    #     offsets = EinsOp('e abc xyz, e abc -> e xyz')(cg.graph_data.lat[cg.edges.graph_i], cg.edges.to_jimage)
    #     recv_pos = cg.nodes.cart[cg.receivers] + offsets

    #     vecs = recv_pos - send_pos
    #     preds = mace.apply(params, cg)
    #     return config.train.loss.regression_loss(preds, batch.graph_data.e_form.reshape(-1, 1), batch.padding_mask)
    
    # res = jax.value_and_grad(loss)(params, batch)
    # jax.block_until_ready(res)

    # from ctypes import cdll
    # libcudart = cdll.LoadLibrary('libcudart.so')

    # libcudart.cudaProfilerStart()
    # for i in range(1):
    #     batch = next(dl)
    #     res = jax.value_and_grad(loss)(params, batch)
    #     jax.block_until_ready(res)
    # libcudart.cudaProfilerStop()
    
    # debug_stat(value=res[0], grad=res[1])

    # debug_structure(batch=batch, module=model, out=out)
    # debug_stat(batch=batch, module=model, out=out)    
    # flax_summary(mace, vecs, cg.nodes.species, cg.edges.sender, cg.edges.receiver)