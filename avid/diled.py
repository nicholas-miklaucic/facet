"""
Generalizable Diffusion with Learned Encoding-Decoding.
"""

from math import perm
from typing import Sequence

import einops
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
from eins import EinsOp
from flax import linen as nn
import optax

from avid.databatch import DataBatch
from avid.diffusion import DiffusionInput, DiffusionModel
from avid.layers import DeepSetEncoder, LazyInMLP, MLPMixer, PermInvariantEncoder
from avid.utils import debug_stat, debug_structure
from avid.vit import ABY_IDENTITY, Encoder, ViTRegressor

# class O3ImageEmbed(nn.Module):
#     """Embeds a 3D image into a sequnce of tokens."""

#     patch_size: int
#     patch_latent_dim: int
#     patch_heads: int
#     pos_embed: nn.Module

#     @tcheck
#     @nn.compact
#     def __call__(self, im: Float[Array, 'I I I C']) -> Float[Array, '_ip3 _dimout']:
#         i = im.shape[0]
#         p = self.patch_size
#         c = im.shape[-1]
#         assert i % p == 0
#         n = i // p

#         patches = im.reshape(n, p, n, p, n, p, c)
#         patches = rearrange(patches, 'n1 p1 n2 p2 n3 p3 c -> (n1 n2 n3) p1 p2 p3 c')

#         embed = jax.vmap(O3Patchify(self.patch_latent_dim, inner_dim=self.patch_heads))(patches)
#         # legendre
#         # pos_embed = self.pos_embed(rearrange(embed, '(i1 i2 i3) c -> i1 i2 i3 c', i1=n, i2=n,
#         # i3=n))
#         # return rearrange(pos_embed, 'i1 i2 i3 c -> (i1 i2 i3) c', i1=n, i2=n, i3=n)

#         # learned
#         pos_embed = self.pos_embed(embed)
#         return pos_embed


def conv_basis(conv_size: int) -> np.ndarray:
    conv_basis = np.zeros((conv_size, conv_size, conv_size), dtype=np.int_)
    n = (conv_size + 2) // 2
    i = 0
    for xyz in np.mgrid[0:n, 0:n, 0:n].reshape(3, -1).T:
        if not np.allclose(np.argsort(xyz), np.arange(3)):
            continue
        for s_xyz in np.mgrid[0:2, 0:2, 0:2].reshape(3, -1).T:
            # flip
            flip_xyz = xyz - 2 * xyz * s_xyz
            for perm in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
                # move -n - n into 0 - conv_size
                xi, yi, zi = flip_xyz[perm] + (n - 1)
                conv_basis[xi, yi, zi] = i
        i += 1
    return conv_basis


class SpeciesTwoWayEmbed(nn.Module):
    """Embedding module that supports recovering densities."""
    n_species: int
    embed_dim: int

    def species_embed_matrix(self):
        """Generates species embedding matrix of orthogonal vectors: (n_species, embed_dim)."""
        raise NotImplementedError

    def __call__(self, data: DataBatch):
        """Embeds the batch as a 3D image, shape batch n n n embed_dim."""
        raise NotImplementedError

    def decode(self, im, data: DataBatch):
        """Projects inputs to obtain the original densities."""
        raise NotImplementedError

class OrthoSpeciesEmbed(SpeciesTwoWayEmbed):
    """Embedding module that supports recovering densities. Uses orthogonal embeddings."""
    n_species: int
    embed_dim: int

    def setup(self):
        if self.embed_dim < self.n_species:
            raise ValueError('Must have more embed dims than species')
        self.mat = jnp.eye(self.embed_dim, self.n_species, dtype=jnp.bfloat16)
        self.ii, self.jj = jnp.tril_indices_from(self.mat, k=-1)
        self.vs = self.param(
            'embed_raw',
            nn.initializers.truncated_normal(stddev=0.2, dtype=jnp.bfloat16, lower=-2, upper=2),
            (self.n_species, self.embed_dim),
        )

    def species_embed_matrix(self):
        """Generates species embedding matrix of orthogonal vectors: (n_species, embed_dim)."""
        vvt = EinsOp('m n1, m n2 -> m n1 n2')(self.vs, self.vs)
        vtv = EinsOp('m n, m n -> m 1 1')(self.vs, self.vs)
        mats = jnp.eye(self.embed_dim, self.embed_dim) - 2 / vtv * vvt
        orth = jnp.linalg.multi_dot(mats)
        return orth[: self.n_species]

    def __call__(self, data: DataBatch):
        """Embeds the batch as a 3D image, shape batch n n n embed_dim."""
        spec = jax.nn.one_hot(data.species, self.n_species, dtype=jnp.bfloat16)
        # debug_structure(dens=data.density, spec=spec, mat=self.species_embed_matrix())
        return einops.einsum(
            data.density,
            spec,
            self.species_embed_matrix(),
            'batch n1 n2 n3 max_spec, batch max_spec n_spec, n_spec emb -> batch n1 n2 n3 emb',
        ).astype(jnp.bfloat16)

    def decode(self, im, data: DataBatch):
        """Projects inputs to obtain the original densities."""
        # TODO use einsop when this doesn't overflow
        proj = jnp.einsum('bxyze,se->bxyzs', im, self.species_embed_matrix())
        proj = jnp.take_along_axis(proj, data.species[:, None, None, None, :], axis=-1)
        return proj

class LossySpeciesEmbed(SpeciesTwoWayEmbed):
    """Embedding module that supports recovering densities."""
    n_species: int
    embed_dim: int

    def setup(self):
        self.emb = nn.Embed(self.n_species, self.embed_dim, dtype=jnp.bfloat16)
        # self.decoder = LazyInMLP(
        #     inner_dims=[64],
        #     out_dim=self.n_species,
        #     name='species_decoder',
        #     inner_act=nn.gelu,
        # )

    def species_embed_matrix(self):
        """Generates species embedding matrix of orthogonal vectors: (n_species, embed_dim)."""
        return self.emb.embedding

    def __call__(self, data: DataBatch):
        """Embeds the batch as a 3D image, shape batch n n n embed_dim."""
        # spec = jax.nn.one_hot(data.species, self.n_species, dtype=jnp.bfloat16)
        # debug_structure(spec=spec, mat=self.species_embed_matrix())
        embs = self.emb(data.species)

        # debug_structure(embs=embs, dens=data.density, mask=data.mask)
        return einops.einsum(
            data.density * data.mask[:, None, None, None, :].astype(data.density.dtype),
            jnp.where(data.mask[..., None], embs, 0),
            'batch n1 n2 n3 max_spec, batch max_spec emb -> batch n1 n2 n3 emb',
        )

    def decode(self, im, data: DataBatch):
        """Projects inputs to obtain the original densities."""
        # dens: b n3 s
        # encoded: b n3 e
        # embs: b s e
        # solve for dens from encoded, embs
        # (b e s)(b s n3) = b e n3

        embs_flip = EinsOp('b s e -> b e s')
        emb_reshape = EinsOp('b n n n e -> b e (n n n)')
        dens_reshape = EinsOp('b n n n s -> b s (n n n)')
        dens_unpack = EinsOp('b s (n=24 n n) -> b n n n s')

        embs = jnp.take(self.emb.embedding, data.species, axis=0) * data.mask[..., None]

        # Least-squares solving gradients are not stable. Instead of solving Ax = b
        # and then comparing x to the actual density, which won't work because dx/dθ is unstable,
        # we compare A @ x_true with b.
        E = embs_flip(jax.lax.stop_gradient(embs))
        ENC = emb_reshape(jax.lax.stop_gradient(im))

        x, resid, rank, s = jax.vmap(jnp.linalg.lstsq)(E.astype(float), ENC.astype(float))

        im_hat = EinsOp('b s e, b n n n s -> b n n n e')(embs, data.density)
        dens = dens_unpack(x)

        out = jnp.take_along_axis(dens, data.species[:, None, None, None, :], axis=-1)
        return out, im_hat


class GConv(nn.Module):
    """Group-equivariant convolution."""

    conv_size: int
    in_features: int
    out_features: int
    stride: int
    transpose: bool = False

    def setup(self):
        self.basis = conv_basis(self.conv_size)
        self.basis = [np.equal(self.basis, k) for k in range(max(self.basis.flatten()) + 1)]
        self.basis = jnp.array(self.basis, dtype=jnp.bfloat16)
        self.subspace_dim = self.basis.shape[0]
        self.kernel = self.param(
            'kernel',
            nn.initializers.normal(stddev=1e-1, dtype=jnp.bfloat16),
            (self.subspace_dim, self.in_features, self.out_features),
        )

        self.project = EinsOp('subspace c_in c_out, subspace n1 n2 n3 -> n1 n2 n3 c_in c_out')

        self.conv = nn.Conv(
            features=self.out_features,
            kernel_size=(self.conv_size, self.conv_size, self.conv_size),
            use_bias=False,
            padding='CIRCULAR',
            dtype=jnp.bfloat16,
            strides=self.stride,
        )

        self.conv_transpose = nn.ConvTranspose(
            features=self.in_features,
            kernel_size=(self.conv_size, self.conv_size, self.conv_size),
            use_bias=False,
            padding='CIRCULAR',
            dtype=jnp.bfloat16,
            strides=[self.stride, self.stride, self.stride],
            transpose_kernel=True,
        )

    def __call__(self, x):
        if self.transpose:
            conv = self.conv_transpose
        else:
            conv = self.conv

        k_proj = self.project(self.kernel, self.basis).astype(jnp.bfloat16)

        conv_out = conv.apply({'params': {'kernel': k_proj}}, x)
        return conv_out


class EncoderDecoder(nn.Module):
    """Embeds species as orthonormal vectors and then applies a patch embedding with learned linear decoder."""

    patch_latent_dim: int
    patch_conv_sizes: Sequence[int]
    patch_conv_strides: Sequence[int]
    patch_conv_features: Sequence[int]
    use_dec_conv: bool
    species_embed_dim: int
    n_species: int
    species_embed_type: str

    def setup(self):
        if self.species_embed_type == 'ortho':
            self.spec_emb = OrthoSpeciesEmbed(
                n_species=self.n_species, embed_dim=self.species_embed_dim
            )
        elif self.species_embed_type == 'lossy':
            self.spec_emb = LossySpeciesEmbed(
                n_species=self.n_species, embed_dim=self.species_embed_dim
            )
        else:
            raise ValueError('Species embed type not recognized:', self.species_embed_type)
        prev_feats = self.species_embed_dim
        enc_convs = []
        dec_convs = []
        for size, stride, feats in zip(
            self.patch_conv_sizes, self.patch_conv_strides, self.patch_conv_features
        ):
            kwargs = dict(conv_size=size, in_features=prev_feats, out_features=feats, stride=stride)
            for convs, transpose in zip((enc_convs, dec_convs), (False, True)):
                if transpose and not self.use_dec_conv:
                    continue
                block = [
                    GConv(transpose=transpose, **kwargs),
                    nn.LayerNorm(dtype=jnp.bfloat16),
                ]
                convs.append(nn.Sequential(block))
            prev_feats = feats

        self.encoder_conv = nn.Sequential(enc_convs)
        if self.use_dec_conv:
            self.decoder_conv = nn.Sequential(dec_convs[::-1])

        self.patch_proj = nn.Sequential([nn.LayerNorm(dtype=jnp.bfloat16), nn.DenseGeneral(
            features=self.patch_latent_dim, axis=-1, dtype=jnp.bfloat16),
        nn.LayerNorm(dtype=jnp.bfloat16)])

        if self.use_dec_conv:
            self.dec_patch_proj = nn.Sequential([
                nn.LayerNorm(dtype=jnp.bfloat16),
                nn.DenseGeneral(features=feats, axis=-1, dtype=jnp.bfloat16),
            ])
        else:
            self.dec_patch_proj = nn.Sequential([nn.LayerNorm(dtype=jnp.bfloat16), nn.DenseGeneral(
                features=int(np.prod(self.patch_conv_strides) ** 3 * self.species_embed_dim),
                axis=-1,
                dtype=jnp.bfloat16,
            )])

    def encode(self, data: DataBatch, training: bool):
        """Returns (spec_emb, enc_conv, patches)"""
        spec_emb = self.spec_emb(data)
        enc = self.encoder_conv(spec_emb)
        patches = self.patch_proj(enc)
        return (spec_emb, enc, patches)

    def patch_decode(self, patches, data: DataBatch, training: bool):
        """Returns (density, rec_spec_emb, spec_emb, dec_conv)"""
        dec = self.dec_patch_proj(patches)
        if self.use_dec_conv:
            spec_emb = self.decoder_conv(dec)
        else:
            spec_emb = EinsOp(
                'b i i i (p p p d) -> b (i p) (i p) (i p) d',
                symbol_values={'d': self.species_embed_dim},
            )(dec)

        dens, rec_spec_emb = self.spec_emb.decode(spec_emb, data)
        return (dens, rec_spec_emb, spec_emb, dec)


class Category:
    """Get category from data."""

    def __call__(self, data: DataBatch):
        """Returns list of integers for category."""
        raise NotImplementedError

    @property
    def num_categories(self) -> int:
        raise NotImplementedError


@dataclass
class EFormCategory(Category):
    num_cats: int = 8
    min_bin: float = -0.5
    max_bin: float = 5.0

    @property
    def bins(self):
        return jnp.linspace(self.min_bin, self.max_bin, self.num_categories - 2)

    def __call__(self, data: DataBatch):
        return jnp.digitize(data.e_form, self.bins)

    @property
    def num_categories(self) -> int:
        return self.num_cats


@dataclass
class SpaceGroupCategory(Category):
    just_cubic: bool = True

    def __call__(self, data: DataBatch):
        if self.just_cubic:
            return data.space_group - 195
        else:
            return data.space_group - 1

    @property
    def num_categories(self) -> int:
        if self.just_cubic:
            return 230 - 194
        else:
            return 230


class DiLED(nn.Module):
    """DiLED model."""

    encoder_decoder: EncoderDecoder
    diffusion: DiffusionModel
    category: Category
    encoder: Encoder | MLPMixer
    perm_encoder: nn.Module
    head: nn.Module
    w: float = 1
    class_dropout: float = 0.5

    @nn.compact
    def __call__(self, data: DataBatch, training: bool):
        """Runs a training step, returning loss. Uses time, noise RNG keys."""
        x_0, conv, patch = self.encoder_decoder.encode(data, training=training)
        enc = patch
        beta_0 = 1e-3
        eps_0 = jax.random.normal(self.make_rng('noise'), shape=enc.shape)
        x_1 = enc + beta_0 * eps_0

        dens, rec_x_0_hat, rec_x_0, dec = self.encoder_decoder.patch_decode(x_1, data, training=training)

        losses = {}

        def voxel_loss(x1, x2):
            return jnp.sqrt(jnp.mean(jnp.square(x1 - x2)))

        # print(data.species)
        # losses['l_dens_rec'] = 2 * voxel_loss(dens, data.density)

        # losses['l_patch_rec'] = 0 * voxel_loss(rec_x_0, x_0)
        # losses['l_spec_rec'] = 0 * voxel_loss(rec_x_0_hat, rec_x_0)        
        # losses['l_patch_reg'] = 0 * voxel_loss(jnp.zeros_like(patch), patch)

        y = self.category(data)

        encoder = self.encoder
        abys = ABY_IDENTITY

        patch_seq = EinsOp('b n n n d -> b (n n n) d')(patch)
        out = encoder(patch_seq, abys, training=training)

        out = DeepSetEncoder(phi=self.perm_encoder)(out, training=training)
        # debug_structure(out=out)

        head = self.head.copy(out_dim=3 + self.category.num_categories)
        out = jax.vmap(lambda x: head(x, training=training))(out)

        # debug_structure(out=out)
        e_form = out[:, 0]
        bandgap = out[:, 1]
        magmom = out[:, 2]
        y_logits = out[:, 3:]
        bandgap = nn.leaky_relu((3 * (bandgap - 1.2)))
        magmom = nn.leaky_relu((3 * (magmom - 0.8)))

        def smooth_l1(x, b=3):
            d = b * x
            l = jnp.log(jnp.cosh(d)) + jnp.log(2) * jnp.square(jnp.tanh(d))
            return l / b
        
        mean_loss = lambda x, y: jnp.mean(smooth_l1(x, y))

        mean_loss = lambda x, y: jnp.sqrt(jnp.mean(jnp.square(x - y)))

        losses['e_form'] = 1 * mean_loss(e_form, data.e_form)
        losses['bandgap'] = 0.1 * mean_loss(bandgap, data.bandgap)
        # losses['magmom_mae'] = 0.2 * jnp.mean(jnp.abs(magmom - data.magmom))
        losses['spg_ce'] = 0.1 * jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(y_logits, y))

        if self.w == 0:
            losses['loss'] = sum(losses.values())
            return losses

        t = jax.random.randint(
            self.make_rng('time'), [1], minval=1, maxval=self.diffusion.schedule.max_t
        )

        y_mask = jnp.where(
            jax.random.uniform(self.make_rng('noise'), (y.shape[0],)) < self.class_dropout,
            -jnp.ones_like(y),
            y,
        )
        eps = jax.random.normal(self.make_rng('noise'), shape=x_1.shape)

        def l_align(eps):
            x_2 = self.diffusion.q_sample(x_1, 2, eps)
            mu_2_1 = self.diffusion.mu_eps_sigma(
                DiffusionInput(x_t=x_2, t=jnp.array([1]), y=y_mask),
                training=training
            )
            loss = voxel_loss(patch, mu_2_1['mu'])
            return loss * 1000

        losses['l_align'] = l_align(eps) / self.diffusion.schedule.max_t

        def l_diffusion(eps):
            x_t = self.diffusion.q_sample(enc, t, eps)
            mu_eps = self.diffusion.mu_eps_sigma(
                DiffusionInput(x_t=x_t, t=jnp.array([t]), y=y_mask),
                training=training
            )
            return voxel_loss(mu_eps['eps'], eps)

        losses['l_diffuse'] = self.w * l_diffusion(eps)

        losses['loss'] = sum(losses.values()) / len(losses)
        return losses
