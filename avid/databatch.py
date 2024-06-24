from flax.struct import dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

@dataclass
class DataBatch:
    """A batch of data."""

    density: Float[Array, 'batch n_grid n_grid n_grid max_spec']
    species: Int[Array, 'batch max_spec']
    mask: Bool[Array, 'batch max_spec']
    e_form: Float[Array, 'batch']
    e_total: Float[Array, 'batch']
    e_hull: Float[Array, 'batch']
    magmom: Float[Array, 'batch']
    cell_density: Float[Array, 'batch']
    bandgap: Float[Array, 'batch']
    space_group: Int[Array, 'batch']
    index: Int[Array, 'batch']