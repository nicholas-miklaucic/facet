"""
Code for dataset metadata.

Constructing models requires knowing things like the number of different elements, average
scale/shift for species, and average number of neighbors. This should be precomputed, and so we need
a representation of that metadata to save and load from memory.
"""

from flax.struct import dataclass
from jaxtyping import ArrayLike, Float, Array, Int, UInt
import jax.numpy as jnp

from facet.utils import debug_structure


@dataclass
class DatasetMetadata:
    """The metadata associated with a dataset: what we need to construct models."""

    # The dataset name.
    dataset_name: str

    # The supported targets: 'energy', 'force', 'stress', 'magmom'.
    supported_targets: tuple[str]

    # The number of batches per group. For datasets with many frames from a single trajectory, all
    # of the frames will be within a single group. This ensures that train/test/validation splits
    # give an appropriate estimate of model performance.
    batches_per_group: UInt[Array, ' num_groups']

    # The number of atoms per batch.
    batch_num_atoms: int

    # The maximum number of edges leaving a single node.
    nearest_k: int

    # The number of graphs per batch.
    batch_num_graphs: int

    # Median energy per atom for the entire dataset.
    shift_energy: float

    # MAE of energy per atom for the entire dataset.
    scale_energy: float

    # The atomic number assigned to each index
    atomic_numbers: UInt[Array, ' num_elements']

    # Element-wise energy per atom for the entire dataset.
    # Regressed from the total values.
    atomwise_shift_energy: Float[Array, ' num_elements']

    # Element-wise MAE energy per atom for the entire dataset.
    # Regressed from the total values.
    atomwise_scale_energy: Float[Array, ' num_elements']

    # The given r_max values.
    r_max_quantile_r: Float[Array, ' num_quantiles']

    # The number of neighbors with the given r_max, by node.
    r_max_quantile_k: Float[Array, ' num_quantiles k+1']

    @property
    def num_groups(self) -> int:
        """Gets the number of groups in the dataset."""
        return self.batches_per_group.size

    @property
    def num_elements(self) -> int:
        """Gets the number of elements in the dataset."""
        return self.atomic_numbers.size

    def avg_num_neighbors(self, r_max: ArrayLike) -> ArrayLike:
        """Gets the average number of neighbors with the given cutoff, in the same shape as
        r_max."""
        return jnp.interp(
            x=r_max,
            xp=self.r_max_quantile_r,
            fp=self.r_max_quantile_k
            @ jnp.arange(0, self.r_max_quantile_k.shape[-1], dtype=self.r_max_quantile_k.dtype),
        )


if __name__ == '__main__':
    from facet.utils import load_pytree

    metadata_dict = load_pytree('precomputed/mp2022/metadata.mpk')
    metadata = DatasetMetadata(**metadata_dict)

    debug_structure(metadata)

    rr = jnp.linspace(5, 8, 13)
    yy = metadata.avg_num_neighbors(rr)

    print(rr.round(2))
    print(yy.round(2))
