"""Code to generate datasets."""

from collections import Counter
from functools import cache
import json
import logging
from typing import Callable, Sequence
from chex import dataclass
from tqdm import tqdm
from jaxtyping import Float, Array
from facet.data.databatch import CrystalGraphs, CrystalData, EdgeData, NodeData, TargetInfo
from pymatgen.core import Structure, Composition
import numpy as np
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
import pickle
import multiprocessing

from facet.layers import edge_vecs
from facet.utils import debug_structure, save_pytree


def get_parts(numbers, batch, chunk_size):
    """Splits the numbers into batches of length batch with as equal a split as possible."""
    # assert len(numbers) % (batch * chunk_size) == 0
    n_batches = len(numbers) // batch
    parts = np.zeros((batch, n_batches), dtype=np.int32)
    part_sizes = np.array([0 for _ in range(n_batches)])

    chunk_i = 0
    for sample_is in np.argsort(-numbers).reshape(batch // chunk_size, chunk_size * n_batches):
        sample_sizes = numbers[sample_is]
        n_filled = np.zeros((n_batches,), dtype=np.int32)
        for sample_i, sample_size in zip(sample_is, sample_sizes):
            next_i = np.argmin(part_sizes + 10000 * (n_filled == chunk_size))
            parts[chunk_i * chunk_size + n_filled[next_i], next_i] += sample_i
            n_filled[next_i] += 1
            part_sizes[next_i] += sample_size
        chunk_i += 1

    return parts, part_sizes


def padded_parts(sizes, num_batch, target_batch_size, extra=0):
    """Partitions the data into groups with minimal wasted space, adding extra padding to make
    batches fit into the desired size until it can be achieved."""
    part_size = num_batch - 1
    data_pad = -len(sizes) % part_size + part_size * extra
    pad_sizes = np.array(list(sizes) + [0] * data_pad)
    parts, part_sizes = get_parts(pad_sizes, part_size, part_size)
    if max(part_sizes) > target_batch_size:
        return padded_parts(sizes, num_batch, target_batch_size, extra=extra + 4)
    else:
        return parts, part_sizes


class KNN:
    """k-NN computation method."""

    def __init__(self, graphs_path: Path):
        """
        graphs_path: pickle file to precomputed graphs
        """
        pass

    def knn_graph(self, struct: Structure, k: int, struct_id: int | None = None) -> EdgeData:
        """Computes the k-NN graph for the periodic structure. struct_id is the index of the
        structure, used for precomputed lookup."""
        raise NotImplementedError


class NaiveKNN(KNN):
    """
    KNN graph creator. Much slower than accelerated versions that aren't yet ported to JAX, so this
    is not recommended to use. Instead, use precomputed lists from mat-graph.
    """

    def _knn_graph_helper(self, struct: Structure, k: int, r_max: float) -> EdgeData:
        graph_ijs = []
        graph_ims = []
        if r_max > np.sqrt(np.sum(np.array(struct.lattice.abc) ** 2)) * 4:
            raise ValueError(f'r_max={r_max} is extremely large for {k}-NN.\nStructure:\n{struct}')

        r = r_max
        for i, nbs in enumerate(struct.get_all_neighbors(r)):
            sites, dists, idxs, ims = zip(*nbs)
            if len(dists) < k:
                return self._knn_graph_helper(struct, k=k, r_max=r * 2)

            chosen = np.argsort(dists)[:k]

            graph_ijs.append(np.array(idxs)[chosen])
            graph_ims.append(np.array(ims)[chosen])

        graph_ijs = np.stack(graph_ijs).astype(np.uint16)
        graph_ims = np.stack(graph_ims).astype(np.int8)

        return EdgeData(jnp.array(graph_ims), jnp.array(graph_ijs))

    def knn_graph(self, struct: Structure, k: int, struct_id: int | None = None) -> EdgeData:
        # start with a reasonable guess for the r_max based on lattice and occupancy
        diag = np.sqrt(np.sum(np.array(struct.lattice.abc) ** 2)) + 1
        r_max = diag * np.cbrt(k / struct.num_sites)
        return self._knn_graph_helper(struct, k=k, r_max=r_max)


class PrecomputedKNN(KNN):
    """
    KNN graph creator that uses precomputed graph files.
    """

    def __init__(self, graphs_path: Path):
        """
        graphs_path: pickle file to precomputed graphs
        """
        super().__init__(graphs_path=graphs_path)
        with open(graphs_path, 'rb') as f:
            self.graphs = pickle.load(f)

    def knn_graph(self, struct: Structure, k: int, struct_id: int | None = None) -> EdgeData:
        if struct_id is None:
            raise ValueError('Precomputed k-NN only works if struct_id is passed')

        graph_data = self.graphs[struct_id]
        return EdgeData(to_jimage=np.array(graph_data['ims']), receiver=np.array(graph_data['ijs']))


def make_data_id_mp2022(df):
    base_id = (
        df['dataset-id']
        .str.replace('mp-', '')
        .str.replace('mvc-', '')
        .str.replace('-GGA', '')
        .str.replace('+U', '')
        .astype(int)
    )

    is_mvc = df['dataset-id'].str.contains('mp-', regex=False).astype(int)
    is_gga = df['dataset-id'].str.contains('-GGA', regex=False).astype(int)
    is_u = df['dataset-id'].str.contains('+U', regex=False).astype(int)

    data_id = base_id * 10 + (is_mvc * 4 + is_gga * 2 + is_u)
    return data_id


def make_data_id_mptrj(df):
    data_id = df['mp_id'].str.replace('mp-', '1').str.replace('mvc-', '2').astype(int) * 1000
    data_id += df['calc'] * 2
    data_id += df['step']
    return data_id


class ElementIndexer:
    """Assigns elements to indices in order of appearance."""

    def __init__(self):
        # 0 for padding
        self.numbers = [0]

    @cache
    def get(self, atomic_num: int) -> int:
        """Gets the index, adding it to the list if not seen before."""
        if atomic_num not in self.numbers:
            self.numbers.append(atomic_num)

        return self.numbers.index(atomic_num)


class RMaxBinner:
    """Computes bins of the r_max values for approximate effective neighbor calculation."""

    def __init__(self, bins: Sequence[float]):
        """Bins is a list of r_max values for binning."""
        self.bins = np.array(bins)
        self.counts: np.ndarray | None = None

    def update(self, dists: Float[Array, ' nodes k']):
        if self.counts is None:
            self.counts = np.zeros((len(self.bins), dists.shape[-1] + 1))
        dists_in_range = dists[..., None] < self.bins[None, None, :]  # nodes k bins

        num_in_range = dists_in_range.sum(axis=1).T  # k[bins nodes]

        for i, counts in enumerate(num_in_range):
            # counts: k[nodes]
            uniq, uniq_counts = np.unique(counts, return_counts=True)
            self.counts[i][uniq] += uniq_counts


class GraphSummarizer:
    """Summarizes graph-level data into a DataFrame that can be processed separately."""

    def __init__(self):
        self.max_species = 0
        self.species = []
        self.total_energies = []
        self.energies = []

    def update(self, species, energy):
        self.max_species = max(self.max_species, max(species))
        self.energies.append(energy / len(species))
        self.total_energies.append(energy)
        self.species.append(Counter(species))

    def to_df(self):
        data = np.zeros((len(self.energies), self.max_species + 3))
        for i, (species, energy, total_energy) in enumerate(
            zip(self.species, self.energies, self.total_energies)
        ):
            for k, v in species.items():
                data[i][k] = v

            data[i][-2] = energy
            data[i][-1] = total_energy

        return pd.DataFrame(
            data, columns=[*map(str, range(self.max_species + 1)), 'energy', 'total_energy']
        )


class BatchMetadataTracker:
    """Keeps track of dataset metadata."""

    def __init__(self):
        self.batches_per_group = []
        self.node_pad_fracs = []
        self.graph_pad_fracs = []

    def new_group(self):
        self.batches_per_group.append(0)

    def update(self, batch: CrystalGraphs):
        self.batches_per_group[-1] += 1
        self.node_pad_fracs.append(batch.padding_mask[batch.nodes.graph_i].sum().item())
        self.graph_pad_fracs.append(batch.padding_mask.sum().item())


@dataclass
class BatchProcessor:
    data_folder: Path
    graphs_folder: Path | None
    knn_strategy: type[KNN]
    data_id_maker: Callable
    indexer: ElementIndexer
    binner: RMaxBinner
    summarizer: GraphSummarizer
    tracker: BatchMetadataTracker
    num_batch: int = 32
    k: int = 16
    num_atoms: int = 32
    energy_key: str = 'corrected_total_energy'

    def create_graph(self, row, data_id: int, edges: EdgeData) -> CrystalGraphs:
        struct: Structure = row['structure']

        species_i = [self.indexer.get(z) for z in struct.atomic_numbers]
        nodes = NodeData(
            species=np.array(species_i, dtype=np.uint8),
            cart=np.array(struct.cart_coords, dtype=np.float32),
            graph_i=np.zeros((struct.num_sites,), dtype=np.uint16),
        )

        data = CrystalData(
            dataset_id=np.array([data_id], dtype=np.uint32),
            abc=np.array([struct.lattice.abc]),
            angles_rad=np.deg2rad(np.array([struct.lattice.angles])),
            lat=np.array([struct.lattice.matrix]),
        )

        if 'force' in row:
            force = row['force']
        else:
            force = nodes.cart * 0

        if 'stress' in row:
            stress = row['stress']
        else:
            stress = np.array([struct.lattice.matrix * 0])

        target = TargetInfo(
            e_form=np.array([row[self.energy_key]]),
            force=force,
            stress=stress,
        )

        cg = CrystalGraphs(
            nodes,
            edges,
            n_node=np.array([struct.num_sites], dtype=np.uint16),
            padding_mask=jnp.ones((1,), dtype=np.bool_),
            graph_data=data,
            target_data=target,
        )

        # debug_structure(cg=cg)

        self.summarizer.update(species_i, target.e_form.item())

        dists = np.sqrt(np.sum(np.square(edge_vecs(cg)), axis=-1) + 1e-6)
        self.binner.update(dists)

        return cg

    def save_raw_metadata(self):
        raw_metadata = {
            'r_max_bins': self.binner.bins.tolist(),
            'r_max_counts': self.binner.counts.tolist(),
            'atomic_numbers': self.indexer.numbers,
            'batches_per_group': self.tracker.batches_per_group,
            'node_pad_frac': self.tracker.node_pad_fracs,
            'graph_pad_frac': self.tracker.graph_pad_fracs,
            'num_batch': self.num_batch,
            'num_atoms': self.num_atoms,
            'k': self.k,
        }
        print(raw_metadata)
        with open(self.data_folder / 'raw_metadata.json', 'w') as f:
            json.dump(raw_metadata, f)

        self.summarizer.to_df().to_feather(self.data_folder / 'energy_data.feather')

    def process_batch(self, batch_name, overwrite: bool = False, max_batches: int = 0):
        path = self.data_folder / 'raw' / f'batch_{batch_name}.pkl'
        df = pd.read_pickle(path)

        if self.graphs_folder is None:
            graph_path = None
        else:
            graph_path = self.graphs_folder / f'batch_{batch_name}.pkl'
        knn = self.knn_strategy(graph_path)

        out_path = self.data_folder / 'batches' / f'group_{batch_name}'
        out_path.mkdir(exist_ok=True, parents=True)

        sizes = [s.num_sites for s in df['structure']]

        data_id = self.data_id_maker(df)

        orig_size = len(sizes)
        parts, part_sizes = padded_parts(sizes, self.num_batch, self.num_batch * self.num_atoms - 1)

        self.tracker.new_group()

        for part_i, partition in tqdm(enumerate(parts.T), total=len(parts.T)):
            if part_i >= max_batches and max_batches != 0:
                return batch_name

            out_fn = out_path / f'{part_i:05}.mpk'
            if out_fn.exists() and not overwrite:
                logging.info(f'Skipping {out_fn}: already exists')
                continue

            cgs = []
            for i in partition:
                if i >= orig_size:
                    # empty pad
                    continue

                row = df.iloc[i]
                edges = knn.knn_graph(row['structure'], self.k, i)

                cgs.append(self.create_graph(row, data_id.iloc[i], edges))

            cg = sum(cgs[1:], start=cgs[0])
            cg = cg.padded(self.num_batch * self.num_atoms, self.k, self.num_batch)
            self.tracker.update(cg)
            save_pytree(cg, out_fn)

        return batch_name
