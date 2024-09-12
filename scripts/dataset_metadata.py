"""Generates the necessary metadata for a dataset."""

from facet.dataset import dataloader
from pyrallis import wrap
from facet.config import MainConfig
from facet.layers import edge_vecs
from itertools import chain
from rich.progress import track
from pymatgen.core import Element
import jax
import json
import numpy as np

@wrap()
def main(config: MainConfig):
    dls = []
    steps = 0
    for split in ('train', 'test', 'valid'):
        split_steps, split_dl = dataloader(config, split=split, infinite=False)
        dls.append(split_dl)
        steps += split_steps
    
    species = set()
    e_forms = np.array([])
    dist_bins = np.linspace(0, 10, 21)
    dist_centers = (dist_bins[1:] + dist_bins[:-1]) / 2
    dist_counts = np.zeros(len(dist_centers), dtype=np.uint64)
    n_nodes = 0
    for step, batch in track(zip(range(steps), chain(*dls)), total=steps):        
        edge_batch = jax.vmap(edge_vecs)(batch)
        for i in range(batch.e_form.shape[0]):            
            node_mask = batch.padding_mask[i][batch.nodes.graph_i[i]]
            species.update(np.unique(np.array(batch.nodes.species[i])[node_mask]))
            e_forms = np.hstack((e_forms, batch.e_form[i][batch.padding_mask[i]].reshape(-1)))            
            edges = np.array(edge_batch[i])
            dists = np.sqrt(np.sum(np.square(edges[node_mask]), axis=-1)).reshape(-1)
            dist_hist, _bins = np.histogram(dists, bins=dist_bins, density=False)            
            dist_counts += dist_hist.astype(np.uint64)
            n_nodes += sum(node_mask).item()

    species = sorted(Element.from_Z(z) for z in species)

    print(np.quantile(e_forms[::7], np.linspace(0, 1, 11)).round(2))
    
    data_size = steps * config.batch_size * config.train_batch_multiple
    e_form_mean = np.mean(e_forms)
    e_form_std = np.std(e_forms)

    elements = [e.symbol for e in species]
    nums = [el.Z for el in species]
    element_inds = np.ones(max(nums) + 1) * 1000
    element_inds[nums] = np.arange(len(elements))
    # we need to set this to something that won't nan because it's how we pad
    element_inds[0] = -1

    print(list(dist_centers), list(dist_counts))

    with open(config.data.dataset_folder / 'metadata.json', 'w') as out:
        json.dump({
            'elements': elements,
            'element_indices': element_inds.astype(int).tolist(),
            'e_form': {
                'mean': e_form_mean,
                'std': e_form_std
            },
            'num_batches': steps * config.train_batch_multiple,
            'batch_size': config.batch_size,  
            'distances': {
                'bins': dist_centers.tolist(),
                'counts': dist_counts.astype(int).tolist()
            },
            'n_nodes': n_nodes,
        }, out, indent=2)


if __name__ == '__main__':
    main()