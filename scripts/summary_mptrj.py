"""Script to create a DataFrame storing metadata for exploration and visualization."""


from collections import Counter
from facet.data.databatch import CrystalGraphs
from facet.data.dataset import dataloader
from pyrallis import wrap
from facet.config import MainConfig
from facet.layers import edge_vecs
from tqdm import tqdm
from pymatgen.core import Element
import jax
import numpy as np
import pandas as pd

@wrap()
def main(config: MainConfig):
    splits = ('train', 'test', 'valid')
        
    species = set()    
    dist_bins = np.linspace(0, 10, 21)
    dist_centers = (dist_bins[1:] + dist_bins[:-1]) / 2
    df = []
    for split in splits:
        split_steps, split_dl = dataloader(config, split=split, infinite=False)
        for step, batch in tqdm(zip(range(split_steps), split_dl), total=split_steps):
            # if len(df) > 100:
            #     break

            batch: CrystalGraphs = batch            
            for stack_i in range(batch.e_form.shape[0]):
                b: CrystalGraphs = jax.tree.map(lambda x: np.array(x[stack_i]), batch)
                edge_batch = edge_vecs(b)
                for i in np.nonzero(b.padding_mask)[0]:
                    row = {
                        'split': split,                    
                        'batch_i': step,
                        'stack_i': stack_i,
                        'graph_i': i,
                        'dataset_id': b.graph_data.dataset_id[i],                        
                    }
                    for k, v in zip('abc', b.graph_data.abc[i]):
                        row[k] = v

                    for k, v in zip(('alpha', 'beta', 'gamma'), b.graph_data.angles_rad[i]):
                        row[k] = np.rad2deg(v)
                        
                    node_mask = b.nodes.graph_i == i                    
                    row['e_form'] = b.e_form[i].item()
                    edges = np.array(edge_batch[node_mask])  # k xyz
                    dists = np.sqrt(np.sum(np.square(edges), axis=-1)).reshape(-1)
                    dist_hist, _bins = np.histogram(dists, bins=dist_bins, density=False)            
                    for center, count in zip(dist_centers, dist_hist):
                        row[f'bin_{center:05.2f}'] = count

                    row['n_nodes'] = np.sum(node_mask).item()

                    species = Counter([Element.from_Z(z.item()).symbol for z in b.nodes.species[node_mask]])
                    row.update(species)

                    df.append(row)

    df = pd.DataFrame(df)
    df['split'] = df['split'].astype('category')
    
    df.to_feather(config.data.dataset_folder / 'summary.feather')


if __name__ == '__main__':
    main()