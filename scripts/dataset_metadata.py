"""Generates the necessary metadata for a dataset."""

from cdv.dataset import dataloader
from pyrallis import wrap
from cdv.config import MainConfig
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
    for step, batch in track(zip(range(steps), chain(*dls)), total=steps):
        for i in range(batch.e_form.shape[0]):
            species.update(np.unique(np.array(batch.nodes.species[i])[batch.padding_mask[i][batch.nodes.graph_i[i]]]))
            e_forms = np.hstack((e_forms, batch.e_form[i][batch.padding_mask[i]].reshape(-1)))

    species = sorted(Element.from_Z(z) for z in species)

    print(np.quantile(e_forms[::7], np.linspace(0, 1, 11)).round(2))
    
    data_size = steps * config.batch_size * config.train_batch_multiple
    e_form_mean = np.mean(e_forms)
    e_form_std = np.std(e_forms)

    elements = [e.symbol for e in species]
    nums = [el.Z for el in species]
    element_inds = np.ones(max(nums) + 1) * 1000
    element_inds[nums] = np.arange(len(elements))

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
        }, out, indent=2)


if __name__ == '__main__':
    main()