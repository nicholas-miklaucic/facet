"""Processes the MP2022 dataset."""

from pathlib import Path
import numpy as np
from tqdm import tqdm
from facet.data.dataset_generation import (
    BatchMetadataTracker,
    BatchProcessor,
    ElementIndexer,
    GraphSummarizer,
    PrecomputedKNN,
    RMaxBinner,
    make_data_id_mp2022,
)

data_folder = Path('precomputed') / 'mp2022'
graphs_folder = Path('/home/nmiklaucic/mat-graph/crystallographic_graph/knns_mp2022')

processor = BatchProcessor(
    data_folder=data_folder,
    graphs_folder=graphs_folder,
    knn_strategy=PrecomputedKNN,
    data_id_maker=make_data_id_mp2022,
    indexer=ElementIndexer(),
    summarizer=GraphSummarizer(),
    binner=RMaxBinner((np.arange(5, 121, 5) / 10).tolist()),
    tracker=BatchMetadataTracker(),
    num_batch=32,
    k=32,
    num_atoms=14,
    energy_key='energy',
)


if __name__ == '__main__' or True:
    import gc
    from rich.prompt import Confirm
    from rich.progress import track
    # if not Confirm.ask('Regenerate batched MP2022 files?'):
    #     raise RuntimeError('Aborted')

    batches = sorted((data_folder / 'raw').glob('batch_*.pkl'))
    names = [batch_fn.stem.removeprefix('batch_') for batch_fn in batches]

    # names = names[:1]

    for name in names:
        res = processor.process_batch(name, overwrite=True, max_batches=0)
        print(f'Finished {res}')
        gc.collect()

    processor.save_raw_metadata()

    print('Done!')
