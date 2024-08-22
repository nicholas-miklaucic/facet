import ijson
import gc
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure

with open('data/MPtrj_2022.9_full.json', 'r') as f:
    objs = ijson.kvitems(f, '', use_float=True)

    data = []
    i = 0
    for _key, vals in tqdm(objs, total=1_590_395):
        for frame_id, val in vals.items():
            task, calc, step = frame_id.rsplit('-', maxsplit=2)
            val['structure'] = Structure.from_dict(val['structure'])
            val['calc'] = int(calc)
            val['step'] = int(step)
            data.append(val)
        i += 1
        if i % 4096 == 0:
            df = pd.DataFrame(data)
            df.to_pickle(f'precomputed/mptrj/raw/batch_{i // 4096:04}.pkl')
            data = []
            del df
            gc.collect()


df = pd.DataFrame(data)
df.to_pickle('precomputed/mptrj/raw/batch_0000.pkl')
df.head()

print(i)