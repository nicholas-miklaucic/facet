
import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import rho_plus as rp

is_dark = True
theme, cs = rp.mpl_setup(is_dark)
rp.plotly_setup(is_dark)


from jarvis.db.figshare import data

data3d = data('dft_3d')

df = pd.DataFrame(data3d)

from jarvis.core.atoms import Atoms
from pymatgen.io.jarvis import JarvisAtomsAdaptor

df['atoms'] = [JarvisAtomsAdaptor.get_structure(Atoms.from_dict(a)) for a in df['atoms']]
df.head()


for col in df.columns:
    if 'na' in df[col].values:
        try:
            df[col] = pd.to_numeric(df[col].replace('na', np.nan))
        except (TypeError, ValueError):
            print(col)

df.select_dtypes(include='number')


from pymatgen.core import Composition
df['spg_number'] = df['spg_number'].astype(int)
df['formula'] = [Composition(f) for f in df['formula']]
df['num_spec'] = [len(f.elements) for f in df['formula']]
df.head()


col_nans = df.select_dtypes('number').isna().mean(axis=0).sort_values()


df['magmom'] = np.where(df.eval('magmom_outcar < 0'), df['magmom_oszicar'], df['magmom_outcar'])
df['magmom'].quantile(np.linspace(0, 1, 9))


clean = df.dropna(axis=1, thresh=0.94 * len(df.index))
clean = clean.drop(columns=['magmom_outcar', 'spg_symbol', 'jid', 'func', 'effective_masses_300K', 'typ', 'spg', 'raw_files', 'reference', 'search', 'elastic_tensor', 'kpoint_length_unit', 'icsd', 'xml_data_link', 'modes', 'encut', 'efg'])

# clean = clean.query('crys == "cubic"').drop(columns=['crys'])
clean = clean.drop(columns=['crys'])


pd.value_counts(clean['dimensionality'])


counts = pd.value_counts(clean['nat'].clip(None, 40)).sort_index()
counts.to_frame().T.iloc[:, 12:30]


clean = clean.query('nat <= 16')


from collections import Counter

els = Counter()

for form in clean['formula']:
    for el in form.elements:
        els[el] += 1

els = pd.Series(els)
filtered = els.sort_values()[-72:]

clean = clean[[all(e in filtered.index for e in f.elements) for f in clean['formula']]]


clean['magmom'] = clean['magmom'].fillna(0)
clean = clean.rename(columns={'optb88vdw_bandgap': 'bandgap', 'nat': 'num_atoms', 'spg_number': 'space_group', 'formation_energy_peratom': 'e_form', 'optb88vdw_total_energy': 'e_total'})


clean.sort_values('e_form')
# cutoff chosen so the number of materials divides 2^9: cutoff is largely arbitrary
clean = clean.query('e_form < 3.7')
clean.to_pickle('precomputed/jarvis_dft3d_cleaned/dataframe.pkl')