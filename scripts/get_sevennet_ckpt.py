"""Downloads and processes the SevenNet checkpoint, converting to a JAX PyTree."""

import urllib.request
from pathlib import Path
import torch
import jax
import numpy as np

CKPT_URL = 'https://github.com/MDIL-SNU/SevenNet/raw/refs/heads/main/sevenn/pretrained_potentials/SevenNet_0__11July2024/checkpoint_sevennet_0.pth'
download_path = Path('data') / 'sevennet_ckpt.pth'
processed_path = Path('precomputed') / 'sevennet_ckpt.npy'

if __name__ == '__main__':
    if not download_path.exists():
        urllib.request.urlretrieve(CKPT_URL, download_path)

    ckpt = torch.load(download_path)

    params = ckpt['model_state_dict']

    def convert_tensor(t):
        if t.size == 0:
            return None
        else:
            return t.detach().cpu().numpy()

    params = jax.tree.map(convert_tensor, params)
    np.save(processed_path, params, allow_pickle=True)

    print('Done!')
