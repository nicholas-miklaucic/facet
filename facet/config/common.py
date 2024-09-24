from chex import dataclass as _dataclass
from functools import partial

dataclass = partial(_dataclass, mappable_dataclass=False, unsafe_hash=True)
