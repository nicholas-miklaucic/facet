"""Utilities."""

from os import PathLike
from pathlib import Path
import re
from abc import ABCMeta
from dataclasses import asdict, is_dataclass
from functools import partial
from types import MappingProxyType

import flax.linen as nn
import humanize
import jax
import jax.numpy as jnp
import numpy as np
import rich
from jaxtyping import jaxtyped
from pymatgen.core import Element
from rich.style import Style
from rich.tree import Tree
from flax.serialization import msgpack_restore, msgpack_serialize

ELEM_VALS = 'Li Be B N O F Na Mg Al Si S K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Rb Sr Y Zr Nb Mo Ru Rh Pd Ag Cd In Sn Sb Te Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi'.split(
    ' '
)

def format_scalar(scalar: int | float, chars=6) -> str:
    """Formats the scalar using no more than the given number of characters, if possible."""
    if scalar < 0:
        return '-' + format_scalar(scalar, chars=chars-1)    
    if isinstance(scalar, int):
        if len(str(scalar)) - chars > 3:
            # egregiously longer, use scientific notation
            used = len('{:.0E}'.format(scalar))
            remaining = max(chars - used, 0)
            return '{:.dE}'.replace('d', str(remaining)).format(scalar)
        else:
            return str(scalar)
    else:
        if scalar == int(scalar):
            return format_scalar(int(scalar), chars=chars - 1) + '.'
        
        decimal = str(scalar)
        if decimal.startswith('0.'):
            frac = decimal.removeprefix('0.')
            wasted = len(frac.lstrip('0')) - len(decimal.replace('.', ''))
        else:
            wasted = len(decimal.replace('.', '').rstrip('0')) - len(decimal.replace('.', ''))

        suffix = '{:.0e}'.format(scalar)[1:]
        if wasted > len(suffix):
            used = len('{:.0E}'.format(scalar))
            remaining = max(chars - used, 0)
            return '{:.dE}'.replace('d', str(remaining)).format(scalar)
        else:
            remaining = max(chars - 2, 0)
            return '{:.df}'.replace('d', str(remaining)).format(scalar)

def item_if_arr(x: int | float | jax.Array) -> float:
    if isinstance(x, (int, float)):
        return x
    else:
        return x.item()


# checker = typechecker(conf=BeartypeConf(is_color=False))
tcheck = partial(jaxtyped, typechecker=None)


def val_to_elem(val: int) -> str:
    return ELEM_VALS[val]


def elem_to_val(elem: str | Element) -> int:
    if isinstance(elem, Element):
        elem = elem.symbol

    return ELEM_VALS.index(elem)


def load_pytree(file: PathLike):
    """Loads a MsgPack serialized PyTree."""
    with open(Path(file), 'rb') as infile:
        return msgpack_restore(infile.read())

def save_pytree(obj, file: PathLike):
    """Saves a MsgPack serialized PyTree."""
    with open(Path(file), 'wb') as out:
        out.write(msgpack_serialize(obj))



class AbstractTreeVisitor(metaclass=ABCMeta):
    def __init__(self):
        pass

    def jax_arr(self, arr: jax.Array):
        raise NotImplementedError()

    def scalar(self, x: int | float):
        raise NotImplementedError()

    def np_arr(self, arr: np.ndarray):
        raise NotImplementedError()


class StatVisitor(AbstractTreeVisitor):
    def __init__(self) -> None:
        super().__init__()

    def jax_arr(self, arr: jax.Array):
        flat = arr.flatten().astype(jnp.float32)
        inds = 1 + 0.01 * jnp.cos(jnp.arange(len(flat), dtype=jnp.float32))
        return f'{(flat * inds).mean().item():.4f}'

    def scalar(self, x: int | float):
        return f'{x:.4f}' + x.__class__.__name__[0]

    def np_arr(self, arr: np.ndarray):
        return self.jax_arr(jnp.array(arr))


class StructureVisitor(AbstractTreeVisitor):
    def __init__(self):
        super().__init__()

    def jax_arr(self, arr: jax.Array):
        return f'{arr.dtype}{list(arr.shape)}'

    def scalar(self, x: int | float):
        return f'{x.__class__.__name__}=' + str(x)

    def np_arr(self, arr: np.ndarray):
        return f'np{list(arr.shape)}'


COLORS = [
    '#00a0ec',
    '#00bc70',
    '#deca00',
    '#ff7300',
    '#d83990',
    '#7555d3',
    '#8ac7ff',
    '#00f0ff',
    '#387200',
    '#aa2e00',
    '#ff7dc6',
    '#e960ff',
]

KWARGS = [dict(color=color) for color in COLORS]
KWARGS[0]['bold'] = True

STYLES = [Style(**kwargs) for kwargs in KWARGS]


def tree_from_dict(base: Tree, obj, depth=0, collapse_single=True):
    style = STYLES[depth % len(STYLES)]
    if isinstance(obj, dict):
        if len(obj) == 1:
            k, v = next(iter(obj.items()))
            base.label = base.label + ' >>> ' + k
            tree_from_dict(base, v, depth)
        else:
            for k, v in obj.items():
                child = base.add(k, style=style)
                tree_from_dict(child, v, depth + 1)
    else:
        base.add(obj, style=style)


def tree_traverse(visitor: AbstractTreeVisitor, obj, max_depth=2, collapse_single=True):
    if isinstance(obj, jax.Array):
        return visitor.jax_arr(obj)
    elif isinstance(obj, np.ndarray):
        return visitor.np_arr(obj)
    elif isinstance(obj, (list, tuple)):
        if max_depth == 0:
            return '[...]'
        else:
            if collapse_single and len(obj) == 1:
                new_depth = max_depth
            else:
                new_depth = max_depth - 1

            return {str(i): tree_traverse(visitor, child, new_depth) for i, child in enumerate(obj)}
    elif isinstance(obj, (float, int)):
        return visitor.scalar(obj)
    elif isinstance(obj, dict):
        if max_depth == 0:
            return '{...}'
        else:
            if collapse_single and len(obj) == 1:
                new_depth = max_depth
            else:
                new_depth = max_depth - 1

            excluded = (('parent', None), ('name', None))
            return {
                k: tree_traverse(visitor, v, new_depth)
                for k, v in obj.items()
                if (k, v) not in excluded
            }
    elif is_dataclass(obj):
        return {obj.__class__.__name__: tree_traverse(visitor, asdict(obj), max_depth)}
    else:
        name = getattr(obj, '__name__', '|')
        return f'{obj.__class__.__name__}={name}'


def show_obj(obj):
    # print_json(data=obj)
    for k, v in obj.items():
        tree = Tree(label=k, style=STYLES[0])
        tree_from_dict(tree, v)
        rich.print(tree)


def _debug_structure(tree_depth=5, **kwargs):
    """Prints out the structure of the inputs."""
    show_obj({f'{k}': tree_traverse(StructureVisitor(), v, tree_depth) for k, v in kwargs.items()})


def _debug_stat(tree_depth=5, **kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries."""
    show_obj({f'{k}': tree_traverse(StatVisitor(), v, tree_depth) for k, v in kwargs.items()})


def debug_structure(*args, **kwargs):
    """Prints out the structure of the inputs. Returns first argument."""
    new_kwargs = {f'arg{i}': arg for i, arg in enumerate(args)}
    new_kwargs.update(**kwargs)
    jax.debug.callback(_debug_structure, **new_kwargs)
    return list(new_kwargs.values())[0]


def debug_stat(*args, **kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries. Returns first argument."""
    new_kwargs = {f'arg{i}': arg for i, arg in enumerate(args)}
    new_kwargs.update(**kwargs)
    jax.debug.callback(_debug_stat, **new_kwargs)
    return list(new_kwargs.values())[0]


def flax_summary(
    mod: nn.Module,
    rngs={},
    *args,
    compute_flops=True,
    compute_vjp_flops=True,
    console_kwargs=None,
    table_kwargs=MappingProxyType({'safe_box': False, 'expand': True, 'box': rich.box.SIMPLE}),
    column_kwargs=MappingProxyType({'justify': 'right'}),
    show_repeated=False,
    depth=None,
    **kwargs,
):
    out = mod.tabulate(
        dict(params=jax.random.key(0), **rngs),
        *args,
        compute_flops=compute_flops,
        compute_vjp_flops=compute_vjp_flops,
        console_kwargs=console_kwargs,
        table_kwargs=table_kwargs,
        column_kwargs=column_kwargs,
        show_repeated=show_repeated,
        depth=depth,
        **kwargs,
    )

    # hack to control numbers so they're formatted reasonably
    # 12580739072 flops is not very helpful

    def human_units(m: re.Match):
        """Format using units, preserving the length with spaces."""
        human = humanize.metric(int(m.group(0))).replace(' ', '')
        pad_num = len(m.group(0)) - len(human)
        return ' ' * pad_num + human

    out = re.sub(r'\d' * 6 + '+', human_units, out)    

    print(out)
