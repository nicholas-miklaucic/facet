import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import rho_plus as rp


from typing import Any

import e3nn_jax as e3nn
from pymatgen.core import Element

import treescope as ts
import treescope.figures as tsf

from pathlib import Path

import pyrallis

from facet.config import MainConfig

from facet.data.databatch import CrystalGraphs
from facet.utils import StatsVisitor, StructureVisitor, tree_traverse
from facet.data.dataset import load_file
from facet.layers import Context


from dataclasses import dataclass

from flax import linen as nn
from treescope import rendering_parts as tsr

from facet.layers import Context, Identity
from facet.utils import signature


def add_with_duplicated_name(d: dict, k, v):
    prefix = 0
    while f'{prefix}_{k}' in d:
        prefix += 1
    d[f'{prefix}_{k}'] = v


class Params(dict):
    def __treescope_repr__(self, path, subtree_renderer):
        size = jax.tree.reduce(
            lambda x, y: x + y, jax.tree.map(lambda x: x.size, dict(self.items())), initializer=0
        )

        children = []
        for k, v in self.items():
            child_path = None if path is None else f'{path}.{k}'
            children.append(
                tsr.build_full_line_with_annotations(
                    tsr.siblings_with_annotations(
                        f'{k}: ',
                        subtree_renderer(v, path=child_path),
                    )
                )
            )
        return tsr.build_foldable_tree_node_from_children(
            prefix=tsr.text('Params{'),
            children=children,
            comma_separated=True,
            suffix='}',
            path=path,
            background_color='salmon',
            first_line_annotation=tsr.abbreviation_color(tsr.text(f'{size} parameters')),
            expand_state=tsr.ExpandState.COLLAPSED,
        )


@dataclass
class ModuleCall:
    module: nn.Module
    input: dict[str, Any]
    params: dict[str, Any]
    children: dict[str, 'ModuleCall']
    output: Any

    def __treescope_repr__(self, path, subtree_renderer):
        attributes = {}
        if len(self.input):
            attributes['input'] = self.input

        if len(self.children):
            attributes['children'] = self.children

        if len(self.params):
            attributes['params'] = Params(**self.params)

        if self.output is not None:
            attributes['output'] = self.output

        children = []
        for k, v in attributes.items():
            child_path = None if path is None else f'{path}.{k}'
            children.append(
                tsr.build_full_line_with_annotations(
                    tsr.siblings_with_annotations(
                        f'{k}: ',
                        subtree_renderer(v, path=child_path),
                    )
                )
            )
        return tsr.build_foldable_tree_node_from_children(
            prefix=tsr.siblings(tsr.maybe_qualified_type_name(type(self.module)), '('),
            children=[tsr.on_separate_lines(children)],
            suffix=')',
            path=path,
            comma_separated=True,
            first_line_annotation=tsr.abbreviation_color(tsr.text(self.module.name or '')),
            background_color=ts.formatting_util.color_from_string(str(type(self.module))),
            expand_state=tsr.ExpandState.COLLAPSED,
        )

        # return ts.repr_lib.render_object_constructor(
        #     object_type=type(self.module),
        #     attributes=attributes,
        #     color=ts.formatting_util.color_from_string(str(type(self.module))),
        #     **kwargs
        # )


def insert(stack, call, path):
    if len(path) == 0:
        i = 0
        while f'{i}' in call.children:
            i += 1
        call.children[f'{i}'] = stack
        return call

    head, *tail = path
    if head in stack.children:
        stack.children[head] = insert(stack.children[head], call, tail)
    else:
        stack.children[head] = call

    return stack


class FlowRecorder:
    def __init__(self):
        self.stack = None
        self.call_chain = []

    def __call__(self, next_fun, args, kwargs, context):
        # print(self.call_chain)
        # print(type(context.module), context.module.path, context.method_name)
        if context.method_name == 'setup' or isinstance(context.module, Identity):
            return next_fun(*args, **kwargs)

        if context.method_name == '__call__':
            path = context.module.path
        else:
            *head, tail = context.module.path
            path = (*head, tail + '.' + context.method_name)

        if path:
            self.call_chain.append(path[-1])

        sig = signature(next_fun)
        bound = sig.bind(*args, **kwargs)

        call = ModuleCall(
            context.module,
            {k: v for k, v in bound.arguments.items() if k != 'ctx'},
            {
                k: v
                for k, v in context.module.variables.get('params', {}).items()
                if not isinstance(v, dict)
            },
            {},
            None,
        )

        if self.stack is None:
            self.stack = call
        else:
            self.stack = insert(self.stack, call, self.call_chain)

        out = context.orig_method(*args, **kwargs)
        call.output = out

        if path:
            self.call_chain.remove(path[-1])

        return out


def visualize_model_flow(
    config: MainConfig,
    params: dict | None = None,
    is_dark: bool = False,
    return_html: bool = True,
    cg: CrystalGraphs | None = None,
):
    theme, cs = rp.mpl_setup(is_dark)
    rp.plotly_setup(is_dark)

    config.batch_size = 32

    if params is None:
        # test initialization
        config.model.resid_init = 'ones'

    model = config.build_regressor()

    if cg is None:
        cg = load_file(config)

    if params is None:
        out, params = model.init_with_output(jr.key(29205), cg=cg, ctx=Context(training=True))

    obj = FlowRecorder()
    ctx = Context(training=False)
    mod = model.bind(params)
    with nn.intercept_methods(obj):
        out = mod(cg=cg, ctx=ctx)

    elements = {z: Element.from_Z(z).symbol for z in range(1, 100)}

    md = mod.dataset_metadata
    atom_nos = md.atomic_numbers

    symbol_perm = np.argsort(atom_nos)
    symbols = [Element.from_Z(z).symbol if z != 0 else '0' for z in sorted(atom_nos)]

    colors = pd.read_csv(
        'https://raw.githubusercontent.com/CorySimon/JMolColors/master/jmolcolors.csv'
    )
    jmol_palette = [(row['R'], row['G'], row['B']) for i, row in colors.iterrows()]

    # x = e3nn.normal('128x0e + 64x1e + 32x2e', leading_shape=(32,))

    if is_dark:
        div = rp.mpl_div_icefire_shift
    else:
        div = rp.mpl_div_coolwarm_shift

    ts.default_diverging_colormap.set_globally((255 * div(jnp.linspace(0, 1, 20))).tolist())

    def render_tensor(arr, abbrev_color=None, **kwargs):
        axis_item_labels = kwargs.get('axis_item_labels', {})
        axis_labels = kwargs.get('axis_labels', {})
        for i, size in enumerate(arr.shape):
            if size == len(md.atomic_numbers):
                axis_labels[i] = 'species'
                axis_item_labels[i] = symbols
                arr = jnp.take(arr, symbol_perm, axis=i)
            elif size == (config.data.batch_n_nodes or md.batch_num_atoms * md.batch_num_graphs):
                node_mask = cg.padding_mask[cg.nodes.graph_i]
                new_shape = (
                    [1 for _ in range(i)] + [size] + [1 for _ in range(i + 1, len(arr.shape))]
                )
                # print(new_shape, node_mask, size)
                kwargs['valid_mask'] = node_mask.reshape(*new_shape)

        kwargs['axis_item_labels'] = axis_item_labels
        kwargs['axis_labels'] = axis_labels

        if arr.dtype == np.int16 and jnp.max(arr) <= 100:
            kwargs['value_item_labels'] = elements
            kwargs['colormap'] = ['grey', *jmol_palette]
            arr = jnp.take(mod.dataset_metadata.atomic_numbers, arr)

        structure = tree_traverse(StructureVisitor(), arr)
        stats = tree_traverse(StatsVisitor(pad=False), arr)

        if abbrev_color is None:
            abbrev = lambda x: tsr.abbreviation_color(tsr.text(x))
        else:
            abbrev = lambda x: tsr.custom_text_color(tsr.text(x), abbrev_color)
        rendering = tsr.build_custom_foldable_tree_node(
            label=abbrev(f'{structure} {stats}'),
            contents=tsr.siblings(
                tsr.fold_condition(
                    expanded=ts.lowering.maybe_defer_rendering(
                        lambda _maybe_exp_state: tsr.siblings(
                            abbrev(':'),
                            tsr.indented_children(
                                [
                                    ts.render_array(
                                        arr,
                                        pixels_per_cell=5,
                                        truncate=True,
                                        **kwargs,
                                    ).treescope_part
                                ]
                            ),
                        ),
                        placeholder_thunk=lambda: tsr.text('Rendering array...'),
                    )
                ),
                # abbrev(">"),
            ),
            expand_state=tsr.ExpandState.COLLAPSED,
        )

        return rendering

    def irrep_structure(arr: e3nn.IrrepsArray) -> str:
        arr_structure = tree_traverse(StructureVisitor(), arr.array[..., 0])
        return f'{arr_structure}[{arr.irreps}] :: '

    def irrep_array_visualizer(value: Any, path):
        if isinstance(value, (np.ndarray, jax.Array)):
            return ts.VisualizationFromTreescopePart(render_tensor(value.squeeze()))
        elif isinstance(value, e3nn.IrrepsArray):
            abs_max = jnp.max(jnp.abs(value.array)).item()
            vmin = -abs_max
            vmax = abs_max

            visualizations = []
            for ir_mul, chunk in zip(value.irreps, value.chunks):
                if chunk is None:
                    continue
                color = cs[ir_mul.ir.l]
                ndim = chunk.ndim
                kwargs = {
                    'abbrev_color': color,
                    'vmin': vmin,
                    'vmax': vmax,
                }

                if ir_mul.ir.l == 0:
                    kwargs['arr'] = chunk.squeeze(-1)
                else:
                    kwargs['arr'] = chunk
                    kwargs['rows'] = [ndim - 1]
                    kwargs['sliders'] = list(range(0, ndim - 2))
                    kwargs['axis_labels'] = {(ndim - 1): str(ir_mul.ir)}

                visualizations.append(render_tensor(**kwargs).renderable)
            return ts.VisualizationFromTreescopePart(
                tsr.build_custom_foldable_tree_node(
                    label=tsr.abbreviation_color(tsr.text(irrep_structure(value))),
                    contents=tsr.indented_children(visualizations, comma_separated=True),
                    path=path,
                    expand_state=tsr.ExpandState.EXPANDED,
                )
            )

    if return_html:
        with ts.active_autovisualizer.set_scoped(irrep_array_visualizer):
            return ts.render_to_html(obj.stack, compressed=False)
    else:
        with ts.active_autovisualizer.set_scoped(irrep_array_visualizer):
            return ts.display(obj.stack)


if __name__ == '__main__':
    from pyrallis.argparsing import wrap

    with open('reports/model_flow.html', 'w') as f:
        f.write(wrap()(visualize_model_flow)())
        print('Done!')
