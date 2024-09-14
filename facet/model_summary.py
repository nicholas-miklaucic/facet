"""Akin to show_model, but captures output instead of printing it."""

import ansi2html
import pyrallis
import rich.terminal_theme
from facet.config import MainConfig
from facet.show_model import show_model
import rich
from contextlib import redirect_stdout
from io import StringIO


def model_summary(config: MainConfig, write_to_file: bool = False, print_summary: bool = False):
    cfg_dict = pyrallis.encode(config)  # type: ignore
    cfg_dict['batch_size'] = 32
    cfg_dict['stack_size'] = 1
    cfg_dict['model']['resid_init'] = 'ones'
    config = pyrallis.decode(MainConfig, cfg_dict)  # type: ignore
    with StringIO() as data, redirect_stdout(data):
        rich.reconfigure(width=200, force_terminal=True, color_system='truecolor')
        cost = show_model(config, make_hlo_dot=False, do_profile=False, show_stat=False)
        output = data.getvalue()

    if print_summary:
        print(output)
    conv = ansi2html.Ansi2HTMLConverter(
        scheme='dracula', title='Model Summary', dark_bg=True, font_size='normal'
    )
    html = conv.convert(output)
    if write_to_file:
        with open('reports/model.html', 'w') as out:
            out.write(html)

    return {'html': html, 'gflops': cost}


@pyrallis.argparsing.wrap()
def model_summary_interactive(config: MainConfig):
    model_summary(config, write_to_file=True, print_summary=True)


if __name__ == '__main__':
    model_summary_interactive()  # type: ignore
