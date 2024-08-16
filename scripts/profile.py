from cdv.train_e_form import MainConfig, run_using_dashboard
from pyrallis import cfgparsing

from cdv.training_runner import run_using_progress

with open('configs/profiling.toml', 'r') as cfg:    
    conf = cfgparsing.load(MainConfig, cfg)

run_using_progress(conf)