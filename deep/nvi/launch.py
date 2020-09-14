import torch.multiprocessing as mp
from torch.cuda import is_available
import os

def single_launch(experiment_fn, **kwargs):
    experiment_fn(node_rank=0,
                  node_total=1,
                  device="cuda" if is_available() else "cpu",
                  **kwargs)

class multigpu_kwargifier:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, i, total_gpus, kwargs):
        return self.fn(**kwargs, node_rank=i, node_total=total_gpus, device=f"cuda:{i}")

def multi_launch(experiment_fn, node_total, **kwargs):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(multigpu_kwargifier(experiment_fn),
             args=(node_total, kwargs),
             nprocs=node_total,
             join=True)
