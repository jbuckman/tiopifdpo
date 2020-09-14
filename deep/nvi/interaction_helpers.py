from collections import defaultdict
import torch

class sequential:
    def __init__(self, **transformations):
        self.transformations = transformations
    def __call__(self, x, **transform_kwargs):
        transform_kwargs = defaultdict(dict, **transform_kwargs)
        for k in transform_kwargs: assert k in self.transformations.keys()
        for k, t in self.transformations.items(): x = t(x, **transform_kwargs[k])
        return x
    def __getitem__(self, item):
        return self.transformations[item]

def uniform_random(action_n, device="cpu"):
    def policy(outcome):
        return torch.randint(0, action_n, [outcome.obs.shape[0]], device=device)
    return policy

def greedy(device="cpu"):
    def greedy(x):
        out = torch.zeros_like(x, device=device)
        out[range(x.shape[0]), x.argmax(axis=-1)] = 1.
        return out
    return greedy

def ε_greedy(ε=None, device="cpu"):
    default_ε = ε
    greedy_policy = greedy(device)
    def policy(x, ε=None):
        ε = default_ε if ε is None else ε
        assert ε is not None, "specify ε"
        greedy_probs = greedy_policy(x)
        random_probs = torch.ones_like(x) / x.shape[1]
        return (1. - ε) * greedy_probs + ε * random_probs
    return policy

def sample(x):
    assert torch.all(x >= 0.) and torch.allclose(x.sum(-1), torch.ones_like(x.sum(-1)))
    return torch.multinomial(x, 1)[:,0]