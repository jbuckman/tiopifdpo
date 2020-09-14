import numpy as np
from core import TabularMDP

def gridworld_P(dims, stochasticity=0.):
    assert len(dims) == 2
    h,w = dims
    state_space_size = dims[0] * dims[1]
    action_space_size = 4
    UP, DOWN, LEFT, RIGHT = range(action_space_size)
    P = np.zeros([state_space_size, action_space_size, state_space_size])
    for state_i in range(state_space_size):
        x,y = state_i % w, state_i // w
        # locate adjactent idxs
        up_i = ((y-1)%h)*w + (x)%w
        down_i = ((y+1)%h)*w + (x)%w
        left_i = ((y)%h)*w + (x-1)%w
        right_i = ((y)%h)*w + (x+1)%w
        # intentional action probabilities
        P[state_i, UP, up_i] = 1. - stochasticity
        P[state_i, DOWN, down_i] = 1. - stochasticity
        P[state_i, LEFT, left_i] = 1. - stochasticity
        P[state_i, RIGHT, right_i] = 1. - stochasticity
        # sometimes the agent moves in a random direction instead
        for i in [up_i, down_i, left_i, right_i]: P[state_i, :, i] += stochasticity * (1 / action_space_size)
    return P

def BasicGridworld(dims, stochasticity=.2, γ=.99):
    state_space_size = dims[0] * dims[1]
    action_space_size = 4
    R = np.random.beta(1,3,[state_space_size, action_space_size])
    P = gridworld_P(dims, stochasticity)
    s0 = np.ones(state_space_size) / state_space_size
    return TabularMDP(state_space_size=state_space_size,
                      action_space_size=action_space_size,
                      R=R, P=P, γ=γ, s0=s0)