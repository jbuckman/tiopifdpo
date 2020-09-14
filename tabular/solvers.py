import numpy as np
from utils import sharpen_to_onehot
from core import TabularMDP, DataBuffer

### helper functions

def policy_iteration(mdp, starting_policy=None, early_stopping=100):
    Q_values = np.random.uniform(0, 1/(1-mdp.γ), [mdp.state_space_size, mdp.action_space_size])
    policy = sharpen_to_onehot(Q_values, n_idx=mdp.action_space_size)
    new_policy = starting_policy
    update_n = 0
    while new_policy is None or not np.allclose(policy, new_policy):
        if new_policy is not None: policy = new_policy
        state_mean_rewards = np.sum(policy * mdp.R_bar, axis=1)
        next_state_dist = np.sum(np.expand_dims(policy, -1) * mdp.P, axis=1)
        state_values = np.linalg.solve(np.eye(mdp.state_space_size) - mdp.γ * next_state_dist, state_mean_rewards)
        Q_values = mdp.R_bar + mdp.γ * np.sum(mdp.P * state_values.reshape([1, 1, mdp.state_space_size]), axis=2)
        new_policy = sharpen_to_onehot(Q_values, n_idx=mdp.action_space_size)
        update_n += 1
        if early_stopping is not None and update_n >= early_stopping: break
    return Q_values

def value_iteration(mdp, n=2000, get_intermediate_policies=False):
    if get_intermediate_policies: intermediate_policies = []
    Q_values = np.random.uniform(0, 1/(1-mdp.γ), [mdp.state_space_size, mdp.action_space_size])
    for i in range(n):
        if get_intermediate_policies: intermediate_policies.append(sharpen_to_onehot(Q_values, n_idx=mdp.action_space_size))
        new_Q_values = mdp.R_bar + mdp.γ * np.sum(mdp.P * Q_values.max(axis=-1).reshape([1, 1, mdp.state_space_size]), axis=2)
        Q_values = new_Q_values
    if get_intermediate_policies: return Q_values, intermediate_policies
    return Q_values

def eps_greedify(policy, ε):
    return (1. - ε) * policy + ε * np.ones_like(policy) / policy.shape[-1]

### Algorithms

class BaseAlgorithm(object):
    CHEATING = False
    def __init__(self, mdp_info, *args, **kwargs):
        state_space_size, action_space_size, γ, s0 = mdp_info
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.γ = γ
        self.s0 = s0
        if self.CHEATING: self.mdp = kwargs['mdp']

    def ingest_data(self, data): pass
    def get_policy(self): pass

class UniformRandom(BaseAlgorithm):
    def get_policy(self, get_intermediate_policies=False):
        policy = np.ones([self.state_space_size, self.action_space_size]) / self.action_space_size
        if get_intermediate_policies: return policy, []
        return policy

class SingleRandomEpsGreedy(BaseAlgorithm):
    def __init__(self, mdp_info, ε=.05, *args, **kwargs):
        super().__init__(mdp_info, *args, **kwargs)
        self.ε = ε
        base = sharpen_to_onehot(np.random.random([self.state_space_size, self.action_space_size]), n_idx=self.action_space_size)
        self.policy = base * (1. - self.ε) + (self.ε / self.action_space_size)
    def get_policy(self, get_intermediate_policies=False):
        if get_intermediate_policies: return self.policy, []
        return self.policy

class PI(BaseAlgorithm):
    CHEATING = True
    def get_policy(self):
        Q_values = self.get_Q_values()
        policy = sharpen_to_onehot(Q_values, n_idx=self.action_space_size)
        return policy
    def get_Q_values(self):
        return policy_iteration(self.mdp)


class VI(BaseAlgorithm):
    CHEATING = True
    def get_policy(self, get_intermediate_policies=False):
        if get_intermediate_policies:
            Q_values, intermediate_policies = value_iteration(self.mdp, get_intermediate_policies=get_intermediate_policies)
            policy = sharpen_to_onehot(Q_values, n_idx=self.mdp.action_space_size)
            return policy, intermediate_policies
        else:
            Q_values = value_iteration(self.mdp, get_intermediate_policies=get_intermediate_policies)
            policy = sharpen_to_onehot(Q_values, n_idx=self.mdp.action_space_size)
            return policy

    def get_Q_values(self):
        return value_iteration(self.mdp)

class QLearning(BaseAlgorithm):
    def __init__(self, mdp_info, lr=.01, *args, **kwargs):
        super().__init__(mdp_info)
        self.lr = lr
        self.Q = np.zeros([self.state_space_size, self.action_space_size])
        self.return_range = [0, 1/(1.-self.γ)]
    def ingest_data(self, data):
        for s, a, r, next_s in data:
            self.Q[s, a] = (1. - self.lr) * self.Q[s, a] + self.lr * (r + self.γ * np.max(self.Q[s]))
            if self.return_range is not None: self.Q = np.clip(self.Q, *self.return_range)
            self.lr *= .999995
    def get_policy(self, get_intermediate_policies=False):
        if get_intermediate_policies: return sharpen_to_onehot(self.Q, n_idx=self.action_space_size), []
        return sharpen_to_onehot(self.Q, n_idx=self.action_space_size)

class Imitation(BaseAlgorithm):
    def __init__(self, mdp_info, early_stopping=None, *args, **kwargs):
        super().__init__(mdp_info)
        self.mdp_info = mdp_info
        self.early_stopping = early_stopping
        self.rb = DataBuffer(self.state_space_size, self.action_space_size, self.mdp_info[-1])
        self.policy = None
    def ingest_data(self, data):
        for datum in data: self.rb.add_data(*datum)
    def get_policy(self):
        return self.rb.to_empirical_policy()

class ReplayPI(BaseAlgorithm):
    def __init__(self, mdp_info, early_stopping=None, *args, **kwargs):
        super().__init__(mdp_info)
        self.mdp_info = mdp_info
        self.early_stopping = early_stopping
        self.rb = DataBuffer(self.state_space_size, self.action_space_size, self.mdp_info[-1])
        self.policy = None
    def ingest_data(self, data):
        for datum in data: self.rb.add_data(*datum)
        R_hat, P_hat = self.rb.to_empirical_R_P()
        self.empirical_mdp = TabularMDP(self.state_space_size, self.action_space_size, R_hat, P_hat, self.γ, self.s0)
    def get_policy(self):
        Q_values = self.get_Q_values()
        policy = sharpen_to_onehot(Q_values, n_idx=self.action_space_size)
        return policy
    def get_Q_values(self):
        return policy_iteration(self.empirical_mdp)

class ReplayPPI(BaseAlgorithm):
    def __init__(self, mdp_info, safety=10., *args, **kwargs):
        super().__init__(mdp_info)
        self.mdp_info = mdp_info
        self.safety = safety
        self.rb = DataBuffer(self.state_space_size, self.action_space_size, self.mdp_info[-1])
    def ingest_data(self, data):
        for datum in data: self.rb.add_data(*datum)
        R_hat, P_hat = self.rb.to_empirical_R_P()
        R_hat -= self.safety / np.sqrt(self.rb.data_counts)
        R_hat = np.clip(R_hat, 0., 1.)
        self.empirical_mdp = TabularMDP(self.state_space_size, self.action_space_size, R_hat, P_hat, self.γ, self.s0)
    def get_policy(self):
        Q_values = self.get_Q_values()
        policy = sharpen_to_onehot(Q_values, n_idx=self.action_space_size)
        return policy
    def get_Q_values(self):
        return policy_iteration(self.empirical_mdp)

class ReplaySPPI(BaseAlgorithm):
    def __init__(self, mdp_info, safety=10., *args, **kwargs):
        super().__init__(mdp_info)
        self.mdp_info = mdp_info
        self.safety = safety
        self.rb = DataBuffer(self.state_space_size, self.action_space_size, self.mdp_info[-1])
    def ingest_data(self, data):
        for datum in data: self.rb.add_data(*datum)
        R_hat, P_hat = self.rb.to_empirical_R_P()
        self.empirical_mdp = TabularMDP(self.state_space_size, self.action_space_size, R_hat, P_hat, self.γ, self.s0)
    def get_policy(self):
        Q_values = self.get_Q_values()
        policy = self.compute_penalized_policy(Q_values)
        return policy
    def get_Q_values(self):
        Q_values = np.zeros([self.state_space_size, self.action_space_size])
        policy = None
        for i in range(1000):
            new_policy = self.compute_penalized_policy(Q_values)
            if policy is not None and np.allclose(policy, new_policy): break
            policy = new_policy
            state_mean_rewards = np.sum(policy * self.empirical_mdp.R_bar, axis=1)
            next_state_dist = np.sum(np.expand_dims(policy, -1) * self.empirical_mdp.P, axis=1)
            safety_penalties = np.sqrt((self.safety * policy**2. * self.rb.data_counts**-1).sum(axis=-1))
            state_values = np.linalg.solve(np.eye(self.state_space_size) - self.γ * next_state_dist, (state_mean_rewards - safety_penalties).clip(0.,1.))
            Q_values = self.empirical_mdp.R_bar + self.γ * np.sum(self.empirical_mdp.P * state_values.reshape([1, 1, self.state_space_size]), axis=2)
        return Q_values
    def compute_penalized_policy(self, Q_values):
        a = Q_values
        b = self.safety / self.rb.data_counts**.5
        a1, a2, a3, a4, a5 = [a[:,i]  for i in range(5)]
        b1, b2, b3, b4, b5 = [b[:,i]  for i in range(5)]
        new_policy = np.zeros_like(Q_values)
        new_policy[:, 0] = ((-(b1*b2*b3*b4*b5)/(a1**2*b2*b3*b4 + a2**2*b1*b3*b4 + a3**2*b1*b2*b4 + a4**2*b1*b2*b3 + a1**2*b2*b3*b5 + a2**2*b1*b3*b5 + a3**2*b1*b2*b5 + a5**2*b1*b2*b3 + a1**2*b2*b4*b5 + a2**2*b1*b4*b5 + a4**2*b1*b2*b5 + a5**2*b1*b2*b4 + a1**2*b3*b4*b5 + a3**2*b1*b4*b5 + a4**2*b1*b3*b5 + a5**2*b1*b3*b4 + a2**2*b3*b4*b5 + a3**2*b2*b4*b5 + a4**2*b2*b3*b5 + a5**2*b2*b3*b4 - b1*b2*b3*b4 - b1*b2*b3*b5 - b1*b2*b4*b5 - b1*b3*b4*b5 - b2*b3*b4*b5 - 2*a1*a2*b3*b4*b5 - 2*a1*a3*b2*b4*b5 - 2*a1*a4*b2*b3*b5 - 2*a1*a5*b2*b3*b4 - 2*a2*a3*b1*b4*b5 - 2*a2*a4*b1*b3*b5 - 2*a2*a5*b1*b3*b4 - 2*a3*a4*b1*b2*b5 - 2*a3*a5*b1*b2*b4 - 2*a4*a5*b1*b2*b3))**(1/2)*(a1*b2*b3*b4 + a1*b2*b3*b5 + a1*b2*b4*b5 + a1*b3*b4*b5 - a2*b3*b4*b5 - a3*b2*b4*b5 - a4*b2*b3*b5 - a5*b2*b3*b4))/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5) + (b2*b3*b4*b5)/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5)
        new_policy[:, 1] = ((-(b1*b2*b3*b4*b5)/(a1**2*b2*b3*b4 + a2**2*b1*b3*b4 + a3**2*b1*b2*b4 + a4**2*b1*b2*b3 + a1**2*b2*b3*b5 + a2**2*b1*b3*b5 + a3**2*b1*b2*b5 + a5**2*b1*b2*b3 + a1**2*b2*b4*b5 + a2**2*b1*b4*b5 + a4**2*b1*b2*b5 + a5**2*b1*b2*b4 + a1**2*b3*b4*b5 + a3**2*b1*b4*b5 + a4**2*b1*b3*b5 + a5**2*b1*b3*b4 + a2**2*b3*b4*b5 + a3**2*b2*b4*b5 + a4**2*b2*b3*b5 + a5**2*b2*b3*b4 - b1*b2*b3*b4 - b1*b2*b3*b5 - b1*b2*b4*b5 - b1*b3*b4*b5 - b2*b3*b4*b5 - 2*a1*a2*b3*b4*b5 - 2*a1*a3*b2*b4*b5 - 2*a1*a4*b2*b3*b5 - 2*a1*a5*b2*b3*b4 - 2*a2*a3*b1*b4*b5 - 2*a2*a4*b1*b3*b5 - 2*a2*a5*b1*b3*b4 - 2*a3*a4*b1*b2*b5 - 2*a3*a5*b1*b2*b4 - 2*a4*a5*b1*b2*b3))**(1/2)*(a2*b1*b3*b4 + a2*b1*b3*b5 + a2*b1*b4*b5 - a1*b3*b4*b5 - a3*b1*b4*b5 - a4*b1*b3*b5 - a5*b1*b3*b4 + a2*b3*b4*b5))/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5) + (b1*b3*b4*b5)/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5)
        new_policy[:, 2] = ((-(b1*b2*b3*b4*b5)/(a1**2*b2*b3*b4 + a2**2*b1*b3*b4 + a3**2*b1*b2*b4 + a4**2*b1*b2*b3 + a1**2*b2*b3*b5 + a2**2*b1*b3*b5 + a3**2*b1*b2*b5 + a5**2*b1*b2*b3 + a1**2*b2*b4*b5 + a2**2*b1*b4*b5 + a4**2*b1*b2*b5 + a5**2*b1*b2*b4 + a1**2*b3*b4*b5 + a3**2*b1*b4*b5 + a4**2*b1*b3*b5 + a5**2*b1*b3*b4 + a2**2*b3*b4*b5 + a3**2*b2*b4*b5 + a4**2*b2*b3*b5 + a5**2*b2*b3*b4 - b1*b2*b3*b4 - b1*b2*b3*b5 - b1*b2*b4*b5 - b1*b3*b4*b5 - b2*b3*b4*b5 - 2*a1*a2*b3*b4*b5 - 2*a1*a3*b2*b4*b5 - 2*a1*a4*b2*b3*b5 - 2*a1*a5*b2*b3*b4 - 2*a2*a3*b1*b4*b5 - 2*a2*a4*b1*b3*b5 - 2*a2*a5*b1*b3*b4 - 2*a3*a4*b1*b2*b5 - 2*a3*a5*b1*b2*b4 - 2*a4*a5*b1*b2*b3))**(1/2)*(a3*b1*b2*b4 + a3*b1*b2*b5 - a1*b2*b4*b5 - a2*b1*b4*b5 - a4*b1*b2*b5 - a5*b1*b2*b4 + a3*b1*b4*b5 + a3*b2*b4*b5))/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5) + (b1*b2*b4*b5)/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5)
        new_policy[:, 3] = ((-(b1*b2*b3*b4*b5)/(a1**2*b2*b3*b4 + a2**2*b1*b3*b4 + a3**2*b1*b2*b4 + a4**2*b1*b2*b3 + a1**2*b2*b3*b5 + a2**2*b1*b3*b5 + a3**2*b1*b2*b5 + a5**2*b1*b2*b3 + a1**2*b2*b4*b5 + a2**2*b1*b4*b5 + a4**2*b1*b2*b5 + a5**2*b1*b2*b4 + a1**2*b3*b4*b5 + a3**2*b1*b4*b5 + a4**2*b1*b3*b5 + a5**2*b1*b3*b4 + a2**2*b3*b4*b5 + a3**2*b2*b4*b5 + a4**2*b2*b3*b5 + a5**2*b2*b3*b4 - b1*b2*b3*b4 - b1*b2*b3*b5 - b1*b2*b4*b5 - b1*b3*b4*b5 - b2*b3*b4*b5 - 2*a1*a2*b3*b4*b5 - 2*a1*a3*b2*b4*b5 - 2*a1*a4*b2*b3*b5 - 2*a1*a5*b2*b3*b4 - 2*a2*a3*b1*b4*b5 - 2*a2*a4*b1*b3*b5 - 2*a2*a5*b1*b3*b4 - 2*a3*a4*b1*b2*b5 - 2*a3*a5*b1*b2*b4 - 2*a4*a5*b1*b2*b3))**(1/2)*(a4*b1*b2*b3 - a1*b2*b3*b5 - a2*b1*b3*b5 - a3*b1*b2*b5 - a5*b1*b2*b3 + a4*b1*b2*b5 + a4*b1*b3*b5 + a4*b2*b3*b5))/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5) + (b1*b2*b3*b5)/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5)
        new_policy[:, 4] = (b1*b2*b3*b4)/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5) - ((-(b1*b2*b3*b4*b5)/(a1**2*b2*b3*b4 + a2**2*b1*b3*b4 + a3**2*b1*b2*b4 + a4**2*b1*b2*b3 + a1**2*b2*b3*b5 + a2**2*b1*b3*b5 + a3**2*b1*b2*b5 + a5**2*b1*b2*b3 + a1**2*b2*b4*b5 + a2**2*b1*b4*b5 + a4**2*b1*b2*b5 + a5**2*b1*b2*b4 + a1**2*b3*b4*b5 + a3**2*b1*b4*b5 + a4**2*b1*b3*b5 + a5**2*b1*b3*b4 + a2**2*b3*b4*b5 + a3**2*b2*b4*b5 + a4**2*b2*b3*b5 + a5**2*b2*b3*b4 - b1*b2*b3*b4 - b1*b2*b3*b5 - b1*b2*b4*b5 - b1*b3*b4*b5 - b2*b3*b4*b5 - 2*a1*a2*b3*b4*b5 - 2*a1*a3*b2*b4*b5 - 2*a1*a4*b2*b3*b5 - 2*a1*a5*b2*b3*b4 - 2*a2*a3*b1*b4*b5 - 2*a2*a4*b1*b3*b5 - 2*a2*a5*b1*b3*b4 - 2*a3*a4*b1*b2*b5 - 2*a3*a5*b1*b2*b4 - 2*a4*a5*b1*b2*b3))**(1/2)*(a1*b2*b3*b4 + a2*b1*b3*b4 + a3*b1*b2*b4 + a4*b1*b2*b3 - a5*b1*b2*b3 - a5*b1*b2*b4 - a5*b1*b3*b4 - a5*b2*b3*b4))/(b1*b2*b3*b4 + b1*b2*b3*b5 + b1*b2*b4*b5 + b1*b3*b4*b5 + b2*b3*b4*b5)
        new_policy += (1. - new_policy.sum(-1, keepdims=True))/5
        print(a[0])
        print(b[0])
        print(new_policy[0], sum(new_policy[0]), new_policy.min(), new_policy.max())
        import code; code.interact(local=locals())
        return new_policy

class ReplayProximalPPI(BaseAlgorithm):
    def __init__(self, mdp_info, safety=10., *args, **kwargs):
        super().__init__(mdp_info)
        self.mdp_info = mdp_info
        self.safety = safety
        self.rb = DataBuffer(self.state_space_size, self.action_space_size, self.mdp_info[-1])
    def ingest_data(self, data):
        for datum in data: self.rb.add_data(*datum)
        R_hat, P_hat = self.rb.to_empirical_R_P()
        self.empirical_mdp = TabularMDP(self.state_space_size, self.action_space_size, R_hat, P_hat, self.γ, self.s0)
        self.empirical_policy = self.rb.to_empirical_policy()
    def get_policy(self):
        Q_values = self.get_Q_values()
        policy = self.compute_penalized_policy(Q_values)
        return policy
    def get_Q_values(self):
        Q_values = np.zeros([self.state_space_size, self.action_space_size])
        policy = None
        for i in range(1000):
            new_policy = self.compute_penalized_policy(Q_values)
            if policy is not None and np.allclose(policy, new_policy): break
            policy = new_policy
            state_mean_rewards = np.sum(policy * self.empirical_mdp.R_bar, axis=1)
            next_state_dist = np.sum(np.expand_dims(policy, -1) * self.empirical_mdp.P, axis=1)
            safety_penalties = (self.safety * np.maximum(policy - self.empirical_policy, 0).sum(-1))
            state_values = np.linalg.solve(np.eye(self.state_space_size) - self.γ * next_state_dist, state_mean_rewards - safety_penalties)
            Q_values = self.empirical_mdp.R_bar + self.γ * np.sum(self.empirical_mdp.P * state_values.reshape([1, 1, self.state_space_size]), axis=2)
        return Q_values
    def compute_penalized_policy(self, Q_values):
        fill_value = Q_values.max(-1, keepdims=True) - self.safety
        actions_from_emp = Q_values > fill_value
        new_policy = np.cast[np.float](actions_from_emp) * self.empirical_policy
        new_policy += (1. - new_policy.sum(-1, keepdims=True)) * sharpen_to_onehot(Q_values, n_idx=self.action_space_size)
        return new_policy