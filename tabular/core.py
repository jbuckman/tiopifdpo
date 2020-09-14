import numpy as np
import torch
from utils import sample_along_axis, gaussian_cdf

class TabularMDP(object):
    def __init__(self, state_space_size, action_space_size, R, P, γ, s0):
        assert R.shape == (state_space_size, action_space_size)
        assert P.shape == (state_space_size, action_space_size, state_space_size)
        assert np.allclose(np.sum(P, axis=-1), 1.)
        assert 0 <= γ and γ < 1
        assert np.allclose(np.sum(s0), 1.)

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.R = R ## array of shape [|S|, |A|], of bernoulli parameters
        self.P = P ## array of shape [|S|, |A|, |S|]
        self.γ = γ
        self.s0 = s0 ## array of shape [|S|] that sums to 1
        self.R_bar = self.R

    @property
    def info(self):
        return self.state_space_size, self.action_space_size, self.γ, self.s0

    def stationary_dist(self, policy):
        assert policy.shape == (self.state_space_size, self.action_space_size)
        policy_transition_mat = np.sum(self.P * np.expand_dims(policy, -1), axis=1)
        a = np.eye(policy_transition_mat.shape[0]) - policy_transition_mat
        a = np.vstack((a.T, np.ones(policy_transition_mat.shape[0])))
        b = np.matrix([0] * policy_transition_mat.shape[0] + [1]).T
        dist = np.linalg.lstsq(a, b, rcond=-1)[0]
        dist = np.array(dist).reshape([self.state_space_size]) / np.sum(dist)
        dist = np.expand_dims(dist, 1) * policy
        dist[dist < 1e-10] = 0.
        dist /= np.sum(dist)
        return dist

    def evaluate_policy(self, policy):
        state_mean_rewards = np.sum(policy * self.R_bar, axis=1)
        next_state_dist = np.sum(np.expand_dims(policy, -1) * self.P, axis=1)
        state_values = np.linalg.solve(np.eye(self.state_space_size) - self.γ * next_state_dist, state_mean_rewards)
        value = np.sum(self.s0 * state_values)
        return value

    def sample_data(self, policy, count=1, state_dist=None):
        if state_dist is None: state_dist = np.sum(self.stationary_dist(policy), -1)
        s = np.random.choice(self.state_space_size, count, p=state_dist)
        a = sample_along_axis(policy[s], -1)
        r = np.cast[np.int](np.random.uniform(0,1,s.size) < self.R[s, a])
        next_s = sample_along_axis(self.P[s,a], -1)
        return list(zip(s.tolist(), a.tolist(), r.tolist(), next_s.tolist()))

    def sample_data_everywhere(self, count=1):
        s = np.array([s for s in range(self.state_space_size) for a in range(self.action_space_size)] * count)
        a = np.array([a for s in range(self.state_space_size) for a in range(self.action_space_size)] * count)
        r = np.cast[np.int](np.random.uniform(0, 1, s.size) < self.R[s, a])
        next_s = sample_along_axis(self.P[s,a], -1)
        return list(zip(s.tolist(), a.tolist(), r.tolist(), next_s.tolist()))

class DataBuffer(object):
    def __init__(self, state_space_size, action_space_size, reward_range, max_size=None):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.reward_range = reward_range
        self.rhat_mat = np.zeros([self.state_space_size, self.action_space_size])
        self.phat_mat = np.zeros([self.state_space_size, self.action_space_size, self.state_space_size])
        self.data_counts = np.zeros([self.state_space_size, self.action_space_size])
        ## if the max size is defined, we need to keep track of the order by which these points were acquired
        self.max_size = max_size
        if self.max_size is not None:
            self.all_s = []
            self.all_a = []
            self.all_r = []
            self.all_next_s = []

    def add_data(self, s, a, r, next_s):
        self.rhat_mat[s, a] += r
        self.phat_mat[s, a, next_s] += 1.
        self.data_counts[s,a] += 1.
        if self.max_size is not None:
            self.all_s.append(s)
            self.all_a.append(a)
            self.all_r.append(r)
            self.all_next_s.append(next_s)
            if len(self.all_s) > self.max_size:
                sn, an, rn, next_sn = [x.pop(0) for x in (self.all_s, self.all_a, self.all_r, self.all_next_s)]
                self.rhat_mat[sn, an] -= rn
                self.phat_mat[sn, an, next_sn] -= 1.
                self.data_counts[sn, an] -= 1.

    def to_empirical_R_P(self, r_default=None, p_default=None):
        ## given an MDP where we assume we don't know R, P, and some data, create an empirical MDP
        emdp_r = self.rhat_mat / self.data_counts
        # emdp_r = np.stack([emdp_r, np.zeros_like(emdp_r)], -1)
        emdp_p = self.phat_mat / self.data_counts[:,:,None]
        ## fill in the 0s
        if r_default is None:
            r_default = np.random.uniform(0., 1., emdp_r.shape)
        if p_default is None:
            p_default = np.ones(emdp_p.shape) / self.state_space_size
        mask = self.data_counts == 0
        emdp_r[mask] = r_default[mask]
        emdp_p[mask] = p_default[mask]
        return emdp_r, emdp_p

    def to_empirical_policy(self):
        base = self.data_counts / self.data_counts.sum(axis=-1, keepdims=True)
        mask = self.data_counts.sum(axis=-1) == 0
        base[mask] = 1./self.action_space_size
        return base

    @classmethod
    def from_data(self, state_space_size, action_space_size, reward_range, data, max_size=None):
        buff = DataBuffer(state_space_size, action_space_size, reward_range, max_size)
        for datum in data: buff.add_data(*datum)
        return buff

class ImageTabularMDP:
    def __init__(self, mdp : TabularMDP, data):
        self.underlying_mdp = mdp
        self.data_map = data
        assert data.shape[0] == self.underlying_mdp.state_space_size

    def map_to_img(self, state_i):
        if isinstance(state_i, int):
            state_i = torch.tensor([state_i])
        if isinstance(state_i, np.ndarray):
            state_i = torch.tensor(state_i)
        out = self.data_map[state_i, torch.randint_like(state_i, 0, self.data_map.shape[1])].type(torch.float32) / 255.
        return out

    def evaluate_policy(self, policy, restarts=1000, duration=100):
        G = []
        s = np.random.choice(self.underlying_mdp.state_space_size, p=self.underlying_mdp.s0, size=restarts)
        for i in range(duration):
            a = policy(self.map_to_img(s))[0].detach().cpu().numpy()
            r = self.underlying_mdp.R_bar[s,a]
            G.append(self.underlying_mdp.γ**i * r)
            s = (np.random.rand(restarts, 1) < self.underlying_mdp.P[s,a].cumsum(-1)).argmax(-1)
        return np.mean(sum(G))

    def sample_transition_batch(self, batch_size):
        s = np.random.choice(self.underlying_mdp.state_space_size, size=batch_size)
        a = np.random.choice(self.underlying_mdp.action_space_size, size=batch_size)
        r = self.underlying_mdp.R_bar[s,a]
        s_prime = (np.random.rand(batch_size, 1) < self.underlying_mdp.P[s,a].cumsum(-1)).argmax(-1)
        return (self.map_to_img(s), a, r, self.map_to_img(s_prime))
