from collections import namedtuple
import torch, cv2
import numpy as np

EnvResponse = namedtuple('EnvResponse', ['reward', 'discount', 'obs', 'initial'])

class MinAtarEnv:
    engine = 'minatar'
    games = {'asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders'}
    def __init__(self, game, agent_device="cpu", _train=True, train_n=1, eval_n=50, sticky_action_prob=0.1, difficulty_ramping=True):
        from minatar import Environment
        self.hps = dict(train_n=train_n, eval_n=eval_n, sticky_action_prob=sticky_action_prob, difficulty_ramping=difficulty_ramping)
        self.game = game
        self.n = train_n if _train else eval_n
        self.envs = [Environment(game, sticky_action_prob, difficulty_ramping) for _ in range(self.n)]
        self.agent_device = agent_device

    @property
    def obs_shape(self): return tuple(self.envs[0].state_shape()[-1:] + self.envs[0].state_shape()[:-1])
    @property
    def obs_dtype(self): return torch.bool
    @property
    def action_n(self): return 6

    def _preprocess_obs(self, obs):
        return obs.permute(0,3,1,2)

    def reset(self):
        for env in self.envs: env.reset()
        obs = self._preprocess_obs(self.obs())
        initial = torch.tensor([True]*self.n, device=self.agent_device)
        return EnvResponse(reward=None, discount=None, obs=obs, initial=initial)

    def step(self, action):
        reward, episode_terminal = self.act(action)
        reward = reward.type(torch.float32)
        discount = (1. - episode_terminal.type(torch.float32))
        for env, termination in zip(self.envs, episode_terminal):
            if termination: env.reset()
        obs = self._preprocess_obs(self.obs())
        initial = episode_terminal
        return EnvResponse(reward=reward, discount=discount, obs=obs, initial=initial)

    def act(self, action):
        action = action.cpu().numpy()
        outs = [e.act(a) for e, a in zip(self.envs, action)]
        return tuple(torch.tensor(np.stack(x), device=self.agent_device) for x in zip(*outs))

    def obs(self):
        return torch.tensor(np.stack([e.state() for e in self.envs], 0), device=self.agent_device)


class GymAtariEnv:
    engine = "atari"
    games = {'adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk', 'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kaboom', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'}
    def __init__(self, game, agent_device="cpu", _train=True, train_n=1, eval_n=5, frameskip=4, initial_random_actions=20, sticky_action_prob=0.25, clip_rewards=False, max_frames_per_episode=108000):
        from gym import make
        self.hps = dict(train_n=train_n, eval_n=eval_n, frameskip=frameskip, initial_random_actions=initial_random_actions,sticky_action_prob=sticky_action_prob,clip_rewards=clip_rewards,max_frames_per_episode=max_frames_per_episode)
        self.game = game
        self.agent_device = agent_device
        self.n = train_n if _train else eval_n
        self.frameskip = frameskip
        self.initial_random_actions = initial_random_actions
        self.sticky_action_prob = sticky_action_prob
        self.clip_rewards = clip_rewards
        self.max_frames_per_episode = max_frames_per_episode
        self.envs = [make(''.join(word.capitalize() for word in self.game.split("_")) + "NoFrameskip-v4") for _ in range(self.n)]
        self.screen_buffer = np.zeros((self.n, self.frameskip, 210, 160), dtype=np.uint8)
        self.last_action = [None] * self.n
        self.ep_frames = [1] * self.n

    @property
    def obs_shape(self): return (1, 84, 84)
    @property
    def obs_dtype(self): return torch.uint8
    @property
    def action_n(self): return self.envs[0].action_space.n

    def _load_obs_into_buffer(self, idx):
        for i, env in enumerate(self.envs):
            env.ale.getScreenGrayscale(self.screen_buffer[i, idx])

    def _reset_one_env(self, i):
        self.envs[i].reset()
        self.screen_buffer[i] = 0.
        self.last_action[i] = None
        self.ep_frames[i] = 1
        for _ in range(self.initial_random_actions):
            self.envs[i].step(np.random.randint(0, self.action_n))

    def obs(self):
        np.max(self.screen_buffer, axis=1, out=self.screen_buffer[:,0])
        # return torch.tensor(ndi.zoom(self.screen_buffer[:,0], (1, 84/210, 84/160))[:,None], device=self.agent_device)
        return torch.tensor([cv2.resize(self.screen_buffer[i,0], (84, 84), interpolation=cv2.INTER_AREA) for i in range(self.n)], device=self.agent_device)[:,None]

    def act(self, action):
        action = action.cpu().numpy()
        stucks = np.random.random((self.frameskip, self.n))
        r = [0.] * self.n
        t = [False] * self.n
        for frame_i in range(self.frameskip):
            for env_i, (e, a) in enumerate(zip(self.envs, action)):
                if self.last_action[env_i] is None or stucks[frame_i,env_i] > self.sticky_action_prob:
                    self.last_action[env_i] = a
                else:
                    a = self.last_action[env_i]
                _, reward, terminal, _ = e.step(a)
                self.ep_frames[env_i] += 1
                if not t[env_i]: r[env_i] += reward
                t[env_i] = t[env_i] or terminal or (self.ep_frames[env_i] > self.max_frames_per_episode)
            if frame_i + 1 < self.frameskip: self._load_obs_into_buffer(frame_i)
        return torch.tensor(r, device=self.agent_device), torch.tensor(t, device=self.agent_device)

    def reset(self):
        for i in range(self.n): self._reset_one_env(i)
        self._load_obs_into_buffer(-1)
        obs = self.obs()
        initial = torch.tensor([True]*self.n, device=self.agent_device)
        return EnvResponse(reward=None, discount=None, obs=obs, initial=initial)

    def step(self, action):
        reward, episode_terminal = self.act(action)
        reward = reward.type(torch.float32)
        discount = (1. - episode_terminal.type(torch.float32))
        for i, termination in enumerate(episode_terminal):
            if termination: self._reset_one_env(i)
        self._load_obs_into_buffer(-1)
        obs = self.obs()
        initial = episode_terminal
        return EnvResponse(reward=reward, discount=discount, obs=obs, initial=initial)


class CuleAtariEnv:
    engine = "atari"
    games = {'adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival', 'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk', 'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kaboom', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan', 'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon'}
    def __init__(self, game, agent_device="cpu", n=20, sticky_action_prob=0.25, clip_rewards=False):
        raise Exception("unimplemented")
        import torchcule.atari
        self.game = game
        self.device = agent_device
        self.n = n
        self.env = torchcule.atari.Env(self.game.capitalize() + "NoFrameskip-v4", self.n,
                                       color_mode='gray', rescale=True,
                                       device=self.device, frameskip=4,
                                       repeat_prob=sticky_action_prob,
                                       clip_rewards=clip_rewards, episodic_life=False,
                                       max_noop_steps=30, max_episode_length=10000)

    @property
    def obs_shape(self): return self.env.observation_space.shape
    @property
    def obs_dtype(self): return torch.uint8
    @property
    def action_n(self): return self.env.action_space.n

    def _preprocess_obs(self, obs):
        return obs.permute(0,3,1,2)

    def reset(self):
        obs = self._preprocess_obs(self.env.reset())
        initial = torch.tensor([True]*self.n, device=self.device)
        return EnvResponse(reward=None, discount=None, obs=obs, initial=initial)

    def step(self, action):
        reward, episode_terminal = self.act(action)
        reward = reward.type(torch.float32)
        discount = (1. - episode_terminal.type(torch.float32))
        for env, termination in zip(self.envs, episode_terminal):
            if termination: env.reset()
        obs = self._preprocess_obs(self.get_state())
        initial = episode_terminal
        return EnvResponse(reward=reward, discount=discount, obs=obs, initial=initial)

    def act(self, action):
        action = action.cpu().numpy()
        outs = [e.act(a) for e, a in zip(self.envs, action)]
        return tuple(torch.tensor(np.stack(x), device=self.device) for x in zip(*outs))
