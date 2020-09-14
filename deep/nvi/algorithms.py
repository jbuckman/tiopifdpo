from collections import defaultdict
from copy import deepcopy
import dill as pickle
from functools import partial
import nvi.nns as nns
from nvi.interaction_helpers import uniform_random, greedy, ε_greedy, sequential, sample
from nvi.memory import ShortTermMemory, LongTermMemory
from nvi.experiment import COLLECT, EVALUATE_SCORE, EVALUATE_ERROR
import torch, time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class Algorithm:
    def __init__(self, env, logger, counter, node_rank, node_total, device, **kwargs):
        self.env = env
        ## properties of the environment
        self.OBS_SPACE = (self.env.obs_shape, self.env.obs_dtype)
        self.FLAG_SPACE = (tuple(), torch.bool)
        self.ACTION_SPACE = (tuple(), torch.int64)
        self.REWARD_SPACE = (tuple(), torch.float32)
        self.DISCOUNT_SPACE = (tuple(), torch.float32)
        self.ACTION_DIST_SPACE = ((self.env.action_n,), torch.float32)
        self.VALUE_SPACE = (tuple(), torch.float32)
        self.logger = logger
        self.counter = counter
        self.node_rank = node_rank
        self.node_total = node_total
        self.device = device
        self.hps = {}
        self.collect_hps = True
        self.set_defaults()
        for k,v in kwargs.items(): self.__setattr__(k, v)
        self.collect_hps = False
        self.agenda = self.main()
        next(self.agenda)

    def __setattr__(self, key, value):
        if hasattr(self, "collect_hps") and self.collect_hps: self.hps[key] = value
        super().__setattr__(key, value)

    def next(self, new_data):
        return self.agenda.send(new_data)

    def set_defaults(self): pass

    def main(self): raise Exception("unimplemented")

class ActRandomly(Algorithm):
    def main(self):
        last_log = 0
        policy = uniform_random(self.env.action_n, device=self.device)
        env_outcome = yield
        _t = time.time()
        while True:
            if self.counter.env_steps - last_log > 1000:
                print("collect", time.time() - _t)
                _t = time.time()
                yield EVALUATE_SCORE, policy
                print("eval", time.time() - _t)
                _t = time.time()
                last_log = self.counter.env_steps
            env_outcome = yield COLLECT, policy(env_outcome)

class NVI(Algorithm):
    def set_defaults(self):
        self.multigpu_info = (0,1)
        ## algorithm hyperparameters
        self.eval_every_n = 5
        self.nn_layers = [32, 64, 64, 128, 128]
        self.lr = 3e-4
        self.ε = .01
        self.γ = .99
        self.α = .0
        self.huber_thresh = 5.0
        self.explore_α = False
        self.clip_rewards = False
        self.warmup_steps = None
        self.warmup_updates = None
        self.network_reset = True
        self.collection_steps = 20000
        self.min_gradient_steps = 2500
        self.min_gradient_epochs = 5
        self.early_stopping_thresh = False
        self.early_stopping_decay = .9
        self.memory_size = 100000
        self.buffer_rows = None # defaults to .1*memory_size
        self.batch_size = 256 * self.node_total
        self.obs_per_state = 1
        ## custom defaults for minatar
        if self.env.engine == "minatar":
            self.batch_size = 1024 * self.node_total
            self.warmup_updates = 100
            self.warmup_steps = 5000
            self.collection_steps = 1000
            self.min_gradient_steps = 10000
            self.min_gradient_epochs = 100
        ## custom defaults for atari
        if self.env.engine == "atari":
            self.obs_per_state = 4
            self.clip_rewards = (-1., 1.)
            self.warmup_updates = 100
            self.warmup_steps = 20000
            self.collection_steps = 4000
            self.min_gradient_epochs = 25
            self.memory_size = 2000000 * self.node_total

    def main(self):
        ## initialize multigpu process group
        if self.node_total > 1: dist.init_process_group("nccl", rank=self.node_rank, world_size=self.node_total)
        ## add fields to our counter
        self.counter.add_fields("bellman_updates", "gradient_steps")
        ## adjust the state shape to reflect our frame-stacking
        state_shape = (self.env.obs_shape[0]*self.obs_per_state, self.env.obs_shape[1], self.env.obs_shape[2])
        ## compute the per-node batch size
        self.batch_size_per_node = self.batch_size // self.node_total

        ## define our Q-function and its optimizer
        Qfn = nns.QConvNet(state_shape, self.env.action_n, layers=self.nn_layers).to(self.device)
        if self.node_total > 1:
            local_Qfn = Qfn
            Qfn = DDP(Qfn, device_ids=[self.node_rank])
        parameters = list(Qfn.parameters())
        if self.explore_α != False: ## make a separate exploration Qfn
            explore_Qfn = nns.QConvNet(state_shape, self.env.action_n, layers=self.nn_layers).to(self.device)
            if self.node_total > 1:
                local_explore_Qfn = explore_Qfn
                explore_Qfn = DDP(explore_Qfn, device_ids=[self.node_rank])
            parameters += list(explore_Qfn.parameters())
        else:
            explore_Qfn = Qfn
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        ## define our exploration and exploitation policies
        exploration_policy = sequential(stack=self.make_framestacker(),
                                        statify=self.obs_stack_to_state,
                                        vfn=explore_Qfn,
                                        policy=ε_greedy(ε=self.ε, device=self.device),
                                        sample=sample)

        ## define our long-term memory
        dataset = LongTermMemory({"obs": self.OBS_SPACE, "initial": self.FLAG_SPACE, "action": self.ACTION_SPACE, "reward": self.REWARD_SPACE, "discount": self.DISCOUNT_SPACE, "current_exploit_value_estimate": self.VALUE_SPACE, "current_explore_value_estimate": self.VALUE_SPACE, "current_policy_action_dist": self.ACTION_DIST_SPACE}, memory_batch=self.env.n, memory_size=self.memory_size // (self.env.n * self.node_total), precontext_rows=self.obs_per_state-1, buffer_rows=self.buffer_rows, device=self.device)

        ## begin by querying the environment for the start-state
        outcome = yield
        self.incorporate_outcome(dataset, outcome)

        ## perform reinforcement learning
        for update_i in self.counter.loop("bellman_updates"):
            ## evaluate current value function
            if self.node_rank == 0 and self.eval_every_n and self.counter.bellman_updates % self.eval_every_n == 0:
                report = yield EVALUATE_ERROR, (self.make_exploitation_policy(Qfn, update_i), self.make_running_Qfn(Qfn), self.γ)
                self.precompute_values(dataset, Qfn, zero_out=update_i == 0)
                dataset_report = self.compute_dataset_td_error(dataset, Qfn, self.α)
                self.logger.write("bellman_update", _flush=True, **self.counter.dict(), **report, **dataset_report)
                print(f"EVAL    | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in dict(**report, **dataset_report).items()]), flush=True)

            # collect data from the environment
            if self.counter.bellman_updates == 0 or self.warmup_updates is None or self.counter.bellman_updates > self.warmup_updates:
                collection_goal = self.collection_steps * (update_i) + (self.collection_steps if self.warmup_steps is None else self.warmup_steps)
                while self.counter.env_steps < collection_goal:
                    action = exploration_policy(outcome, policy=dict(ε=self.ε if update_i > 0 else 1.))
                    dataset.set(-1, action=action)
                    outcome = yield COLLECT, action
                    if self.node_rank == 0: self.periodically_run("dbg_print_coll", 5, print, f"COLLECT | {self.counter.view()} | goal={collection_goal}", flush=True)
                    self.incorporate_outcome(dataset, outcome)

            ## initialize iterator for current data
            idx_iterator = dataset.idx_iterator(self.batch_size_per_node, infinite=True, discard_partial=True)

            ## cache the values predicted by the current value function
            self.precompute_values(dataset, Qfn, zero_out=update_i==0)
            if self.explore_α != False: self.precompute_values(dataset, explore_Qfn, explore=True, zero_out=update_i==0)

            ## reset neural network
            optimizer.__setstate__(defaultdict(dict))
            if self.network_reset:
                if self.node_total == 1:
                    Qfn.reset_parameters()
                    if self.explore_α != False: explore_Qfn.reset_parameters()
                else:
                    if self.node_rank == 0: local_Qfn.reset_parameters()
                    for param in Qfn.parameters(): dist.broadcast(param, 0)
                    if self.explore_α != False:
                        if self.node_rank == 0: local_explore_Qfn.reset_parameters()
                        for param in explore_Qfn.parameters(): dist.broadcast(param, 0)

            ## compute application of bellman optimality operator
            for gradient_i, idx in enumerate(self.counter.loop("gradient_steps", idx_iterator)):
                data = dataset.get(idx,
                                   state=self.state_fetcher, action='action',
                                   reward='reward', discount='discount',
                                   next_state=partial(self.state_fetcher, offset=1),
                                   next_exploit_state_value=LongTermMemory.basic_fetcher('current_exploit_value_estimate', offset=1),
                                   next_explore_state_value=LongTermMemory.basic_fetcher('current_explore_value_estimate', offset=1),)
                gradient_err, gradient_log = self.perform_td_update(data, Qfn=Qfn, α=self.α, detach=False)
                if self.explore_α != False:
                    explore_gradient_err, explore_gradient_log = self.perform_td_update(data, Qfn=explore_Qfn, α=self.explore_α, detach=False)
                    gradient_err += explore_gradient_err
                    gradient_log.update(**{f"explore_{k}": v for k, v in explore_gradient_log.items()})
                optimizer.zero_grad()
                gradient_err.backward()
                optimizer.step()
                self.logger.write("gradient_step", _subsample_rate=100, **self.counter.dict(), **gradient_log)
                if self.node_rank == 0: self.periodically_run("dbg_print_train", 5, print, f"TRAIN   | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in gradient_log.items()]), flush=True)
                if gradient_i > max(self.min_gradient_steps, self.min_gradient_epochs * dataset.count // self.batch_size_per_node): break

    def make_exploitation_policy(self, Qfn, policy_update_i):
        if policy_update_i == 0: return uniform_random(self.env.action_n, device=self.device)
        return sequential(stack=self.make_framestacker(),
                          statify=self.obs_stack_to_state,
                          vfn=Qfn,
                          policy=greedy(device=self.device),
                          sample=sample)
    def make_running_Qfn(self, Qfn):
        return sequential(stack=self.make_framestacker(),
                          statify=self.obs_stack_to_state,
                          vfn=Qfn)

    def perform_td_update(self, data, Qfn, α=None, detach=True):
        Q_values = Qfn(data.state, detach=detach)
        guess = Q_values[range(data.state.shape[0]), data.action]
        target = data.reward + self.γ * data.discount * data.next_exploit_state_value
        td_errors = self.huber(guess, target, thresh=self.huber_thresh)
        td_error = td_errors.mean()
        loss = td_error
        log_info = dict(loss=loss.item(), td_error=td_error.item(), mean_guess=guess.mean().item(), mean_target=target.mean().item())
        return loss, log_info

    def compute_dataset_td_error(self, dataset, Qfn, α=None):
        errs, guesses, targets = [], [], []
        explore_errs, explore_guesses, explore_targets = [], [], []
        for idx in dataset.idx_iterator(self.batch_size_per_node, required_postcontext=1):
            data = dataset.get(idx,
               state=self.state_fetcher, action='action',
               reward='reward', discount='discount',
               next_state=partial(self.state_fetcher, offset=1),
               next_exploit_state_value=LongTermMemory.basic_fetcher('current_exploit_value_estimate', offset=1),
               next_explore_state_value=LongTermMemory.basic_fetcher('current_explore_value_estimate', offset=1), )
            _, info = self.perform_td_update(data, Qfn=Qfn, α=α)
            errs.append(info["td_error"] * idx.shape[0])
            guesses.append(info["mean_guess"] * idx.shape[0])
            targets.append(info["mean_target"] * idx.shape[0])
            if "explore_td_error" in info:
                explore_errs.append(info["explore_td_error"] * idx.shape[0])
                explore_guesses.append(info["explore_mean_guess"] * idx.shape[0])
                explore_targets.append(info["explore_mean_target"] * idx.shape[0])
        report = dict(dataset_td_error=sum(errs) / dataset.count, dataset_mean_guess=sum(guesses) / dataset.count, dataset_mean_target=sum(targets) / dataset.count)
        if self.explore_α != False: report.update(explore_td_error=sum(explore_errs) / dataset.count, dataset_mean_explore_guess=sum(explore_guesses) / dataset.count, dataset_mean_explore_target=sum(explore_targets) / dataset.count)
        return report

    def precompute_values(self, dataset, Qfn, zero_out=False, explore=False):
        if zero_out: ## we use zero-targets for the very first bellman
            dataset.current_exploit_value_estimate = torch.zeros_like(dataset.current_exploit_value_estimate)
            dataset.current_explore_value_estimate = torch.zeros_like(dataset.current_explore_value_estimate)
        else:
            for idx in dataset.idx_iterator(self.batch_size_per_node, required_postcontext=0):
                data = dataset.get(idx, state=self.state_fetcher)
                values = Qfn(data.state)
                if self.clip_rewards and self.γ < 1.: values = values.clamp(self.clip_rewards[0]/(1.-self.γ), self.clip_rewards[1]/(1.-self.γ))
                if not explore:
                    dataset.set(idx, current_exploit_value_estimate=values.max(-1)[0])
                else:
                    dataset.set(idx, current_explore_value_estimate=values.max(-1)[0])

    def state_fetcher(self, data, idx, offset=0):
        if self.env.engine == "minatar": return self._minatar_state_fetcher(data, idx, offset)
        else:                            return self._ale_state_fetcher(data, idx, offset)

    def obs_stack_to_state(self, data):
        if self.env.engine == "minatar": return self._minatar_obs_stack_to_state(data.obs)
        else:                            return self._ale_obs_stack_to_state(data.obs, data.initial)

    def _minatar_state_fetcher(self, data, idx, offset=0):
        return self._minatar_obs_stack_to_state(data.obs[idx[:, 0] + offset, idx[:, 1]])

    def _minatar_obs_stack_to_state(self, obs):
        return obs.type(torch.float32)

    def _ale_state_fetcher(self, data, idx, offset=0):
        range_idx = idx[:,0,None] + torch.arange(-self.obs_per_state+1, 1, device=self.device).reshape((1,)*len(idx[:,0].shape) + (self.obs_per_state,)) + offset
        obs = data.obs[range_idx,idx[:,1,None],0]
        initial = data.initial[range_idx,idx[:,1,None]]
        return self._ale_obs_stack_to_state(obs, initial)

    def _ale_obs_stack_to_state(self, obs_stack, initial):
        ## obs are [batch, time] + obs_shape, and initial is [batch, time]. batch is optional, though.
        shifted_initial = torch.roll(initial.type(torch.float32), -1, -1)
        shifted_initial[...,-1] = 0.
        mask = 1. - torch.flip(torch.cumsum(torch.flip(shifted_initial, [-1]), -1), [-1]).clamp(0.,1.)
        float_mask = mask.reshape(mask.shape + (1,1))
        state = float_mask * obs_stack.type(torch.float32) / 255.
        return state

    def incorporate_outcome(self, dataset, outcome):
        if dataset.count > 0:
            dataset.set(-1, reward=outcome.reward.clamp(*self.clip_rewards) if self.clip_rewards else outcome.reward,
                            discount=outcome.discount)
        dataset.add_data(obs=outcome.obs, initial=outcome.initial)

    def make_framestacker(self):
        return ShortTermMemory(self.obs_per_state, {"obs": self.OBS_SPACE, "initial": self.FLAG_SPACE}, device=self.device)

    def consider_early_stopping(self, early_stopping_thresh, early_stopping_decay):
        def stopper():
            val = yield False
            running_average = val
            i = 0
            while True:
                if early_stopping_thresh is None: yield False
                running_average = early_stopping_decay * running_average + (1 - early_stopping_decay) * val
                val = yield running_average < early_stopping_thresh and i > 1. / (1. - early_stopping_thresh)
                i += 1
        s = stopper()
        s.send(None)
        return s

    def periodically_run(self, unique_id, time_delay, fn, *args, **kwargs):
        if not hasattr(self, "last_run_times"): self.last_run_times = {}
        if unique_id not in self.last_run_times: self.last_run_times[unique_id] = time.time()
        if time.time() - self.last_run_times[unique_id] > time_delay:
            fn(*args, **kwargs)
            self.last_run_times[unique_id] = time.time()

    def huber(self, predicted, true, thresh):
        errors = torch.abs(predicted - true)
        mask = errors < thresh
        return ((0.5 * mask * (errors ** 2)) + ~mask * errors).mean()

class FixedDataNVI(NVI):
    def set_defaults(self):
        super().set_defaults()
        self.dataset_path = ""
        self.min_gradient_epochs = 50
        self.total_bellman_updates = 250

    def main(self, Qfn=None, dataset=None):
        print(f"Node Device: {self.device}")

        ## initialize multigpu process group
        if self.node_total > 1: dist.init_process_group("nccl", rank=self.node_rank, world_size=self.node_total)
        ## add fields to our counter
        self.counter.add_fields("bellman_updates", "gradient_steps")
        ## compute the per-node batch size
        self.batch_size_per_node = self.batch_size // self.node_total

        ## adjust the state shape to reflect our frame-stacking
        state_shape = (self.env.obs_shape[0]*self.obs_per_state, self.env.obs_shape[1], self.env.obs_shape[2])
        ## define our Q-function and its optimizer
        if Qfn is None:
            Qfn = nns.QConvNet(state_shape, self.env.action_n, layers=self.nn_layers).to(self.device)
        if self.node_total > 1:
            local_Qfn = Qfn
            Qfn = DDP(Qfn, device_ids=[self.node_rank])
        optimizer = torch.optim.Adam(list(Qfn.parameters()), lr=self.lr)

        ## define our long-term memory
        if dataset is None:
            print(f"Loading dataset from {self.dataset_path}")
            with open(self.dataset_path, "rb") as f:
                dataset = pickle.load(f)
            if dataset.device != self.device:
                dataset.send_to(self.device)

        ## begin the evaluation loop
        yield
        ## initialize iterator for data
        idx_iterator = dataset.idx_iterator(self.batch_size_per_node, infinite=True, discard_partial=True)

        ## perform reinforcement learning
        for update_i in self.counter.loop("bellman_updates", self.total_bellman_updates):
            ## evaluate current value function
            if self.node_rank == 0 and self.eval_every_n and self.counter.bellman_updates % self.eval_every_n == 0:
                report = yield EVALUATE_ERROR, (self.make_exploitation_policy(Qfn, update_i), self.make_running_Qfn(Qfn), self.γ)
                self.precompute_values(dataset, Qfn, zero_out=update_i == 0)
                dataset_report = self.compute_dataset_td_error(dataset, Qfn, self.α)
                self.logger.write("bellman_update", _flush=True, **self.counter.dict(), **report, **dataset_report)
                print(f"EVAL    | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in dict(**report, **dataset_report).items()]), flush=True)

            ## cache the values predicted by the current value function
            self.precompute_values(dataset, Qfn, zero_out=update_i==0)

            ## reset neural network
            optimizer.__setstate__(defaultdict(dict))
            if self.network_reset:
                if self.node_total == 1:
                    Qfn.reset_parameters()
                else:
                    if self.node_rank == 0: local_Qfn.reset_parameters()
                    for param in Qfn.parameters(): dist.broadcast(param, 0)

            ## compute application of bellman optimality operator
            for gradient_i, idx in enumerate(self.counter.loop("gradient_steps", idx_iterator)):
                data = dataset.get(idx,
                                   state=self.state_fetcher, action='action',
                                   reward='reward', discount='discount',
                                   next_state=partial(self.state_fetcher, offset=1),
                                   next_exploit_state_value=LongTermMemory.basic_fetcher('current_exploit_value_estimate', offset=1),
                                   next_explore_state_value=LongTermMemory.basic_fetcher('current_explore_value_estimate', offset=1),)
                gradient_err, gradient_log = self.perform_td_update(data, Qfn=Qfn, α=self.α, detach=False)
                optimizer.zero_grad()
                gradient_err.backward()
                optimizer.step()
                self.logger.write("gradient_step", _subsample_rate=100, **self.counter.dict(), **gradient_log)
                if self.node_rank == 0: self.periodically_run("dbg_print_train", 5, print, f"TRAIN   | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in gradient_log.items()]), flush=True)
                if gradient_i > max(self.min_gradient_steps, self.min_gradient_epochs * dataset.count // self.batch_size_per_node): break

        if self.node_rank == 0:
            report = yield EVALUATE_ERROR, (self.make_exploitation_policy(Qfn, self.total_bellman_updates), self.make_running_Qfn(Qfn), self.γ)
            self.precompute_values(dataset, Qfn, zero_out=False)
            dataset_report = self.compute_dataset_td_error(dataset, Qfn, self.α)
            self.logger.write("bellman_update", _flush=True, **self.counter.dict(), **report, **dataset_report)
            print(f"EVAL    | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in dict(**report, **dataset_report).items()]), flush=True)

class PessimisticFixedDataNVI(FixedDataNVI):
    def set_defaults(self):
        super().set_defaults()
        self.α = .25
        self.empirical_policy_steps = 100000

    def main(self):
        ## initialize multigpu process group
        if self.node_total > 1: dist.init_process_group("nccl", rank=self.node_rank, world_size=self.node_total)
        ## add fields to our counter
        self.counter.add_fields("bellman_updates", "imitation_gradient_steps", "gradient_steps")
        ## adjust the state shape to reflect our frame-stacking
        state_shape = (self.env.obs_shape[0]*self.obs_per_state, self.env.obs_shape[1], self.env.obs_shape[2])
        ## compute the per-node batch size
        self.batch_size_per_node = self.batch_size // self.node_total

        ## define our Q-function
        Qfn = nns.QConvNet(state_shape, self.env.action_n, layers=self.nn_layers).to(self.device)
        ## define our empirical policy and its optimizer
        empirical_policy = nns.QConvNet(state_shape, self.env.action_n, layers=self.nn_layers).to(self.device)
        if self.node_total > 1:
            local_empirical_policy = empirical_policy
            empirical_policy = DDP(empirical_policy, device_ids=[self.node_rank])
        parameters = list(empirical_policy.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        print(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        if dataset.device != self.device:
            dataset.send_to(self.device)
        ## initialize iterator for data
        idx_iterator = dataset.idx_iterator(self.batch_size_per_node, infinite=True, discard_partial=True)

        ## perform reinforcement learning
        for gradient_i, idx in enumerate(self.counter.loop("imitation_gradient_steps", idx_iterator)):
            data = dataset.get(idx, state=self.state_fetcher, action='action')
            imitation_err, imitation_log = self.perform_imitation_update(data, empirical_policy, detach=False)
            optimizer.zero_grad()
            imitation_err.backward()
            optimizer.step()
            self.logger.write("imitation_gradient_step", _subsample_rate=100, **self.counter.dict(), **imitation_log)
            if self.node_rank == 0: self.periodically_run("dbg_print_imitation_train", 5, print, f"TRAIN   | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in imitation_log.items()]), flush=True)
            if gradient_i > self.empirical_policy_steps: break

        self.empirical_policy = empirical_policy
        yield from super().main(Qfn=Qfn, dataset=dataset)

    def perform_imitation_update(self, data, empirical_policy, detach=True):
        policy_logits = empirical_policy(data.state, detach=detach)
        imitation_losses = torch.nn.functional.cross_entropy(policy_logits, data.action, reduction='none')
        loss = imitation_losses.mean()
        log_info = dict(loss=loss.item(),)
        return loss, log_info

    def precompute_values(self, dataset, Qfn, zero_out=False, explore=False):
        if zero_out: ## we use zero-targets for the very first bellman
            dataset.current_exploit_value_estimate = torch.zeros_like(dataset.current_exploit_value_estimate)
            dataset.current_explore_value_estimate = torch.zeros_like(dataset.current_explore_value_estimate)
        else:
            for idx in dataset.idx_iterator(self.batch_size_per_node, required_postcontext=0):
                data = dataset.get(idx, state=self.state_fetcher)
                Q_values = Qfn(data.state)
                empirical_actions = torch.softmax(self.empirical_policy(data.state), -1)
                selected_policy = self.proximally_maximal_policy(Q_values, empirical_actions)
                if self.clip_rewards and self.γ < 1.: Q_values = Q_values.clamp(self.clip_rewards[0]/(1.-self.γ), self.clip_rewards[1]/(1.-self.γ))
                values = (Q_values * selected_policy).sum(-1) - self.α * torch.nn.functional.relu(selected_policy - empirical_actions).sum(-1)
                dataset.set(idx, current_exploit_value_estimate=values)

    def make_exploitation_policy(self, Qfn, policy_update_i):
        if policy_update_i == 0: return uniform_random(self.env.action_n, device=self.device)
        def penalized_greedy(state):
            Q_values = Qfn(state)
            empirical_actions = torch.softmax(self.empirical_policy(state), -1)
            return self.proximally_maximal_policy(Q_values, empirical_actions)
        return sequential(stack=self.make_framestacker(),
                          statify=self.obs_stack_to_state,
                          policy=penalized_greedy,
                          sample=sample)

    def proximally_maximal_policy(self, Q_values, empirical_actions):
        thresh = Q_values.max(-1, keepdims=True)[0] - self.α
        out = torch.zeros_like(empirical_actions, device=empirical_actions.device)
        out[Q_values > thresh] = empirical_actions[Q_values > thresh]
        out[range(Q_values.shape[0]), Q_values.argmax(axis=-1)] += 1. - out.sum(-1)
        out = out.clamp(0., 1.)
        return out

class FixedDataImitation(FixedDataNVI):
    def set_defaults(self):
        super().set_defaults()
        self.empirical_policy_steps = 100000

    def main(self):
        ## initialize multigpu process group
        if self.node_total > 1: dist.init_process_group("nccl", rank=self.node_rank, world_size=self.node_total)
        ## add fields to our counter
        self.counter.add_fields("bellman_updates", "imitation_gradient_steps", "gradient_steps")
        ## adjust the state shape to reflect our frame-stacking
        state_shape = (self.env.obs_shape[0]*self.obs_per_state, self.env.obs_shape[1], self.env.obs_shape[2])
        ## compute the per-node batch size
        self.batch_size_per_node = self.batch_size // self.node_total

        ## define our Q-function and its optimizer
        empirical_policy = nns.QConvNet(state_shape, self.env.action_n, layers=self.nn_layers).to(self.device)
        if self.node_total > 1:
            local_empirical_policy = empirical_policy
            empirical_policy = DDP(empirical_policy, device_ids=[self.node_rank])
        parameters = list(empirical_policy.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        print(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        if dataset.device != self.device:
            dataset.send_to(self.device)
        ## initialize iterator for data
        idx_iterator = dataset.idx_iterator(self.batch_size_per_node, infinite=True, discard_partial=True)

        ## perform reinforcement learning
        for gradient_i, idx in enumerate(self.counter.loop("imitation_gradient_steps", idx_iterator)):
            data = dataset.get(idx, state=self.state_fetcher, action='action')
            imitation_err, imitation_log = self.perform_imitation_update(data, empirical_policy, detach=False)
            optimizer.zero_grad()
            imitation_err.backward()
            optimizer.step()
            self.logger.write("imitation_gradient_step", _subsample_rate=100, **self.counter.dict(), **imitation_log)
            if self.node_rank == 0: self.periodically_run("dbg_print_imitation_train", 5, print, f"TRAIN   | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in imitation_log.items()]), flush=True)
            if gradient_i > self.empirical_policy_steps: break

        yield
        report = yield EVALUATE_SCORE, self.make_exploitation_policy(empirical_policy)
        self.logger.write("imitation_policy", _flush=True, **self.counter.dict(), **report)
        print(f"EVAL    | {self.counter.view()} | " + ' '.join([f"{k}={v:.3}" for k, v in dict(**report).items()]), flush=True)

    def perform_imitation_update(self, data, empirical_policy, detach=True):
        policy_logits = empirical_policy(data.state, detach=detach)
        imitation_losses = torch.nn.functional.cross_entropy(policy_logits, data.action, reduction='none')
        loss = imitation_losses.mean()
        log_info = dict(loss=loss.item(),)
        return loss, log_info

    def make_exploitation_policy(self, policy):
        return sequential(stack=self.make_framestacker(),
                          statify=self.obs_stack_to_state,
                          policy=lambda x: torch.softmax(policy(x), -1),
                          sample=sample)
