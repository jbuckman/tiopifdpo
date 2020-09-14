COLLECT, EVALUATE_SCORE, EVALUATE_ERROR, DONE = range(4)
import datetime, os
from nvi import environments, algorithms, drop
import torch

def formatdict(d): return '-'.join([f"{k}-{v}" for k,v in d.items()])

def make_path(logdir, name, trial, env, env_params, algo, algo_params):
    if name is None:
        name = f"env-{env}" + \
               (f"-{formatdict(env_params)}" if env_params else "") + \
               f"-algo-{algo}" + \
               (f"-{formatdict(algo_params)}" if algo_params else "") + \
               (f"--{trial}" if trial is not None else "")
        path = os.path.join(logdir, name)
    else:
        name = f"{name}--{trial}" if trial is not None else name
        path = os.path.join(logdir, name)
    return path


def evaluate(env, policy, device="cpu"):
    G = torch.tensor(0., device=device)
    D = torch.tensor(1., device=device)
    outcome = env.reset()
    while D.max() > 1e-2:
        outcome = env.step(policy(outcome))
        G = G + D * outcome.reward
        D = D * outcome.discount
    return dict(score=G.mean().item())

def evaluate_error(env, policy, vfn, γ, device="cpu"):
    cumulative_discount = torch.tensor([1.] * env.n, device=device)
    cumulative_discounted_return = torch.tensor([0.] * env.n, device=device)
    cumulative_discounts = []
    discounts_acquired = []
    predicted_qvalues = []
    actual_qvalues = []
    outcome = env.reset()
    while cumulative_discount.max() > 1e-2:
        action = policy(outcome)
        outcome = env.step(action)
        discounted_reward = cumulative_discount * outcome.reward
        predicted_qvalue = vfn(outcome)[range(env.n), action]
        predicted_qvalues.append(predicted_qvalue)
        actual_qvalues.append(torch.tensor([0.] * env.n, device=device))
        discounts_acquired.append(torch.tensor([1.] * env.n, device=device))
        for i in range(len(actual_qvalues)):
            actual_qvalues[i] += discounts_acquired[i] * discounted_reward
            discounts_acquired[i] *= outcome.discount * γ
        cumulative_discounted_return += discounted_reward
        cumulative_discounts.append(cumulative_discount)
        cumulative_discount = cumulative_discount * outcome.discount
    error = sum(discount * torch.abs(guess - actual) for discount, guess, actual in zip(cumulative_discounts, predicted_qvalues, actual_qvalues)) / sum(cumulative_discounts)
    return dict(score=cumulative_discounted_return.mean().item(),
                interaction_td_error=error.mean().item(),
                interaction_mean_guess=torch.mean(sum(discount * guess for discount, guess in zip(cumulative_discounts, predicted_qvalues)) / sum(cumulative_discounts)).item(),
                interaction_mean_target=torch.mean(sum(discount * actual for discount, actual in zip(cumulative_discounts, actual_qvalues)) / sum(cumulative_discounts)).item())

def SimpleRegretExperiment(env="minatar_breakout", algo="NVI", logdir="logs", name=None,
                           trial=None, total_steps=5000000, env_params={}, algo_params={},
                           device="cpu", node_rank=0, node_total=1):
    # set up logger and counter
    if node_rank == 0:
        logger = drop.Logger(make_path(logdir, name, trial, env, env_params, algo, algo_params))
    else: ## make a "fake" logger that does nothing if we arent the master process
        logger = drop.Logger(None)
    counter = drop.Counter()
    counter.add_fields("episodes", "env_steps")

    # set up environment
    engine, game = env.split("_")[0], '_'.join(env.split("_")[1:])
    if engine == "minatar": env_class = environments.MinAtarEnv
    elif engine == "ale": env_class = environments.GymAtariEnv
    elif engine == "cule": env_class = environments.CuleAtariEnv
    else: raise Exception(f'engine must be "minatar" or "ale" or "cule", not "{engine}"')
    train_env, test_env = env_class(game, agent_device=device, _train=True, **env_params), env_class(game, agent_device=device, _train=False, **env_params)

    # set up learner
    learner = getattr(algorithms, algo)(train_env, logger, counter, node_rank=node_rank, node_total=node_total, total_env_steps=total_steps, device=device, **algo_params)

    # log hps
    logger.write_hps(dict(name=name, experiment="SimpleRegretExperiment", total_steps=total_steps,
                          launch_time=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                          env=env, env_hps=train_env.hps,
                          algo=algo, algo_hps=learner.hps))

    # run main RL loop, evaluating simple regret
    response = train_env.reset()
    while counter.env_steps < total_steps:
        mode, argument = learner.next(response)
        ## collect a transition from the environment
        if mode == COLLECT:
            action = argument
            new_data = train_env.step(action)
            counter.increment("env_steps", train_env.n * node_total)
            counter.increment("episodes", new_data.initial.sum().item() * node_total)
            response = new_data
        ## this means we want to evaluate the performance of the policy
        elif mode == EVALUATE_SCORE:
            policy = argument
            report = evaluate(test_env, policy, device=device)
            response = report
        ## this means we want to evaluate the error of the vfn
        elif mode == EVALUATE_ERROR:
            policy, vfn, γ = argument
            report = evaluate_error(test_env, policy, vfn, γ, device=device)
            response = report

def FixedDataExperiment(env="minatar_breakout", algo="NVI", logdir="logs", name=None,
                        trial=None, total_steps=5000000, env_params={}, algo_params={},
                        device="cpu", node_rank=0, node_total=1):
    # set up logger and counter
    if node_rank == 0:
        logger = drop.Logger(make_path(logdir, name, trial, env, env_params, algo, algo_params))
    else: ## make a "fake" logger that does nothing if we arent the master process
        logger = drop.Logger(None)
    counter = drop.Counter()
    counter.add_fields("episodes", "env_steps")

    # set up environment
    engine, game = env.split("_")[0], '_'.join(env.split("_")[1:])
    if engine == "minatar": env_class = environments.MinAtarEnv
    elif engine == "ale": env_class = environments.GymAtariEnv
    elif engine == "cule": env_class = environments.CuleAtariEnv
    else: raise Exception(f'engine must be "minatar" or "ale" or "cule", not "{engine}"')
    train_env, test_env = env_class(game, agent_device=device, _train=True, **env_params), env_class(game, agent_device=device, _train=False, **env_params)

    # set up learner
    learner = getattr(algorithms, algo)(train_env, logger, counter, node_rank=node_rank, node_total=node_total, total_env_steps=total_steps, device=device, **algo_params)

    # log hps
    logger.write_hps(dict(name=name, experiment="SimpleRegretExperiment", total_steps=total_steps,
                          launch_time=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"),
                          env=env, env_hps=train_env.hps,
                          algo=algo, algo_hps=learner.hps))

    response = None
    while True:
        mode, argument = learner.next(response)
        ## this means we want to end
        if mode == DONE:
            break

        ## this means we want to evaluate the performance of the policy
        elif mode == EVALUATE_SCORE:
            policy = argument
            report = evaluate(test_env, policy, device=device)
            response = report

        ## this means we want to evaluate the error of the vfn
        elif mode == EVALUATE_ERROR:
            policy, vfn, γ = argument
            report = evaluate_error(test_env, policy, vfn, γ, device=device)
            response = report
