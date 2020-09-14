import numpy as np
import solvers, mdps
from utils import exponential_interpolate
import time

ALGORITHMS = {
    "Random Policy": lambda mdp: solvers.UniformRandom(mdp.info),
    "Optimal Policy": lambda mdp: solvers.PI(mdp.info, mdp=mdp), # to get the optimal policy, we cheat
    "Imitation": lambda mdp: solvers.Imitation(mdp.info),
    "Naïve FDPO": lambda mdp: solvers.ReplayPI(mdp.info),
    "UA Pessimistic FDPO": lambda mdp: solvers.ReplayPPI(mdp.info, safety=1.),
    "Proximal Pessimistic FDPO": lambda mdp: solvers.ReplayProximalPPI(mdp.info, safety=1.),
}

DIMS = [8,8]
EXPLORATION = .5
TOTAL_DATA = 2e5
TRIALS = 1000
MAX_EVALS = 25

eval_idxs = exponential_interpolate(n=MAX_EVALS, max=TOTAL_DATA)
results = [["algo", "data_amount", "trial_i", "return"]]
for trial_i in range(TRIALS):
    print("Trial %d/%d" % (trial_i, TRIALS), flush=True)
    mdp = mdps.BasicGridworld(DIMS)
    initial_data_size = 0
    remaining_eval_idxs = eval_idxs
    # initial_data_size = mdp.state_space_size * mdp.action_space_size
    # remaining_eval_idxs = [0,initial_data_size]+eval_idxs[(np.array(eval_idxs) >= initial_data_size).argmax():-1]
    # collection_policy = solvers.SingleRandomEpsGreedy(mdp.info, ε=EXPLORATION).get_policy()
    collection_policy = solvers.eps_greedify(solvers.PI(mdp.info, mdp=mdp).get_policy(), EXPLORATION)
    learners = {algo: fn(mdp) for algo, fn in ALGORITHMS.items()}
    for prev_data_amount, new_data_amount in zip(remaining_eval_idxs, remaining_eval_idxs[1:]):
        # if prev_data_amount == 0:
        #     fresh_data = mdp.sample_data_everywhere()
        # else:
        #     fresh_data = mdp.sample_data(policy=collection_policy, count=new_data_amount - prev_data_amount)
        fresh_data = mdp.sample_data(policy=collection_policy, count=new_data_amount - prev_data_amount)
        for algo in learners:
            learners[algo].ingest_data(fresh_data)
            policy = learners[algo].get_policy()
            policy_return = mdp.evaluate_policy(policy)
            results.append([algo, new_data_amount, trial_i, policy_return])

with open("results/performance_comparisons.csv", "w") as f:
    s = "\n".join([",".join([str(x) for x in line]) for line in results])
    f.write(s)