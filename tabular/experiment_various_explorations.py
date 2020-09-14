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
TOTAL_DATA = 2000
TRIALS = 1000
MAX_EVALS = 25

eval_idxs = exponential_interpolate(n=MAX_EVALS, max=TOTAL_DATA)
results = [["algo", "exploration", "trial_i", "return"]]
for trial_i in range(TRIALS):
    print("Trial %d/%d" % (trial_i, TRIALS), flush=True)
    mdp = mdps.BasicGridworld(DIMS)
    exploration_amounts = np.linspace(0., 1., MAX_EVALS)
    # initial_data_size = mdp.state_space_size * mdp.action_space_size
    initial_data_size = 0
    for exploration in exploration_amounts:
        learners = {algo: fn(mdp) for algo, fn in ALGORITHMS.items()}
        collection_policy = solvers.eps_greedify(solvers.PI(mdp.info, mdp=mdp).get_policy(), exploration)
        # initial_data = mdp.sample_data_everywhere()
        other_data = mdp.sample_data(policy=collection_policy, count=TOTAL_DATA - initial_data_size, state_dist=mdp.s0)
        for algo in learners:
            # learners[algo].ingest_data(initial_data)
            learners[algo].ingest_data(other_data)
            policy = learners[algo].get_policy()
            policy_return = mdp.evaluate_policy(policy)
            results.append([algo, exploration, trial_i, policy_return])

with open("results/various_explorations.csv", "w") as f:
    s = "\n".join([",".join([str(x) for x in line]) for line in results])
    f.write(s)