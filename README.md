Code accompanying "The Importance of Pessimism in Fixed-Dataset Policy Optimization" by Buckman, Gelada, Bellemare.
https://arxiv.org/abs/2009.06799

The tabular experiments can be replicated with simply `experiment_performance_comparison.py` and `experiment_various_explorations.py`.

Running the deep learning experiments requires quite a few more steps.

1) Use `dqn_data_collection.py` to train a standard DQN model to convergence, checkpointing the resulting near-optimal policy.
2) Use `policy_data_collection.py`, pointing at a specific checkpointed policy, and setting a specific epsilon, to collect a dataset with epsilon-greedy, and save it.
3) Use `main.py` to run an FDPO experiment.
4) Use `compile_logs.py` to extract the final performance numbers from the logs of many completed experiments, and compile them into a table suitable for plotting.
