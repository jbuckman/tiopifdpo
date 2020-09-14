import argparse, inspect
from nvi import environments, algorithms
from nvi.experiment import SimpleRegretExperiment, FixedDataExperiment
from nvi.launch import single_launch, multi_launch

def all_object_names(module):
    return {key for key, value in inspect.getmembers(module, inspect.isclass) if value.__module__ == module.__name__}
def load_to_dict(s): return eval(f"dict({s})")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--trial", default=None, type=int)
    parser.add_argument("--env", choices=[f"minatar_{game}" for game in environments.MinAtarEnv.games] + [f"ale_{game}" for game in environments.GymAtariEnv.games], default="minatar_breakout")
    # parser.add_argument("--env", choices=[f"minatar_{game}" for game in environments.MinAtarEnv.games] + [f"ale_{game}" for game in environments.GymAtariEnv.games] + [f"cule_{game}" for game in environments.CuleAtariEnv.games], default="minatar_breakout")
    parser.add_argument("--algo", choices=all_object_names(algorithms), default="FixedDataNVI")
    parser.add_argument("--env_params", type=load_to_dict, default="")
    parser.add_argument("--algo_params", type=load_to_dict, default="")
    parser.add_argument("--total_steps", type=int, default=5000000)
    parser.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args()

    kwargs = dict(env=args.env, algo=args.algo, logdir=args.logdir, name=args.name,
                  trial=args.trial, total_steps=args.total_steps,
                  env_params=args.env_params, algo_params=args.algo_params)

    if args.gpus is None or args.gpus == 1:
        from ipdb import launch_ipdb_on_exception
        with launch_ipdb_on_exception():
            single_launch(FixedDataExperiment, **kwargs)
    else:
        multi_launch(FixedDataExperiment, node_total=args.gpus, **kwargs)
