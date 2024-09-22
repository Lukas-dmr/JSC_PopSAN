import os, inspect

from hrl_gym.environments.HRLGymEnv import HRLGymEnv
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import json
import importlib
import ray
import pybullet as p
import numpy as np
from hrl_gym.helper_scripts.docs import load_config, load_hyperparam
from hrl_gym.helper_scripts.training_manager import TrainingManager
from ray.tune.registry import register_env

from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()
# Print only tensorflow errors and no warnings


def train(args):
    """
    Train an RLib agent using one of the following algorithms dqn, ppo or sac
    Args:
        args: containing agent_name, algo (name of algorithm), (optional) load_model (path to model, which should be loaded)
    """

    ray.shutdown()
    ray.init(num_gpus=0, num_cpus=4)

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    
    config = load_config(currentdir)
    hyperparam = load_hyperparam(currentdir, args.algo) 
    config["project_path"] = currentdir
    config["active_algo"] = args.algo
    
    register_env('HRLEnv-v0', lambda config: HRLGymEnv(config))


    hyperparam_eps = {}
    hyperparam_eps = hyperparam["config"+str(1)]  
    TrainingManager(args, config, hyperparam_eps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    train_pars = subparser.add_parser('train', help=str)
    train_pars.add_argument('--algo', type=str, default='ppo', required=False)
    train_pars.set_defaults(func=train)
    args = parser.parse_args()
    args.func(args)
