import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig

algo = PPO.from_checkpoint("/home/asiimov/Desktop/master_code/HRL_RobotGym/trained_agents/PPO/dnn_agent/checkpoint_009804/checkpoint-9804")