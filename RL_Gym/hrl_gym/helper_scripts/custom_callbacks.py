from typing import Dict
import numpy as np
import os

from ray.rllib.agents.callbacks import DefaultCallbacks
from hrl_gym.helper_scripts.docs import doc_training, doc_config

 

class StatusCallback(DefaultCallbacks):
    """
    Print and save status information about training progress

    Note: Do not call this class directly, instead pass it to config["callbacks"] = StatusCallback
    """

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.init = True
        self.last_eval_mean = None
        self.dir_path = os.getcwd()

    def on_train_result(self, *, trainer, result: dict, **kwargs):

        config = trainer.get_config()

        if self.init: 
            doc_config(config["env_config"]["active_algo"], config["env_config"]["project_path"],config["env_config"])
            self.init = False

        info = []
        if 'evaluation' in result:
               self.last_eval_mean = ("#Mean Eval Reward: " + str(round(result['evaluation']['episode_reward_mean'], 2)))

        if trainer._episodes_total % 100 < 10:
           
            crossbar = "##################################################" 
            progress_in_perc = (result["timesteps_total"]/config["env_config"]["total_timesteps"])*100
            info.append("# Percentage of finished training: " + str(round(progress_in_perc,2)) + "%")
            info.append("# Curr. Steps: " + str(result["timesteps_total"]))
            info.append("# Mean Trainings Reward: " + str(round(result['episode_reward_mean'], 2)))
            
            if self.last_eval_mean is not None:
                info.append("#Mean Eval Reward: " + str(self.last_eval_mean))

            if config["env_config"]["active_algo"] == "dqn": 
                # Is only a linear approximation, because if multiple workers are active, each worker has an individual epsilon value
                epsilon = 1-(result["timesteps_total"]/config["env_config"]["exploration_config"]["epsilon_timesteps"])
                if epsilon < 0: epsilon = 0
                info.append("# Exploration: " + str(epsilon)
)
            for idx, line in enumerate(info):
                while len(line) < 49:
                    line = line + " "
                line = line + "#"
                info[idx] = line

            print(crossbar)
            doc_training(self.dir_path, crossbar)
            for line in info: 
                print(line)
                doc_training(self.dir_path, line)

            print(crossbar)
            doc_training(self.dir_path, crossbar)
            doc_training(self.dir_path, "\n \n \n")
          


