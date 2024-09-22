from gym.envs.registration import register

register(id='HRLEnv-v0',
         entry_point='hrl_gym.environments:HRLGymEnv',
)
