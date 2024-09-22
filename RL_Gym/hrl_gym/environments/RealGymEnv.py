import numpy as np
import rospy
import time

import gym
from gym.utils import seeding
from gym import spaces

from hrl_gym.models.real_world import Real_World

class RealGymEnv(gym.Env):
    """
    Position control can change the position, but not the orientation. 
    Also it decides when to grasp by itself
    """

    def __init__(self, config):

        self._config = config

        # Show current eef position and target position
        self._debug = config["debug_info"]

        # Env parameters
        self._step_cnt = 0 
        self._episode_cnt = 0
        self._target_pos = None

        # Max number of steps per episode
        self._max_steps = config["env_param"]["max_steps_per_episode"]

        # Min and max joint velocity
        self._min_qv = config["env_param"]["min_joint_velocity"]
        self._max_qv = config["env_param"]["max_joint_velocity"]

        self.world = Real_World()
        self.robot = self.world.robot

        self.action_space = spaces.Box( 
                                        low=np.array(config["env_param"]["min_joint_velocity"]),
                                        high=np.array(config["env_param"]["max_joint_velocity"]),
                                        dtype=np.float32
                                        )


        # Create observation space based on settings inside the config file
        self.observation_space = self._setup_spaces()

    #TODO was passiert wenn roboter pose erreicht hat oder ähnliches der muss ja mal stoppen und still stehen
    def reset(self):
        """
        Reset robots position and generate a new target position

        Returns:
            observation
        """

        #self.robot.set_joint_positions(np.zeros(7))

        self.robot.move_robot_to_start()
        
        self._step_cnt = 0
        self._target_pos = self.world.generate_randome_targetpos(cartesian=self._config["target_in_cartesian"])

        #input("Drücke Enter um eine neue Episode zu starten")
        
        return self.get_observation()
        
    def seed(self, seed=None):
        """
        Generate randome seed
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_observation(self):


        if not self._config["use_snn"]: 

            # Multiple observations are stored in a dictionary
            if type(self.observation_space) == type(spaces.Dict()): observation = dict()

            # If the current cartesian position of the eef is required as an observation
            # add a new entity to the dictionary or just use the eef position alone as an observation
            if self._config["observation"]["cartes_curr"]:
                eef_pos = self.robot.get_eef_pos()
                if type(self.observation_space) == type(spaces.Dict()):
                    observation.update(cartes_curr=np.array([eef_pos.x, eef_pos.y, eef_pos.z], dtype=np.float32))
                else:
                    observation = np.array([eef_pos.x, eef_pos.y, eef_pos.z], dtype=np.float32)

            # Distance between the cartesian eef position and cartesian target position
            if self._config["observation"]["cartes_dist"]:
                cartes_dist = self._get_distance(self.robot.get_eef_pos())
                if type(self.observation_space) == type(spaces.Dict()):
                    observation.update(cartes_dist=np.array(cartes_dist, dtype=np.float32))
                else:
                    observation = np.array(cartes_dist, dtype=np.float32)


            # Current joints velocity
            if self._config["observation"]["q_vel"]:
                joint_vel = self.robot.get_joint_vel()
                joint_vel = [joint_vel[0], joint_vel[1], joint_vel[2], joint_vel[3],joint_vel[4],joint_vel[5]]
                #joint_vel = [joint_vel[0], joint_vel[1], joint_vel[3]]
                if type(self.observation_space) == type(spaces.Dict()):
                    observation.update(q_vel=np.array(joint_vel, dtype=np.float32))
                else:
                    observation = np.array(joint_vel)

            # Current joints angles in rad
            if self._config["observation"]["qpos_curr"]:
                joint_pos = self.robot.get_joint_pos()
                joint_pos = [joint_pos[0], joint_pos[1], joint_pos[2], joint_pos[3],joint_pos[4],joint_pos[5]]
                #joint_pos = [joint_pos[0], joint_pos[1], joint_pos[3]]
                if type(self.observation_space) == type(spaces.Dict()):
                    observation.update(qpos_curr=np.array(joint_pos, dtype=np.float32))
                else:
                    observation = np.array(joint_pos, dtype=np.float32)
        
        else:

            
            cartes_dist = self._get_distance(self.robot.get_eef_pos())
            q_curr = self.robot.get_joint_pos()
            q_vel = self.robot.get_joint_vel()
        
            observation = np.zeros(15, dtype=np.float32)
            observation[:3] = cartes_dist
            observation[3:9] = q_vel[:6]
            observation[9:15] = q_curr[:6]

        return observation

    def step(self, action):
        """
        Execute action that the neural network (agent) chose from the action space

        Args:
            action (from the defined action space)

        Returns:
            observation (from observation space)
            reward (float)
            termination (bool)
            emtpy dict
        """

        if self._step_cnt == 0:
            print("target position: ", self._target_pos)
            print("distance: ", np.linalg.norm(self._get_distance(self.robot.get_eef_pos())))
            #input("last input")

        #for idx, el in enumerate(action):
        #    if abs(el) < 0.05:
        #        action[idx] = 0

        # Execute action inside pybullet
        self.robot.set_joint_positions(action)

        self._step_cnt += 1

        obs = self.get_observation()
        reward = self._reward(action)
        end = self._termination()

        return obs, reward, end, {}

    def _reward(self, action):

        eef_pos = self.robot.get_eef_pos()
        euc_dist =  np.linalg.norm(self._target_pos - np.array([eef_pos.x, eef_pos.y, eef_pos.z]))
        
        
        rd = 0
        if euc_dist <= 0.6 and euc_dist > 0.1: 
            rd = -0.5*euc_dist+0.3
        elif euc_dist <= 0.1:
            rd = np.exp(-520*np.power(euc_dist,2))+0.245

        
        rv = 0
        if euc_dist <= 0.1:
            v_max = 0.5
            vel = np.linalg.norm(self.robot.get_joint_vel())
            reward = -(((vel/v_max))*((1-euc_dist/0.1)))

        rp = 0
        #if self.robot.out_of_bounds:
        #    rp = -1

        reward = 0.7*rd + 0.3*rv + rp

        return np.float32(reward)

    def _termination(self):
        """
        Return True if agent was successful or if current number of timesteps is greater than 
        the allowed number of time steps per episode

        Returns:
            bool
        """

        if self._step_cnt == self._max_steps:
            self._episode_cnt += 1
            self.robot.set_joint_positions(np.zeros(self.robot.num_joints))
            self.debug_function()
            return True
        
        return False
            
    def _get_distance(self, eef_pos):
        """
        Args:
            current eef position or joint angles in rad

        Returns:
            distance between current position/angles and target position/angles
        """
        eef_pos = np.array([eef_pos.x, eef_pos.y, eef_pos.z], dtype=np.float32)

        return self._target_pos-eef_pos
    

    
    def debug_function(self):

        print("----------------------------------------")
        print("target_pos: ", self._target_pos)
        print("eef_pos: ", self.robot.get_eef_pos())
        print("distance: ", np.linalg.norm(self._get_distance(self.robot.get_eef_pos())))
        #print("action: ", action)
        #print("obs: ", obs)
        print("----------------------------------------")

        
        

    def _setup_spaces(self):
        """
        Set up the observationspace according to the config settings

        Returns:
            observation space (gym.spaces)
        """

        if self._config["use_snn"]:
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        # Setup observation space
        elif sum([self._config["observation"]["cartes_curr"], self._config["observation"]["cartes_dist"],
            self._config["observation"]["qpos_curr"], self._config["observation"]["q_vel"]]) > 1:

            spaces_dict = {}
            if self._config["observation"]["cartes_curr"]: spaces_dict.update({'cartes_curr': spaces.Box(low=-500, high=500, shape=(3,), dtype=np.float32)})
            if self._config["observation"]["cartes_dist"]: spaces_dict.update({'cartes_dist': spaces.Box(low=-500, high=500, shape=(3,), dtype=np.float32)})
            if self._config["observation"]["q_vel"]: spaces_dict.update({'q_vel': spaces.Box(low=-500, high=500, shape=(len(self._config["env_param"]["min_joint_velocity"]),), dtype=np.float32)})
            if self._config["observation"]["qpos_curr"]: spaces_dict.update({'qpos_curr': spaces.Box(low=-500, high=500, shape=(len(self._config["env_param"]["min_joint_velocity"]),), dtype=np.float32)})
            observation_space = gym.spaces.Dict(spaces_dict)

        else:
            if self._config["observation"]["cartes_curr"]: observation_space = spaces.Box(low=-500, high=500, shape=(3,), dtype=np.float32)
            elif self._config["observation"]["cartes_dist"]: observation_space = spaces.Box(low=-500, high=500, shape=(3,), dtype=np.float32)
            elif self._config["observation"]["qpos_curr"]: observation_space = spaces.Box(low=-500, high=500, shape=(len(self._config["env_param"]["min_joint_velocity"]),), dtype=np.float32)
            elif self._config["observation"]["q_vel"]: observation_space = spaces.Box(low=-500, high=500, shape=(len(self._config["env_param"]["min_joint_velocity"]),), dtype=np.float32)
            else:
                raise Exception("please set one of three observation spaces to true, inside config.yaml")

        return observation_space
            
    

        


