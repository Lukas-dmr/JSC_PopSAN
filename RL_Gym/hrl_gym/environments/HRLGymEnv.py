import numpy as np
import time
import random
from inputimeout import inputimeout, TimeoutOccurred

import gym
from gym.utils import seeding
from gym import spaces

import pybullet as p
from pybullet_utils import bullet_client

from hrl_gym.models.sim_world import Sim_World

class HRLGymEnv(gym.Env):
    """
    Position control can change the position, but not the orientation. 
    Also it decides when to grasp by itself
    """

    def __init__(self, config):


        self._config = config
        self._timeStep = 1. / 240.
        self._end_ep = False

        # Show current eef position and target position
        self._debug = config["debug_info"]

        # Show visualization in PyBullet GUI
        self._render = config["env_param"]["visualize"]

        # Env parameters
        self._step_cnt = 0 
        self._episode_cnt = 0
        self._time_steps_total = 0
        self._target_pos = None
        self._success = False

        # Used for debugging -> used to show delta between new and last position
        self.old_pos = [0,0,0]
        self._old_vel = np.array([0,0,0,0,0,0])

        # Max number of steps per episode
        self._max_steps = config["env_param"]["max_steps_per_episode"]


        # Min and max joint velocity
        self._min_qv = config["env_param"]["min_joint_velocity"]
        self._max_qv = config["env_param"]["max_joint_velocity"]

        self.physics_client = bullet_client.BulletClient(p.GUI if self._render else p.DIRECT)
        self.world = Sim_World(self._config, self.physics_client)
        self.robot = self.world.robot

        # Create action space based on settings inside the config file
        
        self.action_space = spaces.Box( 
                                        low=np.array(config["env_param"]["min_joint_velocity"]),
                                        high=np.array(config["env_param"]["max_joint_velocity"]),
                                        dtype=np.float32
                                        )
        
        # parameters only used for evaluation
        self.noise_type = "n"
        self.extreme_situation = False
        

        # Create observation space based on settings inside the config file
        self.observation_space = self._setup_spaces()

    def reset(self):
        """
        Reset robots position and generate a new target position

        Returns:
            observation
        """
        self._end_ep = False
        self._step_cnt = 0
        self._success = False

        if self.extreme_situation:
             self._target_pos = self.world.generate_extreme_targetpos()
        else:
            self._target_pos = self.world.generate_randome_targetpos()
        
        self._old_vel = np.array([0,0,0,0,0,0])
        
        self.world.soft_reset()

        self.world.visualize_target(self._target_pos)

        self.robot._curriculum_stage = self.world._curriculum_stage

        
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
                cartes_curr = p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0]
                if type(self.observation_space) == type(spaces.Dict()):
                    observation.update(cartes_curr=np.array(cartes_curr, dtype=np.float32))
                else:
                    observation = np.array(cartes_curr, dtype=np.float32)

            # Distance between the cartesian eef position and cartesian target position
            if self._config["observation"]["cartes_dist"]:
                cartes_dist = self._get_distance(p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0])
                if type(self.observation_space) == type(spaces.Dict()):
                    if self.noise_type != "n":
                        cartes_dist = self.add_noise(cartes_dist)
                    observation.update(cartes_dist=np.array(cartes_dist, dtype=np.float32))
                else:
                    observation = np.array(cartes_dist, dtype=np.float32)

            # Current joints velocity
            if self._config["observation"]["q_vel"]:
            
                q_curr_states = p.getJointStates(self.robot.robot_id, self.robot.joint_ids)
                q_vel = []
                for qvel in q_curr_states:
                    q_vel.append(qvel[1])

                if type(self.observation_space) == type(spaces.Dict()):
                    if self.noise_type != "n":
                        q_vel = self.add_noise(q_vel)
                    observation.update(q_vel=np.array(q_vel, dtype=np.float32))
                else:
                    observation = np.array(q_vel)

            # Current joints angles in rad
            if self._config["observation"]["qpos_curr"]:
                q_curr_states = p.getJointStates(self.robot.robot_id, self.robot.joint_ids)
                q_curr = []
                for state in q_curr_states:
                    q_curr.append(state[0])
                if type(self.observation_space) == type(spaces.Dict()):
                    if self.noise_type != "n":
                        q_curr = self.add_noise(q_curr)
                    observation.update(qpos_curr=np.array(q_curr, dtype=np.float32))
                else:
                    observation = np.array(q_curr)

        else:
            
            
            cartes_curr = np.array(p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0])
            cartes_dist = self._get_distance(p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0])
            q_curr_states = p.getJointStates(self.robot.robot_id, self.robot.joint_ids)
            q_curr = []
            q_vel = []
            for state in q_curr_states: 
                q_curr.append(state[0])
                q_vel.append(state[1])
        
            observation = np.zeros(15, dtype=np.float32)

            if self.noise_type != "n":
                cartes_curr = self.add_noise(cartes_curr)
                cartes_dist = self.add_noise(cartes_dist)
                q_curr = self.add_noise(q_curr)
                q_vel = self.add_noise(q_vel)

            observation[:3] = cartes_dist
            observation[3:9] = q_vel
            observation[9:15] = q_curr

                      
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

        # Execute action inside pybullet
        self.robot.apply_action(action)

        if self.render:
            time.sleep(self._timeStep)

        self._step_cnt += 1
        self._time_steps_total += 1
 
        obs = self.get_observation()
        reward = self._reward(action)
        end = self._termination()

        qstates = p.getJointStates(self.robot.robot_id, self.robot.joint_ids)
        
        torq = []
        for q_torq in qstates:
            torq.append(q_torq[1])

        if self._debug:
            if self._step_cnt == 2:
                input("start")
            print("----------------------------------------")
            print("dist: " + str(np.linalg.norm(self._get_distance(p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0]))))
            print("agent_pos " + str(p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0]))
            print("target " + str(self._target_pos))
            print("----------------------------------------")
            if end: 
                print(self._step_cnt)
                input("ende")

        self.old_pos = np.array(p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0])
        
        return obs, reward, end, {}

    def _reward(self, action):

        curr_eef_pos = p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0]
        euc_dist =  np.linalg.norm(self._target_pos - np.array(curr_eef_pos))
        
        qstates = p.getJointStates(self.robot.robot_id, self.robot.joint_ids)
        
        vel = []
        tor = []
        for q_vel in qstates:
            vel.append(q_vel[1])
            tor.append(q_vel[3])
        
        rd = 0
        if euc_dist <= 0.6 and euc_dist > 0.1: 
            rd = -0.5*euc_dist+0.3
        elif euc_dist <= 0.1:
            rd = np.exp(-520*np.power(euc_dist,2))+0.245
        
        ra=0
        vel = np.array(vel)
        diff = vel-self._old_vel
        acel = []
        for el in diff:
            acel.append(el/(3./240.))

        acel = np.array(acel, dtype=np.float32)

        ra = max(np.absolute(acel))*(-0.005)

        self._old_vel = vel
	
        rp=0
        if curr_eef_pos[2] < 0.3 or curr_eef_pos[2] > 0.7  or self.robot.check_joint_limits() is True:
            rp = -1

        reward = rd + ra + rp


    
        return np.float32(reward)

    def _termination(self):
        """
        Return True if agent was successful or if current number of timesteps is greater than 
        the allowed number of time steps per episode

        Returns:
            bool
        """

        #if self._success or self._step_cnt == self._max_steps:
        if self._step_cnt == self._max_steps:
            self._episode_cnt += 1

            self.world.update_avg_error(self._get_distance(p.getLinkState(self.robot.robot_id, self.robot.robot_gripper_index)[0]))
   
            return True
        else:
            return False
            
    def _get_distance(self, curr_pos):
        """
        Args:
            current eef position or joint angles in rad

        Returns:
            distance between current position/angles and target position/angles
        """
        curr_pos = np.array(curr_pos, dtype=np.float32)
        return self._target_pos-curr_pos
    
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
    
    
    def add_noise(self, obs):

        if self.noise_type == "l":
            noise_std = 0.01
        elif self.noise_type == "m":
            noise_std = 0.025
        elif self.noise_type == "h":
            noise_std = 0.05

        noisy_value = []
        for v in obs:
            noisy_value.append(v + random.gauss(0, noise_std))

        return noisy_value
    

        

