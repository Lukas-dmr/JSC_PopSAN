import os
import importlib
import numpy as np
import pybullet as p


class Sim_World():

    def __init__(self, config, client):

        # Environment parameters
        self.config = config
        self._time_step = 1. / 240.
        self._curriculum_stage = 1
        self._error_collection = [1]
        self._episode_cnt = 0
        self.marker_id = -1
        
        if self.config["debug_info"]: 
            self._curriculum_stage = 4
        

        # Visualisatzion
        self._render = config["env_param"]["visualize"]
        self.physics_client = client

        # import complete kuka robot or just the gripper according to specification inside config.yaml
        file = importlib.import_module("hrl_gym.models.Panda")
        Robot = getattr(file, "Robot")
    
        self.robot = Robot(self.physics_client, config=config, time_step=self._time_step)

    def soft_reset(self):
        """
        Reset robot to starting position and load new objects into the environment
        """

        thresh = 0.02
        avg_error = sum(self._error_collection)/len(self._error_collection)

        if avg_error < thresh and self._curriculum_stage < 4 and self._episode_cnt > 20: 
            print("STUFE WURDE GEUPGRADET: " +str(self._curriculum_stage))
            self._curriculum_stage += 1
            self._error_collection = [1]
            self._episode_cnt = 0

        q_init = [
        0, 0, 0, -1.5, 0, 1.537684, 0.8, 0,
        0, 0.000000, 0, 0
        ]

        curr_min_rot = -(np.pi/8)*self._curriculum_stage
        curr_max_rot = (np.pi/8)*self._curriculum_stage
        q_init[0] = np.random.uniform(low=curr_min_rot, high=curr_max_rot)
        
        self.robot.move_to_start(q_init)

        self._episode_cnt += 1

    def generate_randome_targetpos(self):
        """
        Generates a randome target pos

        Returns:
            target_pos
        """

        dist = self.config["r_max"]-self.config["r_min"]
        mid = self.config["r_min"] + dist/2
        curr_min_border = mid-(dist/8)*self._curriculum_stage
        curr_max_border = mid+(dist/8)*self._curriculum_stage 
        radius = np.random.uniform(low=curr_min_border, high=curr_max_border)
        curr_min_rot = -(np.pi/8)*self._curriculum_stage
        curr_max_rot = (np.pi/8)*self._curriculum_stage
        rot_xy = np.random.uniform(low=curr_min_rot, high=curr_max_rot)
            
        x_targ = np.cos(rot_xy)*radius
        y_targ = np.sin(rot_xy)*radius
        z_targ = np.random.uniform(low=0.35,high=0.6)

        return np.array([x_targ,y_targ,z_targ], dtype=np.float32)
        

    def generate_extreme_targetpos(self):
        
        radius = np.random.choice([np.random.uniform(low=0.75, high=0.8), np.random.uniform(low=0.75, high=0.8)])
        rot_xy = np.random.uniform(low=-1.57, high=1.57)

        x_targ = np.cos(rot_xy)*radius
        y_targ = np.sin(rot_xy)*radius
        z_targ = np.random.uniform(low=0.35,high=0.6)

        return np.array([x_targ,y_targ,z_targ], dtype=np.float32)
        

    def update_avg_error(self, cart_dist):

        euc_dist = np.linalg.norm(cart_dist)

        if len(self._error_collection) < 10:
            self._error_collection.append(euc_dist)
        else:
            del(self._error_collection[0])
            self._error_collection.append(euc_dist)

    def visualize_workspace(self, vis_curr=True):

        rad_div = 10
        start_radius = 0.4
        end_radius = 0.7
        delta_radius = (end_radius - start_radius)/rad_div

        rot_div = 15
        start_rot = -1.57
        end_rot = 1.57
        delta_rot = (end_rot - start_rot)/rot_div

        hight_div = 5
        start_hight = 0.35
        end_hight = 0.7
        delta_hight = (end_hight - start_hight)/hight_div

        dist = self.config["r_max"]-self.config["r_min"]
        mid = self.config["r_min"] + dist/2

        color_code = 4

        for i in range(rad_div):
            for j in range(rot_div):
                for k in range(hight_div):
                    if abs(start_radius) >= 0.51 and abs(start_radius) <= 0.58 and \
                          abs(start_rot) < 0.38:
                        color_code = 1

                    elif (abs(start_radius) >= 0.47 and abs(start_radius) <= 0.62 and \
                            (abs(start_rot) >= 0.38 and abs(start_rot) <= 0.78)):
                            color_code = 2

                    elif (abs(start_radius) >= 0.43 and abs(start_radius) <= 0.66 and \
                            (abs(start_rot) >= 0.78 and abs(start_rot) <= 1.17)):
                            color_code = 3

                    elif (abs(start_radius) >= 0.4 and abs(start_radius) <= 0.7 and \
                            (abs(start_rot) >= 1.17 and abs(start_rot) <= 1.57)):
                            color_code = 4
                            
                     
                    #print("radius: ", start_radius, " rot: ", start_rot, " hight: ", start_hight, " color: ", color_code)
                    p.loadURDF(os.getcwd()+"/hrl_gym/models/spheres/b"+str(color_code)+".urdf", start_radius*np.cos(start_rot), start_radius*np.sin(start_rot), start_hight)
                    start_hight += delta_hight
                
                start_hight = 0.35
                start_rot += delta_rot

            start_rot = -1.57
            start_radius += delta_radius
        
        

    def visualize_target(self, target_pos):
        if self.marker_id != -1:
            p.removeBody(self.marker_id)
        self.marker_id = (self.marker_id) = p.loadURDF(os.getcwd()+"/hrl_gym/models/spheres/b1.urdf", target_pos[0], target_pos[1], target_pos[2], 0.000000, 0.000000, 0.0, 1.0)
        p.setCollisionFilterGroupMask(self.marker_id, 0,0,0)
  
        

