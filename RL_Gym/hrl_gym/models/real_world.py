import os
import importlib
import numpy as np
import pybullet as p

from hrl_gym.models.FR3 import FR3


class Real_World():

    def __init__(self):

        # Environment parameters          
        self.robot = FR3()


    def generate_randome_targetpos(self, cartesian):
        """
        Generates a randome target pos

        Args:
            cartesian (bool): If true, generate target pose in cartesian coordinates

        Returns:
            target_pos
        """

        #TODO Fürs erste geh ich auf nummer sicher 
        radius = np.random.uniform(low=0.4, high=0.7)
        rot_xy = np.random.uniform(low=-np.pi/4,high=np.pi/3)

        x_targ = np.cos(rot_xy)*radius
        y_targ = np.sin(rot_xy)*radius
        z_targ = np.random.uniform(low=0.35,high=0.6)

        return np.array([x_targ,y_targ,z_targ], dtype=np.float32)
        
  
        

