import pybullet as p
import pybullet_data
p.setAdditionalSearchPath(pybullet_data.getDataPath())
import numpy as np
import math
import time

class Robot():
  """
  Class to load and control a Panda robot in Pybullet
  """

  def __init__(self, client, config , time_step=1./240):
    """
    Args:
      client: pybullet client
      render (bool): true if pybullet GUI is used
      urdf_root_path (str): path to root direcotry of urdf objects
      time_step: time to wait between simulation steps
    """

    self.client = client
    self.client.setPhysicsEngineParameter(solverResidualThreshold=0)

    self.config = config
    self.time_step = time_step
    self.render = config["env_param"]["visualize"]
    self.urdf_root_path = "franka_panda/panda.urdf"
    self._curriculum_stage = 1
    
    self.robot_num_joints = 7
    self.robot_gripper_index = 11
    self.joint_ids = [0,1,2,3,4,5]

    self.joint_limits = [2, 1.57, 2, 3, 2, 3]

    #joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001
        ]

    self.hard_reset()

  def hard_reset(self):
    """
    Load panda robot into simulation
    """
    
    # Load robot
    self.robot_id = self.client.loadURDF(self.urdf_root_path,basePosition=[0,0,0], useFixedBase=True)

    # Reset robot position and orientation
    self.client.resetBasePositionAndOrientation(self.robot_id, [-0.00000, 0.000000, 0.070000],
                                      [0.000000, 0.000000, 0.000000, 1.000000])


    self.q_init_pos = [
        0, -0.77, 0, -2.35, 0, 1.537684, 0.8, 0,
        0, 0.000000, 0, 0
    ]
        
    for q_index in range(len(self.q_init_pos)):
      self.client.resetJointState(self.robot_id, q_index, self.q_init_pos[q_index])

    for i in range(100):
      p.stepSimulation()

  def apply_action(self, motorCommands, obj_nr=10):
    
    #self.check_limits()

    # compute new joint angles from joint velocities
    current_joint_states = self.client.getJointStates(self.robot_id, self.joint_ids)
    current_joint_pos = np.array([x[0] for x in current_joint_states])
    new_joint_pos = current_joint_pos + np.array(motorCommands)*0.05

    # set mew joint angles
    self.client.setJointMotorControlArray(self.robot_id, self.joint_ids, self.client.POSITION_CONTROL,
                                targetPositions=new_joint_pos)

    if False:
      # Move set joints to target position
      self.client.setJointMotorControlArray(self.robot_id, self.joint_ids, self.client.VELOCITY_CONTROL,
                                targetVelocities=motorCommands)

    for _ in range(3):
      p.stepSimulation()

    if self.render:
      time.sleep(self.time_step)


  def move_to_start(self, q_init=None):

    """
    Set joints to the initial position
    """

    if type(q_init) == type(None): 
      q_init = self.q_init_pos

    for q_index in range(len(q_init)):
      self.client.resetJointState(self.robot_id, q_index, targetValue=self.q_init_pos[q_index], targetVelocity=0)
      self.client.setJointMotorControl2(self.robot_id, q_index, self.client.POSITION_CONTROL,
                                targetPosition=q_init[q_index])
                    
    for _ in range(500):
      p.stepSimulation()

    if self.render:
          time.sleep(self.time_step)
  
  def check_limits(self):

    eef_pos = np.array(p.getLinkState(self.robot_id, self.robot_gripper_index)[0])
    rad = np.sqrt(eef_pos[0]*eef_pos[0] + eef_pos[1]*eef_pos[1])

    reset = False
    dist = self.config["r_max"]-self.config["r_min"]
    mid = self.config["r_min"] + dist/2
    allowed_angle_range = 1.8
    
  
    curr_min_border = (mid-(dist/8)*self._curriculum_stage)-0.05
    curr_max_border = (mid+(dist/8)*self._curriculum_stage)+0.05
    curr_max_rot = (np.pi/8)*self._curriculum_stage+0.1

    if rad < curr_min_border or rad > curr_max_border:
      if rad < curr_min_border: rad_new = curr_min_border
      if rad > curr_max_border : rad_new = curr_max_border
      rot_xy = np.arcsin(eef_pos[1]/rad)
      eef_pos[0] = np.cos(rot_xy)*rad_new
      eef_pos[1]= np.sin(rot_xy)*rad_new
      reset = True

    if eef_pos[2] < 0.25 or eef_pos[2] > 0.7:
      reset = True
      if eef_pos[2] < 0.3: eef_pos[2] = 0.3
      if eef_pos[2] > 0.65: eef_pos[2] = 0.65

    if eef_pos[0] == 0:
      if eef_pos[1] > 0: curr_angle = np.pi/2
      else: curr_angle = -np.pi/2
    else:
      curr_angle = np.arctan(eef_pos[1]/eef_pos[0])



    if abs(curr_angle) > curr_max_rot:
      if curr_angle > 0: eef_pos[1] = np.sin(curr_max_rot)*rad
      else: eef_pos[1] = np.sin(-curr_max_rot)*rad
      reset = True

    if reset:
      # Ensure Top Down orientation
      orn = self.client.getQuaternionFromEuler([0, math.pi, 0])

      # Get joint values for new eef position
      q_pos = self.client.calculateInverseKinematics(self.robot_id,
                                                    self.robot_gripper_index,
                                                    eef_pos,
                                                    orn,
                                                    jointDamping=self.jd)

      # Move set joints to target position
      for q_idx in range(self.robot_num_joints):
        self.client.setJointMotorControl2(self.robot_id, q_idx, self.client.POSITION_CONTROL,
                                targetPosition=q_pos[q_idx])

      for _ in range(3):
        p.stepSimulation()

      if self.render:
          time.sleep(self.time_step)

    return
    
  def check_joint_limits(self):

    q_state= self.client.getJointStates(self.robot_id, self.joint_ids)
    q_pos = np.array([x[0] for x in q_state])

    for q_idx in range(len(self.joint_ids)):
      if abs(q_pos[q_idx]) > self.joint_limits[self.joint_ids[q_idx]]:
        return True

    return False   
    
      
      
    

