import sys, os
import numpy as np
current_working_dir = os.getcwd().replace("/RL_Gym", "")
sys.path.insert(1, current_working_dir + '/ros_fr3_control_service/scripts/control_script')  # Insert the correct path into sys.path

import rospy, roslaunch

from ros_controller import RobotJoint, StateViewer
from sensor_msgs.msg import JointState

class FR3():
  """
  Class to controll a real FR3 robot
  """

  def __init__(self):
    """
    Args:
      client: pybullet client
      render (bool): true if pybullet GUI is used
      urdf_root_path (str): path to root direcotry of urdf objects
      time_step: time to wait between simulation steps
    """

    rospy.init_node('example_velocity', anonymous=True)
    self.joints = RobotJoint()
    self.state = JointState()
    self.state_viewer = StateViewer()
    self.state.name = ['1', '2', '3', '4', '5', '6', '7']
    self.num_joints = len(self.state.name)
    self.robot_is_stopped = False

    self.positions = []

    self.joint_controller = None

    #self.move_robot_to_start()
    
  def move_robot_to_start(self):

    print("Robot is moving to start position")

    joint_states = np.array(self.get_joint_pos())
    des_joint_states = np.array([0, -0.77, 0, -2.35, 0, 1.537684, 0.8])
    delta =  (des_joint_states-joint_states)/50

    new_joint_pos = joint_states+delta
    for _ in range(50):
      self.state.position = new_joint_pos
      self.joints.set_target(self.state)
      new_joint_pos += delta
      rospy.sleep(0.05)

    print("Robot is ready to use")

    
  def set_joint_positions(self, motor_commands):
      
      #motor_commands = [motor_commands[0],motor_commands[1],0,motor_commands[2],0,0,0]
      motor_commands = [motor_commands[0],motor_commands[1],motor_commands[2],motor_commands[3],motor_commands[4],motor_commands[5],0]

      # Clip velocities to avoid accidents
      motor_commands = self.vel_limit(motor_commands)
  
      if not self.check_workspace_limits():
        rospy.loginfo("Robot is out of workspace limits")
        motor_commands = np.zeros(7)
        self.robot_is_stopped = True
  
      new_joint_pos = self.get_joint_pos() + np.array(motor_commands)*0.05

      #self.positions.append([new_joint_pos[0], new_joint_pos[1], new_joint_pos[3]])
      self.positions.append([new_joint_pos[0], new_joint_pos[1], new_joint_pos[2], new_joint_pos[3], new_joint_pos[4], new_joint_pos[5]])

      self.state.position = new_joint_pos
      self.joints.set_target(self.state)

  def set_joint_velocities(self, motor_commands):
    
    motor_commands = [motor_commands[0],motor_commands[1],motor_commands[2],motor_commands[3],motor_commands[4],motor_commands[5],0]

    # Clip velocities to avoid accidents
    motor_commands = self.vel_limit(motor_commands)

    if not self.check_workspace_limits():
      rospy.loginfo("Robot is out of workspace limits")
      motor_commands = np.zeros(7)
      self.robot_is_stopped = True

    self.state.velocity = motor_commands
    self.joints.set_target(self.state)

  def get_eef_pos(self):
    state = self.state_viewer.get_state()
    eef_pos = state[1].transform.translation
    return eef_pos
  
  def get_joint_pos(self):
    state = self.state_viewer.get_state()
    joint_pos = state[0]
    return joint_pos

  def get_joint_vel(self):
    joint_vel = self.joints.real_state.velocity[:7]
    return joint_vel
  
  def get_joint_acc(self):
    rospy.loginfo(self.joints.real_state)
    joint_acc = self.joints.real_state.effort[:7]
    return joint_acc

  def check_workspace_limits(self):
    
    eef_pos = self.get_eef_pos()

    rad_to_robot_base = np.sqrt(eef_pos.x*eef_pos.x + eef_pos.y*eef_pos.y)
    
    r_min = 0.3
    r_max = 0.7

    if rad_to_robot_base < r_min or rad_to_robot_base > r_max:
      print(1)
      return False

    if eef_pos.z < 0.3 or eef_pos.z > 0.7:
      print(2)
      return False 
    
    if eef_pos.x == 0:
      if eef_pos.y > 0:
        curr_angle = np.pi/2
      else:
        curr_angle = -np.pi/2
    else:
      curr_angle = np.arctan(eef_pos.y/eef_pos.x)

    if curr_angle < -np.pi/4 or curr_angle > np.pi/4:
       print(3)
       return False

    return True

  def vel_limit(self, vel):
    vel = np.clip(vel, -0.8, 0.8)
    return vel
  
  def move_out_of_bounds(self, motor_commands):
    motor_commands = np.clip(motor_commands, -0.2, 0.2)
    self.state.velocity = motor_commands
    self.joints.set_target(self.state)
    rospy.sleep(3)
    self.state.velocity = np.zeros(7)
    self.joints.set_target(self.state)