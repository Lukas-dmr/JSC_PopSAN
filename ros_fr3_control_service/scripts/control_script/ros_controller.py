#!/usr/bin/env python

import os
import threading

import actionlib
import json
import rospy

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from franka_gripper.msg import *
from franka_msgs.msg import FrankaState
import tf2_ros
import franka_gripper.msg
import franka_gripper

from move_robot.srv import *


def transform_to_pose(trans):
    pose = PoseStamped()
    pose.pose.position = trans.transform.translation
    pose.pose.orientation = trans.transform.rotation
    pose.header.stamp = trans.header.stamp
    return pose

# Control in cartesian space
class RobotCart:
    def __init__(self, rate=10):
        self.rate = rospy.Rate(rate)
        self.pose = PoseStamped()
        self.tf_buffer = tf2_ros.Buffer()

        listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1)

        self.start_trans = self.tf_buffer.lookup_transform(
            'fr3_link0', 'fr3_EE', rospy.Time())
        self.pose = transform_to_pose(self.start_trans)

        threading.Thread(target=self.sender).start()

    def sender(self):
        pub = rospy.Publisher(
            '/cartesian_impedance_controller/equilibrium_pose', PoseStamped, queue_size=10)

        while not rospy.is_shutdown():
            # print(pose)
            pub.publish(self.pose)
            # rospy.loginfo(self.pose)
            self.rate.sleep()

# Control in joint space
class RobotJoint:
    def __init__(self, rate=10):
        self.rate = rospy.Rate(rate)
        self.real_state = JointState()
        self.sub = rospy.Subscriber(
            "/joint_states", JointState, self.state_callback)
        rospy.sleep(0.5)
        self.desired_state = self.real_state

        threading.Thread(target=self.sender).start()

    def sender(self):
        pub = rospy.Publisher(
            '/desired_joints', JointState, queue_size=10)

        while not rospy.is_shutdown():
            pub.publish(self.desired_state)
            # rospy.loginfo(self.desired_state)
            self.rate.sleep()

    def state_callback(self, joint_data):
        self.real_state = joint_data

    def set_target(self, joints):
        print("Set target: ", joints.position)
        self.desired_state = joints

    def stop(self):
        self.desired_state = self.real_state
        rospy.sleep(2)

# Control gripper
class Gripper:
    def __init__(self, rate=10):
        self.rate = rospy.Rate(rate)
        self.grasp_client = actionlib.SimpleActionClient(
            '/franka_gripper/grasp', GraspAction)
        self.homing_client = actionlib.SimpleActionClient(
            '/franka_gripper/homing', HomingAction)
        self.moving_client = actionlib.SimpleActionClient(
            '/franka_gripper/move', MoveAction)
        self.stop_client = actionlib.SimpleActionClient(
            '/franka_gripper/stop', StopAction)
        self.grasp_client.wait_for_server()
        self.homing_client.wait_for_server()
        self.moving_client.wait_for_server()
        self.stop_client.wait_for_server()
        print("Gripper ready")

    def grasp(self, width_in, speed_in=0.1, force_in=40.0):
        goal = franka_gripper.msg.GraspGoal(
            width=width_in, speed=speed_in, force=force_in)
        goal.epsilon.inner = 0.03
        goal.epsilon.outer = 0.03
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result()
        print("Grasp result: ", self.grasp_client.get_result())

    def move(self, width_in, speed_in=0.1):
        goal = franka_gripper.msg.MoveGoal(width=width_in, speed=speed_in)
        self.moving_client.send_goal(goal)
        self.moving_client.wait_for_result()

    def homing(self):
        self.homing_client.send_goal(franka_gripper.msg.HomingGoal())
        self.homing_client.wait_for_result()

    def stop(self):
        self.stop_client.send_goal(franka_gripper.msg.StopAction())
        self.stop_client.wait_for_result()


class StateViewer:
    def __init__(self, rate=10, print_data=False):
        self.rate = rospy.Rate(rate)
        self.print_data = print_data
        self.joints = []
        # Measured link-side joint torque sensor signals
        self.tau_J = []
        # Desired link-side joint torque sensor signals without gravity.
        self.tau_J_d = []
        # External torque, filtered.
        self.tau_ext_hat_filtered = []
        # Estimated external wrench (force, torque) acting on stiffness frame, expressed relative to the base frame.
        self.O_F_ext_hat_K = []
        # Estimated external wrench (force, torque) acting on stiffness frame, expressed relative to the stiffness frame.
        self.K_F_ext_hat_K = []
        self.tcp = TransformStamped()

        self.sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState, self.joint_callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(0.1)
        self.tcp = self.tf_buffer.lookup_transform(
            'fr3_link0', 'fr3_EE', rospy.Time())
        threading.Thread(target=self.tcp_thread).start()

    def joint_callback(self, franka_state):
        # print("Franka State: ", franka_state)
        self.joints = franka_state.q
        self.tau_J = franka_state.tau_J
        self.tau_J_d = franka_state.tau_J_d
        self.tau_ext_hat_filtered = franka_state.tau_ext_hat_filtered
        self.O_F_ext_hat_K = franka_state.O_F_ext_hat_K
        self.K_F_ext_hat_K = franka_state.K_F_ext_hat_K

    def tcp_thread(self):
        while not rospy.is_shutdown():
            self.tcp = self.tf_buffer.lookup_transform(
                'fr3_link0', 'fr3_EE', rospy.Time())
            if self.print_data:
                self.print_state()
            self.rate.sleep()

    def get_state(self):
        return [self.joints, self.tcp]

    def print_state(self):
        print("Translation: ", self.tcp.transform.translation)
        print("Rotation: ", self.tcp.transform.rotation)

        print("Joint Values: ", self.joints)
        print("Measured link-side joint torque sensor signals: ", self.tau_J)
        print(
            "Desired link-side joint torque sensor signals without gravity: ", self.tau_J_d)
        print("Ext torque, filtered: ", self.tau_ext_hat_filtered)
        print("Estimated ext wrench on stiffness frame, to base frame: ",
              self.O_F_ext_hat_K)
        print("Estimated ext wrench on stiffness frame, to stiffness frame: ",
              self.K_F_ext_hat_K)
        print("\n \n")

    def save_state(self, path=os.getcwd()+"/data/robot_data.json"):
        translation = (self.tcp.transform.translation.x, self.tcp.transform.translation.y,
                       self.tcp.transform.translation.z)
        quat = (self.tcp.transform.rotation.x, self.tcp.transform.rotation.y,
                self.tcp.transform.rotation.z, self.tcp.transform.rotation.w)

        json_dict = {'translation': translation,
                     'rotation': quat,
                     'joints': self.joints,
                     'info': 'Translation and Rotation of the TCP and the joint state of the robot. Rotation as x,y,z,w'}
        json_object = json.dumps(json_dict, indent=4, sort_keys=True)
        with open(path, "w") as outfile:
            outfile.write(json_object)


def load_state(path, print_data=False):
    with open(path) as json_file:
        data = json.load(json_file)

        joints = data['joints']

        tcp = tf2_ros.TransformStamped()
        translation = data['translation']
        quat = data['rotation']
        tcp.transform.translation = translation
        tcp.transform.rotation = quat

        if print_data:
            print("joint Positions: ", joints)
            print("tcp: ", tcp)
        return joints, tcp
