#!/usr/bin/env python

import os
from enum import Enum

import rospy
import roslaunch

from move_robot.srv import *


class Type(Enum):
    VELOCITY = "velocity"
    NONE = "none"
    STOP = "stop"

    def __str__(self):
        return '%s' % self.value


global type
type = Type.NONE


class ControllerNode:
    def __init__(self):
        self.controller = None
        self.uuid = None

    def start(self, type):
        print("launching controller")

        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)

        cwd = os.getcwd().split("/")
        home_dir = ""

        for el in cwd:
            if el == "catkin_ws":
                break
            if el == "":
                pass
            else:
                home_dir += "/"+el

        print(home_dir + '/catkin_ws/src/franka_ros/franka_example_controllers/launch/joint_velocity_controller.launch')

        cli_args = [home_dir + '/catkin_ws/src/franka_ros/franka_example_controllers/launch/joint_velocity_controller.launch',
                        'robot_ip:=172.16.0.2', "load_gripper:=true", "robot:=fr3"]

        roslaunch_args = cli_args[1:]
        roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        self.controller = roslaunch.parent.ROSLaunchParent(
            self.uuid, roslaunch_file)

        self.controller.start()
        rospy.sleep(2)
        rospy.loginfo("started velocity controller")

    def stop(self):
        try:
            self.controller.shutdown()
        except:
            pass
        rospy.logerr("shutdown")
        rospy.sleep(2)

def start_velocity(req):
    global type
    rospy.loginfo("starting velocity controller")
    type = Type.VELOCITY
    return StartControllerResponse(success=True)


def stop_controller(req):
    global type
    rospy.loginfo("stopping controller")
    type = Type.STOP
    return StopControllerResponse(success=True)


if __name__ == '__main__':
    rospy.init_node('controller_service', anonymous=True)
    rospy.loginfo("started controller service")

    start_velocity_service = rospy.Service(
        'start_velocity', StartController, start_velocity)
    stop_service = rospy.Service(
        'stop_controller', StopController, stop_controller)

    controllerNode = ControllerNode()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # print(type)
        if type == Type.VELOCITY:
            print("lets go")
            controllerNode.start(type)
            type = Type.NONE
        elif type == Type.NONE:
            rate.sleep()
            continue
        else:
            print(type)
            controllerNode.stop()
            type = Type.NONE
