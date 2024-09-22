// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/joint_position_controller.h>

#include <cmath>
#include <thread>
#include <chrono>
#include <fstream>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

// Initialize the controller with the given hardware interface and node handle.
bool JointPositionController::init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) {
  position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();
  if (position_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointPositionController: Error getting position joint interface from hardware!");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("JointVelocityExampleController: Could not get parameter arm_id");
    return false;
  }
  if (arm_id == "panda" || arm_id == "panda_sim") {
      m_minLimits = m_minLimitsPanda;
      m_maxLimits = m_maxLimitsPanda;
    } else if (arm_id == "fr3" || arm_id == "fr3_sim") {
      m_minLimits = m_minLimitsFR3;
      m_maxLimits = m_maxLimitsFR3;
    } else {
      ROS_ERROR("JointVelocityExampleController: Unknown arm_id %s", arm_id.c_str());
      return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names)) {
    ROS_ERROR("JointPositionController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointPositionController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }
  position_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      position_joint_handles_[i] = position_joint_interface_->getHandle(joint_names[i]);
      cur_goal[i] = position_joint_handles_[i].getPosition();
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM(
          "JointPositionController: Exception getting joint handles: " << e.what());
      return false;
    }
  }

  ros::NodeHandle n;
  sub = n.subscribe("desired_joints", 5, &JointPositionController::jointsCallback, this);

  return true;
} 

// Start the controller.
void JointPositionController::starting(const ros::Time& /* time */) {
  for (size_t i = 0; i < 7; ++i) {
    initial_pose_[i] = position_joint_handles_[i].getPosition();
  }
}

// Update joint setpoints.
void JointPositionController::update(const ros::Time& /*time*/, const ros::Duration& period) {
  for (size_t index = 0; index < 7; ++index) { 
      double cur = position_joint_handles_[index].getPosition();
      double delta = computeControlSignal(cur_goal[index], cur, index);      
      position_joint_handles_[index].setCommand(cur+delta);
  }
}

// Compute the control signal to guarantee smooth trajectory.
double JointPositionController::computeControlSignal(double setpoint, double process_variable, int index) {
        std::chrono::high_resolution_clock::time_point current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_diff = current_time - prev_time_;
        double dt = time_diff.count();
        
        double error = setpoint - process_variable;
        m_integral[index] += error * dt;
        double derivative = (error - m_preError[index]) / dt;
        double control_signal = m_kp * error + m_ki * m_integral[index]+ m_kd * derivative;
        m_preError[index] = error;
        prev_time_ = current_time;
        return control_signal;
} 

// Callback function to set new joint setpoints.
void JointPositionController::jointsCallback(const sensor_msgs::JointState::ConstPtr& msg)
{

  // Check input dimension
  if (msg->position.size() < 7 || msg->position.size() > 9) {
    ROS_WARN("Wrong input dimension");
    return;
  }

  // Check joint limits
  const auto &pos = msg->position;
  for (size_t i = 0; i < 7; ++i)
    if (pos[i] < m_minLimits[i] || m_maxLimits[i] < pos[i]) {
      ROS_WARN("Out of limits of joint [%i]", static_cast<int>(i));
      return;
    }

  // Check the difference
  double diff = 0;
  for (size_t i = 0; i < 7; ++i)
    diff += std::abs(pos[i] - cur_goal[i]);
  if (diff > 1) {
    ROS_INFO("reset the integral and error term");
    m_preError = std::vector<double>(7, 0);
    m_integral = std::vector<double>(7, 0);
    m_lastOutput = std::vector<double>(7, 0);
  }

  counter++;
 
  cur_goal = pos;
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointPositionController,
                       controller_interface::ControllerBase)
