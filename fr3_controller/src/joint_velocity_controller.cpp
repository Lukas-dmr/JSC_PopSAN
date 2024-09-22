// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/joint_velocity_controller.h>

#include <cmath>

#include <controller_interface/controller_base.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

// Initialize the controller with the given hardware interface and node handle.
bool JointVelocityController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {  
  velocity_joint_interface_ = robot_hardware->get<hardware_interface::VelocityJointInterface>();
  if (velocity_joint_interface_ == nullptr) {
    ROS_ERROR(
        "JointVelocityExampleController: Error getting velocity joint interface from hardware!");
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
    ROS_ERROR("JointVelocityExampleController: Could not parse joint names");
  }
  if (joint_names.size() != 7) {
    ROS_ERROR_STREAM("JointVelocityExampleController: Wrong number of joint names, got "
                     << joint_names.size() << " instead of 7 names!");
    return false;
  }

  velocity_joint_handles_.resize(7);
  for (size_t i = 0; i < 7; ++i) {
    try {
      velocity_joint_handles_[i] = velocity_joint_interface_->getHandle(joint_names[i]);
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointVelocityExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("JointVelocityExampleController: Could not get state interface from hardware");
    return false;
  }

  ros::NodeHandle n;
  sub = n.subscribe("desired_joints", 5, &JointVelocityController::jointsCallback, this);

  return true;
}

// Start the controller.
void JointVelocityController::starting(const ros::Time& /* time */) {
  elapsed_time_ = ros::Duration(0.0);
}

// Update joint setpoints.
void JointVelocityController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
  double cur, vel;
  for (size_t index = 0; index < 7; ++index) { 
      cur = velocity_joint_handles_[index].getVelocity();
      vel = pid(m_qGoal[index], cur, index);
      velocity_joint_handles_[index].setCommand(vel);
  }
}

void JointVelocityController::stopping(const ros::Time& /*time*/) {
  // WARNING: DO NOT SEND ZERO VELOCITIES HERE AS IN CASE OF ABORTING DURING MOTION
  // A JUMP TO ZERO WILL BE COMMANDED PUTTING HIGH LOADS ON THE ROBOT. LET THE DEFAULT
  // BUILT-IN STOPPING BEHAVIOR SLOW DOWN THE ROBOT.
}

// Calculate the PID command with the given setpoint and current value.
double JointVelocityController::pid(double target, double current, size_t index) {
  double error = (target - current);
    
  double pOut = m_kp * error; // Proportional term

  // Integral term
  m_integral[index] += error * m_dt;
  double iOut = m_ki * m_integral[index];

  // Derivative term
  double derivative = (error - m_preError[index]) / m_dt;
  double dOut = m_kd * derivative;
    
  double output = pOut + iOut + dOut; // Calculate total output

  // Restrict to max/min
  if(output > m_max)
    output = m_max;
  else if(output < -m_max)
    output = -m_max;
  else if(std::abs(output) < m_min)
    output = 0;

  m_preError[index] = error; // Save error to previous error

  return output;
}

// Callback function to set new joint setpoints.
void JointVelocityController::jointsCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
  // Check input dimension
  if (msg->velocity.size() < 7 || msg->velocity.size() > 9) {
    ROS_WARN("Wrong input dimension");
    return;
  }

  // Check joint limits
  const auto &vel = msg->velocity;
  for (size_t i = 0; i < 7; ++i)
    if (vel[i] < -2 || 2 < vel[i]) {
      ROS_WARN("Out of velocity limits of joint [%i]", static_cast<int>(i));
      return;
    }

  double pos; 
  for (size_t i = 0; i < 7; ++i)
  {
    pos = velocity_joint_handles_[i].getPosition();
    if (pos < m_minLimits[i] || m_maxLimits[i] < pos) {
      ROS_WARN("Out of limits of joint [%i]", static_cast<int>(i));
      return;
    }
  }

  // check the difference
  double diff = 0;
  for (size_t i = 0; i < 7; ++i)
    diff += std::abs(vel[i] - m_qGoal[i]);
  if (diff > 1) {
    ROS_INFO("reset the integral and error term");
    m_preError = std::vector<double>(7, 0);
    m_integral = std::vector<double>(7, 0);
    m_lastOutput = std::vector<double>(7, 0);
  }
  
  m_qGoal = vel;
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointVelocityController,
                       controller_interface::ControllerBase)
