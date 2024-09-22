// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <sensor_msgs/JointState.h>

namespace franka_example_controllers {

class JointVelocityController : public controller_interface::MultiInterfaceController<
                                       hardware_interface::VelocityJointInterface,
                                       franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void starting(const ros::Time&) override;
  void stopping(const ros::Time&) override;

 private:
  void jointsCallback(const sensor_msgs::JointState::ConstPtr& msg);
  double pid(double target, double current, size_t index);

  hardware_interface::VelocityJointInterface* velocity_joint_interface_;
  std::vector<hardware_interface::JointHandle> velocity_joint_handles_;
  ros::Duration elapsed_time_;

  double m_dt = 0.001;
  double m_max = 2.5;
  double m_min = 0.01;
  double m_kp = 0.05;
  double m_kd = 0.001;
  double m_ki = 0.0005;
  double m_accScale = 0.1;

  ros::Subscriber sub;
  std::vector<double> m_qGoal = {0, 0, 0, 0, 0, 0, 0};
  std::vector<double> m_minLimitsFR3 = {-2.6437, -1.6837, -2.8007, -2.9421, -2.7065, 0.4445, -2.9159};
  std::vector<double> m_maxLimitsFR3 = {2.6437, 1.6837, 2.8007, -0.0518, 2.7065, 4.4169, 2.9159};
  std::vector<double> m_maxLimitsPanda = {2.7973, 1.6628, 2.7973, 0, 2.8973, 3.6525, 2.7973};
  std::vector<double> m_minLimitsPanda = {-2.7973, -1.6628, -2.7973, -2.9718, -2.7973, 0.1, -2.7973};
  std::vector<double> m_minLimits = {-2.6437, -1.6837, -2.8007, -2.9421, -2.7065, 0.4445, -2.9159};
  std::vector<double> m_maxLimits = {2.6437, 1.6837, 2.8007, -0.0518, 2.7065, 4.4169, 2.9159};

  std::vector<double> m_preError  = std::vector<double>(7, 0);
  std::vector<double> m_integral  = std::vector<double>(7, 0);
  std::vector<double> m_lastOutput  = std::vector<double>(7, 0);
};

}  // namespace franka_example_controllers
