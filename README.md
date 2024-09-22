# Joint Space Control (JSC) using a Population Coded Spking Actor Network (PopSAN)

This work investigates the use of a PopSAn Spiking Neural Network (SNN) for inverse kinematics in a 7-DOF robotic arm, leveraging reinforcement learning (RL) techniques. 
The performance of the SNN, trained in simulation, is compared with a traditional Deep Neural Network (DNN) approach using the Proximal Policy Optimization (PPO) algorithm. 
Both methods were evaluated in simulation and successfully transferred to real FR3 robots, demonstrating the potential advantages of SNNs in practical robotic applications.

## Results

The evaluation showed that the PopSAn Spiking Neural Network (SNN) achieved a success rate of 90% for inverse kinematics tasks, while the Deep Neural Network (DNN) agent reached 99%. 
Although the SNN performed competitively, it did not fully match the DNN’s performance in certain areas. 
These results suggest that with additional hyperparameter tuning, the SNN may be able to achieve performance levels comparable to the DNN agent.

<div align="center">
  <img src="assets/snn_agent_demo.gif" width="200" />
</div>

<div align="center">
  <img src="assets/snn_agent_sim2real_demo.mp4" width="200" />
</div>
