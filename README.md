# Joint Space Control (JSC) using a Population Coded Spking Actor Network (PopSAN)

This work investigates the use of a PopSAN Spiking Neural Network (SNN) for inverse kinematics in a 7-DOF robotic arm, leveraging reinforcement learning (RL) techniques. 
The performance of the SNN, trained in simulation, is compared with a traditional Deep Neural Network (DNN) approach using the Proximal Policy Optimization (PPO) algorithm. 
Both methods were evaluated in simulation and successfully transferred to real FR3 robots, demonstrating the potential advantages of SNNs in practical robotic applications.
<div align="center">
 <img src="assets/popsan_setup.png" width="800" height="500"/>
</div>

## Results

The evaluation showed that the PopSAn Spiking Neural Network (SNN) achieved a success rate of 90% for inverse kinematics tasks, while the Deep Neural Network (DNN) agent reached 99%. 
Although the SNN performed competitively, it did not fully match the DNN’s performance in certain areas. 
These results suggest that with additional hyperparameter tuning, the SNN may be able to achieve performance levels comparable to the DNN agent.

<table align="center">
  <tr>
    <td align="center">
      <img src="assets/snn_agent_demo.gif" width="300" height="300"/>
    </td>
    <td align="center">
      <img src="assets/snn_agent_sim2real_demo.gif" width="300" height="300"/>
    </td>
  </tr>
</table>

## Run the agent

1. Navigate to /conda_envs and execute: ```conda env create -f snn_env.yaml```
2. Activate the conda env: ```conda activate snn_env```
3. Run trained SNN agent ```python evaluate_agents.py eval_snn```




