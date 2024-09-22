import json
#import rospy
import os, sys
import argparse
import importlib
import numpy as np
import pybullet as p

from gym.envs.registration import register

from hrl_gym.helper_scripts.docs import load_config
from hrl_gym.environments.HRLGymEnv import HRLGymEnv
from hrl_gym.environments.RealGymEnv import RealGymEnv
from hrl_gym.helper_scripts.data_processing import MovingAverage, normalize_data, calc_accuracy, cnt_over_threshold, caluclate_results

# PopSAN can only be imported, if called with conda snn_env
try:
    current_working_dir = os.getcwd().replace("/HRL_RobotGym", "")
    sys.path.insert(1, current_working_dir+'/pop-spiking-deep-rl')

    from train_snn import Normalize
    from popsan_drl.popsan_ppo.popsan import PopSpikeActor
    import torch
except:
    try:
        import ray
        from ray.tune.registry import register_env
    except:
        print("Please activate the correct conda environment")

def get_action(obs, agent, snn=False, act_limit=1.0, device='cpu', norm=None):
    """
    Args:
        obs: observation from the environment
        agent: agent to get action from
        snn: flag to use SNN
        act_limit: action limit for the environment
        device: device to run the agent on
        norm: normalization object

    Returns:
        action: action from the agent
    """
    if snn:
        import torch
        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            norm_state = norm.normalize_state(state, update=False)
            a = agent(norm_state, 1)[0][0].cpu().numpy()
            return np.clip(a, -act_limit, act_limit)


    else:
        action = agent.compute_single_action(obs)
    return action
    
def run_agent(agent, env, use_snn=False, act_limit=1.0, device='cpu', norm=None):
    """
    Args:
        agent: agent to evaluate
        env: environment to evaluate the agent in
        use_snn: flag to use SNN
        act_limit: action limit for the environment
        device: device to run the agent on
        norm: normalization object
    """

    accuracy = []
    dist = []

    tracked_traj = []

    tracked_eef_pos = []
    eef_pos_list = []

    tracked_actions = []
    joint_pos_list = []

    single_vel = []
    joint_vel_list = []

    tracked_joint_acel = []
    joint_acel_list = []
    old_vel = np.zeros(6)
    
    obs = env.reset()
    # Run env for 10 episodes to wake up GPU
    while env._episode_cnt < 0:
        obs, _, done, _ = env.step(get_action(obs, agent, use_snn, act_limit, device, norm))
        if done:
            env.reset() 
    
    if not use_snn:
        agent.get_policy().model.inference_times = []
    else:
        agent.inference_times = []
        
    env.world._curriculum_stage = 4
    env._episode_cnt = 0
    obs = env.reset()

    print("Evaluating for 100 episodes...")

    # Start thread for energy measurement
    #sub_proc = subprocess.Popen(os.getcwd()+"/start_energy_measurement.sh")

    ind_tracked_traj = []

    while env._episode_cnt < 25:
        
        action = get_action(obs, agent, use_snn, act_limit, device, norm)
        tracked_actions.append(action)
        
        obs, _, done, _ = env.step(action)
        

        dist_xyz = [0,0,0,0]
        dist_xyz[:3] = env._get_distance(p.getLinkState(env.robot.robot_id, env.robot.robot_gripper_index)[0])
        dist_xyz[3] = np.linalg.norm(dist_xyz[:3])
        eef_pos_list.append(dist_xyz)
        
        current_joint_states = p.getJointStates(env.robot.robot_id, env.robot.joint_ids)

        joint_pos = []
        joint_vel = []
        single = []
        for idx,state in enumerate(current_joint_states):
            
                joint_pos.append(state[0])
                joint_vel.append(state[1])
                if idx == 1:
                    single.append(state[1])
            

        joint_pos_list.append(joint_pos)
        joint_vel_list.append(action)
        single_vel.append(single)

        vel = np.array(joint_vel_list[-1])
        diff = vel-old_vel
        acel = []
        for idx, el in enumerate(diff):
            acel.append(el/(3./240.))

        joint_acel_list.append(acel)

        ind_tracked_traj.append(np.linalg.norm(env._get_distance(p.getLinkState(env.robot.robot_id, env.robot.robot_gripper_index)[0])))

        if done:

            eef_pos = p.getLinkState(env.robot.robot_id, env.robot.robot_gripper_index)[0]
            distance = float(np.linalg.norm(env._get_distance(eef_pos)))
            dist.append(distance)
            accuracy.append(distance)
            #calculate_smoothness(joint_vel_list)
            #calculate_smoothness(joint_pos_list)
            #plot_results(eef_pos_list, joint_pos_list, joint_vel_list, joint_acel_list, tracked_actions=tracked_actions)
            tracked_actions = []
            joint_acel_list = []
            joint_pos_list = []
            joint_vel_list = []
            single_vel  = []
            eef_pos_list = []
            norm_data = normalize_data(ind_tracked_traj)
            tracked_traj.append(norm_data)
            ind_tracked_traj = []
            env.reset()

    # Stop thread
    #sub_proc.wait()

    if not use_snn:
        avg_inference_time = float(np.mean(agent.get_policy().model.inference_times))
    else:
        avg_inference_time = float(np.mean(agent.inference_times))


    results = {}
    results["array_results"] = dist
    results["accuracy"] = calc_accuracy(accuracy)[0]
    results["avg_inference_time"] = avg_inference_time
    results["accuracy_without_over_one"] = calc_accuracy(accuracy)[1]
    results["over_one_cnt"] = cnt_over_threshold(dist)[0]
    results["accuracy_without_over_two"] = calc_accuracy(accuracy)[2]
    results["over_two_cnt"] = cnt_over_threshold(dist)[1]
    results["accuracy_without_over_three"] = calc_accuracy(accuracy)[3]
    results["over_three_cnt"] = cnt_over_threshold(dist)[2]
    results["accuracy_without_over_four"] = calc_accuracy(accuracy)[4]
    results["over_four_cnt"] = cnt_over_threshold(dist)[3]
    results["accuracy_without_over_five"] = calc_accuracy(accuracy)[5]
    results["over_five_cnt"] = cnt_over_threshold(dist)[4]

    #plot_eucledian_distancen(tracked_traj, use_snn)

    #save_results(results, use_snn, agent_name, current_iteration_number)

def run_real_robot(agent, env, use_snn=False, act_limit=1.0, device='cpu', norm=None):
    """
    Args:
        agent: agent to evaluate
        env: environment to evaluate the agent in
        use_snn: flag to use SNN
        act_limit: action limit for the environment
        device: device to run the agent on
        norm: normalization   
    """


    # Set of parameters, which are tracked
    accuracy = []
    dist = []
    tracked_traj = []
    ind_tracked_traj = []
    eef_pos_list = []
    joint_pos_list = []
    joint_vel_list = []
    joint_acel_list = []

    tracked_actions = []
    rate = rospy.Rate(4)

    avg_joint_one = MovingAverage(10)
    avg_joint_two = MovingAverage(10)
    avg_joint_three = MovingAverage(10)
    avg_joint_four = MovingAverage(10)
    avg_joint_five = MovingAverage(10)
    avg_joint_six = MovingAverage(10)

    env._episode_cnt = 0
    obs = env.reset()

    print("Evaluating for 100 episodes...")

    while env._episode_cnt < 25:
        
        action = get_action(obs, agent, use_snn, act_limit, device, norm)
        
        tracked_actions.append(action)

        avg_joint_one.update(action[0])
        avg_joint_two.update(action[1])
        avg_joint_three.update(action[2])
        avg_joint_four.update(action[3])
        avg_joint_five.update(action[4])
        avg_joint_six.update(action[5])
        action = [avg_joint_one.get_average(), avg_joint_two.get_average(), avg_joint_three.get_average(), avg_joint_four.get_average(), avg_joint_five.get_average(), avg_joint_six.get_average()]
        #action = [avg_joint_one.get_average(), avg_joint_two.get_average(), avg_joint_three.get_average()]
        
        obs, _, done, _ = env.step(action)
        
        # Track all kind of different parameters
        dist_xyz = [0,0,0,0]
        dist_xyz[:3] = env._get_distance(env.robot.get_eef_pos())
        dist_xyz[3] = np.linalg.norm(dist_xyz[:3])
        eef_pos_list.append(dist_xyz)
        
        joint_pos_list.append(env.robot.get_joint_pos())
        joint_vel_list.append(env.robot.get_joint_vel())
        joint_acel_list.append(env.robot.get_joint_vel())
        ind_tracked_traj.append(np.linalg.norm(env._get_distance(env.robot.get_eef_pos())))

        if done:
            distance = float(np.linalg.norm(dist_xyz[:3]))
            dist.append(distance)
            accuracy.append(distance)
            #calculate_smoothness(tracked_actions)
            #calculate_smoothness(joint_pos_list)
            #plot_results(eef_pos_list, joint_pos_list, joint_vel_list, joint_acel_list, tracked_actions)
            joint_acel_list = []
            joint_pos_list = []
            joint_vel_list = []
            eef_pos_list = []
            tracked_actions = []
            norm_data = normalize_data(ind_tracked_traj)
            tracked_traj.append(norm_data)
            ind_tracked_traj = []
            results = caluclate_results(dist, accuracy)
            #save_results(results, use_snn, agent_name, current_iteration_number, overwrite=True)
            env.reset()
            #print(env._episode_cnt)
            
        rate.sleep()
                  
    
    #plot_eucledian_distancen(tracked_traj)

    #save_results(results, use_snn, agent_name, current_iteration_number, overwrite=True)

def eval_snn(args):
    """
    Evaluate the spiking neural network agent
    """

    current_working_dir = os.getcwd().replace("/HRL_RobotGym", "")

    config = load_config(os.getcwd())
    config["project_path"] = os.getcwd()
    config["active_algo"] = 'ppo'
    config["use_snn"] = True
    config["env_param"]["visualize"] = True
    config["debug_info"] = False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    register(id='HRLEnv-v0',
         entry_point='hrl_gym.environments:HRLGymEnv', kwargs={'config': config}
    )

    if not args.real:
        env = HRLGymEnv(config)
        env.noise_type = args.noise
        env.extreme_situation = args.extreme
    else:
        env = RealGymEnv(config)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # PopSAN
    popsan = PopSpikeActor(obs_dim, act_dim, en_pop_dim=10, de_pop_dim=10, hidden_sizes=(256, 256),
                           mean_range=(-3, 3), std=0.15, spike_ts=5, device=device).to(device)
    
    norm = Normalize(device)
    popsan.load_state_dict(torch.load(args.load_model, map_location=torch.device('cpu')))

    name = "vorerst_final"
   
    for i in range(args.rep):
        if not args.real:
            run_agent(popsan, env, use_snn=True, act_limit=act_limit, device=device, norm=norm)
        else:
            run_real_robot(popsan, env, use_snn=True, act_limit=act_limit, device=device, norm=norm)

    
    return

def eval_dnn(args):
    """
    Evaluate the deep neural network agent
    """

    currentdir = os.getcwd()+"/trained_agents/PPO"

    # Get path to config and path to checkpoint
    tmp = args.load_model.split("/")
    if tmp[0] == "": 
        tmp = tmp[1]
        agent_path = currentdir+args.load_model
    else: 
        tmp = tmp[0]
        agent_path = currentdir+"/"+args.load_model
    config_path = currentdir+"/"+tmp

    
    config = load_config(config_path)
    config["env_param"]["visualize"] = True

    register_env('HRLEnv-v0', lambda config: HRLGymEnv(config))

    ray.shutdown()
    ray.init(num_cpus=4, num_gpus=0)   

    if not args.real:
        env = HRLGymEnv(config)
        env.noise_type = args.noise
        env.extreme_situation = args.extreme
    else:
        env = RealGymEnv(config)

    # Load saved hyperperameters
    with open(config_path+"/params.json") as json_file:
        saved_HP = json.load(json_file)

    algo = importlib.import_module("ray.rllib.agents."+str(config["active_algo"]))
    trainer_module = getattr(algo, str(config["active_algo"]).upper()+"Trainer")
    

    # Change some Hyperparmeters
    config_algo = algo.DEFAULT_CONFIG.copy()
    config_algo.update(saved_HP)
    config_algo["env"] = None
    config_algo["callbacks"] = None
    config_algo["observation_space"] = env.observation_space
    config_algo["action_space"] = env.action_space
    config_algo["num_workers"] = 1

    # Init Trainer and restore checkpoint
    agent = trainer_module(config=config_algo)
    agent.restore(agent_path)
    
    
    tmp = args.load_model.split("/")
    name = tmp[0]
    if tmp[0] == "":
        name = tmp[1]

    device = 'cpu' 

    for i in range(args.rep):
        if not args.real:
            run_agent(agent, env, name, i, device=device)
        else:
            run_real_robot(agent, env, name, i, device=device)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    snn_pars = subparser.add_parser('eval_snn', help=str)
    snn_pars.add_argument('--load_model', type=str, required=False, default=os.getcwd()+"/trained_agents/snn_agent.pt", help="Path to a previous trained agent/model")
    snn_pars.add_argument('--rep', type=int, required=False, default=5, help="Number of repetitions for evaluation")
    snn_pars.add_argument('--noise', type=str, required=False, default="n", help="strength of noise [(l)ight,(m)edium,(h)eavy]")
    snn_pars.add_argument('--extreme', type=bool, required=False, default=False, help="activate extreme situation")
    snn_pars.add_argument('--real', type=bool, required=False, default=False, help="run agent on real a robot")
    snn_pars.set_defaults(func=eval_snn)

    dnn_pars = subparser.add_parser('eval_dnn', help=str)
    dnn_pars.add_argument('--load_model', type=str, required=False, default="dnn_agent/checkpoint_009804/checkpoint-9804", help="Path to a previous trained agent/model")
    dnn_pars.add_argument('--rep', type=int, required=False, default=5, help="Number of repetitions for evaluation")
    dnn_pars.add_argument('--noise', type=str, required=False, default="n", help="strength of noise [(l)ight,(m)edium,(h)eavy]")
    dnn_pars.add_argument('--extreme', type=bool, required=False, default=False, help="activate extreme situation")
    dnn_pars.add_argument('--real', type=bool, required=False, default=False, help="run agent on real a robot")
    dnn_pars.set_defaults(func=eval_dnn)

    args = parser.parse_args()
    args.func(args)

