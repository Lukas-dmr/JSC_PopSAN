import os
import json
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MovingAverage:
    # Class for smoothing joint velocities of the real robot
    def __init__(self, window_size):
        self.window_size = window_size
        self.memory = [0,0,0,0,0,0,0,0,0,0]

    def update(self, new_value):
        self.memory.append(new_value)
        if len(self.memory) > self.window_size:
            self.memory.pop(0)

    def get_average(self):
        return sum(self.memory) / len(self.memory)

def normalize_data(data):
    min_distance = np.min(data)
    max_distance = np.max(data)
    data = (data - min_distance) / (max_distance - min_distance)
    return data

def plot_eucledian_distancen(data, use_snn):

    

    num_episodes = len(data[0])
    num_runs = len(data)

    mean_trajectory = np.mean(data, axis=0)
    std_trajectory = np.std(data, axis=0)

    if not use_snn:
        plt.plot(range(num_episodes), mean_trajectory, color='red', label='Normalized E.d.')
        plt.fill_between(range(num_episodes), mean_trajectory - std_trajectory, mean_trajectory + std_trajectory,
                 color='coral', alpha=0.4, label='E.d. variance')
        
    else:
        plt.plot(range(num_episodes), mean_trajectory, color='blueviolet', label='Normalized E.d.')
        plt.fill_between(range(num_episodes), mean_trajectory - std_trajectory, mean_trajectory + std_trajectory,
                 color='violet', alpha=0.4, label='E.d. variance')

    # Adding labels and title
    plt.axhline(0, color='black', linestyle='--', label='goal')
    plt.xlabel('time step', fontsize=20)
    plt.ylabel('normalized distance', fontsize=20)
    plt.title('Normalized trajectory', fontsize=24)
    plt.legend(fontsize=20)
    plt.xlim(0, 500)
    plt.ylim(0, 1)

    plt.tick_params(axis='both', which='major', labelsize=20)

    # Displaying the plot
    plt.show()

def cnt_over_threshold(dist):
    
    cnt_over_five = 0
    cnt_over_four = 0
    cnt_over_three = 0
    cnt_over_two = 0
    cnt_over_one = 0

    for el in dist:
        if el > 0.05:
            cnt_over_five += 1
        elif el > 0.04:
            cnt_over_four += 1
        elif el > 0.03:
            cnt_over_three += 1
        elif el > 0.02:
            cnt_over_two += 1
        elif el > 0.01:
            cnt_over_one += 1

    return [cnt_over_one, cnt_over_two, cnt_over_three, cnt_over_four, cnt_over_five]

def remove_el_by_threshold(dist, threshold):
    dist = np.array(dist)
    dist = dist[dist < threshold]
    return dist

def calc_accuracy(dist):
    dist_under_one = remove_el_by_threshold(dist, 0.01)
    dist_under_two = remove_el_by_threshold(dist, 0.02)
    dist_under_three = remove_el_by_threshold(dist, 0.03)
    dist_under_four = remove_el_by_threshold(dist, 0.04)
    dist_under_five = remove_el_by_threshold(dist, 0.05)

    return [float(np.mean(dist)), float(np.mean(dist_under_one)), float(np.mean(dist_under_two)), 
            float(np.mean(dist_under_three)), float(np.mean(dist_under_four)), 
            float(np.mean(dist_under_five))]

def caluclate_results(dist, accuracy):

    results = {}
    results["array_results"] = dist
    results["accuracy"] = calc_accuracy(accuracy)[0]
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
    return results

def save_results(results_dict, use_snn,id,nr, overwrite=False):

    # Create a directory for the results
    if not os.path.exists("eval_results"):
        os.mkdir("eval_results")
    
    # Create a directory for the results
    nn = "snn" if use_snn else "dnn"

    path = "eval_results/"+nn+"_"+str(id)+"_"+str(nr)
    if not os.path.exists(path):
        os.mkdir("eval_results/"+nn+"_"+str(id)+"_"+str(nr))
    else:
        if not overwrite:
            print("Directory already exists, alternative name will be used")
            for i in range(100):
                if not os.path.exists("eval_results/"+nn+"_"+str(id)+"_"+str(nr)+"_"+str(i)):
                    os.mkdir("eval_results/"+nn+"_"+str(id)+"_"+str(nr)+"_"+str(i))
                    path = "eval_results/"+nn+"_"+str(id)+"_"+str(nr)+"_"+str(i)
                    break
    
    # Save the results
    with open(path+"/results.json", 'w') as fp:
        json.dump(results_dict, fp, indent=2)

    print("Results saved to: " + path)

def plot_results(tracked_eef_pos, tracked_joint_pos, tracked_joint_vel, tracked_joint_acel, tracked_actions):
    # Plot the results
    

    if False:
        fig, eef_axs = plt.subplots()
        eef_axs.plot(tracked_eef_pos)
        eef_axs.axhline(0, color='black', linestyle='--', label='goal')
        eef_axs.set_title('End effector position')
        eef_axs.set_xlabel('time step')
        eef_axs.set_ylabel('meter', fontsize=14)
        eef_axs.legend(['x', 'y', 'z', 'eucldian error', 'goal'])
        plt.show()
        

    if True:
        fig, q_axs = plt.subplots()
        q_axs.plot(tracked_joint_pos)
        q_axs.set_title('Joint position', fontsize=18)
        q_axs.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], fontsize=15)
        q_axs.set_xlabel('time step', fontsize=15)
        q_axs.set_ylabel('rad', fontsize=15)
        q_axs.tick_params(axis='both', which='major', labelsize=15)
        q_axs.set_xlim(0, 500)
        q_axs.set_ylim(-3.5, 3.5)
        plt.show()

    if True:
        fig, qd_axs = plt.subplots()
        qd_axs.plot(tracked_joint_vel)
        qd_axs.set_title('Joint velocities', fontsize=18)
        qd_axs.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], fontsize=15)
        qd_axs.set_xlabel('time step', fontsize=15)
        qd_axs.set_ylabel('rad/s', fontsize=15)
        qd_axs.tick_params(axis='both', which='major', labelsize=15)
        qd_axs.set_xlim(0, 500)
        qd_axs.set_ylim(-1, 1)
        plt.show()

    if True:
        fig, qa_axs = plt.subplots()
        qa_axs.plot(tracked_actions)
        qa_axs.set_title('Agent output', fontsize=18)
        qa_axs.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'], fontsize=15)
        qa_axs.set_xlabel('time step', fontsize=15)
        qa_axs.set_ylabel('rad/s', fontsize=15)
        qa_axs.tick_params(axis='both', which='major', labelsize=15)
        qa_axs.set_xlim(0, 500)
        qa_axs.set_ylim(-1, 1)
        plt.show()

    if False:
        fig, qds_axs = plt.subplots()
        qds_axs.plot(single)
        qds_axs.set_title('Joint velocity', fontsize=18)
        qds_axs.legend(['joint 2'], fontsize=15)
        qds_axs.set_xlabel('time step', fontsize=15)
        qds_axs.set_ylabel('rad/s', fontsize=15)
        qds_axs.tick_params(axis='both', which='major', labelsize=15)
        qds_axs.set_xlim(0, 500)
        qds_axs.set_ylim(-1, 1)
        plt.show()

    if False:
        fig, qdd_axs = plt.subplots()
        qdd_axs.plot(tracked_joint_acel)
        qdd_axs.set_title('Joint acceleration')
        qdd_axs.legend(['joint 1', 'joint 2', 'joint 3', 'joint 4', 'joint 5', 'joint 6'])
        qdd_axs.set_xlabel('time step')
        qdd_axs.set_ylabel('rad/s²')
        plt.show()

def total_variation(joint_angles):
    diff = np.diff(joint_angles, axis=0)
    tv = np.sum(np.abs(diff), axis=0)
    return tv


def calculate_smoothness(joint_angles):
    
    joint_angles = np.array(joint_angles)
    total_variation_per_joint = total_variation(joint_angles)

    vars = []

    for i, tv in enumerate(total_variation_per_joint):
        vars.append(tv)
        print(f"Total Variation for Joint {i+1}: {tv:.2f} radians")

    print(np.average(vars))