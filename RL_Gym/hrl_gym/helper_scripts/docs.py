import os, inspect
from os import path as path_os
import yaml
from shutil import copy2
       
def create_directory(agent_name, path):
    """
    Args:
        path: path of directory
       load: (bool) if true
    """

    path = path+"/trained_agents/"+agent_name
    try:
        if path_os.exists(path):
            raise Exception("Agent with the same name already exist, please choose another name")
        elif path_os.exists(path):
            os.rmdir(path)
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created directory: %s " % path)

def doc_config(algo,path,config):
    """
    Copies the belonging config data into the result directory
    """

    all_subdirs = []

    for dir in os.listdir(path+"/trained_agents/"+str(algo.upper())+"/"):
        if not "json" in dir:
            all_subdirs.append(dir)

    
    for idx, dir in enumerate(all_subdirs):
        all_subdirs[idx] = path+"/trained_agents/"+str(algo.upper())+"/" + dir

    try:
        dst = max(all_subdirs, key=os.path.getmtime)+"/config.yaml"
        with open(dst, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
            
    except FileNotFoundError:
        print("File not %s Found" % dst)

    else:
        print("Config successfully stored")

def load_config(main_dir):
    """
    Load a YAML into a dictionary

    Args:
        main_dir: Path to "*\HRL_Gym" directory
    """
        
    config_path = main_dir + "/config.yaml"

    try:
        with open(os.path.expanduser(config_path), 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
        
    return config

def load_hyperparam(main_dir, algo):
    """
    Loads hyperparameters from *\DeepRL_RobotGym\hyperparameters/ into a dictionary

    Args:
        main_dir (str): Path to "*\DeepRL_RobotGym" directory
        algo (str): name of the algorithm for which the hyperparametrs should be loaded (dqn, ppo or sac)
    """
        
    hyperparam_path = main_dir + "/hyperparameters/"+algo+".yaml"

    try:
        with open(os.path.expanduser(hyperparam_path), 'r') as file:
            hyperparam = yaml.load(file, Loader=yaml.FullLoader)
    except:
        print("Please ensure that at */DeepRL_RobotGym/hyperparameters a yaml-file with the name (only lowercase) of the algorithm is placed, e.g. dqn.yaml.")
        
    return hyperparam


def doc_training(folder_name, string):
    """
    Write a string to "training_progress.txt" of the latest changed directory inside the "*/DeepRL_RobotGym/trained_agents/" directory

    Args:
        algo (str): name of the algorithm for which the hyperparametrs should be loaded (dqn, ppo or sac)
        string: info that should be added to the text file
    """

    if os.path.isfile(folder_name+"/training_progress.txt"):
        file = open(folder_name+"/training_progress.txt", "a")
        file.write(string)
    else:
        file = open(folder_name+"/training_progress.txt", "w")
        file.write(string)
    file.write("\n")   
    file.close()        

def doc_config_snn(path):

    try:
        source_file = os.getcwd()+"/config.yaml"
        destination_folder = path

        if os.path.exists(path+"/config.yaml"):
            pass
        else:   
            copy2(source_file, destination_folder)
            print("Config successfully stored")
            
    except FileNotFoundError:
        print("File not %s Found" % path)







