import argparse
import gym
import numpy as np
import torch
import gym_env
from custrom_env.vo_env import VOEnv
from rl.policy.actor_critic import ActorCritic
import os
custom_env_world = 'custom_env.yaml'
existing_world_name = 'existing_env.yaml'
import matplotlib
# matplotlib.use('Agg')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
# env = VOEnv(world_name=custom_env_world, obs_mode = 'ground_truth', display=False)

model_path = '/Users/adithyasamudrala/multi_robot_nav/src/rl/policy_train/model_save/r4_63/r4_63_150.pt'

env = VOEnv(world_name=custom_env_world, obs_mode = 'lidar')

def load_policy(filename, policy_dict=True):
        model = torch.load(filename, map_location=torch.device("mps"))
        model.eval()

        return model

def get_action(x, model):
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32)
        action = model.act(x, 0.0001)
    return action

model = load_policy(model_path, policy_dict=True)

o_list = env.reset()

max_diff = float('-inf')
for i in range(100):
    action_list = []
    for i in range(4):
        a_inc = get_action(o_list[i], model)
        action = a_inc + np.squeeze(env.robot_list[i].velocity_xy)
        action_list.append(action)

    o_list, r_list, d_list, i_list = env.step(action_list)

    env.render()

    # print("rewards: ", r_list)
    # print("custom rewards: ", r_custom_list)
    # max_diff = max(np.max(np.abs(np.array(r_list) - np.array(r_custom_list))), max_diff)
    # print("____________________")

    if any(d_list) == True:
        break
    if all(i_list) == True:
        break

env.end()
# [observation_vo, vo_flag, exp_time, collision_flag, min_dis]
