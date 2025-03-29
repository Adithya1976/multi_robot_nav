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
custom_env = VOEnv(world_name=custom_env_world, obs_mode = 'ground_truth', display=False)


parser = argparse.ArgumentParser(description='drl rvo parameters')

par_env = parser.add_argument_group('par env', 'environment parameters') 
par_env.add_argument('--env_name', default='mrnav-v1')
par_env.add_argument('--world_path', default='train_world.yaml') # dynamic_obs_test.yaml; train_world.yaml
par_env.add_argument('--robot_number', type=int, default=4)
par_env.add_argument('--init_mode', default=3)
par_env.add_argument('--reset_mode', default=3)
par_env.add_argument('--mpi', default=False)

par_env.add_argument('--neighbors_region', default=4)
par_env.add_argument('--neighbors_num', type=int, default=5)   
par_env.add_argument('--reward_parameter', type=float, default=(3.0, 0.3, 0.0, 6.0, 0.3, 3.0, -0, 0), nargs='+')
par_env.add_argument('--env_train', default=True)
par_env.add_argument('--random_bear', default=False)
par_env.add_argument('--random_radius', default=False)
par_env.add_argument('--full', default=False)
parser.add_argument('--dis_mode', type=int, default='3')
args = parser.parse_args()

model_path = '/Users/adithyasamudrala/multi_robot_nav/src/rl/policy_train/model_save/r4_2/r4_2_250.pt'

env = gym.make('mrnav-v1', world_name=existing_world_name, robot_number=args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=args.dis_mode, env_train=False, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, goal_threshold=0.2)


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
o_custom_list = custom_env.reset()

max_diff = float('-inf')
for i in range(100):
    action_list = []
    for i in range(4):
        action = get_action(o_list[i], model)
        action_list.append(action)

    o_list, r_list, d_list, i_list = env.step_ir(action_list, vel_type='omni')
    o_custom_list, r_custom_list, d_custom_list, i_custom_list = custom_env.step(action_list)

    env.render()

    print("rewards: ", r_list)
    print("custom rewards: ", r_custom_list)
    max_diff = max(np.max(np.abs(np.array(r_list) - np.array(r_custom_list))), max_diff)
    print("____________________")

    if any(d_list) == True:
        break
    if all(i_list) == True:
        break

print(max_diff)
# [observation_vo, vo_flag, exp_time, collision_flag, min_dis]
