import argparse
import gym
import torch

from custrom_env.vo_env import VOEnv
from rl.policy.actor_critic import ActorCritic

custom_env_world = 'custem_env.yaml'
existing_world_name = 'existing_world.yaml'
import matplotlib
matplotlib.use('Agg')
env = VOEnv(world_name=custom_env_world, obs_mode = 'ground_truth')


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
args = parser.parse_args(['--train_epoch', '250'])

model_path = '/Users/adithyasamudrala/multi_robot_nav/src/rl/policy_train/model_save/r4_2/r4_2_check_point_250.pt'

existing_env = gym.make('mrnav-v1', world_name=existing_world_name, robot_number=args.robot_number, neighbors_region=args.neighbors_region, neighbors_num=args.neighbors_num, robot_init_mode=args.dis_mode, env_train=False, random_bear=args.random_bear, random_radius=args.random_radius, reward_parameter=args.reward_parameter, goal_threshold=0.2)


def load_policy(self, filename, policy_dict=True):

        if policy_dict == True:
            model = ActorCritic(env.observation_space, env.action_space, self.args.state_dim, self.args.dilnet_input_dim, self.args.dilnet_hidden_dim, self.args.hidden_sizes_ac, self.args.hidden_sizes_v, self.args.activation, self.args.output_activation, self.args.output_activation_v, self.device, self.args.mode)
        
            check_point = torch.load(filename, map_location=torch.device("mps"))
            model.load_state_dict(check_point['model_state'], strict=True)
            model.eval()

        else:
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

for i in range(100):
    action_list = []
    for i in range(env.robot_number):
        action = get_action(env.observation_list[i], model)
        action_list.append(action)
