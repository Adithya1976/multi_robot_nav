import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Box
from torch.distributions.normal import Normal

from rl.policy.dil_models.dil_net import DILNet

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []

    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    return nn.Sequential(*layers)


class ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, state_dim=8, dilnet_input_dim=4, 
    dilnet_hidden_dim=64, hidden_sizes_ac=(256, 256), hidden_sizes_v=(256, 256), 
    activation=nn.ReLU, output_activation=nn.Tanh, output_activation_v= nn.Identity, device='mps', mode='GRU', drop_p=0):
        super().__init__()

        self.device = device
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()
        
        obs_dim = (dilnet_hidden_dim + state_dim)

        dil_net = DILNet(state_dim, dilnet_input_dim, dilnet_hidden_dim, device=device, mode=mode)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = GaussianActor(obs_dim, action_space.shape[0], hidden_sizes_ac, activation, output_activation, dil_net=dil_net, device=device)

        # build value function
        self.v = Critic(obs_dim, hidden_sizes_v, activation, output_activation_v, dil_net=dil_net, device=device)


    def step(self, obs, std_factor=1):
        with torch.no_grad():
            pi_dis = self.pi._distribution(obs, std_factor)
            a = pi_dis.sample()
            logp_a = self.pi._log_prob_from_distribution(pi_dis, a)
            v = self.v(obs)

            if self.device != 'cpu':
                a = a.cpu()
                logp_a = logp_a.cpu()
                v = v.cpu()

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs, std_factor=1):
        return self.step(obs, std_factor)[0]

class GaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation, dil_net=None, device='mps'):
        super().__init__()

        self.dil_net = dil_net
        self.device = device
        self.net_out=mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

        log_std = -1 * np.ones(act_dim, dtype=np.float32)

        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std, device=torch.device(self.device)))
        self.net_out=self.net_out.to(self.device)

    def _distribution(self, obs, std_factor=1):

        if isinstance(obs, list):
            obs = self.dil_net.obs_list(obs)
            net_out = self.net_out(obs)
        else:
            obs = self.dil_net.obs(obs)
            net_out = self.net_out(obs)
        
        mu = net_out 
        std = torch.exp(self.log_std)
        std = std_factor * std
        
        return Normal(mu, std)
        
    def _log_prob_from_distribution(self, pi, act):

        act = act.to(self.device)

        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
    def forward(self, obs, act=None, std_factor=1):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, std_factor)
        logp_a = None

        if act is not None:   
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, output_activation, dil_net=None, device='mps'):
        super().__init__()
        self.device = device

        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation, output_activation)
        self.v_net = self.v_net.to(self.device)

        self.dil_net = dil_net

    def forward(self, obs):

        if isinstance(obs, list):
            obs = self.dil_net.obs_list(obs)
        else:
            obs = self.dil_net.obs(obs)
        v = torch.squeeze(self.v_net(obs), -1)

        return v 