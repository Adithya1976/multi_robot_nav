from pyexpat import model
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from rl.policy.dil_models.graph_neural_network import GraphNeuralNetwork
from rl.policy.dil_models.rnn import GRU, LSTM, BiGRU, BiLSTM
from rl.policy.dil_models.deep_sets import DeepSet
from rl.policy.dil_models.set_transformer import SetTransformer
from torch_geometric.data import Data, Batch

model_dict = {
    'LSTM': LSTM,
    'BiLSTM': BiLSTM,
    'GRU': GRU,
    'BiGRU': BiGRU,
    'SetTransformer': SetTransformer,
    'DeepSet': DeepSet,
    'GNN': GraphNeuralNetwork
}

class DILNet(nn.Module):
    def __init__(self, state_dim, input_dim, hidden_dim, device='mps', mode='GRU'):
        super().__init__()
        
        self.state_dim = state_dim
        self.mode = mode
        self.device = device

        if mode != 'GNN':
            self.model = model_dict[mode](input_dim, hidden_dim, device=device)
        else:
            self.model = model_dict[mode](embed_dim=32, gnn_hidden_dim=256, state_embed_dim=256, device=device)
        self.model = self.model.to(device)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        des_dim = state_dim + hidden_dim
        self.ln = nn.LayerNorm(des_dim)

        self.model = self.model.to(device)
        self.ln = self.ln.to(device)
        
        self.robot_encoder = EgoEncoder(state_dim, device=device)
        self.obstacle_encoder = ObstacleEncoder(input_dim, device=device)
        self.robot_encoder = self.robot_encoder.to(device)
        self.obstacle_encoder = self.obstacle_encoder.to(device)

    def obs(self, obs):
        if self.mode == "GNN":
            return self.obs_gnn(obs)

        obs = torch.as_tensor(obs, dtype=torch.float32)  

        obs=obs.to(self.device) 

        moving_state = obs[self.state_dim:]
        robot_state = obs[:self.state_dim]
        mov_len = int(moving_state.size()[0] / self.input_dim)
        dilnet_input = torch.reshape(moving_state, (1, mov_len, self.input_dim))

        model_out = self.model(dilnet_input)
        
        out = torch.cat((robot_state, model_out))
        out = self.ln(out)

        return out  

    def obs_list(self, obs_list):

        if self.mode == "GNN":
            return self.obs_list_gnn(obs_list)
        
        mov_lens = [int((len(obs_tensor)-self.state_dim)/self.input_dim) for obs_tensor in obs_list]
        obs_pad = pad_sequence(obs_list, batch_first = True)
        robot_state_batch = obs_pad[:, :self.state_dim] 

        robot_state_batch=robot_state_batch.to(self.device)

        def obs_tensor_reform(obs_tensor):
            mov_tensor = obs_tensor[self.state_dim:]
            mov_tensor_len = int(len(mov_tensor)/self.input_dim)
            re_mov_tensor = torch.reshape(mov_tensor, (mov_tensor_len, self.input_dim)) 
            return re_mov_tensor
        
        re_mov_list = list(map(lambda o: obs_tensor_reform(o), obs_list))
        re_mov_pad = pad_sequence(re_mov_list, batch_first = True)

        re_mov_pad=re_mov_pad.to(self.device)

        if self.mode == 'SetTransformer':
            # Create a key_padding_mask for the padded moving state.
            # Shape: (batch_size, max_mov_len) with True indicating padded positions.
            batch_size, max_len, _ = re_mov_pad.size()
            key_padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
            for i, l in enumerate(mov_lens):
                if l < max_len:
                    key_padding_mask[i, l:] = True
            key_padding_mask = key_padding_mask.to(self.device)
            dilnet_input = re_mov_pad  # Use the padded tensor directly.
            model_out = self.model(dilnet_input, key_padding_mask=key_padding_mask)
        else:
            # For other models, use pack_padded_sequence.
            dilnet_input = pack_padded_sequence(re_mov_pad, mov_lens, batch_first=True, enforce_sorted=False)
            model_out = self.model(dilnet_input)
        
        fc_obs_batch = torch.cat((robot_state_batch, model_out), 1)
        fc_obs_batch = self.ln(fc_obs_batch)

        return fc_obs_batch

    def obs_gnn(self, obs):
        robot_state = obs[:self.state_dim]
        moving_state = obs[self.state_dim:]
        mov_len = int(moving_state.size()[0] / self.input_dim)
        # list of tensers of length input_dim
        mov_list = torch.split(moving_state, self.input_dim)

        robot_state = robot_state.to(self.device)

        robot_state_embed = self.robot_encoder(robot_state)
        obs_embed_list = [self.obstacle_encoder(mov.to(self.device)) for mov in mov_list]
        nodes = [robot_state_embed]
        nodes.extend(obs_embed_list)
        nodes = torch.stack(nodes)
        start_idxes = [0]*len(obs_embed_list)
        end_idxes = list(range(1, len(obs_embed_list)+1))
        edges = torch.tensor([start_idxes, end_idxes])
        data = Data(x=nodes, edge_index=edges).to(self.device)
        model_out = self.model(data.x, data.edge_index, data.batch)
        return model_out.squeeze()
    
    def obs_list_gnn(self, obs_list):
        data_list = []
        for obs in obs_list:
            robot_state = obs[:self.state_dim]
            moving_state = obs[self.state_dim:]
            mov_len = int(moving_state.size()[0] / self.input_dim)
            # list of tensers of length input_dim
            mov_list = torch.split(moving_state, self.input_dim)

            robot_state = robot_state.to(self.device)

            robot_state_embed = self.robot_encoder(robot_state)
            obs_embed_list = [self.obstacle_encoder(mov.to(self.device)) for mov in mov_list]
            nodes = [robot_state_embed]
            nodes.extend(obs_embed_list)
            nodes = torch.stack(nodes)
            start_idxes = [0]*len(obs_embed_list)
            end_idxes = list(range(1, len(obs_embed_list)+1))
            edges = torch.tensor([start_idxes, end_idxes])
            data = Data(x=nodes, edge_index=edges).to(self.device)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        model_out = self.model(batch.x, batch.edge_index, data.batch)
        return model_out

class EgoEncoder(nn.Module):
    def __init__(self, input_dim=6, embed_dim=32, device = 'mps'):
        """
        MLP encoder for ego-robot node features in a graph.
        - input_dim: Dimension of input node features (default 6).
        - embed_dim: Dimension of the output embedding.
        """
        super(EgoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.encoder = self.encoder.to(
            device
        )
    
    def forward(self, x):
        """
        Forward pass for the encoder.
        - x: Input node features (shape [num_nodes, input_dim]).
        """
        return self.encoder(x)

class ObstacleEncoder(nn.Module):
    def __init__(self, input_dim=8, embed_dim=32, device = 'mps'):
        """
        MLP encoder for obstacle node features in a graph.
        - input_dim: Dimension of input node features (default 8).
        - embed_dim: Dimension of the output embedding.
        """
        super(ObstacleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU()
        )
        self.encoder = self.encoder.to(
            device
        )
    
    def forward(self, x):
        """
        Forward pass for the encoder.
        - x: Input node features (shape [num_nodes, input_dim]).
        """
        return self.encoder(x)