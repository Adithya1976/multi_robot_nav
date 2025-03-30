import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphNeuralNetwork(nn.Module):
    def __init__(
        self, 
        embed_dim=32, 
        gnn_hidden_dim=256, 
        state_embed_dim=256, 
        device = 'mps'
    ):
        """
        PyTorch Geometric model for autonomous navigation in dynamic environments.
        - ego_feat_dim: Dimension of ego-robot node features (default 6).
        - obs_feat_dim: Dimension of obstacle node features (default 8).
        - embed_dim: Dimension of shared node embedding after encoders.
        - gnn_hidden_dim: Hidden dimension for GNN layers.
        - state_embed_dim: Dimension of the graph-level state embedding (also output of final GNN layer).
        - action_dim: Dimension of the continuous action output.
        """
        super(GraphNeuralNetwork, self).__init__()
        # Encoder MLPs for ego and obstacle features to shared embedding
        # Graph Neural Network layers (GraphSAGE in this example)
        self.gnn1 = SAGEConv(in_channels=embed_dim, out_channels=gnn_hidden_dim).to(device)
        self.gnn2 = SAGEConv(in_channels=gnn_hidden_dim, out_channels=state_embed_dim).to(device)
        # (Optional) If a deeper GNN is desired, more layers can be added similarly.
        # MLP heads for action (policy) and value (critic)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass for the model.
        Expects `data` to be a PyG Data or Batch object with:
          - data.x: node feature matrix (if all nodes have same feature length, e.g., 8 via padding).
          - data.edge_index: edge list (shape [2, num_edges]).
          - data.batch: batch indices for each node (if batching multiple graphs).
          - Optionally, data.ego_mask or a similar indicator to identify the ego node(s).
        """
        # Determine device and batch context
        x = self.gnn1(x, edge_index)
        x = x.relu()
        x = self.gnn2(x, edge_index)
        x = F.relu(x)
        
        # ** Graph-level pooling **
        # Aggregate node embeddings to a single graph embedding per graph (state embedding)
        graph_emb = global_mean_pool(x, batch)  # shape: [batch_size, state_embed_dim]
        
        return graph_emb

