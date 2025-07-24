import torch
from torch import nn
import torch.nn.functional as F

from layers import (
    GraphAttentionLayer,
    OmniGNNLayer,
    TemporalEncoder,
    NodePredictor,
)

class OmniGNN(nn.Module):
    """
    The full OmniGNN model that processes multirelational graph snapshots over time.
    """
    def __init__(
        self,
        in_features,
        hidden_dim,
        edge_attr_dim,
        meta_paths,
        temporal_heads=8,
        temporal_layers=4,
        pred_output_dim=1,
        window_size=12,
        n_nodes=11,
        n_stocks=10
    ): 
        super().__init__()
        self.window_size = window_size
        self.meta_paths = meta_paths
        self.n_nodes = n_nodes
        self.n_stocks = n_stocks

        # Layer 1: Multi-relational hierarchical graph embedding layer(s)
        self.gat_layers = nn.ModuleList([
            OmniGNNLayer(in_features, hidden_dim, n_heads=2, edge_attr_dim=edge_attr_dim, meta_paths=meta_paths),
            OmniGNNLayer(hidden_dim, hidden_dim, n_heads=2, edge_attr_dim=edge_attr_dim, meta_paths=meta_paths),
            OmniGNNLayer(hidden_dim, hidden_dim, n_heads=2, edge_attr_dim=edge_attr_dim, meta_paths=meta_paths)
        ])

        # Layer 2: Interday temporal extraction layer
        self.temporal_encoder = TemporalEncoder(embed_dim=hidden_dim, n_heads=temporal_heads, n_layers=temporal_layers)

        # Layer 3: Prediction Layer
        self.node_predictor = NodePredictor(input_dim=hidden_dim, output_dim=pred_output_dim, n_nodes=n_stocks)

    def forward(self, history_features, adj_matrices, edge_attrs):
        """
        history_features: (B, window, n_nodes, in_features)
        adj_matrices: dict of {meta_path: (window, n_nodes, n_nodes)}
        edge_attrs: dict of {meta_path: (window, n_nodes, n_nodes, edge_attr_dim)}
        """
        B, window, n_nodes, _ = history_features.shape
        embeddings = []

        for t in range(self.window_size):
            x_t = history_features[:, t] # (B, N, F)
            adj_t_dict = {mp: adj_matrices[mp][:, t] for mp in self.meta_paths} # (B, N, N)
            edge_t_dict = {mp: edge_attrs[mp][:, t] for mp in self.meta_paths} # (B, N, N, E)

            h = x_t
            for layer in self.gat_layers:
                h = F.elu(layer(h, adj_t_dict, edge_t_dict))

            embeddings.append(h.unsqueeze(1))

        past_embeddings = torch.cat(embeddings, dim=1)  # (B, W, N, HiddenDim)

        alibi_bias = self.temporal_encoder.layers[0].get_alibi_bias(self.window_size, history_features.device).unsqueeze(0)

        z_t = torch.stack([
            self.temporal_encoder(past_embeddings[:, :, i, :], alibi_bias=alibi_bias)
            for i in range(n_nodes)
        ], dim=1)

        return self.node_predictor(z_t)  # (B, N)