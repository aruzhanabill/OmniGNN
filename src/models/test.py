import torch
from omnignn import OmniGNN

# === Dummy Configuration ===
batch_size = 2
window_size = 12
n_nodes = 11
n_stocks = 10
in_features = 64
hidden_dim = 128
edge_attr_dim = 8
meta_paths = ['A', 'B', 'C']  # Example meta-path identifiers
pred_output_dim = 1

# === Dummy Inputs ===
history_features = torch.randn(batch_size, window_size, n_nodes, in_features)

adj_matrices = {
    mp: torch.randint(0, 2, (batch_size, window_size, n_nodes, n_nodes)).float()
    for mp in meta_paths
}

edge_attrs = {
    mp: torch.randn(batch_size, window_size, n_nodes, n_nodes, edge_attr_dim)
    for mp in meta_paths
}

# === Model Instantiation ===
model = OmniGNN(
    in_features,            # input feature dim
    hidden_dim,
    edge_attr_dim,
    meta_paths,
    temporal_heads=4,
    temporal_layers=2,
    pred_output_dim=pred_output_dim,
    window_size=window_size,
    n_nodes=n_nodes,
    n_stocks=n_stocks
)

# === Forward Pass ===
output = model(history_features, adj_matrices, edge_attrs)

print("Output shape:", output.shape)  # Expecting (B, N) or (B, N * pred_output_dim)
print("Output:", output)
