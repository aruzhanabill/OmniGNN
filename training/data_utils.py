import torch

def normalize_tensor_per_node(X):
        if X.dim() == 3:  # (T, N, F)
            mu = X.mean(dim=0) # (N, F)
            sigma = X.std(dim=0) + 1e-8 # (N, F)
            X_norm = (X - mu.unsqueeze(0)) / sigma.unsqueeze(0)
        elif X.dim() == 4: # (B, W, N, F)
            mu = X.mean(dim=(0, 1)) # (N, F)
            sigma = X.std(dim=(0, 1)) + 1e-8
            X_norm = (X - mu.unsqueeze(0).unsqueeze(0)) / sigma.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Expected input of shape (T, N, F) or (B, W, N, F)")
        return X_norm, mu, sigma

def scale_labels_tensor_per_node(y):
        mu = y.mean(dim=0) # (N,)
        sigma = y.std(dim=0) + 1e-8 # (N,)
        y_norm = (y - mu.unsqueeze(0)) / sigma.unsqueeze(0)
        return y_norm, mu, sigma

def normalize_edges_minmax(edge_time, meta_paths, idx_train):
        """
        Min-max normalize edges per relation type and per edge dimension based on training indices.
        """
        edge_time_norm = {}
        for mp in meta_paths:
            edges = edge_time[mp]  # (T, N, N, E)
            train_edges = edges[idx_train]  # (T_train, N, N, E)

            min_val = train_edges.amin(dim=(0,1,2), keepdim=True)  # (1,1,1,E)
            max_val = train_edges.amax(dim=(0,1,2), keepdim=True)

            denom = (max_val - min_val).clamp(min=1e-8)

            normalized_edges = (edges - min_val) / denom
            edge_time_norm[mp] = normalized_edges.float()

            print(f"[Edge Normalization] {mp}: min={min_val.flatten().cpu().numpy()}, max={max_val.flatten().cpu().numpy()}")
        
        return edge_time_norm

def prepare_data(feats, labels, start_idx, end_idx, adj_time, edge_time, meta_paths, window_size):
        """Extract rolling windows from [start_idx, end_idx)."""

        X, y = [], []
        adj_samples  = {mp: [] for mp in meta_paths}
        edge_samples = {mp: [] for mp in meta_paths}

        for t in range(start_idx + window_size, end_idx):
            X.append(feats[t - window_size:t])  # W x N x F
            y.append(labels[t])                      # N
            for mp in meta_paths:
                adj_samples[mp].append(adj_time[mp][t - window_size:t])
                edge_samples[mp].append(edge_time[mp][t - window_size:t])

        X = torch.stack(X).float() # B x W x N x F
        y = torch.stack(y).float() # B x N
        adj = {mp: torch.stack(adj_samples[mp]).float() for mp in meta_paths}   # B x W x N x N
        edge = {mp: torch.stack(edge_samples[mp]).float() for mp in meta_paths} # B x W x N x N x E
        return X, y, adj, edge       