import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr

def get_topk_indices(predictions, top_k):
    """Returns a boolean mask of top-K% predictions per day."""
    n_days, n_stocks = predictions.shape
    k = int(n_stocks * top_k)
    topk_mask = np.zeros_like(predictions, dtype=bool)
    partition_indices = np.argpartition(predictions, -k, axis=1)[:, -k:]
    for i in range(n_days):
        topk_mask[i, partition_indices[i]] = True
    return topk_mask

def compute_metrics(y_true, y_pred, top_k=0.3):
    topk_mask = get_topk_indices(y_pred, top_k)

    topk_returns = (y_true * topk_mask).sum(axis=1) / topk_mask.sum(axis=1)

    ir = topk_returns.mean() / (topk_returns.std() + 1e-8)

    cr = topk_returns.sum()

    precision_per_day = ((y_true > 0) & topk_mask).sum(axis=1) / topk_mask.sum(axis=1)
    prec_at_k = precision_per_day.mean()

    sign_match = (np.sign(y_true) == np.sign(y_pred))
    sign_accuracy = sign_match[y_true != 0].mean()

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    return {
        "mse": mean_squared_error(y_true_flat, y_pred_flat),
        "sign_accuracy": sign_accuracy,
        "pearson": pearsonr(y_true_flat, y_pred_flat)[0],
        "ic": spearmanr(y_pred_flat, y_true_flat).correlation,
        "ir": ir,
        "cr": cr,
        "prec@k": prec_at_k
    }

def evaluate_inverse(model, X, y_scaled, adj, edge, mu, sigma, n_stocks, meta_paths, device):
    """Inverse-transforms predictions, computes evaluation metrics."""
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for i in range(0, X.shape[0], X.shape[0] if X.shape[0] < 16 else 16):  # handles batch size
            x_b = X[i : i + 16].to(device)
            y_b = y_scaled[i : i + 16].to(device)
            adj_b = {mp: adj[mp][i : i + 16].to(device) for mp in meta_paths}
            edge_b = {mp: edge[mp][i : i + 16].to(device) for mp in meta_paths}

            out = model(x_b, adj_b, edge_b)[:, :n_stocks].cpu()
            y_b = y_b[:, :n_stocks].cpu()

            preds.append(out * sigma[:n_stocks] + mu[:n_stocks])
            trues.append(y_b * sigma[:n_stocks] + mu[:n_stocks])

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()

    metrics = compute_metrics(y_true, y_pred, top_k=0.3)
    return metrics, y_true, y_pred