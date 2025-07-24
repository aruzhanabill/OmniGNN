import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from omnignn.model import OmniGNN
from evaluation.metrics import evaluate_inverse
from evaluation.plot import plot_pred_vs_true, plot_loss_curve

torch.manual_seed(7)
np.random.seed(7)

class Backtester:
    """Rolling‑window back‑tester for a Multirelational Dynamic GNN."""
    def __init__(
        self,
        features_time,  # torch.Tensor (T, N, F) = 996 x 11 x 16
        labels_time,    # torch.Tensor (T, N) = 996 x 11
        adj_time,       # dict of T x N x N = {"SS": 996 x 11 x 11, ... }
        edge_time,      # dict of T x N x N x E = # {"SS": 996 x 11 x 11 x 2, ... }
        meta_paths,     # list of mp names = ["SS", "SIS"]
        all_dates,      # pd.Series of datetime
        window_size,    # int
        device,         # torch.device
        stock_names,    # list of stock names = ["MSFT", ... , "PANW"]
        train_months=6, 
        val_months=2,
        test_months=2, 
        batch_size=16
    ):
        # --- datasets -----------------------------------------------------
        self.features_time = features_time
        self.labels_time = labels_time
        self.adj_time = adj_time 
        self.edge_time = edge_time 
        self.meta_paths = meta_paths 
        self.all_dates = pd.to_datetime(all_dates)
        # --- rolling‑window / training cfg --------------------------------
        self.window_size = window_size
        self.device = device
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.batch_size = batch_size
        # --- bookkeeping --------------------------------------------------
        self.n_nodes = features_time.shape[1] # 11
        self.feature_dim = features_time.shape[2] # 16
        self.edge_dim = edge_time[meta_paths[0]].shape[-1] # 2
        self.stock_keys = stock_names # ["MSFT", ... , "PANW"]
        self.n_stocks = len(stock_names) # 10

    def _normalize_tensor_per_node(self, X):
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
    
    def _scale_labels_tensor_per_node(self, y):
        mu = y.mean(dim=0) # (N,)
        sigma = y.std(dim=0) + 1e-8 # (N,)
        y_norm = (y - mu.unsqueeze(0)) / sigma.unsqueeze(0)
        return y_norm, mu, sigma
    
    def _normalize_edges_minmax(self, edge_time, idx_train):
        """
        Min-max normalize edges per relation type and per edge dimension based on training indices.
        """
        edge_time_norm = {}
        for mp in self.meta_paths:
            edges = edge_time[mp]  # (T, N, N, E)
            train_edges = edges[idx_train]  # (T_train, N, N, E)

            min_val = train_edges.amin(dim=(0,1,2), keepdim=True)  # (1,1,1,E)
            max_val = train_edges.amax(dim=(0,1,2), keepdim=True)

            denom = (max_val - min_val).clamp(min=1e-8)

            normalized_edges = (edges - min_val) / denom
            edge_time_norm[mp] = normalized_edges.float()

            print(f"[Edge Normalization] {mp}: min={min_val.flatten().cpu().numpy()}, max={max_val.flatten().cpu().numpy()}")
        
        return edge_time_norm
    
    def _prepare_data(self, feats, labels, start_idx, end_idx, edge_time_override=None):
        """Extract rolling windows from [start_idx, end_idx)."""
        edge_source = edge_time_override if edge_time_override else self.edge_time

        X, y = [], []
        adj_samples  = {mp: [] for mp in self.meta_paths}
        edge_samples = {mp: [] for mp in self.meta_paths}

        for t in range(start_idx + self.window_size, end_idx):
            X.append(feats[t - self.window_size:t])  # W x N x F
            y.append(labels[t])                      # N
            for mp in self.meta_paths:
                adj_samples[mp].append(self.adj_time[mp][t - self.window_size:t])
                edge_samples[mp].append(edge_source[mp][t - self.window_size:t])

        X = torch.stack(X).float() # B x W x N x F
        y = torch.stack(y).float() # B x N
        adj = {mp: torch.stack(adj_samples[mp]).float() for mp in self.meta_paths}   # B x W x N x N
        edge = {mp: torch.stack(edge_samples[mp]).float() for mp in self.meta_paths} # B x W x N x N x E
        return X, y, adj, edge         
    
    def _build_omnignn(self):
        print(f"Initializing OmniGNN - Meta-paths: {self.meta_paths} | Edge dims: {self.edge_dim}")
        return OmniGNN(
            in_features=self.feature_dim,
            hidden_dim=32, # hyperparameter
            edge_attr_dim=self.edge_dim,
            meta_paths=self.meta_paths,
            window_size=self.window_size,
            n_nodes=self.n_nodes
        ).to(self.device)

    def _train_one_epoch(self, model, optimizer, criterion, X, y, adj, edge):
        model.train()
        total_loss = 0
        perm = torch.randperm(X.shape[0])
        for i in range(0, X.shape[0], self.batch_size):
            idx = perm[i:i + self.batch_size]
            x_batch = X[idx].to(self.device)
            y_batch = y[idx].to(self.device)
            adj_batch  = {mp: adj[mp][idx].to(self.device) for mp in self.meta_paths}
            edge_batch = {mp: edge[mp][idx].to(self.device) for mp in self.meta_paths}

            optimizer.zero_grad()
            outputs = model(x_batch, adj_batch, edge_batch)
            loss = criterion(outputs[:, :self.n_stocks], y_batch[:, :self.n_stocks])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        total_batches = max(1, (X.shape[0] + self.batch_size - 1) // self.batch_size)
        return total_loss / total_batches
    
    def _write_metrics(self, path, tag, t0, t3, train_res, test_res, write_header):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Tag", "WindowStart", "WindowEnd", "Split", 
                                "MSE", "SignAcc", "Pearson", "IC", "IR", "CR", "Prec@K"])
            for split_name, res in [("Train", train_res), ("Test", test_res)]:
                writer.writerow([tag, t0.date(), t3.date(), split_name,
                                res["mse"], res["sign_accuracy"], res["pearson"],
                                res["ic"], res["ir"], res["cr"], res["prec@k"]])

    # ---------------------------------------------------------------------
    # Public API -----------------------------------------------------------
    # ---------------------------------------------------------------------
    
    def run(self, epochs=10, lr=0.001):
        tag = "."
        results_dir = os.path.join("results", tag)
        os.makedirs(results_dir, exist_ok=True)
        predictions_path = os.path.join(results_dir, "predictions.csv")
        write_pred_header = not os.path.exists(predictions_path)

        t0 = self.all_dates[0]
        while True:
            t1 = t0 + pd.DateOffset(months=self.train_months)
            t2 = t1 + pd.DateOffset(months=self.val_months)
            t3 = t2 + pd.DateOffset(months=self.test_months)
            if t3 > self.all_dates[-1]: break

            print(f"\nRolling Window: Train {t0.date()} to {t1.date()} | Test {t2.date()} to {t3.date()} with tag {tag}")

            mask = lambda s, e: (self.all_dates >= s) & (self.all_dates <= e)
            train_mask = mask(t0, t2)
            val_mask = mask(t1, t2)
            test_mask = mask(t2, t3)

            feats = self.features_time
            labels = self.labels_time

            idx_train = torch.where(torch.tensor(train_mask))[0]
            idx_val = torch.where(torch.tensor(val_mask))[0]
            idx_test = torch.where(torch.tensor(test_mask))[0]

            feats_train = feats[idx_train]
            labels_train = labels[idx_train]
            feats_val = feats[idx_val]
            labels_val = labels[idx_val]
            feats_test = feats[idx_test]
            labels_test = labels[idx_test]

            feats_train, mu_f, sigma_f = self._normalize_tensor_per_node(feats_train)
            labels_train, mu_y, sigma_y = self._scale_labels_tensor_per_node(labels_train)
            feats_val = (feats_val - mu_f.unsqueeze(0)) / sigma_f.unsqueeze(0)
            labels_val = (labels_val - mu_y.unsqueeze(0)) / sigma_y.unsqueeze(0)
            feats_test = (feats_test - mu_f.unsqueeze(0)) / sigma_f.unsqueeze(0)
            labels_test = (labels_test - mu_y.unsqueeze(0)) / sigma_y.unsqueeze(0)
            
            s_train, e_train = 0, feats_train.shape[0]
            s_val, e_val = 0, feats_val.shape[0]
            s_test, e_test = 0, feats_test.shape[0]

            edge_norm = self._normalize_edges_minmax(self.edge_time, idx_train)
            train_X, train_y, adj, edge = self._prepare_data(feats_train, labels_train, s_train, e_train, edge_norm)
            val_X, val_y, val_adj, val_edge = self._prepare_data(feats_val, labels_val, s_val, e_val)
            test_X, test_y, test_adj, test_edge = self._prepare_data(feats_test, labels_test, s_test, e_test)

            model = self._build_omnignn()
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.MSELoss()
            loss_history, val_loss_history = [], []

            best_val_loss, patience, wait, best_model_state = float("inf"), 50, 0, None
            for epoch in range(epochs):
                train_loss = self._train_one_epoch(model, optimizer, criterion, train_X, train_y, adj, edge)
                model.eval()
                with torch.no_grad():
                    val_preds = model(val_X.to(self.device),
                                    {mp: val_adj[mp].to(self.device) for mp in self.meta_paths},
                                    {mp: val_edge[mp].to(self.device) for mp in self.meta_paths})
                    val_loss = criterion(val_preds[:, :self.n_stocks], val_y[:, :self.n_stocks].to(self.device)).item()
                loss_history.append(train_loss)
                val_loss_history.append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                # --- early stopping ---
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Early stopping at epoch {epoch+1} with best val loss: {best_val_loss:.4f}")
                        break
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                
            loss_plot_path = os.path.join(results_dir, f"Loss_{t2.date()}_to_{t3.date()}_{tag}.png")
            plot_loss_curve(loss_history, val_loss_history, save_path=loss_plot_path)

            print("\nFinal Training Evaluation:")
            train_res, _, _ = evaluate_inverse(model, train_X, train_y, adj, edge, mu_y, sigma_y, self.n_stocks, self.meta_paths, self.device)
            print(f"Train MSE: {train_res['mse']:.4f} | Sign Acc: {train_res['sign_accuracy']:.4f} | Pearson: {train_res['pearson']:.4f} |"
                  f"IC: {train_res['ic']:.4f} | IR: {train_res['ir']:.4f} | CR: {train_res['cr']:.4f} | Prec@K: {train_res['prec@k']:.4f}")
            print("\nFinal Test Evaluation:")
            test_res, y_true_te, y_pred_te = evaluate_inverse(model, test_X, test_y, test_adj, test_edge, mu_y, sigma_y, self.n_stocks, self.meta_paths, self.device)
            print(f"Test MSE: {test_res['mse']:.4f} | Sign Acc: {test_res['sign_accuracy']:.4f} | Pearson: {test_res['pearson']:.4f} |" 
                  f"IC: {test_res['ic']:.4f} | IR: {test_res['ir']:.4f} | CR: {test_res['cr']:.4f} | Prec@K: {test_res['prec@k']:.4f}")
            plot_path = os.path.join(results_dir, f"pred_vs_true_{t2.date()}_to_{t3.date()}_{tag}.png")
            plot_pred_vs_true(y_true_te, y_pred_te, self.stock_keys, save_path=plot_path)

            # cache predictions to csv in results/tag
            rows = []
            num_samples, num_stocks = y_true_te.shape
            idx_test_np = idx_test.numpy()
            test_dates = self.all_dates.values[idx_test_np]
            for i in range(num_samples):
                date = pd.Timestamp(test_dates[i + self.window_size]).date()
                for j in range(num_stocks):
                    rows.append({
                        "WindowStart": t2.date(),
                        "WindowEnd": t3.date(),
                        "Date": date,
                        "Stock": self.stock_keys[j],
                        "y_true": y_true_te[i, j],
                        "y_pred": y_pred_te[i, j]
                    })
            df_preds = pd.DataFrame(rows)
            df_preds.to_csv(predictions_path, mode="a", header=write_pred_header, index=False)
            write_pred_header = False # only write the header once

            metrics_path = os.path.join("results/metrics_log.csv")
            write_header = not os.path.exists(metrics_path)

            self._write_metrics(metrics_path, tag, t0, t3, train_res, test_res, write_header)

            t0 = t0 + pd.DateOffset(months=self.test_months)