import os
import csv
import pandas as pd
import torch
from evaluation.metrics import evaluate_inverse
from evaluation.plot import plot_pred_vs_true, plot_loss_curve

def write_metrics(path, tag, t0, t3, train_res, test_res, write_header):
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Tag", "WindowStart", "WindowEnd", "Split", 
                                "MSE", "SignAcc", "Pearson", "IC", "IR", "CR", "Prec@K"])
            for split_name, res in [("Train", train_res), ("Test", test_res)]:
                writer.writerow([tag, t0.date(), t3.date(), split_name,
                                res["mse"], res["sign_accuracy"], res["pearson"],
                                res["ic"], res["ir"], res["cr"], res["prec@k"]])
                
def save_test_predictions(y_true, y_pred, idx_test, all_dates, stock_keys, window_start, window_end, window_size, path, write_header):
    """
    Save predictions vs true labels to CSV.
    """
    rows = []
    num_samples, num_stocks = y_true.shape
    test_dates = all_dates.values[idx_test.numpy()]

    for i in range(num_samples):
        date = pd.Timestamp(test_dates[i + window_size]).date()
        for j in range(num_stocks):
            rows.append({
                "WindowStart": window_start,
                "WindowEnd": window_end,
                "Date": date,
                "Stock": stock_keys[j],
                "y_true": y_true[i, j],
                "y_pred": y_pred[i, j]
            })

    df_preds = pd.DataFrame(rows)
    df_preds.to_csv(path, mode="a", header=write_header, index=False)