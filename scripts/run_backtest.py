# scripts/run_backtest.py

import argparse
from src.pipeline.backtester import run_backtest

def parse_args():
    parser = argparse.ArgumentParser(description="Run OmniGNN backtest")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_backtest(config_path=args.config)