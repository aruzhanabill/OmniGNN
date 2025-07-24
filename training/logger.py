import logging
import os

def setup_logging(log_dir="logs", log_filename="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level to INFO
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),   # Log to file
            logging.StreamHandler()          # Also log to console
        ]
    )