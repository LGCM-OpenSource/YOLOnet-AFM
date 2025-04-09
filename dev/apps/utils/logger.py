# /home/arthur/lgcm/projects/Unet_AFM/dev/apps/utils/logger.py
import logging
import os
from datetime import datetime
from .data_path import create_dir # Assuming create_dir is in data_path.py

LOG_DIR = "logs"
create_dir(LOG_DIR)

def setup_logger(log_name='process_log', log_file=None, level=logging.INFO):
    """Sets up a logger instance."""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"{log_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level) # Set the minimum level of messages to handle

    # Prevent adding multiple handlers if logger already exists
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Create console handler (optional, for seeing logs in the terminal too)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Example of getting a logger instance
# You can call this function from other modules to get the same logger
def get_logger(log_name='process_log'):
    """Gets the pre-configured logger instance."""
    return logging.getLogger(log_name)

