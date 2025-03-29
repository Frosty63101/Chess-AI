"""
config.py

This file stores global constants and configuration settings used
across the entire project.
"""

import os

# Path to save the trained chess model
MODEL_PATH = "chess_model.pth"

# Folder for model backups
BACKUP_MODEL_PATH = "backup_models"

# Default path to your Stockfish binary
STOCKFISH_DEFAULT_PATH = os.path.join(
    "stockfish-windows-x86-64-avx2",
    "stockfish",
    "stockfish-windows-x86-64-avx2.exe"
)

# Directory where the piece images are stored
IMAGE_DIR = "images"

# Total possible moves in our action space (move_to_index(...) sized)
ACTION_SIZE = 4672

# Min and max evaluation bounds
MIN_EVAL = -10.0
MAX_EVAL = 10.0

# Maximum search depth for killer moves
MAX_DEPTH = 100

# Max time in seconds for move selection
MAX_TIME = 3.0

# If your system has a GPU, we will prefer using "cuda" automatically in the code
