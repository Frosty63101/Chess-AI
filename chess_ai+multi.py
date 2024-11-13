import os
import sys
import math
import random
import threading
import tkinter as tk
from tkinter import messagebox
from collections import deque, defaultdict
from tracemalloc import start
from typing import Dict, Optional

import chess
from keyboard import replay
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from PIL import Image, ImageTk
from stockfish import Stockfish
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.amp.grad_scaler import GradScaler
from chess.polyglot import zobrist_hash

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue  # For logging

import time

print(chess.__version__)

# Constants
MODEL_PATH = "chess_model.pth"
BACKUP_MODEL_PATH = "backup_models/"
STOCKFISH_DEFAULT_PATH = "stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"  # Update this path as needed
IMAGE_DIR = "images"  # Directory containing piece images
ACTION_SIZE = 4672  # Total possible moves in our action space
MIN_EVAL = -10.0  # Defined min evaluation
MAX_EVAL = 10.0   # Defined max evaluation
MAX_DEPTH = 100   # Maximum depth for killer moves
MAX_TIME = 1.0    # Max time in seconds for move selection

# Global model instance
model_instance = None  # Will be initialized in main()

# Locks for thread safety
transposition_table_lock = threading.Lock()
eval_cache_lock = threading.Lock()
board_tensor_cache_lock = threading.Lock()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

def iterative_deepening(board: chess.Board, ai: 'ChessAI', max_time: float = MAX_TIME) -> Optional[chess.Move]:
    """Performs iterative deepening search within a time limit."""
    best_move = None
    depth = 1
    start_time = time.time()
    while True:
        time_elapsed = time.time() - start_time
        if time_elapsed >= max_time:
            break
        ai.search_stop = False
        try:
            best_move = ai.search(board, depth, start_time, max_time)
        except TimeoutError:
            break
        depth += 1
    return best_move

def principal_variation_search(board: chess.Board, depth: int, alpha: float, beta: float, maximizing_player: bool, ai: 'ChessAI', start_time: float, max_time: float) -> float:
    """Principal Variation Search algorithm with alpha-beta pruning and time management."""
    # Check for timeout
    if time.time() - start_time >= max_time or ai.search_stop:
        ai.search_stop = True
        raise TimeoutError("Search timed out")

    # Transposition Table Lookup
    transposition_key = zobrist_hash(board)
    with transposition_table_lock:
        if transposition_key in ai.transposition_table:
            entry = ai.transposition_table[transposition_key]
            if entry['depth'] >= depth:
                return entry['value']

    if board.is_game_over() or depth == 0:
        eval = quiescence_search(board, alpha, beta, ai, start_time, max_time)
        with transposition_table_lock:
            ai.transposition_table[transposition_key] = {'value': eval, 'depth': depth}
        return eval

    moves = ai.order_moves(board, depth)
    if not moves:
        return evaluate_board(board, ai)

    first_child = True
    for move in moves:
        board.push(move)
        try:
            if first_child:
                score = -principal_variation_search(board, depth - 1, -beta, -alpha, not maximizing_player, ai, start_time, max_time)
            else:
                score = -principal_variation_search(board, depth - 1, -alpha - 1, -alpha, not maximizing_player, ai, start_time, max_time)
                if alpha < score < beta:
                    score = -principal_variation_search(board, depth - 1, -beta, -score, not maximizing_player, ai, start_time, max_time)
        except TimeoutError:
            board.pop()
            raise TimeoutError("Search timed out")
        board.pop()

        if score >= beta:
            with transposition_table_lock:
                ai.transposition_table[transposition_key] = {'value': beta, 'depth': depth}
            return beta
        if score > alpha:
            alpha = score
        first_child = False

        # Check for timeout after each move
        if time.time() - start_time >= max_time or ai.search_stop:
            ai.search_stop = True
            raise TimeoutError("Search timed out")

    with transposition_table_lock:
        ai.transposition_table[transposition_key] = {'value': alpha, 'depth': depth}
    return alpha

def quiescence_search(board: chess.Board, alpha: float, beta: float, ai: 'ChessAI', start_time: float, max_time: float) -> float:
    """Extends the search at leaf nodes to capture sequences to avoid the horizon effect."""
    # Check for timeout
    if time.time() - start_time >= max_time or ai.search_stop:
        ai.search_stop = True
        raise TimeoutError("Search timed out")

    stand_pat = evaluate_board(board, ai)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in ai.order_moves_quiescence(board):
        board.push(move)
        try:
            score = -quiescence_search(board, -beta, -alpha, ai, start_time, max_time)
        except TimeoutError:
            board.pop()
            raise TimeoutError("Search timed out")
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def evaluate_board(board: chess.Board, ai: 'ChessAI') -> float:
    """Evaluates the board using material balance and the neural network, with caching."""
    board_hash = zobrist_hash(board)
    with eval_cache_lock:
        if board_hash in ai.eval_cache:
            return ai.eval_cache[board_hash]

    # Material evaluation
    material_score = material_evaluation(board)

    # Neural network evaluation
    state = board_to_tensor_cached(board, ai)
    with torch.no_grad():
        _, value = ai.model(state)  # type: ignore
        nn_eval = value.item()

    # Clamp neural network evaluation between -10 and 10
    nn_eval = max(min(nn_eval, MAX_EVAL), MIN_EVAL)

    # Combine evaluations
    eval_score = nn_eval + material_score
    eval_score = max(min(eval_score, MAX_EVAL), MIN_EVAL)

    # Cache the evaluation
    with eval_cache_lock:
        ai.eval_cache[board_hash] = eval_score

    return eval_score

# Cache for board tensor conversions
board_tensor_cache = {}

def board_to_tensor_cached(board: chess.Board, ai: 'ChessAI') -> torch.Tensor:
    """Converts a chess board to a tensor representation with caching."""
    board_hash = zobrist_hash(board)
    with board_tensor_cache_lock:
        if board_hash in board_tensor_cache:
            return board_tensor_cache[board_hash]
        tensor = board_to_tensor(board).unsqueeze(0).to(ai.device)
        board_tensor_cache[board_hash] = tensor
    return tensor

class ChessNet(nn.Module):
    """Neural network model for evaluating chess positions."""
    def __init__(self):
        super(ChessNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_block = nn.Sequential(
            nn.Conv2d(13, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(256) for _ in range(19)
        ])
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, ACTION_SIZE)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def initialize_model(device: torch.device):
    """Initializes the global model instance and loads pre-trained weights if available."""
    global model_instance
    model_instance = ChessNet()
    print(f"Using device: {device}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model_instance = nn.DataParallel(model_instance)
    model_instance.to(device)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            if isinstance(model_instance, nn.DataParallel):
                model_instance.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model_instance.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a fresh model.")
    else:
        print("No saved model found. Starting with a fresh model.")

class ChessAI:
    """Encapsulates the chess AI functionalities, including the neural network, and move selection."""
    def __init__(self, stockfish_path: str, device: torch.device):
        self.device = device
        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.set_skill_level(10)
        self.stockfish.update_engine_parameters({
            "Threads": 3,
            "Minimum Thinking Time": 5,
            "Hash": 16,
            "UCI_LimitStrength": True,
            "UCI_Elo": 1350  # Adjust as needed
        })

        # Initialize history and PV tables
        self.pv_table: Dict[str, chess.Move] = {}
        self.history_table: Dict[chess.Move, int] = defaultdict(int)
        self.model = model_instance
        self.transposition_table = {}
        self.eval_cache = {}
        self.killer_moves = [[] for _ in range(MAX_DEPTH)]
        self.search_stop = False
        self.max_workers = 4

    def get_stockfish_skill(self):
        return self.stockfish.get_parameters()['Skill Level']

    def select_best_move(self, board: chess.Board, max_time: float = MAX_TIME) -> Optional[chess.Move]:
        """Selects the best move using iterative deepening and PVS within a time limit."""
        return iterative_deepening(board, self, max_time)

    def search(self, board: chess.Board, depth: int, start_time: float, max_time: float) -> Optional[chess.Move]:
        """Searches for the best move at a given depth."""
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        maximizing_player = board.turn == chess.WHITE
        best_eval = -float('inf') if maximizing_player else float('inf')
        moves = self.order_moves(board, depth)
        if not moves:
            return None

        for move in moves:
            time_elapsed = time.time() - start_time
            if time_elapsed >= max_time or self.search_stop:
                self.search_stop = True
                raise TimeoutError("Search timed out")

            board.push(move)
            try:
                eval = -principal_variation_search(board, depth - 1, -beta, -alpha, not maximizing_player, self, start_time, max_time)
            except TimeoutError:
                board.pop()
                break
            board.pop()

            if maximizing_player:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move
                beta = min(beta, eval)

            if beta <= alpha:
                break

        return best_move

    def order_moves(self, board: chess.Board, depth: int) -> list:
        """Orders moves using the model's policy network and heuristics."""
        state = board_to_tensor_cached(board, self)
        with torch.no_grad():
            policy_logits, _ = self.model(state)  # type: ignore
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        moves = list(board.legal_moves)
        move_scores = []
        for move in moves:
            idx = move_to_index(move)
            score = policy[idx]
            # Add history heuristic
            score += self.history_table.get(move, 0)
            # Add killer move bonus
            if move in self.killer_moves[depth]:
                score += 10000
            # Use MVV/LVA for captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                attacker_piece = board.piece_at(move.from_square)
                if captured_piece and attacker_piece:
                    mvv_lva_score = (captured_piece.piece_type * 10) - attacker_piece.piece_type
                    score += mvv_lva_score
            move_scores.append((move, score))

        ordered_moves = [move for move, _ in sorted(move_scores, key=lambda x: x[1], reverse=True)]
        return ordered_moves

    def order_moves_quiescence(self, board: chess.Board) -> list:
        """Orders moves for quiescence search (captures only)."""
        moves = [move for move in board.legal_moves if board.is_capture(move)]
        move_scores = []
        for move in moves:
            captured_piece = board.piece_at(move.to_square)
            attacker_piece = board.piece_at(move.from_square)
            if captured_piece and attacker_piece:
                mvv_lva_score = (captured_piece.piece_type * 10) - attacker_piece.piece_type
                move_scores.append((move, mvv_lva_score))
            else:
                move_scores.append((move, 0))
        ordered_moves = [move for move, _ in sorted(move_scores, key=lambda x: x[1], reverse=True)]
        return ordered_moves

    def save_model(self, optimizer=None, scheduler=None, loss_history=None, performance_history=None, replay_buffer=None):
        """Saves the model, optimizer state, and replay buffer to the specified path."""
        checkpoint = {
            'model_state_dict': model_instance.module.state_dict() if isinstance(model_instance, nn.DataParallel) else model_instance.state_dict(), # type: ignore
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_history': loss_history,
            'performance_history': performance_history,
            'replay_buffer': list(replay_buffer) if replay_buffer else None,
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"Model and state saved to {MODEL_PATH}")

    
    def periodically_save_model(self, optimizer=None, scheduler=None, loss_history=None, performance_history=None, replay_buffer=None):
        """Saves the model and optimizer state to the specified path."""
        if not os.path.exists(BACKUP_MODEL_PATH):
            os.makedirs(BACKUP_MODEL_PATH)
        checkpoint = {
            'model_state_dict': model_instance.module.state_dict() if isinstance(model_instance, nn.DataParallel) else model_instance.state_dict(), # type: ignore
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_history': loss_history,
            'performance_history': performance_history,
            'replay_buffer': list(replay_buffer) if replay_buffer else None,
        }
        save_path = os.path.join(BACKUP_MODEL_PATH, f"{time.time()}.pth")
        torch.save(checkpoint, save_path)
        print(f"Model and state saved to {BACKUP_MODEL_PATH+str(time.time())+'.pth'}")

def material_evaluation(board: chess.Board) -> float:
    """Calculates the material balance of the board."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
    material_score = white_material - black_material
    material_score /= 39
    return material_score

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Converts a chess board to a tensor representation with piece values."""
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q':10, 'k':11
    }
    piece_values = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
    }
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    value_tensor = np.zeros((1, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_symbol = piece.symbol()
            channel = piece_to_channel[piece_symbol]
            value = piece_values[piece_symbol]
            x = square % 8
            y = 7 - (square // 8)
            board_tensor[channel, y, x] = 1
            value_tensor[0, y, x] = value
    combined_tensor = np.concatenate((board_tensor, value_tensor), axis=0)
    return torch.tensor(combined_tensor, dtype=torch.float32)

def move_to_index(move: chess.Move) -> int:
    """Converts a move to an index in the policy output."""
    NUM_SQUARES = 64
    NUM_PROMOTION_PIECES = 4
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    if promotion is None:
        index = from_square * NUM_SQUARES + to_square
    else:
        promotion_index = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}[promotion]
        index = NUM_SQUARES * NUM_SQUARES + (from_square - 8) * NUM_PROMOTION_PIECES + promotion_index
    return index

def index_to_move(idx: int, board: chess.Board) -> Optional[chess.Move]:
    """Converts an index in the policy output to a move."""
    NUM_SQUARES = 64
    NUM_PROMOTION_PIECES = 4

    if idx < NUM_SQUARES * NUM_SQUARES:
        from_square = idx // NUM_SQUARES
        to_square = idx % NUM_SQUARES
        move = chess.Move(from_square, to_square)
    else:
        idx -= NUM_SQUARES * NUM_SQUARES
        from_square = idx // NUM_PROMOTION_PIECES + 8
        promotion_index = idx % NUM_PROMOTION_PIECES
        promotion_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promotion_index]
        to_square = from_square + 8 if board.turn == chess.WHITE else from_square - 8
        move = chess.Move(from_square, to_square, promotion=promotion_piece)

    if move in board.legal_moves:
        return move
    else:
        return None

def periodic_evaluation(ai: ChessAI, episodes: int = 3, skill_level: int = 10):
    """Evaluate the AI's performance against Stockfish and collect game data for training."""
    stockfish = ai.stockfish
    stockfish.set_skill_level(skill_level)
    device = ai.device
    ai.model = model_instance
    ai.model.eval()  # type: ignore

    win_count = 0
    draw_count = 0
    loss_count = 0

    # Initialize a list to store training data
    evaluation_memory = []

    for _ in range(episodes):
        board = chess.Board()
        board.turn = chess.WHITE
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # AI's move
                best_move = ai.select_best_move(board)
                if best_move is None:
                    best_move = random.choice(list(board.legal_moves))
                board.push(best_move)

                # Collect data
                state = board_to_tensor(board).unsqueeze(0).to(device)
                # Get Stockfish evaluation
                stockfish.set_fen_position(board.fen())
                stockfish_eval = stockfish.get_evaluation()
                if stockfish_eval['type'] == 'cp':
                    stockfish_value = stockfish_eval['value'] / 100.0
                elif stockfish_eval['type'] == 'mate':
                    stockfish_value = np.sign(stockfish_eval['value']) * 10.0
                else:
                    stockfish_value = 0.0
                # Adjust evaluation from the AI's perspective
                stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)
                target_eval = torch.tensor([stockfish_value], dtype=torch.float32).to(device)

                # Store the experience
                evaluation_memory.append((state, target_eval))

                if board.is_game_over():
                    break
            else:
                # Stockfish's move
                stockfish.set_fen_position(board.fen())
                stockfish_move = stockfish.get_best_move()
                if stockfish_move:
                    stockfish_move_obj = chess.Move.from_uci(stockfish_move)
                    board.push(stockfish_move_obj)
                else:
                    break

                # Collect data
                state = board_to_tensor(board).unsqueeze(0).to(device)
                # Get Stockfish evaluation
                stockfish.set_fen_position(board.fen())
                stockfish_eval = stockfish.get_evaluation()
                if stockfish_eval['type'] == 'cp':
                    stockfish_value = stockfish_eval['value'] / 100.0
                elif stockfish_eval['type'] == 'mate':
                    stockfish_value = np.sign(stockfish_eval['value']) * 10.0
                else:
                    stockfish_value = 0.0
                # Adjust evaluation from the AI's perspective (since it's the opponent's move)
                stockfish_value = -stockfish_value
                stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)
                target_eval = torch.tensor([stockfish_value], dtype=torch.float32).to(device)

                # Store the experience
                evaluation_memory.append((state, target_eval))

                if board.is_game_over():
                    break

        # Determine game outcome
        result = board.result()
        if result == "1-0":
            win_count += 1
        elif result == "0-1":
            loss_count += 1
        else:
            draw_count += 1

    # Report evaluation results
    print(f"Evaluation against Stockfish (Skill Level {skill_level}):")
    print(f"Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")

    ai.transposition_table.clear()
    ai.pv_table.clear()
    ai.history_table.clear()

    return win_count, draw_count, loss_count, evaluation_memory

#region TRAINING

INITIAL_EPISODES = 0

def train(
    ai: ChessAI,
    episodes: int = 1000,
    lr: float = 0.00001,
    stop_event: Optional[threading.Event] = None,
    current_loss: Optional[list] = None,
    current_episode: Optional[list] = None,
    evaluation_interval: int = 25,
    batch_size: int = 128,
    buffer_size: int = 50000,
    loss_history: Optional[list] = None,
    performance_history: Optional[list] = None
):
    """Trains the neural network using experience replay with Mixed Precision."""
    device = ai.device
    model = model_instance.to(device) # type: ignore
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_history = [] if loss_history is None else loss_history
    performance_history = [] if performance_history is None else performance_history
    
    scaler = GradScaler('cuda')  # Initialize GradScaler for AMP

    replay_buffer = deque(maxlen=buffer_size)

    stockfish = ai.stockfish
    stockfish_skill = 1
    # Limit Stockfish search depth for faster evaluation
    stockfish.update_engine_parameters({
        "Threads": 3,
        "Minimum Thinking Time": 5,
        "Hash": 16,
        "UCI_LimitStrength": True,
        "UCI_Elo": 1350  # Adjust as needed
    })
    stockfish.set_skill_level(stockfish_skill)
    criterion = nn.SmoothL1Loss()
    start_episode = 0

    # Load optimizer and scheduler state if available
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")
        # Load other necessary states if needed
        if 'loss_history' in checkpoint and checkpoint['loss_history'] is not None:
            loss_history.extend(checkpoint['loss_history'])
        if 'performance_history' in checkpoint and checkpoint['performance_history'] is not None:
            performance_history.extend(checkpoint['performance_history'])
        if 'replay_buffer' in checkpoint and checkpoint['replay_buffer'] is not None:
            # Move tensors to the correct device
            replay_buffer.extend([
                (state.to(device), target.to(device)) for state, target in checkpoint['replay_buffer']
            ])
        start_episode = len(loss_history)
        print(f"Resuming training from episode {start_episode + 1}.")
    else:
        print("Starting training from scratch.")
    if os.path.exists("stockfish_skill.txt"):
        with open("stockfish_skill.txt", "r") as f:
            stockfish_skill = int(f.read().strip())
        stockfish.set_skill_level(stockfish_skill)

    # Cache for Stockfish evaluations
    stockfish_eval_cache = {}
    
    INITIAL_EPISODES = start_episode

    for episode in range(start_episode, episodes+start_episode):
        if (episode + 1) % 150 == 0:
            if stockfish_skill < 20:
                stockfish_skill += 1
        stockfish.set_skill_level(stockfish_skill)
        time_start = time.perf_counter()

        if stop_event is not None and stop_event.is_set():
            ai.save_model(optimizer, scheduler, loss_history, performance_history, replay_buffer)
            with open("stockfish_skill.txt", "w") as f:
                f.write(str(stockfish_skill))
            print(f"Training stopped by user. Model saved to {MODEL_PATH}")
            return
        
        if (episode + 1) % evaluation_interval == 0:
            ai.periodically_save_model(optimizer, scheduler, loss_history, performance_history, replay_buffer)
            print('Model saved for backup purposes.')
            print(f"\n--- Evaluating after {episode + 1} episodes ---")
            win, draw, loss_result, evaluation_memory = periodic_evaluation(ai, episodes=3, skill_level=stockfish_skill)
            if performance_history is not None:
                performance_history.append((win, draw, loss_result))
            else:
                print("performance_history is None")

            # Add evaluation data to replay buffer
            replay_buffer.extend(evaluation_memory)

            # Training step
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, targets = zip(*batch)
                states = torch.cat([s.to(device) for s in states])
                targets = torch.stack([t.to(device) for t in targets])

                optimizer.zero_grad()
                with autocast('cuda'):
                    _, outputs = model(states)  # Now 'states' is a tensor of shape (batch_size, 13, 8, 8)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()  # Scale loss for backprop
                scaler.step(optimizer)          # Step optimizer
                scaler.update()                 # Update scaler

                scheduler.step(loss)

                if current_loss is not None:
                    current_loss[0] = loss.item()
                    if loss_history is not None:
                        loss_history.append(loss.item())
            else:
                loss = None  # If not enough samples, set loss to None

            if current_episode is not None:
                current_episode[0] = episode + 1

            end_time = time.perf_counter()

            if loss is not None:
                print(f"Periodic Evaluation Episode {episode + 1}/{start_episode + episodes} - Loss: {loss.item():.4f} - Time: {end_time - time_start:.2f}s")
            else:
                print(f"Episode {episode + 1}/{start_episode + episodes} - Loss: N/A - Time: {end_time - time_start:.2f}s")
            continue

        episode_memory = []
        while len(episode_memory) < batch_size:
            board = chess.Board()
            while not board.is_game_over():
                # Stockfish's move (White)
                stockfish.set_fen_position(board.fen())
                stockfish_move = stockfish.get_best_move()
                if stockfish_move:
                    move = chess.Move.from_uci(stockfish_move)
                else:
                    move = random.choice(list(board.legal_moves))
                board.push(move)

                # Get evaluation after the move
                fen = board.fen()
                if fen in stockfish_eval_cache and 'evaluation' in stockfish_eval_cache[fen]:
                    stockfish_value = stockfish_eval_cache[fen]['evaluation']
                else:
                    stockfish.set_fen_position(fen)
                    stockfish_eval = stockfish.get_evaluation()
                    if stockfish_eval['type'] == 'cp':
                        stockfish_value = stockfish_eval['value'] / 100.0
                    elif stockfish_eval['type'] == 'mate':
                        stockfish_value = np.sign(stockfish_eval['value']) * 10.0
                    else:
                        stockfish_value = 0.0
                    stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)
                    stockfish_eval_cache[fen] = {'evaluation': stockfish_value}

                # Adjust evaluation to be from current player's perspective
                if board.turn == chess.BLACK:
                    stockfish_value = -stockfish_value

                # Normalize and combine evaluations
                stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)

                # Add material evaluation (already normalized between -1 and 1)
                material_score = material_evaluation(board)
                combined_value = stockfish_value + material_score

                # Clamp the combined value to ensure it stays within [-10, 10]
                combined_value = max(min(combined_value, MAX_EVAL), MIN_EVAL)
                target_eval = torch.tensor([combined_value], dtype=torch.float32).to(device)

                # Store the experience
                state = board_to_tensor(board).unsqueeze(0).to(device)
                episode_memory.append((state, target_eval))

                if board.is_game_over() or len(episode_memory) >= batch_size:
                    break

                # Stockfish's move (Black)
                stockfish.set_fen_position(board.fen())
                stockfish_move = stockfish.get_best_move()
                if stockfish_move:
                    move = chess.Move.from_uci(stockfish_move)
                else:
                    move = random.choice(list(board.legal_moves))
                board.push(move)

                # Get evaluation after the move
                fen = board.fen()
                if fen in stockfish_eval_cache and 'evaluation' in stockfish_eval_cache[fen]:
                    stockfish_value = stockfish_eval_cache[fen]['evaluation']
                else:
                    stockfish.set_fen_position(fen)
                    stockfish_eval = stockfish.get_evaluation()
                    if stockfish_eval['type'] == 'cp':
                        stockfish_value = stockfish_eval['value'] / 100.0
                    elif stockfish_eval['type'] == 'mate':
                        stockfish_value = np.sign(stockfish_eval['value']) * 10.0
                    else:
                        stockfish_value = 0.0
                    stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)
                    stockfish_eval_cache[fen] = {'evaluation': stockfish_value}

                # Adjust evaluation to be from current player's perspective
                if board.turn == chess.WHITE:
                    stockfish_value = -stockfish_value

                # Normalize and combine evaluations
                stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)

                # Add material evaluation
                material_score = material_evaluation(board)
                combined_value = stockfish_value + material_score

                # Clamp the combined value
                combined_value = max(min(combined_value, MAX_EVAL), MIN_EVAL)
                target_eval = torch.tensor([combined_value], dtype=torch.float32).to(device)

                # Store the experience
                state = board_to_tensor(board).unsqueeze(0).to(device)
                episode_memory.append((state, target_eval))

                if board.is_game_over() or len(episode_memory) >= batch_size:
                    break

            # Extend the replay buffer with the episode memory
            replay_buffer.extend(episode_memory)

        # Training step
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, targets = zip(*batch)
            states = torch.cat([s.to(device) for s in states])
            targets = torch.stack([t.to(device) for t in targets])
            
            optimizer.zero_grad()
            with autocast('cuda'):  # Enable autocast for mixed precision
                _, outputs = model(states)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()  # Scale loss for backprop
            scaler.step(optimizer)          # Step optimizer
            scaler.update()                 # Update scaler
            
            scheduler.step(loss)
            
            if current_loss is not None:
                current_loss[0] = loss.item()
                if loss_history is not None:
                    loss_history.append(loss.item())
        else:
            loss = None  # If not enough samples, set loss to None

        if current_episode is not None:
            current_episode[0] = episode + 1

        end_time = time.perf_counter()

        if loss is not None:
            print(f"Episode {episode + 1}/{episodes + start_episode} - Loss: {loss.item():.4f} - Time: {end_time - time_start:.2f}s")
        else:
            print(f"Episode {episode + 1}/{episodes + start_episode} - Loss: N/A - Time: {end_time - time_start:.2f}s")

    ai.save_model(optimizer, scheduler, loss_history, performance_history, replay_buffer)
    with open("stockfish_skill.txt", "w") as f:
        f.write(str(stockfish_skill))
    print("Training completed and model saved.")

#endregion

def play_against_stockfish(ai: ChessAI, skill_level: int = 10):
    """Launches a GUI for playing against Stockfish."""
    device = ai.device
    model = model_instance  # Use the global model
    model.to(device)  # type: ignore
    model.eval()  # type: ignore

    game_window = tk.Toplevel()
    game_window.title("Chess AI vs Stockfish")

    # Initialize necessary variables
    board = chess.Board()
    board.turn = chess.WHITE

    # Initialize move number, moves, evaluations, and board positions
    move_number = 1
    moves_san = []
    evaluations = []
    boards = [board.fen()]  # Initial position

    # Load piece images
    piece_images = {}
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K',
                'p', 'n', 'b', 'r', 'q', 'k']
    for piece in pieces:
        if piece.isupper():
            filename = os.path.join(IMAGE_DIR, f"{piece}.png")
        else:
            filename = os.path.join(IMAGE_DIR, f"_{piece}.png")
        try:
            piece_image = Image.open(filename).resize((50, 50), Image.Resampling.LANCZOS)
            piece_images[piece] = piece_image
        except FileNotFoundError:
            messagebox.showerror("Image Error", f"Image file for {piece} not found at {filename}.")
            return

    # Set up GUI layout
    board_canvas = tk.Canvas(game_window, width=400, height=400)
    board_canvas.pack(side=tk.LEFT)

    info_frame = tk.Frame(game_window)
    info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Move list display
    move_list = tk.Text(info_frame, width=30, height=20)
    move_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    move_list.insert('end', "Move List\nAI vs Stockfish\n")

    stockfish_eval_label = tk.Label(info_frame, text="Stockfish Evaluation: N/A")
    stockfish_eval_label.pack()
    
    ai_eval_label = tk.Label(info_frame, text="AI Evaluation: N/A")
    ai_eval_label.pack()

    # Resume Game button
    def resume_game():
        board.set_fen(boards[-1])
        update_board()
        if evaluations:
            model_eval, stockfish_value = evaluations[-1]
            stockfish_eval_label.config(text=f"Stockfish Evaluation: {stockfish_value:.2f}")

    def reset_game():
        nonlocal board, move_number, moves_san, evaluations, boards
        board = chess.Board()
        board.turn = chess.WHITE
        move_number = 1
        moves_san = []
        evaluations = []
        boards = [board.fen()]
        move_list.delete('1.0', tk.END)
        move_list.insert('end', "Move List\nAI vs Stockfish\n")
        update_board()
        stockfish_eval_label.config(text="Stockfish Evaluation: N/A")
        ai_eval_label.config(text="AI Evaluation: N/A")

    resume_button = tk.Button(info_frame, text="Resume Game", command=resume_game)
    resume_button.pack()

    reset_button = tk.Button(info_frame, text="Reset Game", command=reset_game)
    reset_button.pack()

    board_images = []

    def update_board():
        board_image = Image.new("RGB", (400, 400), "white")
        draw_board(board, board_image)
        board_tk = ImageTk.PhotoImage(board_image)
        board_images.append(board_tk)  # Prevent garbage collection
        board_canvas.create_image(0, 0, anchor="nw", image=board_tk)

    def draw_board(board: chess.Board, image: Image.Image):
        square_size = 50
        for rank in range(8):
            for file in range(8):
                color = "lightgray" if (rank + file) % 2 == 0 else "darkgreen"
                x0 = file * square_size
                y0 = (7 - rank) * square_size
                x1 = x0 + square_size
                y1 = y0 + square_size
                image.paste(color, (x0, y0, x1, y1))

                # Draw the pieces
                square = chess.square(file, 7 - rank)
                piece = board.piece_at(square)
                if piece:
                    piece_symbol = piece.symbol()
                    piece_image = piece_images.get(piece_symbol)
                    if piece_image:
                        image.paste(piece_image, (x0, y0), piece_image)

    def next_move():
        nonlocal board, move_number
        model_eval = 0.0
        stockfish_value = 0.0
        if board.turn == chess.WHITE:
            # AI's move
            chosen_move = ai.select_best_move(board)  # Adjust depth as needed
            if chosen_move is None:
                chosen_move = random.choice(list(board.legal_moves))
            move_san = board.san(chosen_move)  # Get SAN before pushing the move
            board.push(chosen_move)
            moves_san.append(move_san)
            boards.append(board.fen())

            # Get evaluations
            # Model evaluation
            state = board_to_tensor(board).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value = model_instance(state)  # type: ignore
                model_eval = max(min(value.item(), MAX_EVAL), MIN_EVAL)

            # Stockfish evaluation
            ai.stockfish.set_fen_position(board.fen())
            stockfish_eval = ai.stockfish.get_evaluation()
            if stockfish_eval['type'] == 'cp':
                stockfish_value = stockfish_eval['value'] / 100.0
            elif stockfish_eval['type'] == 'mate':
                stockfish_value = np.sign(stockfish_eval['value']) * 10.0
            else:
                stockfish_value = 0.0

            # Clamp stockfish_value
            stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)
            evaluations.append((model_eval, stockfish_value))

            # Update move list
            move_index = len(moves_san) - 1
            move_text = f"{move_number:3}. {move_san:8} "

            # Get start index before inserting text
            move_list.insert('end', move_text)
            end_index = move_list.index('end -1c')

            tag_name = f"move_{move_index}"
            move_list.tag_add(tag_name, f'end - {len(move_text)}c', end_index)
            move_list.tag_bind(tag_name, "<Button-1>", on_move_click)

            board.turn = chess.BLACK
            if board.is_game_over():
                move_button.config(state=tk.DISABLED)
                result = board.result()
                messagebox.showinfo("Game Over", f"Game over: {result}")
                game_window.destroy()
                return

        else:
            # Stockfish's move
            ai.stockfish.set_fen_position(board.fen())
            stockfish_move = ai.stockfish.get_best_move()
            if stockfish_move:
                stockfish_move_obj = chess.Move.from_uci(stockfish_move)
                move_san = board.san(stockfish_move_obj)  # Get SAN before pushing the move
                board.push(stockfish_move_obj)
                moves_san.append(move_san)
                boards.append(board.fen())

                # Get evaluations
                # Model evaluation
                state = board_to_tensor(board).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, value = model_instance(state)  # type: ignore
                    model_eval = value.item()

                # Stockfish evaluation
                ai.stockfish.set_fen_position(board.fen())
                stockfish_eval = ai.stockfish.get_evaluation()
                if stockfish_eval['type'] == 'cp':
                    stockfish_value = stockfish_eval['value'] / 100.0
                elif stockfish_eval['type'] == 'mate':
                    stockfish_value = np.sign(stockfish_eval['value']) * 10.0
                else:
                    stockfish_value = 0.0
                evaluations.append((model_eval, stockfish_value))

                # Update move list
                move_index = len(moves_san) - 1
                move_text = f"{move_san} "

                # Get start index before inserting text
                move_list.insert('end', move_text)
                end_index = move_list.index('end -1c')

                tag_name = f"move_{move_index}"
                move_list.tag_add(tag_name, f'end - {len(move_text)}c', end_index)
                move_list.tag_bind(tag_name, "<Button-1>", on_move_click)

                move_list.insert('end', "\n")
                move_number += 1

                board.turn = chess.WHITE
                if board.is_game_over():
                    move_button.config(state=tk.DISABLED)
                    result = board.result()
                    messagebox.showinfo("Game Over", f"Game over: {result}")
                    game_window.destroy()
                    return

        # Update evaluations labels
        stockfish_eval_label.config(text=f"Stockfish Evaluation: {stockfish_value:.2f}")
        ai_eval_label.config(text=f"AI Evaluation: {model_eval:.4f}")

        # Update board display
        update_board()

    def on_move_click(event):
        """Handles clicking on a move in the move list to jump to that position."""
        index = move_list.index(f"@{event.x},{event.y}")
        tags = move_list.tag_names(index)
        for tag in tags:
            if tag.startswith("move_"):
                move_index = int(tag.split("_")[1])
                break
        else:
            return
        fen = boards[move_index + 1]  # +1 because boards[0] is initial position
        board.set_fen(fen)
        update_board()
        if move_index < len(evaluations):
            model_eval, stockfish_value = evaluations[move_index]
            stockfish_eval_label.config(text=f"Stockfish Evaluation: {stockfish_value:.2f}")
            ai_eval_label.config(text=f"AI Evaluation: {model_eval:.4f}")

    def auto_play():
        """Automates the play by alternating moves between AI and Stockfish."""
        if auto_play_toggle.get() and not board.is_game_over():
            next_move()
            game_window.after(300, auto_play)  # Adjust delay as needed

    def on_toggle_auto_play():
        """Toggles the auto-play feature."""
        if auto_play_toggle.get():
            auto_play()

    # Auto-play controls
    auto_play_label = tk.Label(game_window, text="Auto-Play:")
    auto_play_label.pack(side=tk.BOTTOM)
    auto_play_toggle = tk.BooleanVar()
    auto_play_checkbox = tk.Checkbutton(game_window, variable=auto_play_toggle, command=on_toggle_auto_play)
    auto_play_checkbox.pack(side=tk.BOTTOM)

    # Move button
    move_button = tk.Button(game_window, text="Next Move", command=next_move)
    move_button.pack(side=tk.BOTTOM)

    update_board()
    game_window.mainloop()

def exit_program(root: tk.Tk, stop_event: threading.Event, training_thread: Optional[threading.Thread]):
    """Gracefully exits the program by stopping training and closing the GUI."""
    stop_event.set()
    if training_thread is not None and training_thread.is_alive():
        training_thread.join()
    root.quit()
    root.destroy()
    sys.exit()

def play_against_model(ai: ChessAI):
    """Allows the user to play against the AI model."""
    device = ai.device
    model = model_instance  # Use the global model
    model.to(device)  # type: ignore
    model.eval()  # type: ignore

    game_window = tk.Toplevel()
    game_window.title("Play Against AI Model")

    # Initialize necessary variables
    board = chess.Board()
    board.turn = chess.WHITE

    # Initialize move number, moves, evaluations, and board positions
    move_number = 1
    moves_san = []
    evaluations = []
    boards = [board.fen()]  # Initial position

    # Load piece images
    piece_images = {}
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K',
              'p', 'n', 'b', 'r', 'q', 'k']
    for piece in pieces:
        if piece.isupper():
            filename = os.path.join(IMAGE_DIR, f"{piece}.png")
        else:
            filename = os.path.join(IMAGE_DIR, f"_{piece}.png")
        try:
            piece_image = Image.open(filename).resize((50, 50), Image.Resampling.LANCZOS)
            piece_images[piece] = piece_image
        except FileNotFoundError:
            messagebox.showerror("Image Error", f"Image file for {piece} not found at {filename}.")
            return

    # Set up GUI layout
    board_canvas = tk.Canvas(game_window, width=400, height=400)
    board_canvas.pack(side=tk.LEFT)
    
    info_frame = tk.Frame(game_window)
    info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Move list display
    move_list = tk.Text(info_frame, width=30, height=20)
    move_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    move_list.insert('end', "Move List\nYou vs AI Model\n")
    
    ai_eval_label = tk.Label(info_frame, text="AI Evaluation: N/A")
    ai_eval_label.pack()
    
    board_images = []
    selected_square = None  # Keep track of selected square

    def drawHighlightedSquare(square):
        square_size = 50
        file, rank = chess.square_file(square), 7 - chess.square_rank(square)
        x0 = file * square_size
        y0 = rank * square_size
        x1 = x0 + square_size
        y1 = y0 + square_size
        board_canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2, tags="highlight")

    def removeHighlightedSquare():
        board_canvas.delete("highlight")

    def prompt_promotion():
        promo_window = tk.Toplevel(game_window)
        promo_window.title("Choose Promotion")
        promo_window.grab_set()
        promotion_choice = tk.StringVar()
        promotion_choice.set("q")
        options = [("Queen", "q"), ("Rook", "r"), ("Bishop", "b"), ("Knight", "n")]
        tk.Label(promo_window, text="Choose promotion piece:").pack(pady=10)
        for text, value in options:
            tk.Radiobutton(promo_window, text=text, variable=promotion_choice, value=value).pack(anchor=tk.W)
        def confirm():
            promo_window.destroy()
        tk.Button(promo_window, text="OK", command=confirm).pack(pady=10)
        game_window.wait_window(promo_window)
        promotion_map = {
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT
        }
        return promotion_map.get(promotion_choice.get(), chess.QUEEN)

    def on_click(event):
        nonlocal selected_square, move_number
        file, rank = event.x // 50, 7 - (event.y // 50)
        square = chess.square(file, rank)
        removeHighlightedSquare()
        if selected_square is None:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                selected_square = square
                drawHighlightedSquare(square)
        elif selected_square == square:
            selected_square = None
        else:
            piece = board.piece_at(selected_square)
            if piece and piece.piece_type == chess.PAWN:
                is_promotion = (
                    (piece.color == chess.WHITE and chess.square_rank(square) == 7) or
                    (piece.color == chess.BLACK and chess.square_rank(square) == 0)
                )
            else:
                is_promotion = False
            if is_promotion:
                promotion = prompt_promotion()
                if promotion is None:
                    selected_square = None
                    return
                move = chess.Move(selected_square, square, promotion=promotion)
            else:
                move = chess.Move(selected_square, square)
            if move in board.legal_moves:
                # Get SAN before pushing the move
                move_san = board.san(move)
                board.push(move)
                moves_san.append(move_san)
                boards.append(board.fen())
                selected_square = None
                update_board()
                # Update move list
                move_index = len(moves_san) - 1
                if board.turn == chess.BLACK:
                    move_text = f"{move_number:3}. {moves_san[-1]:8} "
                else:
                    move_text = f"{moves_san[-1]}\n"
                    move_number += 1

                # Get start index before inserting text
                move_list.insert('end', move_text)
                end_index = move_list.index('end -1c')

                tag_name = f"move_{move_index}"
                move_list.tag_add(tag_name, f'end - {len(move_text)}c', end_index)
                move_list.tag_bind(tag_name, "<Button-1>", on_move_click)

                # Check for game over
                if board.is_game_over():
                    result = board.result()
                    messagebox.showinfo("Game Over", f"Game over: {result}")
                    game_window.destroy()
                    return

                # AI's turn
                ai_move()
            else:
                messagebox.showinfo("Invalid Move", "This move is not allowed.")
            selected_square = None
            update_board()

    def on_right_click(event):
        nonlocal selected_square
        if selected_square:
            removeHighlightedSquare()
            selected_square = None

    def update_board():
        board_image = Image.new("RGB", (400, 400), "white")
        draw_board(board, board_image)
        board_tk = ImageTk.PhotoImage(board_image)
        board_images.append(board_tk)  # Prevent garbage collection
        board_canvas.create_image(0, 0, anchor="nw", image=board_tk)

    def draw_board(board: chess.Board, image: Image.Image):
        square_size = 50
        for rank in range(8):
            for file in range(8):
                # Flip the rank to have white at the bottom
                actual_rank = 7 - rank
                square = chess.square(file, actual_rank)
                color = "lightgray" if (rank + file) % 2 == 0 else "darkgreen"
                x0 = file * square_size
                y0 = rank * square_size  # y0 corresponds to the flipped rank
                x1 = x0 + square_size
                y1 = y0 + square_size
                image.paste(color, (x0, y0, x1, y1))

                # Draw the pieces
                piece = board.piece_at(square)
                if piece:
                    piece_symbol = piece.symbol()
                    piece_image = piece_images.get(piece_symbol)
                    if piece_image:
                        image.paste(piece_image, (x0, y0), piece_image)

    def ai_move():
        nonlocal move_number
        # AI's move
        chosen_move = ai.select_best_move(board)  # Adjust depth as needed
        if chosen_move is None:
            chosen_move = random.choice(list(board.legal_moves))
        board.push(chosen_move)
        move_san = board.san(chosen_move)
        moves_san.append(move_san)
        boards.append(board.fen())
        update_board()

        # Update move list
        move_index = len(moves_san) - 1
        if board.turn == chess.WHITE:
            move_text = f"{moves_san[-1]}\n"
            move_number += 1
        else:
            move_text = f"{move_number:3}. ... {moves_san[-1]:8} "

        # Get start index before inserting text
        move_list.insert('end', move_text)
        end_index = move_list.index('end -1c')

        tag_name = f"move_{move_index}"
        move_list.tag_add(tag_name, f'end - {len(move_text)}c', end_index)
        move_list.tag_bind(tag_name, "<Button-1>", on_move_click)

        # Get AI evaluation
        state = board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = model_instance(state)  # type: ignore
            model_eval = value.item()
            model_eval = max(min(model_eval, MAX_EVAL), MIN_EVAL)

        ai_eval_label.config(text=f"AI Evaluation: {model_eval:.4f}")

        # Check for game over
        if board.is_game_over():
            result = board.result()
            messagebox.showinfo("Game Over", f"Game over: {result}")
            game_window.destroy()
            return

    def on_move_click(event):
        """Handles clicking on a move in the move list to jump to that position."""
        index = move_list.index(f"@{event.x},{event.y}")
        tags = move_list.tag_names(index)
        for tag in tags:
            if tag.startswith("move_"):
                move_index = int(tag.split("_")[1])
                break
        else:
            return
        fen = boards[move_index + 1]  # +1 because boards[0] is initial position
        board.set_fen(fen)
        update_board()

    def reset_game():
        nonlocal board, move_number, moves_san, evaluations, boards, selected_square
        board = chess.Board()
        board.turn = chess.WHITE
        move_number = 1
        moves_san = []
        evaluations = []
        boards = [board.fen()]
        selected_square = None
        move_list.delete('1.0', tk.END)
        move_list.insert('end', "Move List\nYou vs AI Model\n")
        update_board()
        ai_eval_label.config(text="AI Evaluation: N/A")

    reset_button = tk.Button(info_frame, text="Reset Game", command=reset_game)
    reset_button.pack()

    update_board()

    board_canvas.bind("<Button-1>", on_click)
    board_canvas.bind("<Button-3>", on_right_click)
    game_window.mainloop()

# Custom class to redirect stdout and stderr to a queue
class RedirectText:
    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, message):
        if message.strip() != "":
            self.log_queue.put(message+'\n')

    def flush(self):
        pass  # No need to implement flush for this use-case

def main():
    """Main function to initialize the Chess AI Trainer GUI."""
    stockfish_path = STOCKFISH_DEFAULT_PATH  # Update to your Stockfish path if needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_model(device)
    ai = ChessAI(stockfish_path, device)

    stop_event = threading.Event()
    current_loss = [0.0]
    current_episode = [0]
    training_thread = None

    # Histories for plotting
    loss_history = []
    performance_history = []

    # Initialize the log queue
    log_queue = queue.Queue()

    root = tk.Tk()
    root.title("Chess AI Trainer")
    root.geometry("800x800")  # Increased size to accommodate plots and log

    # Create frames for layout
    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    log_frame = tk.Frame(root)
    log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Episodes input
    episodes_label = tk.Label(control_frame, text="Number of Episodes:")
    episodes_label.grid(row=0, column=0, padx=5, pady=5)

    episodes_entry = tk.Entry(control_frame)
    episodes_entry.grid(row=0, column=1, padx=5, pady=5)
    episodes_entry.insert(0, "1000")

    # Stockfish difficulty input
    stockfish_difficulty_label = tk.Label(control_frame, text="Stockfish Difficulty (1-20):")
    stockfish_difficulty_label.grid(row=1, column=0, padx=5, pady=5)

    stockfish_difficulty_entry = tk.Entry(control_frame)
    stockfish_difficulty_entry.grid(row=1, column=1, padx=5, pady=5)
    stockfish_difficulty_entry.insert(0, "10")

    # Start Training button
    def start_training():
        nonlocal training_thread
        try:
            episodes = int(episodes_entry.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a positive integer for episodes.")
            return

        # Disable start button and enable stop button
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

        # Start the training thread with loss_history and performance_history
        training_thread = threading.Thread(
            target=train,
            args=(
                ai,
                episodes,
                0.001,  # Updated learning rate
                stop_event,
                current_loss,
                current_episode,
                100,  # evaluation_interval
                64,   # batch_size
                50000, # buffer_size
                loss_history,       # Pass loss_history
                performance_history  # Pass performance_history
            )
        )
        training_thread.start()

        update_gui()

    start_button = tk.Button(control_frame, text="Start Training", command=start_training)
    start_button.grid(row=0, column=2, padx=5, pady=5)

    # Stop Training button
    def stop_training():
        stop_event.set()
        messagebox.showinfo("Training", "Training will stop after the current episode.")

    stop_button = tk.Button(control_frame, text="Stop Training", command=stop_training)
    stop_button.grid(row=1, column=2, padx=5, pady=5)
    stop_button.config(state=tk.DISABLED)

    # Play against Stockfish button
    def play():
        try:
            difficulty = int(stockfish_difficulty_entry.get())
            if not (1 <= difficulty <= 20):
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please enter an integer between 1 and 20 for difficulty.")
            return
        play_against_stockfish(ai, skill_level=difficulty)

    play_button = tk.Button(control_frame, text="Play Against Stockfish", command=play)
    play_button.grid(row=0, column=3, padx=5, pady=5)
    
    # Play against AI Model button
    def play_against_model_button():
        play_against_model(ai)

    play_model_button = tk.Button(control_frame, text="Play Against AI Model", command=play_against_model_button)
    play_model_button.grid(row=0, column=4, padx=5, pady=5)

    # Close button
    close_button = tk.Button(control_frame, text="Close", command=lambda: exit_program(root, stop_event, training_thread))
    close_button.grid(row=1, column=3, padx=5, pady=5)

    # Labels for displaying current loss and episode
    loss_label = tk.Label(control_frame, text="Current Loss: N/A")
    loss_label.grid(row=2, column=0, padx=5, pady=5)

    episode_label = tk.Label(control_frame, text="Current Episode: 0")
    episode_label.grid(row=2, column=1, padx=5, pady=5)

    # Labels for CPU and Memory usage
    cpu_label = tk.Label(control_frame, text="CPU Usage: N/A")
    cpu_label.grid(row=2, column=2, padx=5, pady=5)

    memory_label = tk.Label(control_frame, text="Memory Usage: N/A")
    memory_label.grid(row=2, column=3, padx=5, pady=5)
    
    device_label = tk.Label(control_frame, text=f"Device: {device}")
    device_label.grid(row=3, column=0, padx=5, pady=5)
    
    training_diff_label = tk.Label(control_frame, text="Training Difficulty: N/A")
    training_diff_label.grid(row=3, column=1, padx=5, pady=5)

    # Setup Matplotlib Figures
    fig_loss, ax_loss = plt.subplots(figsize=(5, 4))
    ax_loss.set_title("Loss over Episodes")
    ax_loss.set_xlabel("Episodes")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)

    fig_perf, ax_perf = plt.subplots(figsize=(5, 4))
    ax_perf.set_title("Performance Metrics")
    ax_perf.set_xlabel("Evaluations")
    ax_perf.set_ylabel("Count")
    ax_perf.grid(True)
    ax_perf.legend(['Wins', 'Draws', 'Losses'])

    # Create Canvas for Loss Plot
    canvas_loss = FigureCanvasTkAgg(fig_loss, master=plot_frame)
    canvas_loss.draw()
    canvas_loss.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create Canvas for Performance Plot
    canvas_perf = FigureCanvasTkAgg(fig_perf, master=plot_frame)
    canvas_perf.draw()
    canvas_perf.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Setup Logging: Redirect stdout and stderr to the log_queue
    redirect = RedirectText(log_queue)
    sys.stdout = redirect
    sys.stderr = redirect

    # Create the Text widget for logs
    log_text = tk.Text(log_frame, wrap='word', height=10, state='disabled', bg='white', fg='black')
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure a scrollbar for the log_text widget
    log_scrollbar = tk.Scrollbar(log_text)
    log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.config(yscrollcommand=log_scrollbar.set)
    log_scrollbar.config(command=log_text.yview)

    # Function to update the log_text widget from the log_queue
    def update_log():
        while not log_queue.empty():
            try:
                message = log_queue.get_nowait()
                log_text.config(state='normal')
                log_text.insert(tk.END, message)
                log_text.see(tk.END)
                log_text.config(state='disabled')
            except queue.Empty:
                pass
        root.after(100, update_log)  # Check the queue every 100 ms

    # Start updating the log
    update_log()

    # Function to update plots
    def update_plots():
        # Update Loss Plot
        if loss_history:
            ax_loss.cla()
            ax_loss.set_title("Loss over Episodes")
            ax_loss.set_xlabel("Episodes")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True)
            ax_loss.plot(loss_history, color='blue')
            canvas_loss.draw()

        # Update Performance Plot
        if performance_history:
            wins, draws, losses = zip(*performance_history)
            evaluations_range = range(1, len(wins) + 1)
            ax_perf.cla()
            ax_perf.set_title("Performance Metrics")
            ax_perf.set_xlabel("Evaluations")
            ax_perf.set_ylabel("Count")
            ax_perf.grid(True)
            ax_perf.plot(evaluations_range, wins, label='Wins', color='green')
            ax_perf.plot(evaluations_range, draws, label='Draws', color='orange')
            ax_perf.plot(evaluations_range, losses, label='Losses', color='red')
            ax_perf.legend()
            canvas_perf.draw()

    # Function to update GUI elements
    def update_gui():
        loss_label.config(text=f"Current Loss: {current_loss[0]:.4f}")
        episode_label.config(text=f"Current Episode: {current_episode[0] + INITIAL_EPISODES+1}")
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
        memory_label.config(text=f"Memory Usage: {memory_usage}%")
        training_diff_label.config(text=f"Training Difficulty: {ai.get_stockfish_skill()}")  

        # Update plots
        update_plots()

        if training_thread is not None and training_thread.is_alive():
            root.after(1000, update_gui)
        else:
            start_button.config(state=tk.NORMAL)
            stop_button.config(state=tk.DISABLED)

    root.mainloop()

    # Ensure the training thread is properly terminated on exit
    if not stop_event.is_set() and training_thread is not None and training_thread.is_alive():
        training_thread.join()
    print("Program exited.")

if __name__ == "__main__":
    main()
