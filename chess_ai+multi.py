import os
import sys
import math
import random
import threading
import tkinter as tk
from tkinter import messagebox
from collections import deque, defaultdict
from typing import Dict, Optional

import chess
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image, ImageTk
from stockfish import Stockfish
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue  # For logging

import time

# Constants
MODEL_PATH = "chess_model+multi.pth"
STOCKFISH_DEFAULT_PATH = "stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"  # Update this path as needed
IMAGE_DIR = "images"  # Directory containing piece images
ACTION_SIZE = 4672  # Total possible moves in our action space

# Global model instance
model_instance = None  # Will be initialized in main()

class TranspositionEntry:
    """Data structure for transposition table entries."""
    def __init__(self, eval_score: Optional[float] = None, freq: int = 0, pv_move: Optional[chess.Move] = None):
        self.eval = eval_score
        self.freq = freq
        self.pv_move = pv_move

class Node:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}  # action: Node
        self.visit_count = 0
        self.total_value = 0
        self.prior_prob = prior_prob

    def select_child(self):
        C_PUCT = 1.0  # Exploration constant
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            u = C_PUCT * child.prior_prob * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            q = child.total_value / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)

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

class ChessNet(nn.Module):
    """Neural network model for evaluating chess positions."""
    def __init__(self):
        super(ChessNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_block = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1),
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
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class MCTS:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def search(self, board, num_simulations=800):
        root = Node(parent=None, prior_prob=1.0)
        for _ in range(num_simulations):
            node = root
            state = board.copy()
            # Selection
            path = []
            while node is not None and node.children:
                action, node = node.select_child()
                state.push(action)
                path.append(node)
            # Expansion
            if not state.is_game_over():
                policy, value = self.evaluate_state(state)
                action_probs = []
                for move in state.legal_moves:
                    idx = move_to_index(move)
                    prob = policy[idx]
                    action_probs.append((move, prob))
                if node is not None:
                    node.expand(action_probs)
            else:
                # Game over
                value = self.game_over_value(state)
            # Backpropagation
            self.backpropagate(node, value)
        # Choose action with highest visit count
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        policy = np.zeros(ACTION_SIZE)
        for action, child in root.children.items():
            idx = move_to_index(action)
            policy[idx] = child.visit_count
        policy = policy / np.sum(policy)
        return best_move, policy

    def evaluate_state(self, state):
        state_tensor = board_to_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return policy, value.item()

    def game_over_value(self, state):
        # Returns +1, -1, or 0 for win, loss, or draw from current player's perspective
        result = state.result()
        if result == '1-0':
            return 1 if state.turn == chess.WHITE else -1
        elif result == '0-1':
            return -1 if state.turn == chess.WHITE else 1
        else:
            return 0

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Switch perspectives
            node = node.parent

# Initialize the model globally and load weights
def initialize_model(device: torch.device):
    global model_instance
    model_instance = ChessNet()
    print(f"Using device: {device}")
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model_instance = nn.DataParallel(model_instance)
    model_instance.to(device)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
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
        self.stockfish.set_skill_level(10)  # Default skill level

        # Initialize history and PV tables
        self.pv_table: Dict[str, chess.Move] = {}
        self.history_table: Dict[chess.Move, int] = defaultdict(int)
        self.model = model_instance
        self.transposition_table = {}

    def select_best_move(self, board: chess.Board, max_depth: int = 3) -> Optional[chess.Move]:
        best_move = None
        for depth in range(1, max_depth + 1):
            current_best_move = self.iterative_deepening_search(board, depth)
            if current_best_move:
                best_move = current_best_move
        return best_move

    def iterative_deepening_search(self, board: chess.Board, depth: int) -> Optional[chess.Move]:
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')
        best_move = None

        # Prioritize PV move if available
        ordered_moves = []
        board_fen = board.fen()
        if board_fen in self.pv_table:
            pv_move = self.pv_table[board_fen]
            if pv_move in board.legal_moves:
                ordered_moves = [pv_move] + [m for m in self.order_moves(board) if m != pv_move]
        else:
            ordered_moves = self.order_moves(board)

        for move in ordered_moves:
            board.push(move)
            eval_score = self.minimax_evaluate_helper(
                board,
                depth - 1,
                -float('inf'),
                float('inf'),
                board.turn != chess.WHITE,
                self.transposition_table
            )
            board.pop()

            if board.turn == chess.WHITE:
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            else:
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move

        if best_move:
            # Update history and PV tables
            self.history_table[best_move] += 1
            self.pv_table[board_fen] = best_move
        return best_move

    def minimax_evaluate_helper(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        transposition_table: Dict[str, float]
    ) -> float:
        """Helper function for Minimax evaluation with alpha-beta pruning."""
        board_fen = board.fen()
        if board_fen in transposition_table:
            return transposition_table[board_fen]

        if depth == 0 or board.is_game_over():
            eval_score = cached_board_evaluation(board_fen)
            transposition_table[board_fen] = eval_score
            return eval_score

        if is_maximizing:
            max_eval = -float('inf')
            for move in self.order_moves(board):
                board.push(move)
                eval_score = self.minimax_evaluate_helper(
                    board, depth - 1, alpha, beta, False, transposition_table
                )
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            transposition_table[board_fen] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.order_moves(board):
                board.push(move)
                eval_score = self.minimax_evaluate_helper(
                    board, depth - 1, alpha, beta, True, transposition_table
                )
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            transposition_table[board_fen] = min_eval
            return min_eval

    def order_moves(self, board: chess.Board) -> list:
        """Orders moves to improve alpha-beta pruning efficiency."""
        ordered_moves = []
        for move in board.legal_moves:
            if board.is_capture(move):
                ordered_moves.insert(0, move)  # Prioritize captures
            elif board.gives_check(move):
                ordered_moves.insert(0, move)  # Prioritize checks
            else:
                ordered_moves.append(move)
        # Further prioritize based on history heuristic
        ordered_moves.sort(key=lambda move: self.history_table.get(move, 0), reverse=True)
        return ordered_moves

    def save_model(self, optimizer=None, scheduler=None, loss_history=None, performance_history=None):
        """Saves the model and optimizer state to the specified path."""
        checkpoint = {
            'model_state_dict': model_instance.module.state_dict() if isinstance(model_instance, nn.DataParallel) else model_instance.state_dict(), # type: ignore
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_history': loss_history,
            'performance_history': performance_history,
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"Model and state saved to {MODEL_PATH}")

@lru_cache(maxsize=20000)
def cached_board_evaluation(fen: str) -> float:
    """Evaluates the board using the global neural network model."""
    board = chess.Board(fen)
    state = board_to_tensor(board).unsqueeze(0).to(model_instance.device) # type: ignore
    with torch.no_grad():
        _, value = model_instance(state) # type: ignore
        eval_score = value.item()
    return eval_score

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Converts a chess board to a tensor representation."""
    piece_to_channel = {
        'P': 0,  # White Pawn
        'N': 1,  # White Knight
        'B': 2,  # White Bishop
        'R': 3,  # White Rook
        'Q': 4,  # White Queen
        'K': 5,  # White King
        'p': 6,  # Black Pawn
        'n': 7,  # Black Knight
        'b': 8,  # Black Bishop
        'r': 9,  # Black Rook
        'q':10,  # Black Queen
        'k':11   # Black King
    }
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_symbol = piece.symbol()
            channel = piece_to_channel[piece_symbol]
            x = square % 8
            y = 7 - (square // 8)
            board_tensor[channel, y, x] = 1
    return torch.tensor(board_tensor, dtype=torch.float32)

def periodic_evaluation(ai: ChessAI, episodes: int = 5, skill_level: int = 10):
    """Evaluate the AI's performance against Stockfish."""
    stockfish = ai.stockfish
    stockfish.set_skill_level(skill_level)
    device = ai.device
    ai.model = model_instance
    ai.model.eval() # type: ignore

    win_count = 0
    draw_count = 0
    loss_count = 0

    for _ in range(episodes):
        board = chess.Board()
        board.turn = chess.WHITE
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # AI's move
                best_move = ai.select_best_move(board, max_depth=3)
                if best_move is None:
                    best_move = random.choice(list(board.legal_moves))
                board.push(best_move)
            else:
                # Stockfish's move
                stockfish.set_fen_position(board.fen())
                stockfish_move = stockfish.get_best_move()
                if stockfish_move:
                    stockfish_move_obj = chess.Move.from_uci(stockfish_move)
                    board.push(stockfish_move_obj)
                else:
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

    return win_count, draw_count, loss_count

def move_to_index(move: chess.Move) -> int:
    """Converts a move to an index in the policy output."""
    # Initialize constants
    NUM_SQUARES = 64
    NUM_PROMOTION_PIECES = 4  # Queen, Rook, Bishop, Knight

    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    if promotion is None:
        # Normal move
        index = from_square * NUM_SQUARES + to_square
    else:
        # Promotion move
        promotion_index = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}[promotion]
        index = NUM_SQUARES * NUM_SQUARES + (from_square - 8) * NUM_PROMOTION_PIECES + promotion_index
    return index

def index_to_move(idx: int, board: chess.Board) -> Optional[chess.Move]:
    """Converts an index in the policy output to a move."""
    NUM_SQUARES = 64
    NUM_PROMOTION_PIECES = 4  # Queen, Rook, Bishop, Knight

    if idx < NUM_SQUARES * NUM_SQUARES:
        from_square = idx // NUM_SQUARES
        to_square = idx % NUM_SQUARES
        move = chess.Move(from_square, to_square)
    else:
        idx -= NUM_SQUARES * NUM_SQUARES
        from_square = idx // NUM_PROMOTION_PIECES + 8
        promotion_index = idx % NUM_PROMOTION_PIECES
        promotion_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promotion_index]
        # For promotion, to_square is one rank ahead for pawns
        to_square = from_square + 8 if board.turn == chess.WHITE else from_square - 8
        move = chess.Move(from_square, to_square, promotion=promotion_piece)

    if move in board.legal_moves:
        return move
    else:
        return None

def train(
    ai: ChessAI,
    episodes: int = 1000,
    lr: float = 0.0001,  # Reduced learning rate
    stop_event: Optional[threading.Event] = None,
    current_loss: Optional[list] = None,
    current_episode: Optional[list] = None,
    evaluation_interval: int = 100,
    batch_size: int = 32,
    buffer_size: int = 10000,
    loss_history: Optional[list] = None,       # New parameter
    performance_history: Optional[list] = None  # New parameter
):
    """Trains the neural network using experience replay."""
    device = ai.device
    model = model_instance  # Use the global model
    model.to(device) # type: ignore
    stockfish = ai.stockfish
    stockfish.set_skill_level(20)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # type: ignore
    criterion = nn.SmoothL1Loss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10) # type: ignore
    loss_history = [] if loss_history is None else loss_history
    performance_history = [] if performance_history is None else performance_history
    epsilon_start = None

    # Load optimizer and scheduler state if available
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")
        # Load transposition_table and history_table if needed
        if 'transposition_table' in checkpoint:
            ai.transposition_table = {}
            for fen, entry in checkpoint['transposition_table'].items():
                if isinstance(entry, TranspositionEntry):
                    ai.transposition_table[fen] = entry.eval
                else:
                    ai.transposition_table[fen] = entry  # Assuming it's already a float
            print("Transposition table loaded and converted.")
        if 'history_table' in checkpoint:
            ai.history_table = checkpoint['history_table']
            print("History table loaded.")
        if 'loss_history' in checkpoint and checkpoint['loss_history'] is not None:
            loss_history.extend(checkpoint['loss_history'])
        if 'performance_history' in checkpoint and checkpoint['performance_history'] is not None:
            performance_history.extend(checkpoint['performance_history'])
    if os.path.exists("epsilon_start.txt"):
        with open("epsilon_start.txt", "r") as f:
            epsilon_start = float(f.read().strip())

    # Replay buffer for experience replay
    replay_buffer = deque(maxlen=buffer_size)

    # Epsilon decay parameters
    epsilon = 1.0
    if epsilon_start is None:
        epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

    # Normalization parameters
    MIN_EVAL = -10.0  # Defined min evaluation
    MAX_EVAL = 10.0   # Defined max evaluation

    for episode in range(episodes):
        time_start = time.perf_counter()
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        if stop_event is not None and stop_event.is_set():
            ai.save_model(optimizer, scheduler, loss_history, performance_history)
            with open("epsilon_start.txt", "w") as f:
                f.write(str(epsilon))
            print(f"Training stopped by user. Model saved to {MODEL_PATH}")
            return

        board = chess.Board()
        episode_memory = []

        while not board.is_game_over():
            state = board_to_tensor(board).unsqueeze(0).to(device)

            # Select move (epsilon-greedy strategy)
            if random.uniform(0, 1) < epsilon:
                chosen_move = random.choice(list(board.legal_moves))
            else:
                chosen_move = ai.select_best_move(board, max_depth=3)  # Adjust depth as needed
                if chosen_move is None:
                    chosen_move = random.choice(list(board.legal_moves))
            board.push(chosen_move)

            # Stockfish evaluation of the new board state
            stockfish.set_fen_position(board.fen())
            stockfish_eval = stockfish.get_evaluation()
            if stockfish_eval['type'] == 'cp':
                stockfish_value = stockfish_eval['value'] / 100.0
            elif stockfish_eval['type'] == 'mate':
                stockfish_value = np.sign(stockfish_eval['value']) * 10.0
            else:
                stockfish_value = 0.0

            stockfish_value = max(min(stockfish_value, MAX_EVAL), MIN_EVAL)
            normalized_value = (stockfish_value - MIN_EVAL) / (MAX_EVAL - MIN_EVAL) * 2 - 1
            target_eval = torch.tensor([normalized_value], dtype=torch.float32).to(device)

            # Store the experience
            episode_memory.append((state, target_eval))

        replay_buffer.extend(episode_memory)
        loss = None
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, targets = zip(*batch)
            states = torch.cat(states)
            targets = torch.stack(targets)

            model.train().to(device) # type: ignore
            optimizer.zero_grad()
            _, outputs = model(states.to(device)) # type: ignore
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # type: ignore
            optimizer.step()

            scheduler.step(loss)

            if current_loss is not None:
                current_loss[0] = loss.item()
                if loss_history is not None:
                    loss_history.append(loss.item())

        if current_episode is not None:
            current_episode[0] = episode + 1

        # Periodic evaluation
        if (episode + 1) % evaluation_interval == 0:
            print(f"\n--- Evaluating after {episode + 1} episodes ---")
            win, draw, loss_result = periodic_evaluation(ai, episodes=5, skill_level=10)
            if performance_history is not None:
                performance_history.append((win, draw, loss_result))

        end_time = time.perf_counter()
        
        if loss is not None:
            print(f"Episode {episode + 1}/{episodes} - Loss: {loss.item():.4f} - Time: {end_time - time_start:.2f}s")
        else:
            print(f"Episode {episode + 1}/{episodes} - Loss: N/A - Time: {end_time - time_start:.2f}s")

    ai.save_model(optimizer, scheduler, loss_history, performance_history)
    with open("epsilon_start.txt", "w") as f:
        f.write(str(epsilon))
    print("Training completed and model saved.")

def play_against_stockfish(ai: ChessAI, skill_level: int = 10):
    """Launches a GUI for playing against Stockfish."""
    device = ai.device
    model = model_instance  # Use the global model
    model.to(device) # type: ignore
    model.eval() # type: ignore

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
            chosen_move = ai.select_best_move(board, max_depth=3)  # Adjust depth as needed
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
                if model_instance is not None:
                    _, value = model_instance(state)
                    model_eval = value.item()
                else:
                    model_eval = 0.0  # Default value or handle the error appropriately

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
                    _, value = model_instance(state) # type: ignore
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
                0.0001,  # Updated learning rate
                stop_event,
                current_loss,
                current_episode,
                100,  # evaluation_interval
                32,   # batch_size
                10000, # buffer_size
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
        episode_label.config(text=f"Current Episode: {current_episode[0]}")
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
        memory_label.config(text=f"Memory Usage: {memory_usage}%")

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
