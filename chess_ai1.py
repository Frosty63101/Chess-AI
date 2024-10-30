import os
import sys
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
from PIL import Image, ImageTk, ImageDraw
from stockfish import Stockfish
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue  # New import for logging

# Constants
MODEL_PATH = "chess_model1.pth"
STOCKFISH_DEFAULT_PATH = "stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"  # Update this path as needed
IMAGE_DIR = "images"  # Directory containing piece images

# Global model instance
model_instance = None  # Will be initialized in main()

class TranspositionEntry:
    """Data structure for transposition table entries."""
    def __init__(self, eval_score: Optional[float] = 0.0, freq: int = 0, pv_move: Optional[chess.Move] = None):
        self.eval = eval_score
        self.freq = freq
        self.pv_move = pv_move

class ChessNet(nn.Module):
    """Neural network model for evaluating chess positions."""
    def __init__(self):
        super(ChessNet, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convolutional layers with increased depth and feature maps
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Residual Blocks for capturing patterns
        self.residual_block1 = self._make_residual_block(512)
        self.residual_block2 = self._make_residual_block(512)
        self.residual_block3 = self._make_residual_block(512)  # Additional residual block

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 12, 8, 8))

        # Fully connected layers with increased units and Dropout for regularization
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.4)

    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x += self.positional_encoding  # Adding positional encoding
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))

        # Apply Residual Blocks
        x = self._apply_residual(x, self.residual_block1)
        x = self._apply_residual(x, self.residual_block2)
        x = self._apply_residual(x, self.residual_block3)  # Additional block

        # Flatten for fully connected layers
        x = x.view(-1, 512 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def _apply_residual(self, x, block):
        residual = x
        x = block(x)
        x += residual  # Skip connection
        return x

# Initialize the model globally and load weights
def initialize_model(device: torch.device):
    global model_instance
    model_instance = ChessNet().to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model_instance.eval()
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a fresh model.")
    else:
        print("No saved model found. Starting with a fresh model.")

class ChessAI:
    """Encapsulates the chess AI functionalities, including the neural network, Stockfish integration, and move selection."""
    def __init__(self, stockfish_path: str, device: torch.device):
        self.device = device
        self.stockfish = Stockfish(stockfish_path)
        self.stockfish.set_skill_level(10)  # Default skill level

        # Transposition and history tables
        self.transposition_table: Dict[str, TranspositionEntry] = defaultdict(TranspositionEntry)
        self.pv_table: Dict[str, chess.Move] = {}
        self.history_table: Dict[chess.Move, int] = defaultdict(int)

    def save_model(self):
        """Saves the global model to the specified path."""
        torch.save(model_instance.state_dict(), MODEL_PATH) # type: ignore
        print(f"Model saved to {MODEL_PATH}")

    def select_best_move(self, board: chess.Board, max_depth: int = 10) -> Optional[chess.Move]:
        best_move = None
        for depth in range(1, max_depth + 1):
            current_best_move = self.iterative_deepening_search(board, depth)
            if current_best_move:
                best_move = current_best_move
        return best_move

    def iterative_deepening_search(self, board: chess.Board, depth: int) -> Optional[chess.Move]:
        """Performs iterative deepening search to find the best move."""
        best_eval = -float('inf')
        best_move = None
        # Prioritize PV move if available
        ordered_moves = []
        if board.fen() in self.pv_table:
            pv_move = self.pv_table[board.fen()]
            if pv_move in board.legal_moves:
                ordered_moves = [pv_move] + [m for m in self.order_moves(board) if m != pv_move]
        else:
            ordered_moves = self.order_moves(board)

        for move in ordered_moves:
            board.push(move)
            eval_score = self.minimax_evaluate_helper(board, depth - 1, -float('inf'), float('inf'), False)
            board.pop()
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
        if best_move:
            # Update history and PV tables
            self.history_table[best_move] += 1
            self.pv_table[board.fen()] = best_move
        return best_move

    def minimax_evaluate_helper(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool
    ) -> float:
        """Helper function for Minimax evaluation with alpha-beta pruning."""
        board_fen = board.fen()
        entry = self.transposition_table[board_fen]

        if entry.eval is not None:
            # Use cached evaluation
            return entry.eval

        if depth == 0 or board.is_game_over():
            eval_score = cached_board_evaluation(board_fen)
            self.transposition_table[board_fen].eval = eval_score
            return eval_score

        if is_maximizing:
            max_eval = -float('inf')
            for move in self.order_moves(board):
                board.push(move)
                eval_score = self.minimax_evaluate_helper(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            self.transposition_table[board_fen].eval = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.order_moves(board):
                board.push(move)
                eval_score = self.minimax_evaluate_helper(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            self.transposition_table[board_fen].eval = min_eval
            return min_eval

    def order_moves(self, board: chess.Board) -> list:
        """Orders moves to improve alpha-beta pruning efficiency."""
        moves = list(board.legal_moves)
        # Evaluate each move and sort based on the evaluation score
        move_scores = {}
        for move in moves:
            board.push(move)
            move_scores[move] = cached_board_evaluation(board.fen())
            board.pop()
        ordered_moves = sorted(moves, key=lambda move: move_scores[move], reverse=True)
        return ordered_moves

@lru_cache(maxsize=20000)
def cached_board_evaluation(fen: str) -> float:
    """Evaluates the board using the global neural network model."""
    board = chess.Board(fen)
    state = board_to_tensor(board).unsqueeze(0).to(model_instance.device) # type: ignore
    with torch.no_grad():
        eval_score = model_instance(state).item() # type: ignore
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
        'q': 10, # Black Queen
        'k': 11  # Black King
    }
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_symbol = piece.symbol()
            channel = piece_to_channel[piece_symbol]
            x = square % 8
            y = square // 8
            board_tensor[channel, y, x] = 1
    return torch.tensor(board_tensor, dtype=torch.float32)

def periodic_evaluation(ai: ChessAI, episodes: int = 5, skill_level: int = 10):
    """Evaluate the AI's performance against Stockfish."""
    stockfish = ai.stockfish
    stockfish.set_skill_level(skill_level)
    device = ai.device
    ai.model_instance = model_instance  # Ensure AI uses the global model # type: ignore
    ai.model_instance.eval() # type: ignore

    win_count = 0
    draw_count = 0
    loss_count = 0

    for _ in range(episodes):
        board = chess.Board()
        board.turn = chess.WHITE
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # AI's move
                best_move = ai.select_best_move(board, max_depth=10)  # Adjust depth as needed
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

def train(
    ai: ChessAI,
    episodes: int = 1000,
    lr: float = 0.001,
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
    optimizer = optim.Adam(model.parameters(), lr=lr) # type: ignore
    criterion = nn.SmoothL1Loss()

    # Replay buffer for experience replay
    replay_buffer = deque(maxlen=buffer_size)

    # Epsilon decay parameters
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995

    for episode in range(episodes):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        if stop_event is not None and stop_event.is_set():
            ai.save_model()
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
                chosen_move = ai.select_best_move(board, max_depth=10)  # Adjust depth as needed
                if chosen_move is None:
                    chosen_move = random.choice(list(board.legal_moves))
            board.push(chosen_move)

            # Stockfish evaluation of the new board state
            stockfish.set_fen_position(board.fen())
            stockfish_eval = stockfish.get_evaluation()
            if stockfish_eval['type'] == 'cp':
                stockfish_value = stockfish_eval['value'] / 100.0
            elif stockfish_eval['type'] == 'mate':
                stockfish_value = np.sign(stockfish_eval['value']) * 1000.0
            else:
                stockfish_value = 0.0

            stockfish_value = np.clip(stockfish_value, -10, 10)
            target_eval = torch.tensor([stockfish_value], dtype=torch.float32).to(device)

            # Store the experience
            episode_memory.append((state, target_eval))

        replay_buffer.extend(episode_memory)
        loss = None
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, targets = zip(*batch)
            states = torch.cat(states)
            targets = torch.stack(targets)

            model.train() # type: ignore
            optimizer.zero_grad()
            outputs = model(states.to(device)) # type: ignore
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

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

        if loss is not None:
            print(f"Episode {episode + 1}/{episodes} - Loss: {loss.item():.4f}")
        else:
            print(f"Episode {episode + 1}/{episodes} - Loss: N/A")

    ai.save_model()
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
            chosen_move = ai.select_best_move(board, max_depth=10)  # Adjust depth as needed
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
                    model_eval = model_instance(state).item()
                else:
                    model_eval = 0.0  # Default value or handle the error appropriately

            # Stockfish evaluation
            ai.stockfish.set_fen_position(board.fen())
            stockfish_eval = ai.stockfish.get_evaluation()
            if stockfish_eval['type'] == 'cp':
                stockfish_value = stockfish_eval['value'] / 100.0
            elif stockfish_eval['type'] == 'mate':
                stockfish_value = np.sign(stockfish_eval['value']) * 1000.0
            else:
                stockfish_value = 0.0
            evaluations.append((model_eval, stockfish_value))

            # Update move list
            move_index = len(moves_san) - 1
            move_text = f"{move_number:3}. {move_san:8} "

            # Get start index before inserting text
            start_index = move_list.index('end -1c')
            move_list.insert('end -1c', move_text)
            end_index = move_list.index('end -1c')

            tag_name = f"move_{move_index}"
            move_list.tag_add(tag_name, start_index, end_index)
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
                    model_eval = model_instance(state).item() # type: ignore

                # Stockfish evaluation
                ai.stockfish.set_fen_position(board.fen())
                stockfish_eval = ai.stockfish.get_evaluation()
                if stockfish_eval['type'] == 'cp':
                    stockfish_value = stockfish_eval['value'] / 100.0
                elif stockfish_eval['type'] == 'mate':
                    stockfish_value = np.sign(stockfish_eval['value']) * 1000.0
                else:
                    stockfish_value = 0.0
                evaluations.append((model_eval, stockfish_value))

                # Update move list
                move_index = len(moves_san) - 1
                move_text = f"{move_san} "

                # Get start index before inserting text
                start_index = move_list.index('end -1c')
                move_list.insert('end -1c', move_text)
                end_index = move_list.index('end -1c')

                tag_name = f"move_{move_index}"
                move_list.tag_add(tag_name, start_index, end_index)
                move_list.tag_bind(tag_name, "<Button-1>", on_move_click)

                move_list.insert('end -1c', "\n")
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

# New: Custom class to redirect stdout and stderr to a queue
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
                0.001,
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
