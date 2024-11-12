import tkinter as tk
from tkinter import messagebox
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
from collections import defaultdict
import math
import queue
import random
from typing import Optional, Dict

# Import for Zobrist hashing
from chess.polyglot import zobrist_hash

# Constants
ACTION_SIZE = 4672
MODEL_PATH = "chess_model.pth"  # Updated to match the second program's model path
MAX_EVAL = 10.0
MIN_EVAL = -10.0
MAX_DEPTH = 100
MAX_TIME = 10.0

# Global model instance
model_instance = None

# Locks for thread safety
eval_cache_lock = threading.Lock()
board_tensor_cache_lock = threading.Lock()
transposition_table_lock = threading.Lock()

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
            nn.Linear(256, 1)  # Removed Tanh to match the second program's model
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
    model_instance.to(device)
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model_instance.load_state_dict(state_dict)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a fresh model.")
    else:
        print("No saved model found. Starting with a fresh model.")

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

def board_to_tensor_cached(board: chess.Board, ai: 'ChessAI') -> torch.Tensor:
    """Converts a chess board to a tensor representation with caching."""
    board_hash = zobrist_hash(board)
    with board_tensor_cache_lock:
        if board_hash in ai.board_tensor_cache:
            return ai.board_tensor_cache[board_hash]
        tensor = board_to_tensor(board).unsqueeze(0).to(ai.device)  # type: ignore
        ai.board_tensor_cache[board_hash] = tensor
    return tensor

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

class ChessAI:
    """Encapsulates the chess AI functionalities, including the neural network and move selection."""
    def __init__(self, device: torch.device):
        self.device = device
        self.model = model_instance
        self.history_table: Dict[chess.Move, int] = defaultdict(int)
        self.pv_table: Dict[str, chess.Move] = {}
        self.eval_cache = {}
        self.transposition_table = {}
        self.board_tensor_cache = {}
        self.killer_moves = [[] for _ in range(MAX_DEPTH)]
        self.search_stop = False
    
    def select_best_move(self, board: chess.Board, max_time: float = MAX_TIME):
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

    def order_moves(self, board: chess.Board, depth: int):
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

    def order_moves_quiescence(self, board: chess.Board):
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

class ChessGUI:
    def __init__(self, ai: ChessAI, master):
        self.master = master
        self.master.title("Chess - Play Against AI")
        self.ai = ai
        self.board = chess.Board()
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()
        self.board_images = []
        self.load_piece_images()
        self.move_history = []
        self.current_move_index = 0
        self.opening_book = {
            'e2e4 e7e5': "King's Pawn Game",
            'e2e4 e7e5 g1f3': "King's Knight Opening",
            'e2e4 e7e5 g1f3 b8c6': "Italian Game",
            'e2e4 e7e5 g1f3 b8c6 f1c4': "Giuoco Piano",
            'd2d4 d7d5': "Queen's Pawn Game",
            'd2d4 d7d5 c1g5': "Trompowsky Attack",
            # Add more openings as needed
        }

        # Initialize message and opening_label before updating the board
        self.message = tk.Label(master, text="Your move", font=("Arial", 12))
        self.message.pack()
        self.opening_label = tk.Label(master, text="", font=("Arial", 12))
        self.opening_label.pack()
        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        self.prev_button = tk.Button(self.button_frame, text="Previous", command=self.on_prev)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.button_frame, text="Next", command=self.on_next)
        self.next_button.pack(side=tk.LEFT)

        self.resume_button = tk.Button(self.button_frame, text="Resume", command=self.on_resume)
        self.resume_button.pack(side=tk.LEFT)

        # Now you can safely update the board
        self.update_board()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.selected_square = None
        self.ai_thread = None
        self.premove = None
        self.ai_busy = False
        self.queue = queue.Queue()
    
    def load_piece_images(self):
        self.piece_images = {}
        pieces = "PNBRQKpnbrqk"
        for piece in pieces:
            if piece.isupper():
                img_path = os.path.join("images", f"{piece.lower()}.png")
                if not os.path.exists(img_path):
                    messagebox.showerror("Error", f"Image file {img_path} not found.")
                    self.master.destroy()
                self.piece_images[piece] = Image.open(img_path).resize((50, 50), Image.Resampling.LANCZOS)
            else:
                img_path = os.path.join("images", f"_{piece}.png")
                if not os.path.exists(img_path):
                    messagebox.showerror("Error", f"Image file {img_path} not found.")
                    self.master.destroy()
                self.piece_images[piece] = Image.open(img_path).resize((50, 50), Image.Resampling.LANCZOS)
    
    def update_board(self):
        # Rebuild the board up to current_move_index
        self.board = chess.Board()
        for move in self.move_history[:self.current_move_index]:
            self.board.push(move)
        # Then draw the board
        board_image = Image.new("RGB", (400, 400), "white")
        self.draw_board(self.board, board_image)
        board_tk = ImageTk.PhotoImage(board_image)
        self.board_images.append(board_tk)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=board_tk)
        # Update the message
        if self.current_move_index < len(self.move_history):
            self.message.config(text="Viewing move {}/{}".format(self.current_move_index, len(self.move_history)))
        else:
            if self.board.is_game_over():
                result = self.board.result()
                self.message.config(text=f"Game Over: {result}")
            else:
                if self.board.turn == chess.WHITE:
                    self.message.config(text="Your move")
                else:
                    self.message.config(text="AI's move")
        # Display the opening if any
        self.display_opening()
    
    def display_opening(self):
        move_sequence = ' '.join([move.uci() for move in self.move_history[:self.current_move_index]])
        matching_opening = None
        max_length = 0
        for opening_moves, opening_name in self.opening_book.items():
            if move_sequence.startswith(opening_moves):
                if len(opening_moves) > max_length:
                    matching_opening = opening_name
                    max_length = len(opening_moves)
        if matching_opening:
            self.opening_label.config(text=f"Opening: {matching_opening}")
        else:
            self.opening_label.config(text="")
    
    def draw_board(self, board, image):
        square_size = 50
        for rank in range(7, -1, -1):
            for file in range(8):
                color = "lightgray" if (rank + file) % 2 == 0 else "darkgreen"
                x0 = file * square_size
                y0 = (7 - rank) * square_size
                x1 = x0 + square_size
                y1 = y0 + square_size
                square_image = Image.new("RGB", (square_size, square_size), color)
                image.paste(square_image, (x0, y0))
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece:
                    piece_symbol = piece.symbol()
                    piece_image = self.piece_images[piece_symbol]
                    image.paste(piece_image, (x0, y0), piece_image)
    
    def drawHighlightedSquare(self, square):
        square_size = 50
        file, rank = chess.square_file(square), 7 - chess.square_rank(square)
        x0 = file * square_size
        y0 = rank * square_size
        x1 = x0 + square_size
        y1 = y0 + square_size
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2, tags="highlight")
    
    def removeHighlightedSquare(self):
        self.canvas.delete("highlight")
    
    def on_click(self, event):
        if self.current_move_index != len(self.move_history):
            messagebox.showinfo("Info", "Cannot make a move when viewing old positions. Click 'Resume' to return to the latest position.")
            return
        file, rank = event.x // 50, 7 - (event.y // 50)
        square = chess.square(file, rank)
        self.drawHighlightedSquare(square)
        if self.selected_square is None:
            self.selected_square = square
        elif self.selected_square == square:
            self.removeHighlightedSquare()
            self.selected_square = None
        else:
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                is_promotion = (
                    (piece.color == chess.WHITE and chess.square_rank(square) == 7) or
                    (piece.color == chess.BLACK and chess.square_rank(square) == 0)
                )
            else:
                is_promotion = False
            if is_promotion:
                promotion = self.prompt_promotion()
                if promotion is None:
                    self.removeHighlightedSquare()
                    self.selected_square = None
                    return
                move = chess.Move(self.selected_square, square, promotion=promotion)
            else:
                move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                if self.ai_busy:
                    self.premove = move
                    self.message.config(text="Premove registered")
                else:
                    self.board.push(move)
                    # Add the move to move_history
                    self.move_history.append(move)
                    self.current_move_index += 1
                    self.update_board()
                    if self.board.is_game_over():
                        result = self.board.result()
                        self.message.config(text=f"Game Over: {result}")
                    else:
                        self.message.config(text="AI's move")
                        self.start_ai_move()
            else:
                messagebox.showinfo("Invalid Move", "This move is not allowed.")
            self.removeHighlightedSquare()
            self.selected_square = None
    
    def prompt_promotion(self):
        promo_window = tk.Toplevel(self.master)
        promo_window.title("Choose Promotion")
        promo_window.grab_set()
        result_dict: dict[str, Optional[int]] = {'result': None}
        promotion_choice = tk.StringVar()
        promotion_choice.set("q")
        options = [("Queen", "q"), ("Rook", "r"), ("Bishop", "b"), ("Knight", "n")]
        tk.Label(promo_window, text="Choose promotion piece:").pack(pady=10)
        for text, value in options:
            tk.Radiobutton(promo_window, text=text, variable=promotion_choice, value=value).pack(anchor=tk.W)
        def confirm():
            promo = promotion_choice.get()
            promo_window.destroy()
            promotion_map = {
                "q": chess.QUEEN,
                "r": chess.ROOK,
                "b": chess.BISHOP,
                "n": chess.KNIGHT
            }
            return_value = promotion_map.get(promo, chess.QUEEN)
            result_dict['result'] = return_value
        tk.Button(promo_window, text="OK", command=confirm).pack(pady=10)
        self.master.wait_window(promo_window)
        return result_dict.get('result', None)
    
    def on_right_click(self, event):
        if self.selected_square:
            self.removeHighlightedSquare()
            self.selected_square = None
    
    def on_prev(self):
        if self.current_move_index > 0:
            self.current_move_index -= 1
            self.update_board()
    
    def on_next(self):
        if self.current_move_index < len(self.move_history):
            self.current_move_index += 1
            self.update_board()
    
    def on_resume(self):
        self.current_move_index = len(self.move_history)
        self.update_board()
    
    def start_ai_move(self):
        if self.ai_thread and self.ai_thread.is_alive():
            return
        self.ai_busy = True
        self.ai_thread = threading.Thread(target=self.ai_move, daemon=True)
        self.ai_thread.start()
        self.master.after(100, self.check_ai_thread)
    
    def check_ai_thread(self):
        if self.ai_thread and self.ai_thread.is_alive():
            self.master.after(100, self.check_ai_thread)
        else:
            best_move = self.queue.get()
            if best_move is not None:
                self.board.push(best_move)
                # Add to move_history
                self.move_history.append(best_move)
                self.current_move_index += 1
                self.update_board()
                if self.board.is_game_over():
                    result = self.board.result()
                    self.message.config(text=f"Game Over: {result}")
                else:
                    self.message.config(text="Your move")
                    if self.premove:
                        self.board.push(self.premove)
                        # Add the move to move_history
                        self.move_history.append(self.premove)
                        self.current_move_index += 1
                        self.update_board()
                        self.premove = None
                        if self.board.is_game_over():
                            result = self.board.result()
                            self.message.config(text=f"Game Over: {result}")
                        else:
                            self.message.config(text="AI's move")
                            self.start_ai_move()
            else:
                self.message.config(text="AI did not find a move.")
            self.ai_busy = False
    
    def ai_move(self):
        best_move = self.ai.select_best_move(board=self.board)
        self.queue.put(best_move)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_model(device)
    ai = ChessAI(device)
    root = tk.Tk()
    app = ChessGUI(ai, root)
    root.mainloop()
