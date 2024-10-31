import tkinter as tk
from tkinter import messagebox
import chess
from sympy import ordered
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added import
import numpy as np
from PIL import Image, ImageTk
import os
import random
import time
from functools import lru_cache
import threading
from queue import Queue
from typing import Optional, DefaultDict
from collections import defaultdict

ACTION_SIZE = 4672  # Added constant

# Adjusted model path to match the second program's model
MODEL_PATH = "chess_model+.pth"

class ResidualBlock(nn.Module):  # Added class
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

# Global transposition table with evaluations and frequency counts
history_table = defaultdict(int)
pv_table = {}

# Initialize the model globally for multiprocessing
def init_model():
    global model_instance
    model_instance = ChessNet()
    model_instance.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model_instance.to(torch.device("cpu"))
    model_instance.eval()

@lru_cache(maxsize=20000)
def cached_board_evaluation(board_fen, device):
    board = chess.Board(fen=board_fen)
    state = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        _, value = model_instance(state)  # Adjusted to get value from the model
        return value.item()

def board_to_tensor(board):
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
        'q': 10,  # Black Queen
        'k': 11   # Black King
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

def evaluate_move(task):
    move, board_fen, depth = task
    board = chess.Board(board_fen)
    board.push(move)
    eval_score = minimax_evaluate_serial(board, depth)
    return (eval_score, move)

def minimax_evaluate_serial(board, depth):
    transposition_table = {}
    return minimax_evaluate_helper(board, depth, -float('inf'), float('inf'), False, transposition_table)

def minimax_evaluate_helper(board, depth, alpha, beta, is_maximizing, transposition_table):
    board_fen = board.fen()
    if board_fen in transposition_table:
        return transposition_table[board_fen]

    if depth == 0 or board.is_game_over():
        eval_score = cached_board_evaluation(board_fen, torch.device("cpu"))
        transposition_table[board_fen] = eval_score
        return eval_score

    if is_maximizing:
        max_eval = -float('inf')
        ordered_moves = order_moves_serial(board)
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax_evaluate_helper(board, depth - 1, alpha, beta, False, transposition_table)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        transposition_table[board_fen] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        ordered_moves = order_moves_serial(board)
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax_evaluate_helper(board, depth - 1, alpha, beta, True, transposition_table)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        transposition_table[board_fen] = min_eval
        return min_eval

def order_moves_serial(board):
    ordered_moves = []
    for move in board.legal_moves:
        if board.is_capture(move):
            ordered_moves.insert(0, move)  # Prioritize captures
        elif board.gives_check(move):
            ordered_moves.insert(0, move)  # Prioritize checks
        else:
            ordered_moves.append(move)
    return ordered_moves

class ChessGUI:
    def __init__(self, model, master):
        self.master = master
        self.master.title("Chess - Play Against AI")
        
        # Load the model
        self.model = model
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.board = chess.Board()
        
        # Set up the board display
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()
        
        self.board_images = []
        
        self.load_piece_images()
        self.update_board()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        
        # Game messages
        self.message = tk.Label(master, text="Your move", font=("Arial", 12))
        self.message.pack()

        # Track moves
        self.selected_square = None
        
        self.transposition_table: DefaultDict[str, dict] = defaultdict(
            lambda: {'eval': None, 'freq': 0, 'pv_move': None}
        )

        
        # AI computation thread and queue
        self.ai_thread = None
        self.premove = None
        self.ai_busy = False
        self.queue = Queue()

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
        board_image = Image.new("RGB", (400, 400), "white")
        self.draw_board(self.board, board_image)
        board_tk = ImageTk.PhotoImage(board_image)
        self.board_images.append(board_tk)
        
        # Clear previous images and display the new one
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=board_tk)

    def draw_board(self, board, image):
        square_size = 50
        for rank in range(7, -1, -1):  # Start from rank 7 to 0, placing white at the bottom
            for file in range(8):
                color = "lightgray" if (rank + file) % 2 == 0 else "darkgreen"
                x0 = file * square_size
                y0 = (7 - rank) * square_size  # Rank is reversed for correct visual placement
                x1 = x0 + square_size
                y1 = y0 + square_size

                # Draw square color
                square_image = Image.new("RGB", (square_size, square_size), color)
                image.paste(square_image, (x0, y0))

                # Place piece image if present on the square
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

        # Draw a red rectangle around the square with a specific tag
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2, tags="highlight")

    def removeHighlightedSquare(self):
        # Remove the rectangle with the "highlight" tag
        self.canvas.delete("highlight")

    def on_click(self, event):
        # Adjust to match the orientation in draw_board
        file, rank = event.x // 50, 7 - (event.y // 50)  # Ensure this matches board rendering
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
                # Determine if the pawn is moving to the last rank
                is_promotion = (
                    (piece.color == chess.WHITE and chess.square_rank(square) == 7) or
                    (piece.color == chess.BLACK and chess.square_rank(square) == 0)
                )
            else:
                is_promotion = False
    
            if is_promotion:
                # Prompt the user to choose a promotion piece
                promotion = self.prompt_promotion()
                if promotion is None:
                    # User cancelled promotion
                    self.removeHighlightedSquare()
                    self.selected_square = None
                    return
                # Create a move with the selected promotion piece
                move = chess.Move(self.selected_square, square, promotion=promotion)
            else:
                # Create a regular move without promotion
                move = chess.Move(self.selected_square, square)
    
            if move in self.board.legal_moves:
                if self.ai_busy:
                    # If AI is busy, set premove
                    self.premove = move
                    self.message.config(text="Premove registered")
                else:
                    # Push the move to the board
                    self.board.push(move)
                    self.update_board()
                    if self.board.is_game_over():
                        result = self.board.result()
                        self.message.config(text=f"Game Over: {result}")
                    else:
                        self.message.config(text="AI's move")
                        self.start_ai_move()
            else:
                # Move is invalid
                messagebox.showinfo("Invalid Move", "This move is not allowed.")
    
            # Clean up selection and highlighting
            self.removeHighlightedSquare()
            self.selected_square = None


    def prompt_promotion(self):
    # Create a new top-level window
        promo_window = tk.Toplevel(self.master)
        promo_window.title("Choose Promotion")
        promo_window.grab_set()  # Make the window modal

        # Initialize a local variable to store the result
        result_dict = {'result': None}

        # Variable to store the selected promotion
        promotion_choice = tk.StringVar()
        promotion_choice.set("q")  # Default to queen

        # Define promotion options
        options = [("Queen", "q"), ("Rook", "r"), ("Bishop", "b"), ("Knight", "n")]

        tk.Label(promo_window, text="Choose promotion piece:").pack(pady=10)

        for text, value in options:
            tk.Radiobutton(promo_window, text=text, variable=promotion_choice, value=value).pack(anchor=tk.W)

        # Function to handle confirmation
        def confirm():
            promo = promotion_choice.get()
            promo_window.destroy()
            # Map to chess library constants
            promotion_map = {
                "q": chess.QUEEN,
                "r": chess.ROOK,
                "b": chess.BISHOP,
                "n": chess.KNIGHT
            }
            return_value = promotion_map.get(promo, chess.QUEEN)
            # Store the result in a nonlocal variable
            result_dict['result'] = return_value  # type: ignore
    
        # Confirm button
        tk.Button(promo_window, text="OK", command=confirm).pack(pady=10)

        # Wait for the window to close
        self.master.wait_window(promo_window)

        # Retrieve the result
        return result_dict.get('result', None)


    def on_right_click(self, event):
        if self.selected_square:
            self.removeHighlightedSquare()
            self.selected_square = None

    def start_ai_move(self):
        if self.ai_thread and self.ai_thread.is_alive():
            # AI is already thinking
            return
        self.ai_busy = True
        self.ai_thread = threading.Thread(target=self.ai_move, daemon=True)
        self.ai_thread.start()
        self.master.after(100, self.check_ai_thread)

    def check_ai_thread(self):
        if self.ai_thread and self.ai_thread.is_alive():
            self.master.after(100, self.check_ai_thread)
        else:
            # AI has finished its move
            start_time, best_move = self.queue.get()
            self.board.push(best_move)
            self.update_board()
            if self.board.is_game_over():
                result = self.board.result()
                self.message.config(text=f"Game Over: {result}")
            else:
                self.message.config(text="Your move")
                # Execute premove if any
                if self.premove:
                    self.board.push(self.premove)
                    self.update_board()
                    self.premove = None
                    if self.board.is_game_over():
                        result = self.board.result()
                        self.message.config(text=f"Game Over: {result}")
                    else:
                        self.message.config(text="AI's move")
                        self.start_ai_move()
            self.ai_busy = False

    def ai_move(self):
        start_time = time.time()
        best_move = self.select_best_move(board=self.board, max_depth=3)  # Adjust max_depth as needed
        end_time = time.time()
        self.queue.put((start_time, best_move))
        print(f"AI move time: {end_time - start_time:.2f} seconds")

    def select_best_move(self, board, max_depth=3):
        best_move = None
        for depth in range(1, max_depth + 1):
            print(f"Searching at depth {depth}")
            current_best_move = self.iterative_deepening_search(board, depth)
            if current_best_move:
                best_move = current_best_move
        return best_move

    def iterative_deepening_search(self, board, depth):
        best_eval = -float('inf')
        best_move = None
        # Prioritize PV move if available
        ordered_moves = []
        if board.fen() in pv_table:
            pv_move = pv_table[board.fen()]
            if pv_move in board.legal_moves:
                ordered_moves = [pv_move] + [m for m in self.order_moves(board) if m != pv_move]
        else:
            ordered_moves = self.order_moves(board)

        for move in ordered_moves:
            board.push(move)
            eval_score = minimax_evaluate_helper(board, depth - 1, -float('inf'), float('inf'), False, self.transposition_table)
            board.pop()
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
        if best_move:
            # Update history and PV tables
            history_table[best_move] += 1
            pv_table[board.fen()] = best_move
        return best_move

    def order_moves(self, board):
        ordered_moves = []
        for move in board.legal_moves:
            if board.is_capture(move):
                ordered_moves.insert(0, move)  # Prioritize captures
            elif board.gives_check(move):
                ordered_moves.insert(0, move)  # Prioritize checks
            else:
                ordered_moves.append(move)
        # Further prioritize based on history heuristic
        ordered_moves.sort(key=lambda move: history_table[move], reverse=True)
        return ordered_moves

if __name__ == "__main__":
    # Initialize the model for the main process
    model = ChessNet()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        print("Model loaded successfully")
    except FileNotFoundError:
        messagebox.showerror("Error", "Model file not found. Ensure chess_model+.pth is in the program directory.")
        exit()
    
    # Define model_instance globally
    global model_instance
    model_instance = model  # Assign the loaded model to model_instance
    
    root = tk.Tk()
    app = ChessGUI(model, root)
    root.mainloop()
