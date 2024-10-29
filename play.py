import tkinter as tk
from tkinter import messagebox
import chess
from matplotlib.pyplot import pie
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageTk
import os

MODEL_PATH = "chess_model.pth"  # Ensure this path matches the model's location

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Initial Convolutional layers with increased depth and feature maps
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # First Residual Block
        self.residual_block1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        # Second Residual Block
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )

        # Fully connected layers with increased units and Dropout for regularization
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)
        
        # Dropout layer with a probability of 0.4 to prevent overfitting
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Pass through initial convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        
        # Apply first residual block with skip connection
        residual1 = x
        x = self.residual_block1(x)
        x += residual1  # Skip connection
        
        # Apply second residual block with skip connection
        residual2 = x
        x = self.residual_block2(x)
        x += residual2  # Skip connection

        # Flatten the output for fully connected layers
        x = x.view(-1, 512 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for the output layer
        
        return x

def board_to_tensor(board):
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_symbol = piece.symbol()
            channel = piece_to_channel[piece_symbol]
            x, y = square % 8, square // 8
            board_tensor[channel, y, x] = 1
    return torch.tensor(board_tensor, dtype=torch.float32)

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
    
    def load_piece_images(self):
        self.piece_images = {}
        pieces = "PNBRQKpnbrqk"
        for piece in pieces:
            if piece.isupper():
                img_path = os.path.join("images", f"{piece.lower()}.png")
                self.piece_images[piece] = Image.open(img_path).resize((50, 50), Image.Resampling.LANCZOS)
            else:
                img_path = os.path.join("images", f"_{piece}.png")
                self.piece_images[piece] = Image.open(img_path).resize((50, 50), Image.Resampling.LANCZOS)

    def update_board(self):
        board_image = Image.new("RGB", (400, 400), "white")
        self.draw_board(self.board, board_image)
        board_tk = ImageTk.PhotoImage(board_image)
        self.board_images.append(board_tk)
        
        # Display on canvas
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
        # Adjusted to match the orientation in draw_board
        file, rank = event.x // 50, 7 - (event.y // 50)  # Ensure this matches board rendering
        square = chess.square(file, rank)
        
        self.drawHighlightedSquare(square)
        
        if self.selected_square is None:
            self.selected_square = square
        elif self.selected_square == square:
            self.removeHighlightedSquare()
            self.selected_square = None
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board()
                if self.board.is_game_over():
                    result = self.board.result()
                    self.message.config(text=f"Game Over: {result}")
                else:
                    self.message.config(text="AI's move")
                    self.master.after(500, self.ai_move)
            else:
                messagebox.showinfo("Invalid Move", "This move is not allowed.")
            self.removeHighlightedSquare()
            self.selected_square = None
    
    def on_right_click(self, event):
        if self.selected_square:
            self.removeHighlightedSquare()
            self.selected_square = None

    def ai_move(self):
        best_move = self.select_best_move()
        if best_move:
            self.board.push(best_move)
            self.update_board()
            if self.board.is_game_over():
                result = self.board.result()
                self.message.config(text=f"Game Over: {result}")
            else:
                self.message.config(text="Your move")

    def select_best_move(self):
        legal_moves = list(self.board.legal_moves)
        best_eval = -float('inf')
        best_move = None

        for move in legal_moves:
            self.board.push(move)
            state = board_to_tensor(self.board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                eval_score = self.model(state).item()
            if eval_score > best_eval:
                best_eval = eval_score
                best_move = move
            self.board.pop()
        
        return best_move

if __name__ == "__main__":
    root = tk.Tk()
    model = ChessNet()

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        print("Model loaded successfully")
    except FileNotFoundError:
        messagebox.showerror("Error", "Model file not found. Ensure chess_model.pth is in the program directory.")
        root.quit()

    app = ChessGUI(model, root)
    root.mainloop()
