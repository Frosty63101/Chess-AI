from enum import auto
from shutil import move
from tracemalloc import stop
import stockfish
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import chess
import numpy as np
from stockfish import Stockfish
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import threading
import psutil
import random
import sys

MODEL_PATH = "chess_model.pth"

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

def simulate(state, model):
    if state.is_game_over():
        result = state.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0
    input_tensor = board_to_tensor(state).unsqueeze(0)
    with torch.no_grad():
        return model(input_tensor).item()

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

def select_best_move(model, board, device):
    was_training = model.training
    model.eval()
    legal_moves = list(board.legal_moves)
    best_eval = -float('inf')
    best_move = None
    repetition_penalty = 0.1  # Penalty applied if a move results in repetition

    for move in legal_moves:
        board.push(move)
        
        # Check for repetitions to apply penalty
        if board.is_repetition(3):
            penalty = repetition_penalty
        else:
            penalty = 0.0

        # Get the evaluation score from the model
        state = board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            eval_score = model(state).item() - penalty  # Apply penalty if repeated

        # Check if this move is better than the current best
        if eval_score > best_eval:
            best_eval = eval_score
            best_move = move

        board.pop()  # Undo move to explore other options

    # Restore model training mode if it was initially training
    if was_training:
        model.train()
    return best_move if best_move else random.choice(legal_moves)


def train(model, stockfish_path, episodes=1000, lr=0.001, stop_event=None, current_loss=None, current_episode=None, root=None, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stockfish = Stockfish(stockfish_path)
    stockfish.set_skill_level(20)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    # Epsilon decay parameters for exploration-exploitation balance
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.9995

    for episode in range(episodes):
        # Update epsilon
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

        if stop_event is not None and stop_event.is_set():
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Training stopped by user. Model saved to {MODEL_PATH}")
            exit_program()
            sys.exit()

        board = chess.Board()
        states = []
        targets = []

        while not board.is_game_over():
            state = board_to_tensor(board).unsqueeze(0).to(device)
            # Select move
            if random.uniform(0, 1) < epsilon:
                # Exploration: select a random move
                chosen_move = random.choice(list(board.legal_moves))
            else:
                # Exploitation: select the best move according to the model
                chosen_move = select_best_move(model, board, device)
            board.push(chosen_move)

            # Get Stockfish evaluation of the new board state
            stockfish.set_fen_position(board.fen())
            stockfish_eval = stockfish.get_evaluation()
            if stockfish_eval['type'] == 'cp':
                stockfish_value = stockfish_eval['value'] / 100.0
            elif stockfish_eval['type'] == 'mate':
                stockfish_value = np.sign(stockfish_eval['value']) * 1000.0  # Large value for mate
            else:
                stockfish_value = 0.0

            # Clip the value to a reasonable range
            stockfish_value = np.clip(stockfish_value, -10, 10)
            target_eval = torch.tensor([stockfish_value], dtype=torch.float32).to(device)

            # Collect data
            states.append(state)
            targets.append(target_eval)

        # After the game, train the model on collected data
        dataset = torch.utils.data.TensorDataset(torch.cat(states), torch.stack(targets))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        model.train()
        for batch_states, batch_targets in dataloader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)

        avg_loss = total_loss / len(dataset)

        print(f"Episode {episode + 1}/{episodes} - Loss: {avg_loss:.4f}")

        if current_loss is not None:
            current_loss[0] = avg_loss
        if current_episode is not None:
            current_episode[0] = episode + 1
    torch.save(model.state_dict(), MODEL_PATH)
    exit_program()
    sys.exit()


def play_against_stockfish(model, stockfish_path, skill_level=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    game_window = tk.Toplevel()
    game_window.title("Chess AI vs Stockfish")

    image_refs = {}
    stockfish = Stockfish(stockfish_path)
    stockfish.set_skill_level(skill_level)
    
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
            filename = os.path.join('images', f"{piece}.png")
        else:
            filename = os.path.join('images', f"_{piece}.png")
        piece_image = Image.open(filename).resize((50, 50), Image.Resampling.LANCZOS)
        piece_images[piece] = piece_image

    # Set up GUI layout
    board_canvas = tk.Canvas(game_window, width=400, height=400)
    board_canvas.pack(side=tk.LEFT)

    info_frame = tk.Frame(game_window)
    info_frame.pack(side=tk.RIGHT, fill=tk.Y)

    # Move list display
    move_list = tk.Text(info_frame, width=30, height=20)
    move_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    move_list.insert('end', "Move List\n" + "AI vs Stockfish\n")

    stockfish_eval_label = tk.Label(info_frame, text="Stockfish Evaluation: N/A")
    stockfish_eval_label.pack()

    # Resume Game button
    def resume_game():
        board.set_fen(boards[-1])
        update_board()
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
        board_images.append(board_tk)
        board_canvas.create_image(0, 0, anchor="nw", image=board_tk)

    def draw_board(board, image):
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
                    piece_image = piece_images[piece_symbol]
                    image.paste(piece_image, (x0, y0), piece_image)

    def next_move():
        if len(boards) != 1:
            resume_game()
        nonlocal board, move_number
        model_eval = 0.0
        stockfish_value = 0.0
        if board.turn == chess.WHITE:
            # AI's move
            state = board_to_tensor(board).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                evaluations_list = []
                legal_moves = list(board.legal_moves)
                for move in legal_moves:
                    board.push(move)
                    eval_state = board_to_tensor(board).unsqueeze(0).to(device)
                    eval_score = model(eval_state).item()
                    evaluations_list.append((eval_score, move))
                    board.pop()
                best_move = max(evaluations_list, key=lambda x: x[0])[1]
                move_san = board.san(best_move)  # Get SAN before pushing the move
                board.push(best_move)
                moves_san.append(move_san)
                boards.append(board.fen())

                # Get evaluations
                # Model evaluation
                state = board_to_tensor(board).unsqueeze(0).to(device)
                with torch.no_grad():
                    model_eval = model(state).item()

                # Stockfish evaluation
                stockfish.set_fen_position(board.fen())
                stockfish_eval = stockfish.get_evaluation()
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
                    messagebox.showinfo("Game Over", "Game over: AI Won!")
                    game_window.destroy()
                    return 0

        else:
            # Stockfish's move
            stockfish.set_fen_position(board.fen())
            stockfish_move = stockfish.get_best_move()
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
                    model_eval = model(state).item()

                # Stockfish evaluation
                stockfish.set_fen_position(board.fen())
                stockfish_eval = stockfish.get_evaluation()
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
                    messagebox.showinfo("Game Over", "Game over: Stockfish Won!")
                    game_window.destroy()
                    return 0

        # Update evaluations labels
        stockfish_eval_label.config(text=f"Stockfish Evaluation: {stockfish_value:.2f}")

        # Update board display
        update_board()
        return 1

    def on_move_click(event):
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
        model_eval, stockfish_value = evaluations[move_index]
        stockfish_eval_label.config(text=f"Stockfish Evaluation: {stockfish_value:.2f}")
    
    def auto_play():
        if auto_play_toggle.get():
            if next_move():
                game_window.after(1000, auto_play)

    def on_toggle_auto_play():
        if auto_play_toggle.get():
            auto_play()

    auto_play_label = tk.Label(game_window, text="Play against Stockfish:")
    auto_play_label.pack(side=tk.BOTTOM)
    auto_play_toggle = tk.BooleanVar()
    auto_play_checkbox = tk.Checkbutton(game_window, variable=auto_play_toggle, command=on_toggle_auto_play)
    auto_play_checkbox.pack(side=tk.BOTTOM)

    move_button = tk.Button(game_window, text="Next Move", command=next_move)
    move_button.pack(side=tk.BOTTOM)

    update_board()
    game_window.mainloop()


if __name__ == "__main__":
    stockfish_path = "C:/Users/samue/OneDrive/Desktop/Chess AI/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"  # Update to your Stockfish path
    model = ChessNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded from", MODEL_PATH)
    except FileNotFoundError:
        print("No saved model found. Starting training from scratch.")

    stop_event = threading.Event()
    current_loss = [0.0]
    current_episode = [0]
    training_thread = None

    root = tk.Tk()
    root.title("Chess AI Trainer")
    root.geometry("300x300")

    episodes_label = tk.Label(root, text="Number of Episodes:")
    episodes_label.pack()

    episodes_entry = tk.Entry(root)
    episodes_entry.pack()
    episodes_entry.insert(0, "1000")

    def start_training():
        episodes = int(episodes_entry.get())

        global training_thread
        training_thread = threading.Thread(
            target=train,
            args=(model, stockfish_path, episodes, 0.001, stop_event, current_loss, current_episode, root)
        )
        training_thread.start()

        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

        update_gui()

    def play():
        stockfish_difficulty = int(stockfish_difficulty_entry.get())
        if stockfish_difficulty < 1:
            stockfish_difficulty = 1
        elif stockfish_difficulty > 20:
            stockfish_difficulty = 20
        play_against_stockfish(model, stockfish_path, stockfish_difficulty)
    
    def exit_program():
        """Exit the program by stopping threads and closing the GUI."""
        if root:
            # Schedule root.quit() and root.destroy() on the main thread
            root.after(100, root.quit)  # Ensures all Tk windows close on the main thread
            root.after(200, root.destroy)  # Closes the main window on the main thread

        # Use os._exit(0) to fully terminate after main thread has processed events
        root.after(300, os._exit, 0)  # Slight delay ensures destroy is processed first

    start_button = tk.Button(root, text="Start Training", command=start_training)
    start_button.pack()

    def stop_training():
        stop_event.set()
        messagebox.showinfo("Training", "Training will stop after the current episode.")

    stop_button = tk.Button(root, text="Stop Training", command=stop_training)
    stop_button.pack()
    stop_button.config(state=tk.DISABLED)

    loss_label = tk.Label(root, text="Current Loss: N/A")
    loss_label.pack()

    episode_label = tk.Label(root, text="Current Episode: 0")
    episode_label.pack()

    cpu_label = tk.Label(root, text="CPU Usage: N/A")
    cpu_label.pack()

    memory_label = tk.Label(root, text="Memory Usage: N/A")
    memory_label.pack()
    
    stockfish_difficulty_label = tk.Label(root, text="Stockfish Difficulty (1-20):")
    stockfish_difficulty_label.pack()
    stockfish_difficulty_entry = tk.Entry(root)
    stockfish_difficulty_entry.pack()
    stockfish_difficulty_entry.insert(0, "10")
    
    play_button = tk.Button(root, text="Play", command=play)
    play_button.pack()
    
    close_button = tk.Button(root, text="Close", command=exit_program)
    close_button.pack()

    def update_gui():
        loss_label.config(text=f"Current Loss: {current_loss[0]:.4f}")
        episode_label.config(text=f"Current Episode: {current_episode[0]}")
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
        memory_label.config(text=f"Memory Usage: {memory_usage}%")
        if training_thread is not None and training_thread.is_alive():
            root.after(1000, update_gui)
        else:
            start_button.config(state=tk.NORMAL)

    root.mainloop()
    
    if not stop_event.is_set() and training_thread is not None and training_thread.is_alive():
        training_thread.join()  # Wait for thread termination before saving
    print("Program exited.")
