"""
main.py

Launches the Tkinter GUI and orchestrates everything:
  - Allows user to train the AI
  - Allows user to play against Stockfish or the AI model
  - Monitors resource usage & logs
"""

import sys
import os
import queue
import psutil
import time
import random
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")  # We ensure this for the canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

# We'll import from our modules
from config import (MODEL_PATH, BACKUP_MODEL_PATH, STOCKFISH_DEFAULT_PATH, IMAGE_DIR)
from ai import ChessAI
from training import train, periodicEvaluation
import threading

import chess
from PIL import Image, ImageTk
from stockfish import Stockfish
from collections import deque

# We'll maintain some lists that we can pass into train(...) to see updated results
currentLoss = [0.0]
currentEpisode = [0]
lossHistory = []
performanceHistory = []

class RedirectText:
    """
    A small helper to redirect stdout/stderr to a Tkinter-based Text box.
    """
    def __init__(self, logQueue):
        self.logQueue = logQueue

    def write(self, text):
        text = text.strip()
        if text:
            self.logQueue.put(text + "\n")

    def flush(self):
        pass

def playAgainstStockfish(ai: ChessAI, skillLevel: int = 10):
    """
    Launches a new Toplevel window allowing the AI to play against Stockfish.
    White is the AI; Black is Stockfish.
    """
    import chess
    gameWindow = tk.Toplevel()
    gameWindow.title("Chess AI vs Stockfish")

    board = chess.Board()
    board.turn = chess.WHITE

    moveNumber = 1
    movesSAN = []
    evaluations = []
    boards = [board.fen()]

    pieceImages = {}
    for piece in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
        if piece.isupper():
            filename = os.path.join(IMAGE_DIR, f"{piece}.png")
        else:
            filename = os.path.join(IMAGE_DIR, f"_{piece}.png")
        try:
            pieceImage = Image.open(filename).resize((50, 50), Image.Resampling.LANCZOS)
            pieceImages[piece] = pieceImage
        except FileNotFoundError:
            messagebox.showerror("Image Error", f"Couldn't find {filename}.")
            return

    boardCanvas = tk.Canvas(gameWindow, width=400, height=400)
    boardCanvas.pack(side=tk.LEFT)

    infoFrame = tk.Frame(gameWindow)
    infoFrame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    moveList = tk.Text(infoFrame, width=30, height=20)
    moveList.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    moveList.insert('end', "Move List\nAI vs Stockfish\n")

    stockfishEvalLabel = tk.Label(infoFrame, text="Stockfish Eval: N/A")
    stockfishEvalLabel.pack()
    aiEvalLabel = tk.Label(infoFrame, text="AI Eval: N/A")
    aiEvalLabel.pack()

    autoPlayVar = tk.BooleanVar()

    def drawBoard():
        import math
        boardImage = Image.new("RGB", (400, 400), "white")
        squareSize = 50
        for rank in range(8):
            for file in range(8):
                color = "lightgray" if (rank + file) % 2 == 0 else "darkgreen"
                x0 = file * squareSize
                y0 = (7 - rank) * squareSize
                x1 = x0 + squareSize
                y1 = y0 + squareSize
                boardImage.paste(color, (x0, y0, x1, y1))

                square = chess.square(file, 7 - rank)
                piece = board.piece_at(square)
                if piece:
                    pieceSymbol = piece.symbol()
                    pieceImg = pieceImages.get(pieceSymbol)
                    if pieceImg:
                        boardImage.paste(pieceImg, (x0, y0), pieceImg)

        boardTk = ImageTk.PhotoImage(boardImage)
        boardCanvas.create_image(0, 0, anchor="nw", image=boardTk)
        boardCanvas.image = boardTk  # keep a reference

    def nextMove():
        nonlocal moveNumber
        modelEval = 0.0
        stockfishEval = 0.0

        # AI is White
        if board.turn == chess.WHITE:
            chosenMove = ai.selectBestMove(board)
            if chosenMove is None:
                chosenMove = random.choice(list(board.legal_moves))
            moveSAN = board.san(chosenMove)
            board.push(chosenMove)
            movesSAN.append(moveSAN)
            boards.append(board.fen())

            with torch.no_grad():
                from ai import boardToTensor
                state = boardToTensor(board).unsqueeze(0).to(ai.device)
                _, value = ai.model(state)
                modelEval = max(min(value.item(), 10.0), -10.0)

            ai.stockfish.set_fen_position(board.fen())
            fishEval = ai.stockfish.get_evaluation()
            if fishEval['type'] == 'cp':
                stockfishEval = fishEval['value'] / 100.0
            elif fishEval['type'] == 'mate':
                stockfishEval = (fishEval['value'] / abs(fishEval['value'])) * 10.0

            stockfishEvalLabel.config(text=f"Stockfish Eval: {stockfishEval:.2f}")
            aiEvalLabel.config(text=f"AI Eval: {modelEval:.2f}")

            moveList.insert('end', f"{moveNumber}. {moveSAN} ")
            board.turn = chess.BLACK
            if board.is_game_over():
                messagebox.showinfo("Game Over", f"Game over: {board.result()}")
                gameWindow.destroy()
                return

        else:
            # Stockfish's move
            ai.stockfish.set_fen_position(board.fen())
            sfMove = ai.stockfish.get_best_move()
            if sfMove:
                sfMoveObj = chess.Move.from_uci(sfMove)
                moveSAN = board.san(sfMoveObj)
                board.push(sfMoveObj)
                movesSAN.append(moveSAN)
                boards.append(board.fen())
                board.turn = chess.WHITE

                with torch.no_grad():
                    from ai import boardToTensor
                    state = boardToTensor(board).unsqueeze(0).to(ai.device)
                    _, value = ai.model(state)
                    modelEval = value.item()

                ai.stockfish.set_fen_position(board.fen())
                fishEval = ai.stockfish.get_evaluation()
                if fishEval['type'] == 'cp':
                    stockfishEval = fishEval['value'] / 100.0
                elif fishEval['type'] == 'mate':
                    stockfishEval = (fishEval['value'] / abs(fishEval['value'])) * 10.0

                stockfishEvalLabel.config(text=f"Stockfish Eval: {stockfishEval:.2f}")
                aiEvalLabel.config(text=f"AI Eval: {modelEval:.2f}")

                moveList.insert('end', f"{moveSAN}\n")
                moveNumber += 1

                if board.is_game_over():
                    messagebox.showinfo("Game Over", f"Game over: {board.result()}")
                    gameWindow.destroy()
                    return

        drawBoard()

    def toggleAutoPlay():
        if autoPlayVar.get():
            autoPlayStep()

    def autoPlayStep():
        if not board.is_game_over() and autoPlayVar.get():
            nextMove()
            gameWindow.after(500, autoPlayStep)  # 0.5 second delay

        if board.is_game_over():
            messagebox.showinfo("Game Over", f"Game over: {board.result()}")
            gameWindow.destroy()

    moveButton = tk.Button(infoFrame, text="Next Move", command=nextMove)
    moveButton.pack(side=tk.BOTTOM)

    autoplayCheck = tk.Checkbutton(
        infoFrame,
        text="Auto-Play",
        variable=autoPlayVar,
        command=toggleAutoPlay
    )
    autoplayCheck.pack(side=tk.BOTTOM)

    drawBoard()

def playAgainstModel(ai: ChessAI):
    """
    Allows the user to play as White against the model as Black (or vice versa).
    """
    import chess
    gameWindow = tk.Toplevel()
    gameWindow.title("Play Against AI Model")

    board = chess.Board()
    board.turn = chess.WHITE

    moveNumber = 1
    movesSAN = []
    boards = [board.fen()]

    pieceImages = {}
    for piece in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']:
        if piece.isupper():
            filename = os.path.join(IMAGE_DIR, f"{piece}.png")
        else:
            filename = os.path.join(IMAGE_DIR, f"_{piece}.png")
        try:
            pieceImage = Image.open(filename).resize((50, 50), Image.Resampling.LANCZOS)
            pieceImages[piece] = pieceImage
        except FileNotFoundError:
            messagebox.showerror("Image Error", f"Couldn't find {filename}.")
            return

    boardCanvas = tk.Canvas(gameWindow, width=400, height=400)
    boardCanvas.pack(side=tk.LEFT)

    infoFrame = tk.Frame(gameWindow)
    infoFrame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    moveList = tk.Text(infoFrame, width=30, height=20)
    moveList.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    moveList.insert('end', "Move List\nYou vs AI Model\n")

    aiEvalLabel = tk.Label(infoFrame, text="AI Eval: N/A")
    aiEvalLabel.pack()

    boardImages = []
    selectedSquare = None

    def drawBoard():
        boardImage = Image.new("RGB", (400, 400), "white")
        squareSize = 50
        for rank in range(8):
            for file in range(8):
                color = "lightgray" if (rank + file) % 2 == 0 else "darkgreen"
                x0 = file * squareSize
                y0 = rank * squareSize
                x1 = x0 + squareSize
                y1 = y0 + squareSize
                boardImage.paste(color, (x0, y0, x1, y1))

                square = chess.square(file, 7 - rank)
                piece = board.piece_at(square)
                if piece:
                    pieceSymbol = piece.symbol()
                    pieceImg = pieceImages.get(pieceSymbol)
                    if pieceImg:
                        boardImage.paste(pieceImg, (x0, y0), pieceImg)

        boardTk = ImageTk.PhotoImage(boardImage)
        boardCanvas.create_image(0, 0, anchor="nw", image=boardTk)
        boardCanvas.image = boardTk

    def onClick(event):
        nonlocal selectedSquare, moveNumber
        file = event.x // 50
        rank = event.y // 50
        actualRank = 7 - rank
        square = chess.square(file, actualRank)

        if selectedSquare is None:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                selectedSquare = square
                highlightSquare(selectedSquare)
        else:
            if selectedSquare == square:
                unhighlightSquare()
                selectedSquare = None
                return
            move = chess.Move(selectedSquare, square)
            # Pawn promotion check
            attackerPiece = board.piece_at(selectedSquare)
            if attackerPiece and attackerPiece.piece_type == chess.PAWN:
                promotionRank = 7 if attackerPiece.color == chess.WHITE else 0
                if chess.square_rank(square) == promotionRank:
                    # Prompt user for piece
                    promotionPiece = promptPromotion()
                    if promotionPiece:
                        move = chess.Move(selectedSquare, square, promotion=promotionPiece)
                    else:
                        unhighlightSquare()
                        selectedSquare = None
                        return

            if move in board.legal_moves:
                moveSAN = board.san(move)
                board.push(move)
                movesSAN.append(moveSAN)
                boards.append(board.fen())
                unhighlightSquare()
                selectedSquare = None
                drawBoard()

                if board.turn == chess.BLACK:
                    moveList.insert('end', f"{moveNumber}. {moveSAN} ")
                else:
                    moveList.insert('end', f"{moveSAN}\n")
                    moveNumber += 1

                if board.is_game_over():
                    messagebox.showinfo("Game Over", f"Game over: {board.result()}")
                    gameWindow.destroy()
                    return

                # Now let AI move
                aiMove()
            else:
                messagebox.showinfo("Invalid Move", "This move is not allowed.")
                unhighlightSquare()
                selectedSquare = None
            drawBoard()

    highlightTag = "highlight"

    def highlightSquare(sq):
        boardCanvas.delete(highlightTag)
        sqSize = 50
        file = chess.square_file(sq)
        rank = 7 - chess.square_rank(sq)
        x0 = file * sqSize
        y0 = rank * sqSize
        x1 = x0 + sqSize
        y1 = y0 + sqSize
        boardCanvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2, tags=highlightTag)

    def unhighlightSquare():
        boardCanvas.delete(highlightTag)

    def promptPromotion():
        promotionWindow = tk.Toplevel(gameWindow)
        promotionWindow.title("Choose Promotion")
        promotionWindow.grab_set()
        promoChoice = tk.StringVar(value="q")

        def confirm():
            promotionWindow.destroy()

        tk.Label(promotionWindow, text="Promote pawn to:").pack(pady=5)
        for txt, val in [("Queen", "q"), ("Rook", "r"), ("Bishop", "b"), ("Knight", "n")]:
            tk.Radiobutton(promotionWindow, text=txt, variable=promoChoice, value=val).pack(anchor=tk.W)

        tk.Button(promotionWindow, text="OK", command=confirm).pack(pady=5)
        gameWindow.wait_window(promotionWindow)
        pieceMap = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
        return pieceMap.get(promoChoice.get(), chess.QUEEN)

    def aiMove():
        nonlocal moveNumber
        from ai import boardToTensor
        chosenMove = ai.selectBestMove(board)
        if chosenMove is None:
            chosenMove = random.choice(list(board.legal_moves))
        moveSAN = board.san(chosenMove)
        board.push(chosenMove)
        movesSAN.append(moveSAN)
        boards.append(board.fen())

        if board.turn == chess.WHITE:
            moveList.insert('end', f"{movesSAN[-1]}\n")
            moveNumber += 1
        else:
            moveList.insert('end', f"...{movesSAN[-1]} ")

        with torch.no_grad():
            state = boardToTensor(board).unsqueeze(0).to(ai.device)
            _, value = ai.model(state)
            modelEval = max(min(value.item(), 10.0), -10.0)
            aiEvalLabel.config(text=f"AI Eval: {modelEval:.2f}")

        drawBoard()
        if board.is_game_over():
            messagebox.showinfo("Game Over", f"Game over: {board.result()}")
            gameWindow.destroy()

    boardCanvas.bind("<Button-1>", onClick)

    def resetGame():
        nonlocal board, moveNumber, movesSAN, boards, selectedSquare
        board = chess.Board()
        board.turn = chess.WHITE
        moveNumber = 1
        movesSAN = []
        boards = [board.fen()]
        selectedSquare = None
        unhighlightSquare()
        moveList.delete('1.0', tk.END)
        moveList.insert('end', "Move List\nYou vs AI Model\n")
        aiEvalLabel.config(text="AI Eval: N/A")
        drawBoard()

    resetButton = tk.Button(infoFrame, text="Reset Game", command=resetGame)
    resetButton.pack()

    drawBoard()

def main():
    ai = ChessAI(STOCKFISH_DEFAULT_PATH, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    stopEvent = threading.Event()
    trainingThread = None

    root = tk.Tk()
    root.title("Chess AI Trainer")
    root.geometry("1000x800")

    controlFrame = tk.Frame(root)
    controlFrame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    plotFrame = tk.Frame(root)
    plotFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    logFrame = tk.Frame(root)
    logFrame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Episodes
    tk.Label(controlFrame, text="Episodes:").grid(row=0, column=0)
    episodesEntry = tk.Entry(controlFrame)
    episodesEntry.grid(row=0, column=1)
    episodesEntry.insert(0, "1000")

    # Stockfish skill
    tk.Label(controlFrame, text="Stockfish Skill (1-20):").grid(row=1, column=0)
    skillEntry = tk.Entry(controlFrame)
    skillEntry.grid(row=1, column=1)
    skillEntry.insert(0, "10")

    # Buttons
    def startTraining():
        nonlocal trainingThread
        try:
            episodes = int(episodesEntry.get())
            if episodes < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid number of episodes.")
            return

        startButton.config(state=tk.DISABLED)
        stopButton.config(state=tk.NORMAL)

        # Start a new training thread
        trainingThread = threading.Thread(
            target=train,
            args=(
                ai,
                episodes,
                0.0001,  # Learning rate
                stopEvent,
                currentLoss,
                currentEpisode,
                25,  # evaluationInterval
                128, # batchSize
                50000, # bufferSize
                lossHistory,
                performanceHistory
            )
        )
        trainingThread.start()
        updateGUI()

    def stopTraining():
        stopEvent.set()
        messagebox.showinfo("Training", "Training will stop soon after current iteration finishes.")

    startButton = tk.Button(controlFrame, text="Start Training", command=startTraining)
    startButton.grid(row=0, column=2, padx=5, pady=5)

    stopButton = tk.Button(controlFrame, text="Stop Training", command=stopTraining)
    stopButton.grid(row=1, column=2, padx=5, pady=5)
    stopButton.config(state=tk.DISABLED)

    def doPlayVsStockfish():
        try:
            skill = int(skillEntry.get())
            ai.setStockfishSkill(skill)
            playAgainstStockfish(ai, skill)
        except ValueError:
            messagebox.showerror("Error", "Skill must be an integer (1-20).")

    playFishButton = tk.Button(controlFrame, text="Play vs Stockfish", command=doPlayVsStockfish)
    playFishButton.grid(row=0, column=3, padx=5, pady=5)

    def doPlayVsAI():
        playAgainstModel(ai)

    playAIButton = tk.Button(controlFrame, text="Play vs AI Model", command=doPlayVsAI)
    playAIButton.grid(row=0, column=4, padx=5, pady=5)

    def onClose():
        stopEvent.set()
        if trainingThread is not None and trainingThread.is_alive():
            trainingThread.join()
        root.destroy()
        sys.exit(0)

    closeButton = tk.Button(controlFrame, text="Close", command=onClose)
    closeButton.grid(row=1, column=3, padx=5, pady=5)

    lossLabel = tk.Label(controlFrame, text="Loss: N/A")
    lossLabel.grid(row=2, column=0)
    episodeLabel = tk.Label(controlFrame, text="Episode: 0")
    episodeLabel.grid(row=2, column=1)
    cpuLabel = tk.Label(controlFrame, text="CPU Usage: ...")
    cpuLabel.grid(row=2, column=2)
    memLabel = tk.Label(controlFrame, text="RAM Usage: ...")
    memLabel.grid(row=2, column=3)
    deviceLabel = tk.Label(controlFrame, text=f"Device: {ai.device}")
    deviceLabel.grid(row=3, column=0)

    # Setup plots for loss and performance
    figLoss, axLoss = plt.subplots(figsize=(4, 3))
    axLoss.set_title("Loss Over Time")
    axLoss.set_xlabel("Episode")
    axLoss.set_ylabel("Loss")
    axLoss.grid(True)
    canvasLoss = FigureCanvasTkAgg(figLoss, master=plotFrame)
    canvasLoss.draw()
    canvasLoss.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    figPerf, axPerf = plt.subplots(figsize=(4, 3))
    axPerf.set_title("Performance (Wins/Draws/Losses)")
    axPerf.set_xlabel("Evaluation #")
    axPerf.set_ylabel("Count")
    axPerf.grid(True)
    canvasPerf = FigureCanvasTkAgg(figPerf, master=plotFrame)
    canvasPerf.draw()
    canvasPerf.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    logQueue = queue.Queue()
    logText = tk.Text(logFrame, wrap='word', height=10, state='disabled', bg='white')
    logText.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(logFrame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    logText.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=logText.yview)

    redirect = RedirectText(logQueue)
    sys.stdout = redirect
    sys.stderr = redirect

    def updateLog():
        while not logQueue.empty():
            message = logQueue.get_nowait()
            logText.config(state='normal')
            logText.insert(tk.END, message)
            logText.see(tk.END)
            logText.config(state='disabled')
        root.after(200, updateLog)

    updateLog()

    def updateGUI():
        # Show CPU / RAM usage
        lossLabel.config(text=f"Loss: {currentLoss[0]:.4f}")
        episodeLabel.config(text=f"Episode: {currentEpisode[0]}")
        cpuUsage = psutil.cpu_percent()
        memUsage = psutil.virtual_memory().percent
        cpuLabel.config(text=f"CPU Usage: {cpuUsage:.1f}%")
        memLabel.config(text=f"RAM Usage: {memUsage:.1f}%")

        # Update loss plot
        if lossHistory:
            axLoss.cla()
            axLoss.set_title("Loss Over Time")
            axLoss.set_xlabel("Episode")
            axLoss.set_ylabel("Loss")
            axLoss.grid(True)
            axLoss.plot(lossHistory, label="Training Loss")
            canvasLoss.draw()

        # Update performance plot
        if performanceHistory:
            axPerf.cla()
            axPerf.set_title("Performance (Wins/Draws/Losses)")
            axPerf.set_xlabel("Eval #")
            axPerf.set_ylabel("Count")
            axPerf.grid(True)
            wins, draws, losses = zip(*performanceHistory)
            axPerf.plot(range(len(wins)), wins, label='Wins')
            axPerf.plot(range(len(draws)), draws, label='Draws')
            axPerf.plot(range(len(losses)), losses, label='Losses')
            axPerf.legend()
            canvasPerf.draw()

        # Keep polling until user closes or training ends
        if trainingThread is not None and trainingThread.is_alive():
            root.after(1000, updateGUI)
        else:
            startButton.config(state=tk.NORMAL)
            stopButton.config(state=tk.DISABLED)

    root.mainloop()

if __name__ == "__main__":
    main()
