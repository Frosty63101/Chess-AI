"""
training.py

Manages the training of the AdvancedChessNet:
  - Self-play or Stockfish-driven experience generation
  - Collecting state/value pairs
  - Doing gradient updates
  - Periodically evaluating performance
"""

import os
import time
import random
import chess
import torch
import numpy as np
from collections import deque
from typing import Optional
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.functional as F
import csv
import io
from glob import glob
import chess.pgn

from config import (MODEL_PATH, BACKUP_MODEL_PATH, MIN_EVAL, MAX_EVAL,
                    MAX_TIME, ACTION_SIZE)

from ai import (ChessAI, boardToTensor, moveToIndex, materialEvaluation,
                evaluateBoard)
from stockfish import Stockfish
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

def periodicEvaluation(ai: ChessAI, episodes: int = 3, skillLevel: int = 20):
    stockfish = ai.stockfish
    stockfish.set_skill_level(skillLevel)
    device = ai.device

    ai.model.eval()

    winCount, drawCount, lossCount = 0, 0, 0
    evaluationMemory = []

    for _ in range(episodes):
        board = chess.Board()
        board.turn = chess.WHITE

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                bestMove = ai.selectBestMove(board)
                if bestMove is None:
                    bestMove = random.choice(list(board.legal_moves))
                board.push(bestMove)

                state = boardToTensor(board).unsqueeze(0).to(device)
                stockfish.set_fen_position(board.fen())
                fishEval = stockfish.get_evaluation()
                fishVal = fishEval['value'] / 100.0 if fishEval['type'] == 'cp' else np.sign(fishEval['value']) * 10.0
                fishVal = fishEval['value'] / 100.0 if fishEval['type'] == 'cp' else np.sign(fishEval['value']) * 10.0
                fishVal = max(min(fishVal, MAX_EVAL), MIN_EVAL)
                targetEval = torch.tensor([fishVal], dtype=torch.float32).to(device)
                evaluationMemory.append((state, targetEval))

                if board.is_game_over():
                    break
            else:
                stockfish.set_fen_position(board.fen())
                fishMove = stockfish.get_best_move()
                if fishMove:
                    board.push(chess.Move.from_uci(fishMove))
                else:
                    break

                state = boardToTensor(board).unsqueeze(0).to(device)
                stockfish.set_fen_position(board.fen())
                fishEval = stockfish.get_evaluation()
                fishVal = fishEval['value'] / 100.0 if fishEval['type'] == 'cp' else np.sign(fishEval['value']) * 10.0
                fishVal = fishEval['value'] / 100.0 if fishEval['type'] == 'cp' else np.sign(fishEval['value']) * 10.0
                fishVal = -fishVal
                fishVal = max(min(fishVal, MAX_EVAL), MIN_EVAL)
                targetEval = torch.tensor([fishVal], dtype=torch.float32).to(device)
                evaluationMemory.append((state, targetEval))

                if board.is_game_over():
                    break

        result = board.result()
        if result == "1-0":
            winCount += 1
        elif result == "0-1":
            lossCount += 1
        else:
            drawCount += 1

    print(f"[Eval vs Stockfish (Skill={skillLevel})] Wins: {winCount}, Draws: {drawCount}, Losses: {lossCount}")

    ai.transpositionTable.clear()
    ai.pvTable.clear()
    ai.historyTable.clear()

    return winCount, drawCount, lossCount
    return winCount, drawCount, lossCount

def randomChunkCsv(csvFilePath, chunkSize=10000, encoding='utf-8', seed=None):
    """
    Reads up to `chunkSize` lines from a random position in the CSV file,
    then returns (lines, header).
    
    - csvFilePath: path to the CSV.
    - chunkSize: how many lines (excluding the header) to read at most.
    - encoding: file encoding (usually 'utf-8').
    - seed: if you want reproducible random offset, set this.

    Returns:
        (sampleLines, header)
        where `sampleLines` is a list of raw CSV lines (strings),
        and `header` is the CSV header line (string).

    Notes:
    - We do a random seek into the file, discard the partial line, then read `chunkSize`.
    - This only pulls a fraction of the entire file, so it's not strictly uniform.
    - Perfect for "semi-frequent" partial sampling.
    """

    if seed is not None:
        random.seed(seed)

    fileSize = os.path.getsize(csvFilePath)
    if fileSize == 0:
        return [], None

    with open(csvFilePath, 'r', encoding=encoding) as f:
        # 1) Read and store the header
        header = f.readline()
        if not header:
            # Empty file or no header line
            return [], None

        # 2) Pick a random offset in [0, fileSize)
        #    We'll skip offset=0 area because that's near the header.
        #    If your CSV's first line is always a header, you might do random offset in [len(header), fileSize).
        offset = random.randint(len(header), fileSize - 1)

        # 3) Seek to that offset
        f.seek(offset)

        # 4) Read/discard the partial line we are in the middle of
        _ = f.readline()

        # 5) Now read up to `chunkSize` lines from here
        sampleLines = []
        print(f"Reading from offset {offset} in file '{csvFilePath}'")  # Debug info for random sampling
        for i in range(chunkSize):
            print(i, end='')  # Debug: show progress in reading lines
            line = f.readline()
            if not line:
                break
            sampleLines.append(line)
        print()  # Newline for cleaner output after reading lines
        print("closing file")  # Debug info to show we are done reading lines

    return sampleLines, header

def parseSampledCsvLines(sampleLines, header):
    """
    Takes raw lines and the header, and parses them with csv.DictReader.
    Returns a list of dict rows. E.g. for puzzle usage:
        [
          { 'PuzzleId': 'abc123', 'FEN': '....', 'Moves': '....', ... },
          ...
        ]
    """
    print("Parsing sampled CSV lines...")
    csvContent = header + "".join(sampleLines)
    f = io.StringIO(csvContent)
    reader = csv.DictReader(f)
    rows = [row for row in reader]
    return rows

def puzzleToTrainingExamples(row, ai, maxPositions=5):
    """
    Converts a Lichess puzzle row into training examples using the given moves
    without relying on Stockfish evaluations.

    Each training example is a (stateTensor, targetEvalTensor) pair.
    The target is fixed (+1 for White's good move, -1 for Black's), with optional noise.

    Parameters:
    - row: A dictionary representing one puzzle.
    - ai: The ChessAI instance (for device).
    - maxPositions: Max number of positions to extract from the puzzle.

    Returns:
    - A list of (stateTensor, targetEvalTensor) tuples.
    Converts a Lichess puzzle row into training examples using the given moves
    without relying on Stockfish evaluations.

    Each training example is a (stateTensor, targetEvalTensor) pair.
    The target is fixed (+1 for White's good move, -1 for Black's), with optional noise.

    Parameters:
    - row: A dictionary representing one puzzle.
    - ai: The ChessAI instance (for device).
    - maxPositions: Max number of positions to extract from the puzzle.

    Returns:
    - A list of (stateTensor, targetEvalTensor) tuples.
    """
    print(f"Processing puzzle: {row['PuzzleId']} with FEN: {row['FEN']} and moves: {row['Moves']}")
    examples = []


    fen = row['FEN']
    movesStr = row['Moves']

    movesStr = row['Moves']

    try:
        board = chess.Board(fen)
    except ValueError:
        return examples  # Skip puzzles with invalid FEN

        return examples  # Skip puzzles with invalid FEN

    moveUcis = movesStr.strip().split()
    device = ai.device


    moveCount = 0
    for moveUci in moveUcis:
        if moveCount >= maxPositions or board.is_game_over():
            break

        # Convert board to input tensor before applying the move

        # Convert board to input tensor before applying the move
        stateTensor = boardToTensor(board).unsqueeze(0).to(device)

        # Determine target value based on side to move
        baseValue = 1.0 if board.turn == chess.WHITE else -1.0
        noise = random.uniform(-0.05, 0.0)  # Slight noise to prevent overfitting
        targetValue = baseValue + noise
        targetEvalTensor = torch.tensor([targetValue], dtype=torch.float32).to(device)

        # Try applying the move

        # Determine target value based on side to move
        baseValue = 1.0 if board.turn == chess.WHITE else -1.0
        noise = random.uniform(-0.05, 0.0)  # Slight noise to prevent overfitting
        targetValue = baseValue + noise
        targetEvalTensor = torch.tensor([targetValue], dtype=torch.float32).to(device)

        # Try applying the move
        try:
            moveObj = chess.Move.from_uci(moveUci.strip())
            moveObj = chess.Move.from_uci(moveUci.strip())
            if moveObj not in board.legal_moves:
                break
            policyTargetIndex = moveToIndex(moveObj)
            policyTargetTensor = torch.tensor([policyTargetIndex], dtype=torch.long).to(device)
            examples.append((stateTensor, targetEvalTensor, policyTargetTensor, 0))
            policyTargetIndex = moveToIndex(moveObj)
            policyTargetTensor = torch.tensor([policyTargetIndex], dtype=torch.long).to(device)
            examples.append((stateTensor, targetEvalTensor, policyTargetTensor, 0))
            board.push(moveObj)
        except ValueError:
            break


        moveCount += 1


    return examples

def parsePGNGames(folderPath: str, maxGames: int = 10):
    """
    Randomly selects PGN files and parses a few games.
    Returns a list of `chess.pgn.Game` objects.
    """
    allFiles = glob(os.path.join(folderPath, '*.pgn'))
    if not allFiles:
        return []

    selectedFile = random.choice(allFiles)
    games = []
    with open(selectedFile, 'r', encoding='utf-8', errors='ignore') as f:
        while len(games) < maxGames:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games

def gameToTrainingExamples(game: chess.pgn.Game, ai: ChessAI, maxPositions: int = 30):
    """
    Converts a PGN game into training examples using the mainline moves.

    Returns a list of (stateTensor, targetEvalTensor, policyTargetTensor, age) tuples.
    """
    examples = []
    board = game.board()
    device = ai.device

    node = game
    moveCount = 0
    while node.variations and moveCount < maxPositions:
        move = node.variations[0].move
        if move not in board.legal_moves:
            break

        # Convert board to input tensor before applying move
        stateTensor = boardToTensor(board).unsqueeze(0).to(device)
        baseValue = 1.0 if board.turn == chess.WHITE else -1.0
        noise = random.uniform(-0.05, 0.0)
        targetValue = baseValue + noise
        targetEvalTensor = torch.tensor([targetValue], dtype=torch.float32).to(device)
        policyTargetTensor = torch.tensor([moveToIndex(move)], dtype=torch.long).to(device)

        examples.append((stateTensor, targetEvalTensor, policyTargetTensor, 0))
        board.push(move)
        node = node.variations[0]
        moveCount += 1


    return examples

def train(ai: ChessAI, episodes: int = 1000, lr: float = 0.0001,
          stopEvent: Optional[any] = None, currentLoss: Optional[list] = None,
          currentEpisode: Optional[list] = None, evaluationInterval: int = 25,
          batchSize: int = 128, bufferSize: int = 50000,
          lossHistory: Optional[list] = None,
          performanceHistory: Optional[list] = None,
          csvFilePath: str = "lichess_db_puzzle.csv\lichess_db_puzzle.csv",
          chunkSize: int = 1000, maxSampleAge: int = 100):

    device = ai.device
    model = ai.model.to(device)
    model = ai.model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = nn.SmoothL1Loss()
    scaler = GradScaler('cuda')

    if lossHistory is None:
        lossHistory = []
    if performanceHistory is None:
        performanceHistory = []

    replayBuffer = deque(maxlen=bufferSize)
    startEpisode = 0


    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'loss_history' in checkpoint:
            lossHistory.extend(checkpoint['loss_history'])
        if 'performance_history' in checkpoint:
            performanceHistory.extend(checkpoint['performance_history'])
        if 'replay_buffer' in checkpoint:
            for st, tv, policyTarget, age in checkpoint['replay_buffer']:
                replayBuffer.append((st.to(device), tv.to(device), policyTarget, age))
        startEpisode = len(lossHistory)
        print(f"Resuming training from episode {startEpisode + 1}.")

    for episodeIndex in range(startEpisode, episodes + startEpisode):
        if stopEvent and stopEvent.is_set():
            ai.saveModel(optimizer, scheduler, lossHistory, performanceHistory, replayBuffer)
            print("Training stopped by user.")
            return

        timeStart = time.perf_counter()

        # Dynamic noise decay
        noiseLevel = max(0.05 * (1 - episodeIndex / episodes), 0.005)

        episodeMemory = []
        board = chess.Board()
        
        r = random.random()
        
        if r < 0.2 and episodeIndex > 50:
            # 20% chance - sample full games from PGN
            games = parsePGNGames("twic_pgn_files", maxGames=10)
            for game in games:
                gameExamples = gameToTrainingExamples(game, ai, maxPositions=50)
                replayBuffer.extend(gameExamples)
        
        elif r < 0.5 and episodeIndex > 50:
            # 30% chance total (20% PGN, 30% puzzles) - sample puzzles
            sampleLines, header = randomChunkCsv(csvFilePath, chunkSize=chunkSize)
            parsedRows = parseSampledCsvLines(sampleLines, header)
            for row in parsedRows:
                puzzleExamples = puzzleToTrainingExamples(row, ai, maxPositions=5)
                replayBuffer.extend(puzzleExamples)
        else:
            while not board.is_game_over() and len(episodeMemory) < batchSize:
                ai.stockfish.set_fen_position(board.fen())
                fishMove = ai.stockfish.get_best_move()
                if not fishMove:
                    break
                moveObj = chess.Move.from_uci(fishMove)

                # Convert board to tensor and get policy index
                state = boardToTensor(board).unsqueeze(0).to(device)
                policyTargetIndex = moveToIndex(moveObj)

                if not fishMove:
                    break
                moveObj = chess.Move.from_uci(fishMove)

                # Convert board to tensor and get policy index
                state = boardToTensor(board).unsqueeze(0).to(device)
                policyTargetIndex = moveToIndex(moveObj)

                board.push(moveObj)

                state = boardToTensor(board).unsqueeze(0).to(device)
                ai.stockfish.set_fen_position(board.fen())
                fishEval = ai.stockfish.get_evaluation()
                fishVal = fishEval['value'] / 100.0 if fishEval['type'] == 'cp' else np.sign(fishEval['value']) * 10.0
                fishVal += random.uniform(-noiseLevel, noiseLevel)
                fishVal = fishEval['value'] / 100.0 if fishEval['type'] == 'cp' else np.sign(fishEval['value']) * 10.0
                fishVal += random.uniform(-noiseLevel, noiseLevel)
                fishVal = max(min(fishVal, MAX_EVAL), MIN_EVAL)
                target = torch.tensor([fishVal], dtype=torch.float32).to(device)
                policyTargetTensor = torch.tensor([policyTargetIndex], dtype=torch.long).to(device)
                episodeMemory.append((state, target, policyTargetTensor, 0))

            replayBuffer.extend(episodeMemory)
        
        agedBuffer = deque()
        for state, valueTarget, policyTarget, age in replayBuffer:
            if age + 1 < maxSampleAge:
                agedBuffer.append((state, valueTarget, policyTarget, age + 1))
        replayBuffer = agedBuffer  # Replace buffer with aged version

        if len(replayBuffer) >= batchSize:
            # Prioritize high-reward samples
            weights = [abs(value.item()) for _, value, _, _ in replayBuffer]
            batch = random.choices(replayBuffer, weights=weights, k=batchSize)
            states, valueTargets, policyTargets, _ = zip(*batch)
            states = torch.cat(states).to(device)
            valueTargets = torch.stack(valueTargets).to(device)
            policyTargets = torch.cat(policyTargets).to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                policyOut, valueOut = model(states)
                valueLoss = criterion(valueOut, valueTargets)
                policyLoss = F.cross_entropy(policyOut, policyTargets)
                combinedLoss = valueLoss + 0.5 * policyLoss  # You can tune this weight
            scaler.scale(combinedLoss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(combinedLoss)
            newLr = lr * (0.5 ** (episodeIndex // 200))
            for paramGroup in optimizer.param_groups:
                paramGroup['lr'] = newLr

            if currentLoss is not None:
                currentLoss[0] = combinedLoss.item()
            lossHistory.append(combinedLoss.item())

        if currentEpisode is not None:
            currentEpisode[0] = episodeIndex + 1

        if (episodeIndex + 1) % evaluationInterval == 0:
            print("Performing periodic evaluation...")
            w, d, l = periodicEvaluation(ai)
            performanceHistory.append((w, d, l))
            ai.periodicallySaveModel(optimizer, scheduler, lossHistory, performanceHistory, replayBuffer)

        timeEnd = time.perf_counter()
        if lossHistory:
            if len(lossHistory) >= 10:
                meanLoss = np.mean(lossHistory[-10:])
                print(f"Episode {episodeIndex + 1}: Loss={combinedLoss.item():.4f}, Mean(10)={meanLoss:.4f}, Noise={noiseLevel:.4f}, Time={timeEnd - timeStart:.2f}s")
            else:
                print(f"Episode {episodeIndex + 1}: Loss={combinedLoss.item():.4f}, Noise={noiseLevel:.4f}, Time={timeEnd - timeStart:.2f}s")

    ai.saveModel(optimizer, scheduler, lossHistory, performanceHistory, replayBuffer)
    print("Training completed and model saved.")