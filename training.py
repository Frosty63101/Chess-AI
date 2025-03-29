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
import csv
import io

from config import (MODEL_PATH, BACKUP_MODEL_PATH, MIN_EVAL, MAX_EVAL,
                    MAX_TIME, ACTION_SIZE)

from ai import (ChessAI, boardToTensor, moveToIndex, materialEvaluation,
                evaluateBoard)
from stockfish import Stockfish
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

def periodicEvaluation(ai: ChessAI, episodes: int = 3, skillLevel: int = 10):
    """
    Evaluate the AI's performance against Stockfish for a small number of games.
    Collect experience for training. Return (wins, draws, losses, experience).
    """
    stockfish = ai.stockfish
    stockfish.set_skill_level(skillLevel)
    device = ai.device

    # We'll keep the model in eval mode to avoid training-time overhead
    ai.model.eval()

    winCount, drawCount, lossCount = 0, 0, 0
    evaluationMemory = []

    for _ in range(episodes):
        board = chess.Board()
        # Always let the AI be White, Stockfish be Black
        board.turn = chess.WHITE

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                bestMove = ai.selectBestMove(board)
                if bestMove is None:
                    bestMove = random.choice(list(board.legal_moves))
                board.push(bestMove)

                # Gather state/target
                state = boardToTensor(board).unsqueeze(0).to(device)
                stockfish.set_fen_position(board.fen())
                fishEval = stockfish.get_evaluation()
                if fishEval['type'] == 'cp':
                    fishVal = fishEval['value'] / 100.0
                elif fishEval['type'] == 'mate':
                    fishVal = np.sign(fishEval['value']) * 10.0
                else:
                    fishVal = 0.0
                fishVal = max(min(fishVal, MAX_EVAL), MIN_EVAL)
                targetEval = torch.tensor([fishVal], dtype=torch.float32).to(device)
                evaluationMemory.append((state, targetEval))

                if board.is_game_over():
                    break
            else:
                stockfish.set_fen_position(board.fen())
                fishMove = stockfish.get_best_move()
                if fishMove:
                    fishMoveObj = chess.Move.from_uci(fishMove)
                    board.push(fishMoveObj)
                else:
                    break

                state = boardToTensor(board).unsqueeze(0).to(device)
                stockfish.set_fen_position(board.fen())
                fishEval = stockfish.get_evaluation()
                if fishEval['type'] == 'cp':
                    fishVal = fishEval['value'] / 100.0
                elif fishEval['type'] == 'mate':
                    fishVal = np.sign(fishEval['value']) * 10.0
                else:
                    fishVal = 0.0
                # Because it's the opponent's move, from AI's perspective, invert sign
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

    return winCount, drawCount, lossCount, evaluationMemory

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
    Given one puzzle row from the CSV, create a list of (stateTensor, targetEvalTensor) pairs
    by stepping through the puzzle's moves.
    
    :param row: A dict with keys like:
        {
          'PuzzleId': ...,
          'FEN': ...,
          'Moves': "e8d7 a2e6 d7d8 f7f8",  # space-separated UCI moves
          'Rating': ...,
          'RatingDeviation': ...,
          'Popularity': ...,
          'NbPlays': ...,
          'Themes': ...,
          'GameUrl': ...,
          'OpeningTags': ...
        }
    :param ai: Your ChessAI instance (so we can access its Stockfish, device, etc.)
    :param maxPositions: The maximum number of positions from the puzzle we want to store
    
    Returns: a list of (state, targetEval) suitable for your replay buffer.
    """
    print(f"Processing puzzle: {row['PuzzleId']} with FEN: {row['FEN']} and moves: {row['Moves']}")

    # Prepare to store (state, targetEval) pairs
    examples = []
    
    # 1) Extract puzzle data
    fen = row['FEN']
    movesStr = row['Moves']  # e.g. "e8d7 a2e6 d7d8 f7f8"
    
    # 2) Create the board from the puzzle FEN
    try:
        board = chess.Board(fen)
    except ValueError:
        # If the FEN is invalid, just return empty
        return examples
    
    # Split the puzzle's moves by space
    moveUcis = movesStr.strip().split()
    
    # 3) For each move in the puzzle, do:
    #    - Convert board to a state tensor
    #    - Evaluate with Stockfish
    #    - Append (state, targetEval) to examples
    #    - Push the move on the board
    device = ai.device
    moveCount = 0
    
    for moveUci in moveUcis:
        if moveCount >= maxPositions or board.is_game_over():
            break
        
        # Convert current position to a state
        stateTensor = boardToTensor(board).unsqueeze(0).to(device)
        
        # Evaluate with Stockfish
        ai.stockfish.set_fen_position(board.fen())
        fishEval = ai.stockfish.get_evaluation()
        
        if fishEval['type'] == 'cp':
            fishVal = fishEval['value'] / 100.0
        elif fishEval['type'] == 'mate':
            # clamp to ±10.0 if it's a mate
            fishVal = np.sign(fishEval['value']) * 10.0
        else:
            fishVal = 0.0
        
        # Optionally you can incorporate a material bonus or keep it pure
        # fishVal += materialEvaluation(board)
        
        # clamp final value
        fishVal = max(min(fishVal, 10.0), -10.0)
        
        targetEvalTensor = torch.tensor([fishVal], dtype=torch.float32).to(device)
        
        # Add (state, targetEval) to examples
        examples.append((stateTensor, targetEvalTensor))
        
        # Attempt to push the next puzzle move onto the board
        try:
            moveObj = chess.Move.from_uci(moveUci)
            if moveObj not in board.legal_moves:
                # If not legal, puzzle might be malformed or there's a mismatch
                break
            board.push(moveObj)
        except ValueError:
            # Invalid UCI notation
            break
        
        moveCount += 1
    
    return examples

def train(
    ai: ChessAI,
    episodes: int = 1000,
    lr: float = 0.0001,
    stopEvent: Optional[any] = None,
    currentLoss: Optional[list] = None,
    currentEpisode: Optional[list] = None,
    evaluationInterval: int = 25,
    batchSize: int = 128,
    bufferSize: int = 50000,
    lossHistory: Optional[list] = None,
    performanceHistory: Optional[list] = None,
    chunkInterval=10,
    csvFilePath='lichess_db_puzzle.csv\lichess_db_puzzle.csv',
    chunkSize=100,
):
    """
    Main training loop that uses Stockfish to generate experiences for the AI,
    then periodically trains the net on those experiences.
    """
    device = ai.device
    model = ai.model
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    criterion = nn.SmoothL1Loss()
    scaler = GradScaler('cuda')

    if lossHistory is None:
        lossHistory = []
    if performanceHistory is None:
        performanceHistory = []

    replayBuffer = deque(maxlen=bufferSize)

    # Possibly load states if continuing training
    startEpisode = 0
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")
        if 'loss_history' in checkpoint and checkpoint['loss_history']:
            lossHistory.extend(checkpoint['loss_history'])
        if 'performance_history' in checkpoint and checkpoint['performance_history']:
            performanceHistory.extend(checkpoint['performance_history'])
        if 'replay_buffer' in checkpoint and checkpoint['replay_buffer']:
            for st, tv in checkpoint['replay_buffer']:
                # Move them to GPU
                replayBuffer.append((st.to(device), tv.to(device)))
        startEpisode = len(lossHistory)
        print(f"Resuming training from episode {startEpisode + 1}.")

    # Possibly load skill
    stockfishSkill = ai.getStockfishSkill()
    print(f"Current Stockfish skill: {stockfishSkill}")

    for episodeIndex in range(startEpisode, episodes + startEpisode):
        # If user requests stop
        if stopEvent is not None and stopEvent.is_set():
            ai.saveModel(optimizer, scheduler, lossHistory, performanceHistory, replayBuffer)
            print(f"Training stopped by user. Model saved to {MODEL_PATH}")
            return

        # --- New: randomly vary Stockfish skill every X episodes ---
        if episodeIndex % 50 == 0 and episodeIndex != 0:
            newSkill = random.randint(1, 20)
            ai.setStockfishSkill(newSkill)
            print(f"Randomly set Stockfish skill to {newSkill} for variety.")

        timeStart = time.perf_counter()

        # Periodic evaluation
        if (episodeIndex + 1) % evaluationInterval == 0:
            ai.periodicallySaveModel(
                optimizer, scheduler,
                lossHistory, performanceHistory,
                replayBuffer
            )
            print("=== Periodic Backup Saved ===")

            w, d, l, evaluationMemory = periodicEvaluation(ai, episodes=3, skillLevel=stockfishSkill)
            performanceHistory.append((w, d, l))
            replayBuffer.extend(evaluationMemory)

            # Training step
            if len(replayBuffer) >= batchSize:
                batch = random.sample(replayBuffer, batchSize)
                states, targets = zip(*batch)
                states = torch.cat([s for s in states]).to(device)
                targets = torch.stack([t for t in targets]).to(device)

                optimizer.zero_grad()
                with autocast('cuda'):
                    _, outputs = model(states)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(loss)
                if currentLoss is not None:
                    currentLoss[0] = loss.item()
                lossHistory.append(loss.item())
            else:
                loss = None

            if currentEpisode is not None:
                currentEpisode[0] = episodeIndex + 1

            timeEnd = time.perf_counter()
            if loss is not None:
                print(f"Episode {episodeIndex + 1}/{episodes + startEpisode} - Loss: {loss.item():.4f} - Time: {timeEnd - timeStart:.2f}s")
            else:
                print(f"Episode {episodeIndex + 1}/{episodes + startEpisode} - Loss: N/A - Time: {timeEnd - timeStart:.2f}s")

            continue
        
        if (episodeIndex + 1) % chunkInterval == 0:
            print(f"[Episode {episodeIndex+1}] Reading random CSV chunk of size {chunkSize}")
            sampleLines, header = randomChunkCsv(csvFilePath, chunkSize=chunkSize)
            parsedRows = parseSampledCsvLines(sampleLines, header)

            # Now convert those rows to training examples. For puzzles, for example:
            for row in parsedRows:
                puzzleExamples = puzzleToTrainingExamples(row, ai, maxPositions=5)
                replayBuffer.extend(puzzleExamples)

            print(f"Replay buffer size after chunk: {len(replayBuffer)}")

        # Otherwise, run "self-play" with Stockfish for data
        episodeMemory = []
        while len(episodeMemory) < batchSize:
            board = chess.Board()
            while not board.is_game_over():
                # White = Stockfish
                ai.stockfish.set_fen_position(board.fen())
                fishMove = ai.stockfish.get_best_move()
                if fishMove:
                    moveObj = chess.Move.from_uci(fishMove)
                else:
                    moveObj = random.choice(list(board.legal_moves))
                board.push(moveObj)

                fen = board.fen()
                ai.stockfish.set_fen_position(fen)
                fishEval = ai.stockfish.get_evaluation()
                if fishEval['type'] == 'cp':
                    fishVal = fishEval['value'] / 100.0
                elif fishEval['type'] == 'mate':
                    fishVal = np.sign(fishEval['value']) * 10.0
                else:
                    fishVal = 0.0
                fishVal = max(min(fishVal, MAX_EVAL), MIN_EVAL)

                # If it's black to move, from perspective of black we invert
                if board.turn == chess.BLACK:
                    fishVal = -fishVal

                combinedValue = fishVal + materialEvaluation(board)
                combinedValue = max(min(combinedValue, MAX_EVAL), MIN_EVAL)
                targetEval = torch.tensor([combinedValue], dtype=torch.float32).to(device)

                state = boardToTensor(board).unsqueeze(0).to(device)
                episodeMemory.append((state, targetEval))

                if board.is_game_over() or len(episodeMemory) >= batchSize:
                    break

                # Black = Stockfish
                ai.stockfish.set_fen_position(board.fen())
                fishMove = ai.stockfish.get_best_move()
                if fishMove:
                    moveObj = chess.Move.from_uci(fishMove)
                else:
                    moveObj = random.choice(list(board.legal_moves))
                board.push(moveObj)

                fen = board.fen()
                ai.stockfish.set_fen_position(fen)
                fishEval = ai.stockfish.get_evaluation()
                if fishEval['type'] == 'cp':
                    fishVal = fishEval['value'] / 100.0
                elif fishEval['type'] == 'mate':
                    fishVal = np.sign(fishEval['value']) * 10.0
                else:
                    fishVal = 0.0
                fishVal = max(min(fishVal, MAX_EVAL), MIN_EVAL)

                if board.turn == chess.WHITE:
                    fishVal = -fishVal

                combinedValue = fishVal + materialEvaluation(board)
                combinedValue = max(min(combinedValue, MAX_EVAL), MIN_EVAL)
                targetEval = torch.tensor([combinedValue], dtype=torch.float32).to(device)

                state = boardToTensor(board).unsqueeze(0).to(device)
                episodeMemory.append((state, targetEval))

                if board.is_game_over() or len(episodeMemory) >= batchSize:
                    break

        # Put the entire "episodeMemory" into the replayBuffer
        replayBuffer.extend(episodeMemory)

        # Now do the training step from the replayBuffer
        if len(replayBuffer) >= batchSize:
            batch = random.sample(replayBuffer, batchSize)
            states, targets = zip(*batch)
            states = torch.cat([s for s in states]).to(device)
            targets = torch.stack([t for t in targets]).to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                _, outputs = model(states)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(loss)

            if currentLoss is not None:
                currentLoss[0] = loss.item()
            lossHistory.append(loss.item())
        else:
            loss = None

        if currentEpisode is not None:
            currentEpisode[0] = episodeIndex + 1

        timeEnd = time.perf_counter()
        if loss is not None:
            print(f"Episode {episodeIndex + 1}/{episodes + startEpisode} - Loss: {loss.item():.4f} - Time: {timeEnd - timeStart:.2f}s")
        else:
            print(f"Episode {episodeIndex + 1}/{episodes + startEpisode} - Loss: N/A - Time: {timeEnd - timeStart:.2f}s")

    ai.saveModel(optimizer, scheduler, lossHistory, performanceHistory, replayBuffer)
    print("Training completed and model saved.")
