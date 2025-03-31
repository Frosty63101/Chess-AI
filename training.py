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
    correctPolicyCount = 0
    totalPolicyCount = 0
    maxPolicyScore = 0

    evaluationMemory = []

    for _ in range(episodes):
        board = chess.Board()
        board.turn = chess.WHITE

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                bestMove = ai.selectBestMove(board)
                # Evaluate policy accuracy
                stateTensor = boardToTensor(board).unsqueeze(0).to(device)
                with torch.no_grad():
                    policyOutput, _ = ai.model(stateTensor)
                    policyProbs = torch.softmax(policyOutput[0], dim=0).cpu().numpy()

                legalMoves = list(board.legal_moves)
                if legalMoves:
                    bestPredictedMove = max(legalMoves, key=lambda mv: policyProbs[moveToIndex(mv)])
                    predictedIndex = moveToIndex(bestPredictedMove)

                    # Get Stockfish's best move
                    sfBestMoveUci = stockfish.get_best_move()
                    if sfBestMoveUci:
                        sfBestMove = chess.Move.from_uci(sfBestMoveUci)
                        if sfBestMove in board.legal_moves:
                            sfIndex = moveToIndex(sfBestMove)

                            # Only now is it safe to compare!
                            policyProbs = torch.softmax(policyOutput[0], dim=0).cpu()  # ← keep it a tensor
                            topK = torch.topk(policyProbs, k=3).indices.numpy()
                            if sfIndex in topK:
                                rank = list(topK).index(sfIndex) + 1  # 1-based index
                                credit = 1.0 / rank
                                correctPolicyCount += credit
                                maxPolicyScore += 1  # or sum of 1/rank for all topK if using advanced credit system
                            totalPolicyCount += 1

                if bestMove is None:
                    bestMove = random.choice(list(board.legal_moves))
                board.push(bestMove)

                state = boardToTensor(board).unsqueeze(0).to(device)
                stockfish.set_fen_position(board.fen())
                fishEval = stockfish.get_evaluation()
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
    if totalPolicyCount > 0:
        accuracy = maxPolicyScore / totalPolicyCount
        print(f"[Policy Accuracy vs Stockfish] {maxPolicyScore}/{totalPolicyCount} = {accuracy:.2%} with {correctPolicyCount:.2f} credit")
    else:
        print("[Policy Accuracy] No comparisons made.")


    ai.transpositionTable.clear()
    ai.pvTable.clear()
    ai.historyTable.clear()

    return winCount, drawCount, lossCount, accuracy

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
        for i in range(chunkSize):
            line = f.readline()
            if not line:
                break
            sampleLines.append(line)

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
    examples = []


    fen = row['FEN']
    movesStr = row['Moves']

    movesStr = row['Moves']

    try:
        board = chess.Board(fen)
    except ValueError:
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

def moveIndexDisagrees(policyOut, targetIndex):
    predictedIndex = torch.argmax(policyOut).item()
    return predictedIndex != targetIndex.item()

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
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
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
        noiseLevel = max(0.03 * (1 - episodeIndex / episodes), 0.001)

        episodeMemory = []
        board = chess.Board()
        
        r = random.random()
        
        if r < 0.2 and episodeIndex > 20:
            # 20% chance - sample full games from PGN
            games = parsePGNGames("twic_pgn_files", maxGames=100)
            for game in games:
                gameExamples = gameToTrainingExamples(game, ai, maxPositions=50)
                replayBuffer.extend(gameExamples)
        
        elif r < 0.5 and episodeIndex > 20:
            # 30% chance total (20% PGN, 30% puzzles) - sample puzzles
            sampleLines, header = randomChunkCsv(csvFilePath, chunkSize=chunkSize)
            parsedRows = parseSampledCsvLines(sampleLines, header)
            for row in parsedRows:
                puzzleExamples = puzzleToTrainingExamples(row, ai, maxPositions=5)
                replayBuffer.extend(puzzleExamples)
        elif r >= 0.5 or episodeIndex <= 20:
            episodeMemory = []

            # --- Start from random known opening ---
            board = chess.Board()
            openingMoves = []
            try:
                openingGames = parsePGNGames("twic_pgn_files", maxGames=10)
                openingGame = random.choice(openingGames)
                node = openingGame

                pliesToPlay = random.randint(4, 12)
                for _ in range(pliesToPlay):
                    if not node.variations:
                        break
                    move = node.variations[0].move
                    if move in board.legal_moves:
                        board.push(move)
                        node = node.variations[0]
                        openingMoves.append(move)
                    else:
                        break
            except Exception as e:
                print(f"[Opening Load Error] {e}")

            # --- Now let Stockfish continue from this mid-game ---
            while not board.is_game_over() and len(episodeMemory) < batchSize:
                ai.stockfish.set_fen_position(board.fen())
                fishMove = ai.stockfish.get_best_move()
                if not fishMove:
                    break
                
                moveObj = chess.Move.from_uci(fishMove)
                if moveObj not in board.legal_moves:
                    break
                
                # --- 1. Convert board to tensor ---
                stateTensor = boardToTensor(board).unsqueeze(0).to(device)

                # --- 2. Policy Target: Best move index ---
                policyTargetIndex = moveToIndex(moveObj)
                policyTargetTensor = torch.tensor([policyTargetIndex], dtype=torch.long).to(device)

                # --- 3. Push the move ---
                board.push(moveObj)

                # --- 4. Get evaluation for value head ---
                ai.stockfish.set_fen_position(board.fen())
                fishEval = ai.stockfish.get_evaluation()
                fishVal = fishEval['value'] / 100.0 if fishEval['type'] == 'cp' else np.sign(fishEval['value']) * 10.0
                fishVal += random.uniform(-noiseLevel, noiseLevel)
                fishVal = max(min(fishVal, MAX_EVAL), MIN_EVAL)
                targetEvalTensor = torch.tensor([fishVal], dtype=torch.float32).to(device)

                # --- 5. Store the training sample ---
                episodeMemory.append((stateTensor, targetEvalTensor, policyTargetTensor, 0))

            replayBuffer.extend(episodeMemory)

            
            replayBuffer.extend(episodeMemory)
        
        agedBuffer = deque()
        for state, valueTarget, policyTarget, age in replayBuffer:
            if age + 1 < maxSampleAge:
                agedBuffer.append((state, valueTarget, policyTarget, age + 1))
        replayBuffer = agedBuffer  # Replace buffer with aged version

        if len(replayBuffer) >= batchSize:
            # Prioritize high-reward samples
            weights = [abs(value.item()) for (_, value, _, _) in replayBuffer]


            batch = random.choices(replayBuffer, weights=weights, k=batchSize)
            states, valueTargets, policyTargets, _ = zip(*batch)
            states = torch.cat(states).to(device)
            valueTargets = torch.stack(valueTargets).to(device)
            policyTargets = torch.cat(policyTargets).to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                policyOut, valueOut = model(states)
                
                # Compute disagreement-based weights
                with torch.no_grad():
                    predictedIndices = torch.argmax(policyOut, dim=1)
                    disagreements = (predictedIndices != policyTargets).float()  # 1.0 if disagree, 0.0 if agree
                    disagreementWeights = 1.0 + disagreements  # 2.0 for disagreement, 1.0 for agreement


                # 🔍 Top-K Policy Match Tracking
                with torch.no_grad():
                    topK = torch.topk(policyOut, k=5, dim=1).indices  # shape: (batch, 5)
                    matches = (topK == policyTargets.unsqueeze(1)).any(dim=1).float()  # 1.0 if target in top-5
                    topKMatchRate = matches.mean().item()
                    if episodeIndex % 10 == 0:
                        print(f"[Top-5 Policy Match Rate] {topKMatchRate:.2%}")

                valueLoss = criterion(valueOut, valueTargets)

                if torch.any(policyTargets >= ACTION_SIZE) or torch.any(policyTargets < 0):
                    print("❌ Invalid policy target index detected!")
                    print(policyTargets)

                rawPolicyLoss = F.cross_entropy(policyOut, policyTargets, reduction='none')
                weightedPolicyLoss = (rawPolicyLoss * disagreementWeights).mean()
                combinedLoss = (0.25 * valueLoss) + (1.75 * weightedPolicyLoss)

            scaler.scale(combinedLoss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            scheduler.step(combinedLoss)

            if currentLoss is not None:
                currentLoss[0] = combinedLoss.item()
            lossHistory.append(combinedLoss.item())

        if currentEpisode is not None:
            currentEpisode[0] = episodeIndex + 1

        if (episodeIndex + 1) % evaluationInterval == 0:
            print("Performing periodic evaluation...")
            w, d, l, acc = periodicEvaluation(ai)
            performanceHistory.append((w, d, l, acc))
            path = ai.periodicallySaveModel(optimizer, scheduler, lossHistory, performanceHistory, replayBuffer)
            if os.path.exists(path) and ((episodeIndex + 1) % (4 * evaluationInterval) == 0):
                print('resetting optimizer and scheduler')
                device = ai.device
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
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'loss_history' in checkpoint:
                    lossHistory.clear()
                    lossHistory.extend(checkpoint['loss_history'])
                if 'performance_history' in checkpoint:
                    performanceHistory.clear()
                    performanceHistory.extend(checkpoint['performance_history'])
                if 'replay_buffer' in checkpoint:
                    replayBuffer.clear()
                    for st, tv, policyTarget, age in checkpoint['replay_buffer']:
                        replayBuffer.append((st.to(device), tv.to(device), policyTarget, age))
                print("Model checkpoint loaded successfully.")

        timeEnd = time.perf_counter()
        if lossHistory:
            if len(lossHistory) >= 10:
                meanLoss = np.mean(lossHistory[-10:])
                print(f"Episode {episodeIndex + 1}: Loss={combinedLoss.item():.4f}, Value Loss={valueLoss.item():.4f}, Policy Loss={weightedPolicyLoss.item():.4f}, Mean(10)={meanLoss:.4f}, LR={scheduler.get_last_lr()}, Noise={noiseLevel:.4f}, Time={timeEnd - timeStart:.2f}s")
            else:
                print(f"Episode {episodeIndex + 1}: Loss={combinedLoss.item():.4f}, Value Loss={valueLoss.item():.4f}, Policy Loss={weightedPolicyLoss.item():.4f}, Noise={noiseLevel:.4f}, LR={scheduler.get_last_lr()}, Time={timeEnd - timeStart:.2f}s")

    ai.saveModel(optimizer, scheduler, lossHistory, performanceHistory, replayBuffer)
    print("Training completed and model saved.")