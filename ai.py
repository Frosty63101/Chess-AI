"""
ai.py

Implements the ChessAI class responsible for:
  - Searching moves via iterative deepening + PVS
  - Evaluating boards with the neural network
  - Caching to speed up evaluations
  - Maintaining transposition / killer / history tables
"""

import os
import time
import random
import chess
import torch
import threading
import numpy as np

from typing import Dict, Optional
from collections import defaultdict
from chess.polyglot import zobrist_hash
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# We share config constants
from config import (MODEL_PATH, BACKUP_MODEL_PATH, STOCKFISH_DEFAULT_PATH,
                    IMAGE_DIR, ACTION_SIZE, MIN_EVAL, MAX_EVAL, MAX_DEPTH,
                    MAX_TIME)

# We'll import the advanced net
from model import AdvancedChessNet

# Additional imports from the original code
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stockfish import Stockfish
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from collections import deque
from typing import List

# We keep the board -> tensor logic here (or you can factor it out separately).
boardTensorCache = {}
boardTensorCacheLock = threading.Lock()

def boardToTensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess board to a tensor representation.
    Now we have 15 channels:
      0-11: One-hot planes for each piece type
      12:   piece-value plane
      13:   side-to-move plane (all 1s if white to move, all 0s otherwise)
      14:   castling-rights plane (all 1s if the side to move still can castle, else 0)
    """
    pieceToChannel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q':10, 'k':11
    }
    pieceValues = {
        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
        'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0
    }
    # 15 planes: 12 for piece type, 1 for pieceValue, 1 for sideToMove, 1 for castling
    boardTensor = np.zeros((12, 8, 8), dtype=np.float32)
    valueTensor = np.zeros((1, 8, 8), dtype=np.float32)
    sidePlane = np.zeros((1, 8, 8), dtype=np.float32)
    castlingPlane = np.zeros((1, 8, 8), dtype=np.float32)

    # Fill piece planes + piece-value plane
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pieceSymbol = piece.symbol()
            channelIndex = pieceToChannel[pieceSymbol]
            value = pieceValues[pieceSymbol]
            x = square % 8
            y = 7 - (square // 8)
            boardTensor[channelIndex, y, x] = 1.0
            valueTensor[0, y, x] = value

    # Fill side-to-move plane
    # If White to move, set all to 1. If Black to move, remain 0.
    if board.turn == chess.WHITE:
        sidePlane[0, :, :] = 1.0

    # Fill castling plane
    # We just check if side to move can still castle at all
    # (You can do a more fine-grained approach if you like)
    canCastle = False
    if board.turn == chess.WHITE:
        if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
            canCastle = True
    else:
        if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
            canCastle = True

    if canCastle:
        castlingPlane[0, :, :] = 1.0

    combinedTensor = np.concatenate((boardTensor, valueTensor, sidePlane, castlingPlane), axis=0)
    return torch.tensor(combinedTensor, dtype=torch.float32)

def boardToTensorCached(board: chess.Board, ai: 'ChessAI') -> torch.Tensor:
    """
    Converts a chess board to a tensor representation with caching
    to avoid recomputing the same representation many times.
    """
    boardHash = zobrist_hash(board)
    with boardTensorCacheLock:
        if boardHash in boardTensorCache:
            return boardTensorCache[boardHash]
        tensor = boardToTensor(board).unsqueeze(0).to(ai.device)
        boardTensorCache[boardHash] = tensor
    return tensor

def moveToIndex(move: chess.Move) -> int:
    """
    Converts a move into an integer index for the policy output space.
    We assume 64x64 = 4096 for normal from->to moves,
    plus 8*4 = 32 for promotion moves from the 8 possible "from" squares
    times 4 possible promotion pieces.
    """
    NUM_SQUARES = 64
    NUM_PROMOTION_PIECES = 4
    fromSquare = move.from_square
    toSquare = move.to_square
    promotion = move.promotion

    if promotion is None:
        index = fromSquare * NUM_SQUARES + toSquare
    else:
        promotionIndex = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }[promotion]
        index = NUM_SQUARES * NUM_SQUARES + (fromSquare - 8) * NUM_PROMOTION_PIECES + promotionIndex
    return index

def indexToMove(idx: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Inverse of moveToIndex(...). Converts an index to a valid chess.Move
    if it is a legal move on the current board. Otherwise returns None.
    """
    NUM_SQUARES = 64
    NUM_PROMOTION_PIECES = 4

    if idx < NUM_SQUARES * NUM_SQUARES:
        fromSquare = idx // NUM_SQUARES
        toSquare = idx % NUM_SQUARES
        move = chess.Move(fromSquare, toSquare)
    else:
        idx -= NUM_SQUARES * NUM_SQUARES
        fromSquare = idx // NUM_PROMOTION_PIECES + 8
        promotionIndex = idx % NUM_PROMOTION_PIECES
        promotionPiece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][promotionIndex]
        toSquare = fromSquare + 8 if board.turn == chess.WHITE else fromSquare - 8
        move = chess.Move(fromSquare, toSquare, promotion=promotionPiece)

    return move if move in board.legal_moves else None

# We keep locks for concurrency
transpositionTableLock = threading.Lock()
evalCacheLock = threading.Lock()

def materialEvaluation(board: chess.Board) -> float:
    """
    Simple material-based evaluation. 
    Normalized by 39 to keep score roughly in [-1,1].
    """
    pieceValues = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    whiteMaterial = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in pieceValues.items())
    blackMaterial = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in pieceValues.items())
    materialScore = whiteMaterial - blackMaterial
    # 39 is the max typical difference (9+9+5+5+3+3+3+3+1+1...).
    materialScore /= 39.0
    return materialScore


def evaluateBoard(board: chess.Board, ai: 'ChessAI') -> float:
    """
    Uses the neural net plus a learned material weighting. 
    Caches results to avoid recomputation.
    """
    from torch import no_grad
    boardHash = zobrist_hash(board)
    with ai.evalCacheLock:
        if boardHash in ai.evalCache:
            return ai.evalCache[boardHash]

    # Prepare input
    state = boardToTensorCached(board, ai)
    matVal = torch.tensor([materialEvaluation(board)], dtype=torch.float32, device=ai.device)
    
    with no_grad():
        policyOut, valueOut = ai.model(state, materialTensor=matVal)
        # valueOut shape is (1,1)
        value = valueOut.item()

    # We clamp final just to keep it in [MIN_EVAL, MAX_EVAL]
    value = max(min(value, MAX_EVAL), MIN_EVAL)

    with ai.evalCacheLock:
        ai.evalCache[boardHash] = value
    return value


def quiescenceSearch(board: chess.Board, alpha: float, beta: float,
                     ai: 'ChessAI', startTime: float, maxTime: float) -> float:
    """
    Extends the search at leaf nodes to capture sequences to avoid the horizon effect.
    """
    if time.time() - startTime >= maxTime or ai.searchStop:
        ai.searchStop = True
        raise TimeoutError("Search timed out")

    standPat = evaluateBoard(board, ai)
    if standPat >= beta:
        return beta
    if alpha < standPat:
        alpha = standPat

    moves = ai.orderMovesQuiescence(board)
    for move in moves:
        board.push(move)
        try:
            score = -quiescenceSearch(board, -beta, -alpha, ai, startTime, maxTime)
        except TimeoutError:
            board.pop()
            raise
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

def principalVariationSearch(board: chess.Board, depth: int, alpha: float, beta: float,
                             maximizingPlayer: bool, ai: 'ChessAI',
                             startTime: float, maxTime: float) -> float:
    """
    Principal Variation Search (PVS) with alpha-beta and time checks.
    """
    if time.time() - startTime >= maxTime or ai.searchStop:
        ai.searchStop = True
        raise TimeoutError("Search timed out")

    # Transposition table lookup
    transpositionKey = zobrist_hash(board)
    with transpositionTableLock:
        if transpositionKey in ai.transpositionTable:
            entry = ai.transpositionTable[transpositionKey]
            if entry['depth'] >= depth:
                return entry['value']

    if board.is_game_over() or depth == 0:
        evalValue = quiescenceSearch(board, alpha, beta, ai, startTime, maxTime)
        with transpositionTableLock:
            ai.transpositionTable[transpositionKey] = {'value': evalValue, 'depth': depth}
        return evalValue

    moves = ai.orderMoves(board, depth)
    if not moves:
        return evaluateBoard(board, ai)

    firstChild = True
    for move in moves:
        board.push(move)
        try:
            if firstChild:
                score = -principalVariationSearch(board, depth - 1, -beta, -alpha,
                                                  not maximizingPlayer, ai, startTime, maxTime)
            else:
                score = -principalVariationSearch(board, depth - 1, -alpha - 1, -alpha,
                                                  not maximizingPlayer, ai, startTime, maxTime)
                if alpha < score < beta:
                    score = -principalVariationSearch(board, depth - 1, -beta, -score,
                                                      not maximizingPlayer, ai, startTime, maxTime)
        except TimeoutError:
            board.pop()
            raise
        board.pop()

        if score >= beta:
            with transpositionTableLock:
                ai.transpositionTable[transpositionKey] = {'value': beta, 'depth': depth}
            return beta
        if score > alpha:
            alpha = score
        firstChild = False

        if time.time() - startTime >= maxTime or ai.searchStop:
            ai.searchStop = True
            raise TimeoutError("Search timed out")

    with transpositionTableLock:
        ai.transpositionTable[transpositionKey] = {'value': alpha, 'depth': depth}
    return alpha

def iterativeDeepening(board: chess.Board, ai: 'ChessAI', maxTime: float = MAX_TIME) -> Optional[chess.Move]:
    """
    Performs iterative deepening search within a time limit, increasing depth
    until no time remains.
    """
    bestMove = None
    depth = 1
    startTime = time.time()
    while True:
        timeElapsed = time.time() - startTime
        if timeElapsed >= maxTime:
            break
        ai.searchStop = False
        try:
            bestMove = ai.search(board, depth, startTime, maxTime)
        except TimeoutError:
            break
        depth += 1
    return bestMove

class ChessAI:
    """
    Encapsulates the chess AI:
      - The neural network model (AdvancedChessNet)
      - Search routines
      - Stockfish engine for reference moves and evaluations
      - Caches for speed (transposition_table, history_table, eval_cache, etc.)
    """
    def __init__(self, stockfishPath: str, device: torch.device):
        self.device = device
        self.model = None
        self.stockfish = Stockfish(stockfishPath)
        self.stockfish.set_skill_level(10)
        self.stockfish.update_engine_parameters({
            "Threads": 3,
            "Minimum Thinking Time": 5,
            "Hash": 16,
            "UCI_LimitStrength": True,
            "UCI_Elo": 1350
        })

        self.transpositionTable = {}
        self.evalCache = {}
        self.evalCacheLock = threading.Lock()
        self.pvTable: Dict[str, chess.Move] = {}
        self.historyTable: Dict[chess.Move, int] = defaultdict(int)
        self.killerMoves = [[] for _ in range(MAX_DEPTH)]
        self.searchStop = False
        self.maxWorkers = 4

        self.initialize_model()

    def initialize_model(self):
        print(f"Using device: {self.device}")
        net = AdvancedChessNet()
        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
            net = nn.DataParallel(net)
        self.model = net.to(self.device)

        if os.path.exists(MODEL_PATH):
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=self.device)
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with a fresh model.")
        else:
            print("No saved model found. Starting with a fresh model.")

    def getStockfishSkill(self):
        return self.stockfish.get_parameters()['Skill Level']

    def setStockfishSkill(self, skill: int):
        self.stockfish.set_skill_level(skill)

    def selectBestMove(self, board: chess.Board, maxTime: float = MAX_TIME) -> Optional[chess.Move]:
        """
        Chooses the best move. We can decide to use MCTS or fallback to iterativeDeepening.
        Below we show a simple toggle: if the board is complex or you want MCTS, call mctsSearch.
        Otherwise, call iterativeDeepening.
        """
        # For demonstration, always use MCTS:
        return self.mctsSearch(board, simulations=300, cPuct=1.0)

        # Or if you want the old approach:
        # return iterativeDeepening(board, self, maxTime)

    def mctsSearch(self, board: chess.Board, simulations: int = 300, cPuct: float = 1.0) -> Optional[chess.Move]:
        """
        A basic Monte Carlo Tree Search. We'll store (N, W, Q, P) for each node:
            N = visit count
            W = total value
            Q = average value (W / N)
            P = prior (from policy network)
        We'll do 'simulations' playouts, then pick the move with the highest N.
        """
        import math

        # We define a local dictionary for MCTS stats: { zobristHash: {move: [N, W, Q, P]} }
        mctsStats = {}

        def getPolicyAndValue(bd: chess.Board):
            """
            Runs the neural net to get policy (as a distribution) and value for the board.
            Returns (policyDict, valueFloat).
            policyDict: {move: priorProbability}
            valueFloat: in [-10, 10]
            """
            stateTensor = boardToTensorCached(bd, self)
            matVal = torch.tensor([materialEvaluation(bd)], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                policyLogits, valueOut = self.model(stateTensor, materialTensor=matVal)
                policyArr = torch.softmax(policyLogits[0], dim=0).cpu().numpy()
                valueFloat = valueOut.item()
            
            # Build a dictionary for legal moves
            legalMoves = list(bd.legal_moves)
            policyDict = {}
            for mv in legalMoves:
                idx = moveToIndex(mv)
                policyDict[mv] = policyArr[idx]
            return policyDict, valueFloat

        def selectMove(bd: chess.Board) -> chess.Move:
            """
            Selects a move using the UCB formula:
            Q + cPuct * P * sqrt(sum(N)) / (1 + N).
            """
            stateKey = zobrist_hash(bd)
        
            # Ensure the state is in the MCTS stats
            if stateKey not in mctsStats:
                legalMoves = list(bd.legal_moves)
        
                # Initialize stats for each legal move:
                # [N (visits), W (total value), Q (mean value), P (prior probability)]
                mctsStats[stateKey] = {
                    mv: [0, 0.0, 0.0, 1.0 / len(legalMoves)] for mv in legalMoves
                }
        
            movesStats = mctsStats[stateKey]
            totalVisits = sum(movesStats[mv][0] for mv in movesStats)
        
            bestMoveSelected = None
            bestScore = float('-inf')  # cleaner than -999999
            for mv, (n, w, q, p) in movesStats.items():
                # UCB formula balances exploration and exploitation
                ucb = q + cPuct * p * math.sqrt(totalVisits) / (1 + n)
                if ucb > bestScore:
                    bestScore = ucb
                    bestMoveSelected = mv
        
            return bestMoveSelected

        def simulate(bd: chess.Board):
            """
            Recursively run a simulation (selection, expansion, evaluation, backprop).
            Returns a value in [-10, 10].
            """
            stateKey = zobrist_hash(bd)

            # If game is over, return the terminal value
            if bd.is_game_over():
                res = bd.result()
                if res == "1-0":
                    return 1.0
                elif res == "0-1":
                    return -1.0
                else:
                    return 0.0

            # If we haven't expanded this node, do so
            if stateKey not in mctsStats:
                policyDict, valueFloat = getPolicyAndValue(bd)
                # Initialize MCTS stats
                mctsStats[stateKey] = {}
                for mv, prior in policyDict.items():
                    mctsStats[stateKey][mv] = [0, 0.0, 0.0, prior]  # N=0, W=0, Q=0, P=prior
                return valueFloat

            # Otherwise, select a move using UCB
            moveChosen = selectMove(bd)
            bd.push(moveChosen)
            valueNext = simulate(bd)
            bd.pop()

            # Update stats
            movesStats = mctsStats[stateKey]
            n, w, q, p = movesStats[moveChosen]
            n += 1
            w += valueNext
            q = w / n
            movesStats[moveChosen] = [n, w, q, p]
            return valueNext

        # ---- MCTS main loop ----
        for _ in range(simulations):
            simulate(board)

        # After all simulations, pick move with highest N
        stateKeyRoot = zobrist_hash(board)
        if stateKeyRoot not in mctsStats or not board.legal_moves:
            return None

        movesStatsRoot = mctsStats[stateKeyRoot]
        bestMove = None
        bestVisits = -1
        for mv, (n, w, q, p) in movesStatsRoot.items():
            if n > bestVisits:
                bestVisits = n
                bestMove = mv
        return bestMove

    def search(self, board: chess.Board, depth: int, startTime: float, maxTime: float) -> Optional[chess.Move]:
        """
        At a given depth, tries all moves and picks the best one by exploring PVS.
        """
        bestMove = None
        alpha = -float('inf')
        beta = float('inf')
        maximizingPlayer = board.turn == chess.WHITE
        bestEval = -float('inf') if maximizingPlayer else float('inf')

        moves = self.orderMoves(board, depth)
        if not moves:
            return None

        for move in moves:
            if time.time() - startTime >= maxTime or self.searchStop:
                self.searchStop = True
                raise TimeoutError("Search timed out")

            board.push(move)
            try:
                evalScore = -principalVariationSearch(
                    board, depth - 1, -beta, -alpha,
                    not maximizingPlayer, self, startTime, maxTime
                )
            except TimeoutError:
                board.pop()
                break
            board.pop()

            if maximizingPlayer:
                if evalScore > bestEval:
                    bestEval = evalScore
                    bestMove = move
                alpha = max(alpha, evalScore)
            else:
                if evalScore < bestEval:
                    bestEval = evalScore
                    bestMove = move
                beta = min(beta, evalScore)

            if beta <= alpha:
                break

        return bestMove

    def orderMoves(self, board: chess.Board, depth: int) -> List[chess.Move]:
        """
        Orders moves using:
          - The model's policy network (to get a probability for each move)
          - History table
          - Killer moves
          - MVV/LVA for captures
        """
        state = boardToTensorCached(board, self)
        with torch.no_grad():
            policyLogits, _ = self.model(state)
            policy = torch.softmax(policyLogits, dim=1).cpu().numpy()[0]

        moves = list(board.legal_moves)
        moveScores = []
        for move in moves:
            idx = moveToIndex(move)
            score = policy[idx]

            # Add history heuristics
            score += self.historyTable.get(move, 0)

            # Add killer move bonus
            if move in self.killerMoves[depth]:
                score += 10000

            # For captures, do a MVV-LVA bonus
            if board.is_capture(move):
                capturedPiece = board.piece_at(move.to_square)
                attackerPiece = board.piece_at(move.from_square)
                if capturedPiece and attackerPiece:
                    mvvLva = (capturedPiece.piece_type * 10) - attackerPiece.piece_type
                    score += mvvLva

            moveScores.append((move, score))

        orderedMoves = [mv for mv, sc in sorted(moveScores, key=lambda x: x[1], reverse=True)]
        return orderedMoves

    def orderMovesQuiescence(self, board: chess.Board) -> List[chess.Move]:
        """
        During quiescence search, we only consider capture moves.
        """
        moves = [move for move in board.legal_moves if board.is_capture(move)]
        moveScores = []
        for move in moves:
            capturedPiece = board.piece_at(move.to_square)
            attackerPiece = board.piece_at(move.from_square)
            mvvLvaScore = 0
            if capturedPiece and attackerPiece:
                mvvLvaScore = (capturedPiece.piece_type * 10) - attackerPiece.piece_type
            moveScores.append((move, mvvLvaScore))

        orderedMoves = [mv for mv, sc in sorted(moveScores, key=lambda x: x[1], reverse=True)]
        return orderedMoves

    def saveModel(self, optimizer=None, scheduler=None, lossHistory=None,
                  performanceHistory=None, replayBuffer=None):
        """
        Saves the model and training state to disk.
        """
        checkpoint = {
            'model_state_dict': (self.model.module.state_dict() 
                                 if isinstance(self.model, nn.DataParallel) 
                                 else self.model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_history': lossHistory,
            'performance_history': performanceHistory,
            'replay_buffer': list(replayBuffer) if replayBuffer else None,
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"Model and state saved to {MODEL_PATH}")

    def periodicallySaveModel(self, optimizer=None, scheduler=None, lossHistory=None,
                              performanceHistory=None, replayBuffer=None):
        """
        Backup-saves the model to a timestamped .pth file inside BACKUP_MODEL_PATH.
        """
        if not os.path.exists(BACKUP_MODEL_PATH):
            os.makedirs(BACKUP_MODEL_PATH)

        checkpoint = {
            'model_state_dict': (self.model.module.state_dict() 
                                 if isinstance(self.model, nn.DataParallel) 
                                 else self.model.state_dict()),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss_history': lossHistory,
            'performance_history': performanceHistory,
            'replay_buffer': list(replayBuffer) if replayBuffer else None,
        }
        fileName = os.path.join(BACKUP_MODEL_PATH, f"{time.time()}.pth")
        torch.save(checkpoint, fileName)
        print(f"Model and state backup saved to {fileName}")
