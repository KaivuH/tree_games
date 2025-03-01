import chess
import chess.variant
import random
from chessbots import ChessBotInterface

class LeetCodeBot(ChessBotInterface):
    """
    LeetCode implementation of a chess bot
    LeetCode ID: lip_uIzLCKgnM3vQRUkmMJmH
    """
    
    def getBestMove(self, gameState, variant):
        """
        Implements a LeetCode solution to choose the best move
        """
        # Create a board with the appropriate variant
        board = self.getBoardObject(variant)
        
        # Apply all previous moves to reach current board state
        for move in gameState.move_list:
            board.push_uci(move)
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # LeetCode solution for evaluating and selecting the best move
        # This is a simplified chess engine approach
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(move)
            
            # Evaluate the position after the move
            score = self.evaluate_position(board_copy)
            
            # Update best move if this one is better
            if score > best_score:
                best_score = score
                best_move = move
        
        # If we couldn't find a good move, just play a random one
        if best_move is None:
            best_move = random.choice(legal_moves)
            
        return best_move.uci()
    
    def evaluate_position(self, board):
        """
        Simple position evaluation function
        Returns a score from the perspective of the current player
        Higher score is better for the current player
        """
        if board.is_checkmate():
            # If we're checkmated, this is the worst possible position
            if board.turn == chess.WHITE:
                return -10000
            else:
                return 10000
                
        # Material counting with piece values
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0  # King's value doesn't matter for material counting
        }
        
        # Count material for both sides
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Mobility (number of legal moves) bonus
        mobility = len(list(board.legal_moves)) * 10
        
        # Calculate final score from perspective of current player
        material_balance = white_material - black_material
        
        if board.turn == chess.WHITE:
            return material_balance + mobility
        else:
            return -material_balance + mobility

    def getResponseToMessage(self, chatLine):
        """
        Respond to chat messages
        """
        if "hello" in chatLine.text.lower():
            return "Hello! I'm a LeetCode chess bot (lip_uIzLCKgnM3vQRUkmMJmH)"
        elif "help" in chatLine.text.lower():
            return "I'm a chess bot using a custom position evaluation function"
        elif "move" in chatLine.text.lower():
            return "I evaluate positions based on material and mobility"
        return None