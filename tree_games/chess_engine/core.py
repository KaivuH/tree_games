import chess
import chess.svg
from typing import List, Dict, Optional, Tuple, Union

class ChessEngine:
    def __init__(self):
        self.board = chess.Board()
    
    def reset_board(self):
        """Reset the board to starting position."""
        self.board = chess.Board()
        return self.board
    
    def get_board(self):
        """Return the current board state."""
        return self.board
    
    def create_board_from_fen(self, fen: str):
        """Create a board from FEN notation."""
        return chess.Board(fen)
    
    def apply_move(self, move: Union[str, chess.Move]) -> Optional[chess.Board]:
        """Apply a move to a copy of the current board and return the new board."""
        move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
        if move_obj in self.board.legal_moves:
            board_copy = self.board.copy()
            board_copy.push(move_obj)
            return board_copy
        return None
    
    def apply_move_sequence(self, moves: List[Union[str, chess.Move]]) -> Optional[chess.Board]:
        """Apply a sequence of moves to a copy of the current board."""
        board_copy = self.board.copy()
        for move in moves:
            move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
            if move_obj in board_copy.legal_moves:
                board_copy.push(move_obj)
            else:
                return None
        return board_copy
    
    def execute_move(self, move: Union[str, chess.Move]) -> bool:
        """Execute a move on the current board."""
        move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
        if move_obj in self.board.legal_moves:
            self.board.push(move_obj)
            return True
        return False
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves for the current position."""
        return list(self.board.legal_moves)
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_result(self) -> str:
        """Get the game result if the game is over."""
        if not self.board.is_game_over():
            return "Game in progress"
        
        if self.board.is_checkmate():
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
            return "1/2-1/2"
        else:
            return "Unknown result"
    
    def get_board_visual(self) -> str:
        """Return a text representation of the board."""
        return str(self.board)
    
    def get_board_svg(self, size: int = 400) -> str:
        """Return an SVG representation of the board."""
        return chess.svg.board(self.board, size=size)
    
    def get_turn(self) -> str:
        """Return the current turn (white or black)."""
        return "white" if self.board.turn == chess.WHITE else "black"