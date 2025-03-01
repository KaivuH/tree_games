import chess
import chess.svg

from typing import List, Dict, Optional, Tuple, Union, Set, Any
import analysis

class ChessEngine:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []


    def __str__(self) -> str:
        """Return string representation of the current board state."""
        return str(self.board)
    
    def reset_board(self):
        """Reset the board to starting position."""
        self.board = chess.Board()
        self.move_history = []
        return self.board
    
    def get_board(self):
        """Return the current board state."""
        return self.board
    
    def create_board_from_fen(self, fen: str):
        """Create a board from FEN notation."""
        return chess.Board(fen)
    
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
    
    def take_action(self, move: Union[str, chess.Move]) -> bool:
        """Execute a move on the current board."""
        move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
        if move_obj in self.board.legal_moves:
            self.board.push(move_obj)
            self.move_history.append(move_obj)
            return True
        return False
    
    def backtrack(self) -> bool:
        """Undo the last move if possible."""
        if self.move_history:
            self.board.pop()
            self.move_history.pop()
            return True
        return False
    
    def get_move_history(self) -> List[chess.Move]:
        """Get the history of moves played."""
        return self.move_history.copy()
    
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
    
    def get_hanging_pieces(self) -> Dict[str, List[Tuple[chess.Square, chess.Piece]]]:
        """Find all hanging pieces on the current board."""
        return analysis.get_hanging_pieces(self.board)
    
    def is_piece_hanging(self, square: Union[str, chess.Square]) -> bool:
        """Check if a piece on the given square is hanging."""
        if isinstance(square, str):
            # Convert algebraic notation (e.g., 'e4') to square
            square = chess.parse_square(square)
        return analysis.is_piece_hanging(self.board, square)
    
    def get_fork_candidates(self) -> List[Tuple[chess.Square, Set[chess.Square]]]:
        """Find potential fork situations on the current board."""
        return analysis.get_fork_candidates(self.board)
    
    def get_pins(self) -> List[
        Tuple[chess.Square, chess.Square, chess.Square]
    ]:
        """Find all pins on the current board."""
        return analysis.get_pins(self.board)
    
    def get_discover_check_candidates(self) -> List[chess.Square]:
        """Find pieces that could deliver a discovered check if moved."""
        return analysis.check_discover_check_candidates(self.board)
    
    def evaluate_position(self) -> Dict[str, Any]:
        """Evaluate the current position using multiple metrics."""
        return analysis.evaluate_position(self.board)
    
    def get_material_balance(self) -> int:
        """Get the material balance in centipawns (+ve means white advantage)."""
        return analysis.material_balance(self.board)
    
    # def load_board_from_pgn(self, pgn: str) -> chess.Board:
    #     """Load a chess board from PGN format."""
    #     game = chess.pgn.read_game(io.StringIO(pgn))
    #     board = game.end().board()
    #     self.board = board
    #     self.move_history = list(game.mainline_moves())
    #     return board

    

if __name__ == "__main__":
    # Check if a specific piece is hanging
    engine = ChessEngine()
    engine.board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    print(engine.board)
