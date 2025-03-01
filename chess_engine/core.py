import chess
import chess.svg
import re
import random
from typing import List, Dict, Optional, Tuple, Union, Set, Any
from . import analysis
import copy

def load_puzzles_from_pgn(filename):
    puzzles = []
    with open(filename, "r", encoding="utf-8") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            # If it's a puzzle, it should have a SetUp and FEN header
            if game.headers.get("SetUp") == "1" and "FEN" in game.headers:
                fen = game.headers["FEN"]
                # This sets up a board from the puzzle's FEN
                board = chess.Board(fen=fen)
                
                # mainline_moves() yields the principal move sequence
                # for that puzzle. We can store them in SAN or convert them to UCI.
                move_sequence = list(game.mainline_moves())
                
                # If you want them in algebraic SAN:
                san_moves = []
                temp_board = board.copy()
                for mv in move_sequence:
                    san_moves.append(temp_board.san(mv))
                    temp_board.push(mv)
                
                # Or in UCI:
                uci_moves = [mv.uci() for mv in move_sequence]
                
                puzzles.append((fen, san_moves, uci_moves))
    
    return puzzles


def load_random_puzzle(engine, puzzle_text):
    """Load a random puzzle into the chess engine."""
    
    puzzles = load_puzzles(puzzle_text)
    if not puzzles:
        print("No puzzles loaded!")
        return None
    
    print(f"Selecting from {len(puzzles)} puzzles")
    fen, moves = random.choice(puzzles)
    print(f"Selected puzzle with FEN: {fen}")
    print(f"Moves: {moves}")
    
    # Set up the board with the puzzle position
    engine.board = chess.Board(fen)
    engine.move_history = []
    
    # Convert algebraic moves to UCI
    uci_moves = []
    temp_board = chess.Board(fen)
    for move in moves:
        # Clean up any annotations
        clean_move = re.sub(r'[+#]', '', move)
        try:
            san_move = temp_board.parse_san(clean_move)
            uci_moves.append(san_move.uci())
            temp_board.push(san_move)
        except ValueError as e:
            print(f"Error parsing move '{move}': {e}")
            continue
    
    print(f"UCI moves: {uci_moves}")
    
    return {
        'fen': fen,
        'moves': uci_moves,
        'turn': 'white' if engine.board.turn == chess.WHITE else 'black'
    }


class ChessEngine:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []



    def copy(self):
        return copy.deepcopy(self)



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
    
    def backtrack(self, n_moves=1) -> bool:
        """Undo the last move if possible."""
        for _ in range(n_moves):
            if self.move_history:
                self.board.pop()
                self.move_history.pop()
    
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
    

# if __name__ == "__main__":
#     # Check if a specific piece is hanging
#     engine = ChessEngine()
#     # Try both relative and absolute paths
#     try:
#         # First try current directory
#         with open("puzzle_mini.pgn", "r") as file:
#             puzzle_text = file.read()
#         print(f"Successfully loaded from current directory, file size: {len(puzzle_text)} bytes")
#     except FileNotFoundError:
#         try:
#             # Then try with the module path
#             with open("chess_engine/puzzle_mini.pgn", "r") as file:
#                 puzzle_text = file.read()
#             print(f"Successfully loaded from chess_engine directory, file size: {len(puzzle_text)} bytes")
#         except FileNotFoundError:
#             print("Could not find puzzle_mini.pgn in either location!")
#             exit(1)
    
#     puzzle = load_random_puzzle(engine, puzzle_text)
#     if puzzle:
#         engine.board = engine.create_board_from_fen(puzzle['fen'])
#         print(engine.board)
#     else:
#         print("Failed to load puzzle!")

if __name__ == "__main__":
    puzzles = load_puzzles_from_pgn("puzzle_mini.pgn")
    print(puzzles)
    for i, (fen, san_moves, uci_moves) in enumerate(puzzles):
        print(f"Puzzle #{i+1}")
        print("FEN:", fen)
        print("SAN moves:", san_moves)
        print("UCI moves:", uci_moves)
        print("="*40)
        if i > 5:
            break