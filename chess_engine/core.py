import chess
import chess.svg
import re
import random
from typing import List, Dict, Optional, Tuple, Union, Set, Any
from . import analysis
import copy
from datetime import datetime

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

    def evaluate_board(self, color) -> float:
        """
        Evaluate the board position based on material for the given color.
        Returns a score where positive means advantage for the given color.
        
        Piece values:
        - Pawn = 1
        - Knight = 3
        - Bishop = 3
        - Rook = 5
        - Queen = 9
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3, 
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }
        
        score = 0
        
        # Convert color string to chess.Color
        eval_color = chess.WHITE if color.lower() == 'white' else chess.BLACK
        
        # Calculate material balance
        for piece_type in piece_values:
            # Count pieces for the evaluated color
            own_pieces = len(self.board.pieces(piece_type, eval_color))
            # Count opponent pieces
            opp_pieces = len(self.board.pieces(piece_type, not eval_color))
            # Add to score (positive for own pieces, negative for opponent pieces)
            score += piece_values[piece_type] * (own_pieces - opp_pieces)
            
        return score

    def to_pgn(self) -> str:
        """
        Constructs a PGN string from a given starting board and a move history.
        
        Parameters:
            start_board: The chess.Board representing the starting position.
            moves_history: An iterable of tuples (move_num, player, move) where 
                        'move' is in UCI format.
                        
        Returns:
            A string containing the PGN.
        """
        # Create a new PGN game and add headers.
        moves_history = self.move_history
        start_board = self.starting_board
        game = chess.pgn.Game()
        game.headers["Event"] = "Game from Move History"
        game.headers["Site"] = "?"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "White"
        game.headers["Black"] = "Black"
        game.headers["Result"] = "*"
        
        # Use a temporary board initialized with the starting position.
        board = start_board.copy()
        node = game
        
        # Replay moves from the move history.
        print(moves_history)
        for move in moves_history:
            if move not in board.legal_moves:
                raise ValueError(f"Illegal move {move} at move in board state {board.fen()}")
            node = node.add_variation(move)
            board.push(move)
        
        # Convert the PGN game to a string.
        pgn_str = str(game)
        return pgn_str



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
        board = chess.Board(fen)
        self.starting_board = copy.deepcopy(board)
        return board
    
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
        try:
            move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
        except:
            return False
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
    
    def stockfish_evaluate(self, depth: int = 15, time_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the current position using Stockfish engine
        
        Args:
            depth: Search depth for Stockfish (higher means stronger but slower analysis)
            time_limit: Optional time limit in milliseconds
            
        Returns:
            Dictionary with Stockfish evaluation metrics
        """
        return analysis.stockfish_evaluate(self.board, depth, time_limit)
    
    def get_best_move(self, depth: int = 15) -> Optional[str]:
        """
        Get the best move for the current position according to Stockfish
        
        Args:
            depth: Search depth for Stockfish
            
        Returns:
            Best move in UCI format or None if Stockfish is not available
        """
        eval_result = self.stockfish_evaluate(depth)
        if eval_result.get('best_moves') and len(eval_result['best_moves']) > 0:
            return eval_result['best_moves'][0]['Move']
        return None
    
    def get_claude_legal_moves(self, board=None):
        """
        Get all legal moves for the current position.
        
        Args:
            board: Optional chess.Board object. If None, uses the current engine board.
            
        Returns:
            A list of legal moves in UCI notation
        """
        from . import chess_fns
        if board is None:
            board = self.board
        return chess_fns.get_legal_moves(board)
    
    def get_claude_is_check(self, board=None):
        """
        Determine if the side to move is in check.
        
        Args:
            board: Optional chess.Board object. If None, uses the current engine board.
            
        Returns:
            Boolean indicating whether the side to move is in check
        """
        from . import chess_fns
        if board is None:
            board = self.board
        return chess_fns.is_check(board)
    
    def get_claude_attacked_squares(self, side, board=None):
        """
        Count the number of squares attacked by a specific side.
        
        Args:
            side: String indicating which side to analyze ('white' or 'black')
            board: Optional chess.Board object. If None, uses the current engine board.
            
        Returns:
            Integer count of squares attacked by the specified side
        """
        from . import chess_fns
        if board is None:
            board = self.board
        return chess_fns.count_attacked_squares(board, side)
    
    def get_claude_hanging_pieces(self, board=None):
        """
        Identify undefended or hanging pieces on the board.
        
        Args:
            board: Optional chess.Board object. If None, uses the current engine board.
            
        Returns:
            A list of dictionaries with information about hanging pieces
        """
        from . import chess_fns
        if board is None:
            board = self.board
        return chess_fns.find_hanging_pieces(board)
    
    def get_claude_center_control(self, board=None):
        """
        Evaluate control of the center squares (d4, d5, e4, e5).
        
        Args:
            board: Optional chess.Board object. If None, uses the current engine board.
            
        Returns:
            A dictionary with center control metrics
        """
        from . import chess_fns
        if board is None:
            board = self.board
        return chess_fns.evaluate_center_control(board)
    
    def get_claude_fork_opportunities(self, board=None):
        """
        Identify potential fork opportunities for both sides.
        
        Args:
            board: Optional chess.Board object. If None, uses the current engine board.
            
        Returns:
            A list of dictionaries describing potential forks
        """
        from . import chess_fns
        if board is None:
            board = self.board
        return chess_fns.detect_fork_opportunities(board)


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