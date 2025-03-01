import chess
import chess.pgn
from typing import Dict, List, Any, Tuple, Set, Optional

def material_count(board: chess.Board, color: str) -> int:
    """
    Calculate the material count for a given color
    
    Args:
        board: Chess board
        color: 'white' or 'black'
        
    Returns:
        Material value in centipawns
    """
    values = {
        chess.PAWN: 100, 
        chess.KNIGHT: 320, 
        chess.BISHOP: 330,
        chess.ROOK: 500, 
        chess.QUEEN: 900, 
        chess.KING: 0
    }
    
    color_bool = color == 'white'
    
    count = sum(values[piece.piece_type] for piece in board.piece_map().values()
               if piece.color == color_bool)
    return count

def material_balance(board: chess.Board) -> int:
    """
    Calculate the material balance between white and black
    
    Args:
        board: Chess board
        
    Returns:
        Material balance in centipawns (positive means white advantage)
    """
    white_material = material_count(board, 'white')
    black_material = material_count(board, 'black')
    return white_material - black_material


def king_safety(board: chess.Board, color: str) -> float:
    """
    Calculate a king safety score for the given color
    
    Args:
        board: Chess board
        color: 'white' or 'black'
        
    Returns:
        King safety score between 0.0 and 1.0 (higher is safer)
    """
    color_bool = color == 'white'
    
    # Find king position
    king_square = None
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING and piece.color == color_bool:
            king_square = square
            break
    
    if king_square is None:
        return 0.0  # Shouldn't happen in valid chess position
    
    # Count attackers of squares around the king
    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)
    safety_score = 1.0
    
    # Check squares around king
    for f in range(max(0, king_file-1), min(8, king_file+2)):
        for r in range(max(0, king_rank-1), min(8, king_rank+2)):
            square = chess.square(f, r)
            if board.is_attacked_by(not color_bool, square):
                safety_score -= 0.1  # Reduce safety for each attacked square
    
    # Check if king is castled
    queen_side_file = 2  # c-file
    king_side_file = 6   # g-file
    
    if king_file == queen_side_file or king_file == king_side_file:
        safety_score += 0.2  # Bonus for castled king
    
    return max(0.0, min(1.0, safety_score))


def is_piece_hanging(board: chess.Board, square: chess.Square) -> bool:
    """
    Check if a piece on the given square is hanging (undefended or defended 
    by pieces of higher value)
    
    Args:
        board: Chess board
        square: Square to check
        
    Returns:
        True if the piece is hanging, False otherwise
    """
    piece = board.piece_at(square)
    if piece is None:
        return False
    
    piece_color = piece.color
    piece_value = {
        chess.PAWN: 1, 
        chess.KNIGHT: 3, 
        chess.BISHOP: 3,
        chess.ROOK: 5, 
        chess.QUEEN: 9, 
        chess.KING: 99  # effectively infinite
    }[piece.piece_type]
    
    # Find attackers and defenders using built-in methods
    attackers = board.attackers(not piece_color, square)
    defenders = board.attackers(piece_color, square)
    
    # If no attackers, the piece is not hanging
    if not attackers:
        return False
    
    # If no defenders, the piece is hanging
    if not defenders:
        return True
    
    # Get the lowest valued attacker
    attacker_values = []
    for sq in attackers:
        piece_at_square = board.piece_at(sq)
        if piece_at_square is not None:
            attacker_values.append(piece_value[piece_at_square.piece_type])
    
    min_attacker_value = min(attacker_values) if attacker_values else float('inf')
    
    # Get the lowest valued defender
    defender_values = []
    for sq in defenders:
        piece_at_square = board.piece_at(sq)
        if piece_at_square is not None:
            defender_values.append(piece_value[piece_at_square.piece_type])
    
    min_defender_value = min(defender_values) if defender_values else float('inf')
    
    # If the best attacker is worth less than the piece being attacked
    # and there's no adequate defense, the piece is hanging
    return (min_attacker_value < piece_value and 
            min_defender_value > min_attacker_value)

def get_hanging_pieces(
    board: chess.Board
) -> Dict[str, List[Tuple[chess.Square, chess.Piece]]]:
    """
    Find all hanging pieces on the board
    
    Args:
        board: Chess board
        
    Returns:
        Dictionary with 'white' and 'black' keys containing lists of 
        (square, piece) tuples
    """
    hanging = {'white': [], 'black': []}
    
    # Efficiently iterate through all pieces on the board
    for square, piece in board.piece_map().items():
        if is_piece_hanging(board, square):
            color_key = 'white' if piece.color == chess.WHITE else 'black'
            hanging[color_key].append((square, piece))
    
    return hanging

def get_fork_candidates(board: chess.Board) -> List[Tuple[chess.Square, Set[chess.Square]]]:
    """
    Find potential fork situations on the board
    
    Args:
        board: Chess board
        
    Returns:
        List of tuples: (attacker_square, target_squares) where a single piece
        is attacking multiple valuable pieces
    """
    valuable_pieces = {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.KING}
    fork_candidates = []
    
    # Check knight forks first (most common)
    knights = board.pieces(chess.KNIGHT, board.turn)
    for square in knights:
        knight_color = board.piece_at(square).color
        targets = set()
        
        # Get all squares the knight attacks
        attacks = board.attacks(square)
        for target_square in attacks:
            target_piece = board.piece_at(target_square)
            
            # If target is valuable and of opposite color
            if (target_piece and 
                target_piece.piece_type in valuable_pieces and 
                target_piece.color != knight_color):
                targets.add(target_square)
        
        # If knight attacks multiple valuable pieces, it's a fork
        if len(targets) >= 2:
            fork_candidates.append((square, targets))
    
    # Check for other piece types that can fork (queens, rooks, bishops)
    for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP]:
        pieces = board.pieces(piece_type, board.turn)
        for square in pieces:
            piece_color = board.piece_at(square).color
            targets = set()
            
            # Get all squares this piece attacks
            attacks = board.attacks(square)
            for target_square in attacks:
                target_piece = board.piece_at(target_square)
                
                # If target is valuable and of opposite color
                if (target_piece and 
                    target_piece.piece_type in valuable_pieces and 
                    target_piece.color != piece_color):
                    targets.add(target_square)
            
            # If piece attacks multiple valuable pieces, it's a fork
            if len(targets) >= 2:
                fork_candidates.append((square, targets))
    
    return fork_candidates

def get_pins(
    board: chess.Board
) -> List[Tuple[chess.Square, chess.Square, chess.Square]]:
    """
    Find all pins on the board
    
    Args:
        board: Chess board
        
    Returns:
        List of tuples: (pinner_square, pinned_square, target_square)
    """
    pins = []
    
    # Look for pins against both kings
    for king_color in [chess.WHITE, chess.BLACK]:
        # Find the king
        kings = board.pieces(chess.KING, king_color)
        if not kings:
            continue
            
        king_square = next(iter(kings))
        
        # Get all squares with enemy sliding pieces
        enemy_color = not king_color
        enemy_queens = board.pieces(chess.QUEEN, enemy_color)
        enemy_rooks = board.pieces(chess.ROOK, enemy_color)
        enemy_bishops = board.pieces(chess.BISHOP, enemy_color)
        
        # Check for pins by queens (both orthogonal and diagonal)
        for queen_square in enemy_queens:
            ray = chess.ray(queen_square, king_square)
            if ray:  # If queen and king are aligned
                pinned_piece_found = False
                pinned_square = None
                
                # Scan from queen to king
                for sq in ray:
                    if sq == queen_square:
                        continue
                    if sq == king_square:
                        if pinned_piece_found:
                            pins.append((queen_square, pinned_square, king_square))
                        break
                        
                    # Check if there's a piece in between
                    piece = board.piece_at(sq)
                    if piece:
                        if pinned_piece_found:
                            # Second piece found, no pin
                            break
                        elif piece.color == king_color:
                            # Potential pinned piece
                            pinned_piece_found = True
                            pinned_square = sq
                        else:
                            # Enemy piece, no pin
                            break
                            
        # Check for pins by rooks (orthogonal)
        for rook_square in enemy_rooks:
            if chess.square_file(rook_square) == chess.square_file(king_square) or \
               chess.square_rank(rook_square) == chess.square_rank(king_square):
                ray = chess.ray(rook_square, king_square)
                if ray:  # If rook and king are aligned
                    pinned_piece_found = False
                    pinned_square = None
                    
                    # Scan from rook to king
                    for sq in ray:
                        if sq == rook_square:
                            continue
                        if sq == king_square:
                            if pinned_piece_found:
                                pins.append((rook_square, pinned_square, king_square))
                            break
                            
                        # Check if there's a piece in between
                        piece = board.piece_at(sq)
                        if piece:
                            if pinned_piece_found:
                                # Second piece found, no pin
                                break
                            elif piece.color == king_color:
                                # Potential pinned piece
                                pinned_piece_found = True
                                pinned_square = sq
                            else:
                                # Enemy piece, no pin
                                break
        
        # Check for pins by bishops (diagonal)
        for bishop_square in enemy_bishops:
            if abs(chess.square_file(bishop_square) - chess.square_file(king_square)) == \
               abs(chess.square_rank(bishop_square) - chess.square_rank(king_square)):
                ray = chess.ray(bishop_square, king_square)
                if ray:  # If bishop and king are aligned
                    pinned_piece_found = False
                    pinned_square = None
                    
                    # Scan from bishop to king
                    for sq in ray:
                        if sq == bishop_square:
                            continue
                        if sq == king_square:
                            if pinned_piece_found:
                                pins.append((bishop_square, pinned_square, king_square))
                            break
                            
                        # Check if there's a piece in between
                        piece = board.piece_at(sq)
                        if piece:
                            if pinned_piece_found:
                                # Second piece found, no pin
                                break
                            elif piece.color == king_color:
                                # Potential pinned piece
                                pinned_piece_found = True
                                pinned_square = sq
                            else:
                                # Enemy piece, no pin
                                break
    
    return pins

def check_discover_check_candidates(board: chess.Board) -> List[chess.Square]:
    """
    Find pieces that could deliver a discovered check if moved
    
    Args:
        board: Chess board
        
    Returns:
        List of squares with pieces that could deliver discovered checks
    """
    candidates = []
    
    # Get the current turn
    current_color = board.turn
    enemy_color = not current_color
    
    # Find the opponent's king
    enemy_kings = board.pieces(chess.KING, enemy_color)
    if not enemy_kings:
        return []  # No king found (shouldn't happen in a valid position)
        
    king_square = next(iter(enemy_kings))
    
    # Use python-chess's built-in pin detection
    # We're looking for pieces that could be moved to reveal a check
    for blocker_square in chess.SQUARES:
        blocker = board.piece_at(blocker_square)
        if not blocker or blocker.color != current_color:
            continue
            
        # Check if this piece blocks a check from our piece to enemy king
        # We simulate removing it and see if that creates a check
        board_copy = board.copy()
        # Remove the piece (replace with empty square)
        board_copy.remove_piece_at(blocker_square)
        
        # If removing this piece creates a check, it's a discovered check candidate
        if board_copy.is_check():
            candidates.append(blocker_square)
    
    return candidates

def evaluate_position(board: chess.Board) -> Dict[str, Any]:
    """
    Evaluate the current position using multiple metrics
    
    Args:
        board: Chess board
        
    Returns:
        Dictionary with various evaluation metrics
    """
    eval_dict = {
        'material_balance': material_balance(board),
        'white_king_safety': king_safety(board, 'white'),
        'black_king_safety': king_safety(board, 'black'),
        'hanging_pieces': get_hanging_pieces(board),
        'pins': len(get_pins(board)),
        'forks': len(get_fork_candidates(board))
    }
    
    return eval_dict

def stockfish_evaluate(board: chess.Board, depth: int = 15, time_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Evaluate the current position using Stockfish engine
    
    Args:
        board: Chess board
        depth: Search depth for Stockfish
        time_limit: Optional time limit in milliseconds
        
    Returns:
        Dictionary with Stockfish evaluation metrics
    """
    from stockfish import Stockfish
    import os
    
    # Directly use the known Stockfish path
    stockfish_path = "/opt/homebrew/bin/stockfish"
    
    try:
        # Initialize Stockfish with explicit path
        stockfish = Stockfish(path=stockfish_path)
            
        # Configure Stockfish
        stockfish.set_depth(depth)
        
        # Set parameters for stronger analysis
        stockfish.update_engine_parameters({
            "Threads": 4,  # Use more threads for better performance
            "Hash": 256,   # Use larger hash table (in MB)
            "MultiPV": 3   # Get top 3 moves
        })
        
        # Set position
        stockfish.set_fen_position(board.fen())
        
        # Get evaluation
        evaluation = stockfish.get_evaluation()
        
        # Get best moves
        best_moves = stockfish.get_top_moves(3)
        
        return {
            'stockfish_eval': evaluation,
            'best_moves': best_moves,
            'depth': depth,
            'engine_path': stockfish_path
        }
    except Exception as e:
        return {
            'error': f"Error using Stockfish: {str(e)}. Path tried: {stockfish_path}",
            'stockfish_eval': None,
            'best_moves': None,
            'depth': depth
        }
    
    