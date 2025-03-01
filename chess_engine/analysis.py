import chess
from typing import Dict, List, Any, Tuple, Set

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

def center_control(board: chess.Board, color: str) -> int:
    """
    Calculate how many central squares (d4, e4, d5, e5) are controlled by the given color
    
    Args:
        board: Chess board
        color: 'white' or 'black'
        
    Returns:
        Number of central squares controlled (0-4)
    """
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    color_bool = color == 'white'
    
    control_count = 0
    for square in center_squares:
        if board.is_attacked_by(color_bool, square):
            control_count += 1
    return control_count

def development_score(board: chess.Board, color: str) -> float:
    """
    Calculate the development score for the given color
    
    Args:
        board: Chess board
        color: 'white' or 'black'
        
    Returns:
        Development score between 0.0 and 1.0
    """
    # Count developed minor pieces and castling
    color_bool = color == 'white'
    
    # Starting squares for knights and bishops
    knight_start = [chess.B1, chess.G1] if color_bool else [chess.B8, chess.G8]
    bishop_start = [chess.C1, chess.F1] if color_bool else [chess.C8, chess.F8]
    
    # Count developed pieces
    pieces = board.piece_map()
    developed = 0
    total = 4  # 2 knights + 2 bishops
    
    # Check knights
    for square in knight_start:
        piece = pieces.get(square)
        if piece is None or piece.piece_type != chess.KNIGHT:
            developed += 1
    
    # Check bishops
    for square in bishop_start:
        piece = pieces.get(square)
        if piece is None or piece.piece_type != chess.BISHOP:
            developed += 1
    
    # Check if castled
    king_square = chess.E1 if color_bool else chess.E8
    king_piece = pieces.get(king_square)
    if king_piece is None or king_piece.piece_type != chess.KING:
        developed += 1  # King has moved, may have castled
        total += 1
        
    return developed / total if total > 0 else 0.0

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

def evaluate_position(board: chess.Board) -> Dict[str, Any]:
    """
    Comprehensive position evaluation including material, development, king safety, etc.
    
    Args:
        board: Chess board
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate various evaluation metrics
    white_material = material_count(board, 'white')
    black_material = material_count(board, 'black')
    material_diff = white_material - black_material
    
    white_center = center_control(board, 'white')
    black_center = center_control(board, 'black')
    center_diff = white_center - black_center
    
    white_development = development_score(board, 'white')
    black_development = development_score(board, 'black')
    development_diff = white_development - black_development
    
    white_king_safety = king_safety(board, 'white')
    black_king_safety = king_safety(board, 'black')
    king_safety_diff = white_king_safety - black_king_safety
    
    # Normalize to a single score between -1.0 and 1.0
    # Weights for different components can be tuned
    material_weight = 0.5
    center_weight = 0.15
    development_weight = 0.2
    king_safety_weight = 0.15
    
    # Normalize material (assuming max diff could be 3900 centipawns - roughly a queen + rook)
    norm_material = material_diff / 3900.0
    
    # Calculate total score from weighted components
    position_score = (
        material_weight * norm_material + 
        center_weight * (center_diff / 4.0) + 
        development_weight * development_diff + 
        king_safety_weight * king_safety_diff
    )
    
    # Clamp to reasonable range
    position_score = max(-1.0, min(1.0, position_score))
    
    # Create evaluation dictionary
    evaluation = {
        "position_score": position_score,  # -1.0 to 1.0, positive favors White
        "material_balance": material_diff / 100.0,  # In pawns
        "center_control": {
            "white": white_center,
            "black": black_center
        },
        "development": {
            "white": white_development,
            "black": black_development
        },
        "king_safety": {
            "white": white_king_safety,
            "black": black_king_safety
        },
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_insufficient_material": board.is_insufficient_material(),
        "halfmove_clock": board.halfmove_clock,
        "fullmove_number": board.fullmove_number,
    }
    
    return evaluation

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
    
    # Find attackers and defenders
    attackers = board.attackers(not piece_color, square)
    defenders = board.attackers(piece_color, square)
    
    # If no attackers, the piece is not hanging
    if not attackers:
        return False
    
    # If no defenders, the piece is hanging
    if not defenders:
        return True
    
    # Get the lowest valued attacker
    min_attacker_value = min(
        piece_value[board.piece_at(sq).piece_type] for sq in attackers
    )
    
    # Get the lowest valued defender
    min_defender_value = min(
        piece_value[board.piece_at(sq).piece_type] for sq in defenders
    )
    
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
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and is_piece_hanging(board, square):
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
    for square in chess.SQUARES:
        knight = board.piece_at(square)
        if knight and knight.piece_type == chess.KNIGHT:
            targets = set()
            knight_color = knight.color
            
            # Get all squares the knight attacks
            for move in [m for m in board.legal_moves if m.from_square == square]:
                target_square = move.to_square
                target_piece = board.piece_at(target_square)
                
                # If target is valuable and of opposite color
                if (target_piece and 
                    target_piece.piece_type in valuable_pieces and 
                    target_piece.color != knight_color):
                    targets.add(target_square)
            
            # If knight attacks multiple valuable pieces, it's a fork
            if len(targets) >= 2:
                fork_candidates.append((square, targets))
    
    # TODO: Add checks for other piece types that can fork
    
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
    
    # Check for pins by sliding pieces (queen, rook, bishop)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        # Skip if no piece or not a sliding piece
        if not piece or piece.piece_type not in {chess.QUEEN, chess.ROOK, chess.BISHOP}:
            continue
            
        piece_color = piece.color
        
        # Determine attack directions based on piece type
        directions = []
        if piece.piece_type in {chess.QUEEN, chess.ROOK}:
            # Rook-like moves (files and ranks)
            directions.extend([(0, 1), (1, 0), (0, -1), (-1, 0)])
        
        if piece.piece_type in {chess.QUEEN, chess.BISHOP}:
            # Bishop-like moves (diagonals)
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        
        # Check each direction for pins
        for dx, dy in directions:
            file, rank = chess.square_file(square), chess.square_rank(square)
            potential_pinned = None
            
            # Look along the line
            for i in range(1, 8):
                new_file, new_rank = file + i*dx, rank + i*dy
                
                # Check if we're still on the board
                if not (0 <= new_file < 8 and 0 <= new_rank < 8):
                    break
                    
                target_square = chess.square(new_file, new_rank)
                target_piece = board.piece_at(target_square)
                
                if not target_piece:
                    # Empty square, continue checking
                    continue
                elif target_piece.color == piece_color:
                    # Same color piece, no pin possible in this direction
                    break
                elif potential_pinned is None:
                    # First enemy piece found, possible pinned piece
                    potential_pinned = target_square
                else:
                    # Second enemy piece found
                    # If it's a king, we have a pin
                    if target_piece.piece_type == chess.KING:
                        pins.append((square, potential_pinned, target_square))
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
    
    # Find the opponent's king
    king_square = None
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.KING and piece.color != current_color:
            king_square = sq
            break
    
    if king_square is None:
        return []  # No king found (shouldn't happen in a valid position)
    
    # Check from the king outward for potential discovered checks
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    
    # Define the 8 directions (horizontal, vertical, diagonal)
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    
    for dx, dy in directions:
        blocker = None
        attack_piece = None
        
        # Look along the line from king outward
        for i in range(1, 8):
            new_file, new_rank = king_file + i*dx, king_rank + i*dy
            
            # Check if we're still on the board
            if not (0 <= new_file < 8 and 0 <= new_rank < 8):
                break
                
            check_square = chess.square(new_file, new_rank)
            check_piece = board.piece_at(check_square)
            
            if not check_piece:
                # Empty square, continue
                continue
            elif check_piece.color != current_color:
                # Enemy piece, no discovery possible
                break
            elif blocker is None:
                # First friendly piece - potential blocker
                blocker = check_square
            else:
                # Second friendly piece - potential attacker
                piece_type = check_piece.piece_type
                
                # Check if this piece could actually attack along this line
                if ((abs(dx) == 1 and abs(dy) == 0) or 
                        (abs(dx) == 0 and abs(dy) == 1)):
                    # Rook-like direction
                    if piece_type in {chess.ROOK, chess.QUEEN}:
                        attack_piece = check_square
                        break
                elif abs(dx) == 1 and abs(dy) == 1:
                    # Bishop-like direction
                    if piece_type in {chess.BISHOP, chess.QUEEN}:
                        attack_piece = check_square
                        break
                
        # If we found both a blocker and an attacker, the blocker is a candidate
        if blocker is not None and attack_piece is not None:
            candidates.append(blocker)
    
    return candidates