import chess
from typing import Dict, List, Any

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