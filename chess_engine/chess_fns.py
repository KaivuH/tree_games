import chess
import chess.pgn

def get_legal_moves(board: chess.Board) -> list:
    """
    Generates all legal moves from the current position.
    
    Args:
        board: A chess.Board object representing the current position.
        
    Returns:
        A list of legal moves in algebraic notation (e.g., ["e2e4", "d2d4", ...])
    """
    # Get all legal moves and convert to string format
    legal_moves = []
    for move in board.legal_moves:
        legal_moves.append(move.uci())
    
    return legal_moves

def is_check(board: chess.Board) -> bool:
    """
    Determines if the side to move is in check.
    
    Args:
        board: A chess.Board object representing the current position.
        
    Returns:
        Boolean indicating whether the side to move is in check.
    """
    # Check if the side to move is in check
    return board.is_check()

def count_attacked_squares(board: chess.Board, side: str) -> int:
    """
    Counts the number of squares attacked by a specific side.
    
    Args:
        board: A chess.Board object representing the current position.
        side: String indicating which side to analyze ('white' or 'black').
        
    Returns:
        Integer count of squares attacked by the specified side.
    """
    # Determine which side to analyze
    color = chess.WHITE if side.lower() == 'white' else chess.BLACK
    
    # Count the attacked squares
    attacked_count = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(color, square):
            attacked_count += 1
    
    return attacked_count

def find_hanging_pieces(board: chess.Board) -> list:
    """
    Identifies undefended or hanging pieces on the board.
    
    Args:
        board: A chess.Board object representing the current position.
        
    Returns:
        A list of dictionaries, each containing:
        {
            'piece': str,  # Piece type (e.g., 'p', 'n', 'b', 'r', 'q', 'k')
            'color': str,  # 'white' or 'black'
            'square': str,  # Square in algebraic notation (e.g., 'e4')
            'value': int    # Material value of the piece
        }
    """
    # Piece material values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Kings cannot be captured
    }
    
    hanging_pieces = []
    
    # Check each square on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        # Skip empty squares
        if piece is None:
            continue
        
        color = 'white' if piece.color == chess.WHITE else 'black'
        
        # Check if the piece is defended
        piece_square = chess.square_name(square)
        
        # A piece is considered "hanging" if:
        # - it's attacked more times than defended
        # and the attacker is of lower or equal value
        attackers = list(board.attackers(not piece.color, square))
        defenders = list(board.attackers(piece.color, square))
        
        # If there are no attackers, the piece is not hanging
        if not attackers:
            continue
        
        # If there are more attackers than defenders, the piece might be hanging
        if len(attackers) > len(defenders):
            # Check if the lowest-value attacker is of lower or equal value
            min_attacker_value = min([
                piece_values[board.piece_type_at(a)] 
                for a in attackers if board.piece_type_at(a) != chess.KING
            ])
            
            # If low-value attacker or no defenders, the piece is hanging
            if (min_attacker_value <= piece_values[piece.piece_type] 
                    or not defenders):
                hanging_pieces.append({
                    'piece': piece.symbol().lower(),
                    'color': color,
                    'square': piece_square,
                    'value': piece_values[piece.piece_type]
                })
        
    return hanging_pieces

def evaluate_center_control(board: chess.Board) -> dict:
    """
    Evaluates control of the center squares (d4, d5, e4, e5).
    
    Args:
        board: A chess.Board object representing the current position.
        
    Returns:
        A dictionary with center control metrics:
        {
            'white_attacks': int,  # Number of white attacks on center squares
            'black_attacks': int,  # Number of black attacks on center squares
            'white_occupancy': int,  # Number of center squares occupied by white
            'black_occupancy': int,  # Number of center squares occupied by black
            'center_score': float   # Positive: white advantage, negative: black
        }
    """
    # Define center squares
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    
    # Initialize counters
    white_attacks = 0
    black_attacks = 0
    white_occupancy = 0
    black_occupancy = 0
    
    # Count attacks and occupancy for each center square
    for square in center_squares:
        # Check occupancy
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                white_occupancy += 1
            else:
                black_occupancy += 1
        
        # Count attacks
        if board.is_attacked_by(chess.WHITE, square):
            white_attacks += 1
        if board.is_attacked_by(chess.BLACK, square):
            black_attacks += 1
    
    # Calculate center score (positive favors white, negative favors black)
    # Double weight for occupancy compared to attacks
    center_score = (
        (white_attacks - black_attacks) + 
        2 * (white_occupancy - black_occupancy)
    )
    
    return {
        'white_attacks': white_attacks,
        'black_attacks': black_attacks,
        'white_occupancy': white_occupancy,
        'black_occupancy': black_occupancy,
        'center_score': center_score
    }

def detect_fork_opportunities(board: chess.Board) -> list:
    """
    Identifies potential fork opportunities for both sides.
    
    Args:
        board: A chess.Board object representing the current position.
        
    Returns:
        A list of dictionaries describing potential forks:
        [
            {
                'piece': str,  # Piece that can execute the fork
                'color': str,  # 'white' or 'black'
                'from_square': str,  # Current position of the piece
                'to_square': str,   # Square where the fork can be executed
                'targets': list,     # List of pieces that would be targeted
                'value_gain': float  # Potential material gain from the fork
            }
        ]
    """
    # Piece material values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # Kings are valuable for check forks but not for material gain
    }
    
    fork_opportunities = []
    
    # Check each legal move for both sides
    for color in [chess.WHITE, chess.BLACK]:
        # Make a copy of the board to explore moves
        test_board = board.copy()
        
        # If it's not this color's turn, make a null move
        if test_board.turn != color:
            test_board.push(chess.Move.null())
        
        # Generate all legal moves for this color
        for move in test_board.legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            
            # Get the piece making the move
            piece = test_board.piece_at(from_square)
            
            # Skip captures (looking for forks where piece moves to a safe square)
            if test_board.piece_at(to_square) is not None:
                continue
            
            # Make the move on a test board
            test_board_after_move = test_board.copy()
            test_board_after_move.push(move)
            
            # Check if the piece attacks multiple targets after the move
            targets = []
            value_gain = 0
            
            # Check each square for potential targets
            for target_square in chess.SQUARES:
                target_piece = test_board_after_move.piece_at(target_square)
                
                # Skip empty squares or pieces of the same color
                if target_piece is None or target_piece.color == color:
                    continue
                
                # Check if our moved piece attacks this target
                attack_mask = 0
                
                # Generate attack mask based on piece type
                piece_type = test_board_after_move.piece_type_at(to_square)
                
                if piece_type == chess.PAWN:
                    # Pawns attack diagonally
                    attack_mask = chess.BB_PAWN_ATTACKS[color][to_square]
                elif piece_type == chess.KNIGHT:
                    attack_mask = chess.BB_KNIGHT_ATTACKS[to_square]
                elif piece_type == chess.BISHOP:
                    attack_mask = chess.attacks_bb(
                        chess.BISHOP, to_square, test_board_after_move.occupied
                    )
                elif piece_type == chess.ROOK:
                    attack_mask = chess.attacks_bb(
                        chess.ROOK, to_square, test_board_after_move.occupied
                    )
                elif piece_type == chess.QUEEN:
                    attack_mask = chess.attacks_bb(
                        chess.QUEEN, to_square, test_board_after_move.occupied
                    )
                elif piece_type == chess.KING:
                    attack_mask = chess.BB_KING_ATTACKS[to_square]
                
                # Check if the target is in the attack mask
                if (1 << target_square) & attack_mask:
                    # Check if the target is adequately defended
                    attackers = list(
                        test_board_after_move.attackers(color, target_square)
                    )
                    defenders = list(
                        test_board_after_move.attackers(not color, target_square)
                    )
                    
                    # If there are more attackers than defenders, it's a valid target
                    if len(attackers) > len(defenders):
                        targets.append({
                            'piece': target_piece.symbol().lower(),
                            'square': chess.square_name(target_square),
                            'value': piece_values[target_piece.piece_type]
                        })
                        value_gain += piece_values[target_piece.piece_type]
            
            # If we found multiple targets, it's a fork
            if len(targets) >= 2:
                fork_opportunities.append({
                    'piece': piece.symbol().lower(),
                    'color': 'white' if color == chess.WHITE else 'black',
                    'from_square': chess.square_name(from_square),
                    'to_square': chess.square_name(to_square),
                    'targets': targets,
                    'value_gain': value_gain
                })
            
            # Reset the test board
            test_board = board.copy()
            if test_board.turn != color:
                test_board.push(chess.Move.null())
    
    # Sort fork opportunities by value gain (highest first)
    fork_opportunities.sort(key=lambda x: x['value_gain'], reverse=True)
    
    return fork_opportunities