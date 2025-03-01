#!/usr/bin/env python3
import chess
import chess.pgn
import os
import sys
import random
from tree_games.chess_engine.core import ChessEngine


def load_pgn_file(pgn_file_path):
    """
    Load a PGN file and return a list of games.
    
    Args:
        pgn_file_path: Path to the PGN file
        
    Returns:
        List of chess.pgn.Game objects
    """
    games = []
    with open(pgn_file_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    
    print(f"Loaded {len(games)} games from {pgn_file_path}")
    return games


def get_position_at_move(game, move_number=None):
    """
    Get the chess position after a specific move number.
    If move_number is None, returns a random position from the game.
    
    Args:
        game: chess.pgn.Game object
        move_number: Move number to get position at (1-indexed, counting 
                    full moves)
        
    Returns:
        chess.Board representing the position
    """
    board = game.board()
    moves = list(game.mainline_moves())
    
    if move_number is None:
        # Get a random position, but avoid the first 5 moves (opening book)
        # and the last 5 moves (likely endgame)
        if len(moves) <= 10:
            move_idx = random.randint(0, len(moves) - 1)
        else:
            move_idx = random.randint(5, len(moves) - 5)
    else:
        # Calculate the move index (0-indexed)
        move_idx = min((move_number - 1) * 2, len(moves) - 1)
    
    # Play moves up to the selected index
    for i, move in enumerate(moves):
        if i <= move_idx:
            board.push(move)
        else:
            break
    
    return board


def load_into_core(positions, num_positions=5):
    """
    Load positions into the ChessEngine core module and display them.
    
    Args:
        positions: List of chess.Board objects
        num_positions: Number of positions to load
        
    Returns:
        List of ChessEngine objects with loaded positions
    """
    engines = []
    for i, board in enumerate(positions[:num_positions]):
        engine = ChessEngine()
        engine.board = board.copy()
        engines.append(engine)
        
        print(f"\nPosition {i+1}:")
        print(engine.get_board_visual())
        print(f"FEN: {board.fen()}")
        print(f"Turn: {engine.get_turn()}")
        
    return engines


def main():
    pgn_file = "chess_engine/Abdusattorov.pgn"
    
    if not os.path.exists(pgn_file):
        print(f"Error: PGN file {pgn_file} not found.")
        sys.exit(1)
    
    # Load games from PGN file
    games = load_pgn_file(pgn_file)
    
    # Get random games
    if len(games) > 5:
        selected_games = random.sample(games, 5)
    else:
        selected_games = games
    
    # Get random positions from each game
    positions = [get_position_at_move(game) for game in selected_games]
    
    # Load positions into core and display them
    engines = load_into_core(positions)
    
    print("\nTo use a specific position in your code, you can use:")
    print("from tree_games.chess_engine.core import ChessEngine")
    print("engine = ChessEngine()")
    print("engine.board = chess.Board('FEN_STRING_HERE')")
    
    return engines


if __name__ == "__main__":
    main() 