#!/usr/bin/env python3
"""
Utility script to evaluate chess opening positions using Stockfish.
"""

import os
import sys
import argparse
import chess
from core import ChessEngine
from opening_positions import OPENING_POSITIONS, get_opening_names, get_opening_fen, get_random_opening


def evaluate_position(engine, fen, name=None, depth=15):
    """
    Set up a position and evaluate it with Stockfish.
    
    Args:
        engine: ChessEngine instance
        fen: FEN string for the position
        name: Name of the opening (optional)
        depth: Evaluation depth
        
    Returns:
        Evaluation result dictionary
    """
    # Set up the position
    engine.board = engine.create_board_from_fen(fen)
    
    # Print position info
    if name:
        print(f"\n=== {name} ===")
    print(f"FEN: {fen}")
    print(engine.get_board_visual())
    
    # Get Stockfish evaluation
    try:
        eval_result = engine.stockfish_evaluate(depth=depth)
        
        if 'error' in eval_result and eval_result['error']:
            print(f"Error: {eval_result['error']}")
            return None
            
        # Extract evaluation
        stock_eval = eval_result['stockfish_eval']
        best_moves = eval_result['best_moves']
        
        # Print evaluation
        print("\nStockfish Evaluation:")
        print(f"Score: {stock_eval}")
        print(f"Depth: {eval_result['depth']}")
        
        # Print best moves
        print("\nTop moves:")
        for i, move in enumerate(best_moves):
            print(f"{i+1}. {move['Move']} (Centipawn: {move.get('Centipawn', 'N/A')})")
            
        return eval_result
        
    except Exception as e:
        print(f"Error evaluating position: {e}")
        return None
        

def evaluate_all_openings(depth=15):
    """
    Evaluate all opening positions in the database.
    
    Args:
        depth: Evaluation depth
    """
    engine = ChessEngine()
    
    for name, fen in OPENING_POSITIONS.items():
        evaluate_position(engine, fen, name, depth)
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess opening positions with Stockfish")
    parser.add_argument("--opening", type=str, help="Name of opening to evaluate")
    parser.add_argument("--fen", type=str, help="Custom FEN to evaluate")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish evaluation depth")
    parser.add_argument("--list", action="store_true", help="List available openings")
    parser.add_argument("--all", action="store_true", help="Evaluate all openings")
    parser.add_argument("--random", action="store_true", help="Evaluate a random opening")
    args = parser.parse_args()
    
    # Create chess engine
    engine = ChessEngine()
    
    # List openings if requested
    if args.list:
        print("Available openings:")
        for name in get_opening_names():
            print(f"- {name}")
        return
        
    # Evaluate all openings
    if args.all:
        evaluate_all_openings(args.depth)
        return
        
    # Evaluate a random opening
    if args.random:
        name, fen = get_random_opening()
        evaluate_position(engine, fen, name, args.depth)
        return
        
    # Evaluate specified opening
    if args.opening:
        fen = get_opening_fen(args.opening)
        if fen:
            evaluate_position(engine, fen, args.opening, args.depth)
        else:
            print(f"Opening '{args.opening}' not found. Use --list to see available openings.")
        return
        
    # Evaluate custom FEN
    if args.fen:
        evaluate_position(engine, args.fen, depth=args.depth)
        return
        
    # If no options are provided, show help
    parser.print_help()
    
    
if __name__ == "__main__":
    # When run directly, load a random opening and evaluate it
    engine = ChessEngine()
    print("=== Stockfish Opening Evaluation ===\n")
    
    # Get a random opening
    name, fen = get_random_opening()
    print(f"Selected opening: {name}")
    
    # Evaluate the position
    evaluate_position(engine, fen, name, depth=15)
    
    # Also show best move
    best_move = engine.get_best_move()
    if best_move:
        print(f"\nStockfish's best move: {best_move}")
        
        # Make the move to see the resulting position
        move_obj = chess.Move.from_uci(best_move)
        engine.take_action(move_obj)
        
        print("\nPosition after Stockfish's move:")
        print(engine.get_board_visual())
        
        # Evaluate new position
        print("\nEvaluating new position...")
        eval_result = engine.stockfish_evaluate(depth=12)
        if 'stockfish_eval' in eval_result:
            print(f"New evaluation: {eval_result['stockfish_eval']}")