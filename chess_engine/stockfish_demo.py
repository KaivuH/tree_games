#!/usr/bin/env python3
"""
Simple demo script showing Stockfish evaluation integration with core.py.
Evaluates several chess positions directly using the ChessEngine class.
"""

from core import ChessEngine

# Sample positions to evaluate (FEN strings)
POSITIONS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    
    # Ruy Lopez
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    
    # Sicilian Najdorf
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    
    # Position with clear advantage for white
    "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    
    # Position with mate in 2
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
]

def main():
    # Create engine instance
    engine = ChessEngine()
    
    print("=== Stockfish Evaluation Demo ===\n")
    
    # Evaluate each position
    for i, fen in enumerate(POSITIONS):
        print(f"\nPosition #{i+1}:")
        print("FEN:", fen)
        
        # Set up the position
        engine.board = engine.create_board_from_fen(fen)
        print(engine.get_board_visual())
        
        # Get material balance using built-in evaluation
        material = engine.get_material_balance()
        print(f"Material balance: {material} centipawns")
        
        # Evaluate with Stockfish
        print("\nEvaluating with Stockfish...")
        try:
            eval_result = engine.stockfish_evaluate(depth=15)
            
            if 'error' in eval_result and eval_result['error']:
                print(f"Error using Stockfish: {eval_result['error']}")
                continue
                
            # Parse evaluation
            stock_eval = eval_result['stockfish_eval']
            print(f"Stockfish evaluation: {stock_eval}")
            
            # Get best moves
            best_moves = eval_result['best_moves']
            print("\nTop moves:")
            for j, move in enumerate(best_moves):
                print(f"{j+1}. {move['Move']} (Centipawn: {move.get('Centipawn', 'N/A')}, Mate: {move.get('Mate', 'N/A')})")
            
            # Get single best move
            best_move = engine.get_best_move()
            if best_move:
                print(f"\nBest move: {best_move}")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Stockfish is installed on your system.")
            
        print("-" * 50)
            
if __name__ == "__main__":
    main()