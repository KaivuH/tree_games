#!/usr/bin/env python3
import argparse
import json
import sys
import time
import os
import chess
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChessTreeCLI")

class MockLLMAPI:
    """Mock LLM API for testing without real API calls."""
    
    def __call__(self, model: str, messages: List[Dict[str, Any]], 
                temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Mock LLM API call.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with mock response
        """
        # Get user message content (prompt)
        prompt = messages[-1].get("content", "")
        
        # Log prompt for debug
        logger.debug(f"Mock LLM API call with prompt: {prompt[:100]}...")
        
        # Prepare a mock response
        board_str = None
        board_lines = []
        board_found = False
        for line in prompt.split("\n"):
            if "Board State:" in line:
                board_found = True
                continue
            if board_found and line.strip() == "":
                board_found = False
                continue
            if board_found:
                board_lines.append(line.strip())
        
        if board_lines:
            board_str = "\n".join(board_lines)
            
        # Generate a mock analysis based on the board
        analysis = {
            "position_score": 0.0,
            "confidence": 0.8,
            "material_balance": 0.0,
            "development_score": 0.0,
            "tactical_score": 0.0,
            "plans": [
                "Develop minor pieces and control the center",
                "Prepare for castling to ensure king safety"
            ],
            "key_variations": [
                "The standard opening principles apply"
            ]
        }
        
        # Add some dynamic values based on board if available
        if board_str:
            try:
                board = chess.Board()
                
                # Check if we have more white or black pieces (very simplistic)
                white_count = board_str.count("P") + board_str.count("N") + \
                             board_str.count("B") + board_str.count("R") + \
                             board_str.count("Q") + board_str.count("K")
                             
                black_count = board_str.count("p") + board_str.count("n") + \
                             board_str.count("b") + board_str.count("r") + \
                             board_str.count("q") + board_str.count("k")
                
                # Set position score based on material
                material_diff = (white_count - black_count) / 10.0
                analysis["position_score"] = max(-1.0, min(1.0, material_diff))
                analysis["material_balance"] = material_diff
                
                # Add more detailed plan based on position
                if "e4" in prompt:
                    analysis["plans"].append("Consider developing with Nf3 and Bc4 (Italian Game)")
                elif "d4" in prompt:
                    analysis["plans"].append("Look for opportunities to develop with c4 (Queen's Gambit)")
                    
            except Exception as e:
                logger.error(f"Error analyzing board: {str(e)}")
        
        # Format as a mock response
        response_text = json.dumps(analysis, indent=2)
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": response_text
                    }
                }
            ]
        }
        
        # Simulate API delay
        time.sleep(0.5)
        
        return mock_response

def initialize_tree_search(use_mock_api: bool = False):
    """
    Initialize the tree search system.
    
    Args:
        use_mock_api: Whether to use the mock API
    
    Returns:
        TreeSearchController or LLMOrchestrator object
    """
    try:
        # Try to import necessary modules
        logger.info("Initializing tree search system...")
        
        if use_mock_api:
            logger.info("Using mock LLM API")
            from tree_games.tree_search.controller import TreeSearchController
            from tree_games.chess_engine.analysis import evaluate_position
            
            # Create mock LLM API
            llm_api = MockLLMAPI()
            
            # Use simpler TreeSearchController for mock API
            controller = TreeSearchController(llm_caller=None)  # Use built-in evaluation
            return controller
            
        else:
            # Check if API key is available
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found. Using mock API.")
                # Fall back to mock API
                return initialize_tree_search(use_mock_api=True)
            
            # Import with real API
            logger.info("Using OpenAI API")
            from tree_games.llm_interface.orchestrator import LLMOrchestrator
            
            # Create OpenAI API function (simplified)
            def openai_api_caller(model, messages, temperature=0.7, max_tokens=1024):
                import openai
                openai.api_key = api_key
                
                # Call API
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response
            
            # Create orchestrator with real API
            orchestrator = LLMOrchestrator(
                llm_api_caller=openai_api_caller,
                model_name="gpt-4"
            )
            
            # Start worker threads
            orchestrator.start_workers()
            
            return orchestrator
            
    except ImportError as e:
        logger.error(f"Error importing modules: {str(e)}")
        logger.info("Falling back to simple evaluation.")
        
        # Create a simple TreeSearchController with default evaluation
        from tree_games.tree_search.controller import TreeSearchController
        controller = TreeSearchController(llm_caller=None)
        return controller

def display_board(board: chess.Board):
    """
    Display a chess board in the terminal.
    
    Args:
        board: Chess board
    """
    print(board)
    print("\nFEN:", board.fen())
    print("Turn:", "White" if board.turn == chess.WHITE else "Black")
    print("Fullmove number:", board.fullmove_number)
    print("Halfmove clock:", board.halfmove_clock)
    
    # Show legal moves
    print("\nLegal moves:")
    legal_moves = list(board.legal_moves)
    for i, move in enumerate(legal_moves):
        san = board.san(move)
        print(f"{i+1}. {move.uci()} ({san})", end="\t")
        if (i + 1) % 5 == 0:
            print()
    print("\n")

def display_evaluation(evaluation: Dict[str, Any]):
    """
    Display an evaluation in the terminal.
    
    Args:
        evaluation: Evaluation dictionary
    """
    if not evaluation:
        print("No evaluation available")
        return
        
    print("\n=== Evaluation ===")
    
    # Print scores
    print(f"Position score: {evaluation.get('position_score', 0.0):.3f}")
    print(f"Confidence: {evaluation.get('confidence', 0.0):.2f}")
    print(f"Material balance: {evaluation.get('material_balance', 0.0):.2f}")
    print(f"Development score: {evaluation.get('development_score', 0.0):.2f}")
    print(f"Tactical score: {evaluation.get('tactical_score', 0.0):.2f}")
    
    # Print plans
    plans = evaluation.get("plans", [])
    if plans:
        print("\nPlans:")
        for i, plan in enumerate(plans):
            print(f"  {i+1}. {plan}")
    
    # Print variations
    variations = evaluation.get("key_variations", [])
    if variations:
        print("\nKey variations:")
        for i, variation in enumerate(variations):
            print(f"  {i+1}. {variation}")
    
    print("=================\n")

def parse_move_index(index_str: str, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
    """
    Parse a move index or UCI string.
    
    Args:
        index_str: Move index (1-based) or UCI string
        legal_moves: List of legal moves
        
    Returns:
        Move object or None if invalid
    """
    try:
        # Try parsing as index
        index = int(index_str) - 1
        if 0 <= index < len(legal_moves):
            return legal_moves[index]
    except ValueError:
        # Try parsing as UCI
        for move in legal_moves:
            if move.uci() == index_str:
                return move
    
    return None

def print_help():
    """Print help message."""
    help_text = """
    Chess Tree Search CLI Commands:
    
    move <index or uci>  - Make a move by index or UCI notation
    analyze [depth]      - Analyze current position with optional depth
    explore <move> [n]   - Explore a branch starting with move, with n variations
    undo                 - Undo the last move
    display              - Display the current board
    eval                 - Show the current position evaluation
    best                 - Show the best move
    new                  - Start a new game
    quit                 - Exit the application
    help                 - Show this help message
    """
    print(help_text)

def main():
    parser = argparse.ArgumentParser(description="Recursive LLM Chess Tree Search CLI")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM API")
    parser.add_argument("--fen", type=str, help="Initial position in FEN notation")
    args = parser.parse_args()
    
    # Initialize the tree search system
    system = initialize_tree_search(use_mock_api=args.mock)
    
    # Set initial position if provided
    if args.fen:
        try:
            board = chess.Board(args.fen)
            system.engine.board = board
            system.root = system.root.__class__(board)
            print(f"Set initial position: {args.fen}")
        except ValueError:
            print(f"Invalid FEN: {args.fen}")
    
    # Display the board
    display_board(system.engine.board)
    
    # Main command loop
    while True:
        try:
            user_input = input("chess> ").strip()
            
            if not user_input:
                continue
                
            command_parts = user_input.split()
            command = command_parts[0].lower()
            args = command_parts[1:]
            
            if command in ["quit", "exit", "q"]:
                break
                
            elif command == "help":
                print_help()
                
            elif command == "new":
                system.reset()
                print("Started new game.")
                display_board(system.engine.board)
                
            elif command == "display":
                display_board(system.engine.board)
                
            elif command == "move":
                if not args:
                    print("Please specify a move index or UCI notation.")
                    continue
                    
                legal_moves = list(system.engine.board.legal_moves)
                move = parse_move_index(args[0], legal_moves)
                
                if move:
                    san = system.engine.board.san(move)
                    success = system.commit_move(move)
                    if success:
                        print(f"Executed move: {move.uci()} ({san})")
                        display_board(system.engine.board)
                    else:
                        print(f"Failed to execute move: {move.uci()}")
                else:
                    print(f"Invalid move: {args[0]}")
                    
            elif command == "undo":
                success = system.backtrack()
                if success:
                    print("Undid last move.")
                    display_board(system.engine.board)
                else:
                    print("No moves to undo.")
                    
            elif command == "analyze":
                depth = int(args[0]) if args else 3
                print(f"Analyzing position to depth {depth}...")
                
                # Get current evaluation or perform analysis
                results = system.evaluate_all_moves(max_branches=depth)
                
                if "evaluations" in results and results["evaluations"]:
                    for eval_dict in results["evaluations"]:
                        move = eval_dict.get("move", "")
                        if move:
                            try:
                                move_obj = chess.Move.from_uci(move)
                                san = system.engine.board.san(move_obj)
                                print(f"\nAnalysis for move: {move} ({san})")
                            except ValueError:
                                print(f"\nAnalysis for move: {move}")
                        display_evaluation(eval_dict)
                else:
                    print("No evaluations available.")
                    
            elif command == "explore":
                if not args:
                    print("Please specify a move to explore.")
                    continue
                    
                legal_moves = list(system.engine.board.legal_moves)
                move = parse_move_index(args[0], legal_moves)
                
                if not move:
                    print(f"Invalid move: {args[0]}")
                    continue
                    
                branches = int(args[1]) if len(args) > 1 else 3
                
                print(f"Exploring branch for move {move.uci()} with {branches} variations...")
                
                # Perform branch exploration
                evaluation = system.explore_branch(move, max_branches=branches)
                
                # Display results
                print(f"Exploration complete for {move.uci()}")
                display_evaluation(evaluation)
                
            elif command == "eval":
                # Display the current position evaluation
                results = system.evaluate_all_moves(max_branches=1)
                
                if "evaluations" in results and results["evaluations"]:
                    display_evaluation(results["evaluations"][0])
                else:
                    print("No evaluation available.")
                    
            elif command == "best":
                # Get the best move
                best_move = system.get_best_move()
                
                if best_move:
                    try:
                        san = system.engine.board.san(best_move)
                        print(f"Best move: {best_move.uci()} ({san})")
                    except ValueError:
                        print(f"Best move: {best_move.uci()}")
                else:
                    print("No best move available. Try analyzing the position first.")
                    
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for a list of commands.")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
    print("Goodbye!")
    
if __name__ == "__main__":
    main()