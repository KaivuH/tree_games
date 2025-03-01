import chess
from typing import List, Dict, Optional, Any, Union, Callable
import time
import random

from tree_games.chess_engine.core import ChessEngine
from tree_games.chess_engine.analysis import evaluate_position
from tree_games.tree_search.node import TreeNode

class TreeSearchController:
    def __init__(self, llm_caller: Callable = None):
        """
        Initialize tree search controller.
        
        Args:
            llm_caller: Function to call to get LLM evaluations
        """
        self.engine = ChessEngine()
        self.root = TreeNode(self.engine.get_board())
        self.llm_caller = llm_caller
        self.exploration_weight = 1.4
        self.max_branches = 3
        self.max_depth = 4
        self.evaluation_cache = {}  # Cache to store evaluations
    
    def reset(self):
        """Reset the controller to initial state."""
        self.engine.reset_board()
        self.root = TreeNode(self.engine.get_board())
        self.evaluation_cache = {}
    
    def apply_move(self, move: Union[str, chess.Move]) -> bool:
        """
        Apply a move to the main board and update the tree.
        
        Args:
            move: Move to apply
            
        Returns:
            True if move was applied successfully
        """
        if self.engine.execute_move(move):
            # Find the child node corresponding to this move
            move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
            next_node = None
            
            for child in self.root.children:
                if child.move == move_obj:
                    next_node = child
                    break
            
            # If move wasn't previously analyzed, create new node
            if next_node is None:
                self.root = TreeNode(self.engine.get_board())
            else:
                # Use existing node as new root
                self.root = next_node
                self.root.parent = None  # Detach from previous tree
            
            return True
        return False
    
    def explore_branch(self, move: Union[str, chess.Move], max_branches: int = None) -> Optional[Dict[str, Any]]:
        """
        Explore a specific branch by spawning an LLM to analyze it.
        
        Args:
            move: First move to explore
            max_branches: Maximum branching factor for exploration
            
        Returns:
            Evaluation dictionary
        """
        if max_branches is None:
            max_branches = self.max_branches
            
        # Apply move to create branch
        child = self.root.add_child(move)
        if child is None:
            return None
            
        # Use cached evaluation if available
        board_hash = str(child.board)
        if board_hash in self.evaluation_cache:
            evaluation = self.evaluation_cache[board_hash]
            child.add_evaluation(evaluation)
            return evaluation
            
        # Request LLM evaluation of the branch
        evaluation = None
        if self.llm_caller:
            branch_context = {
                "board": child.board,
                "move_path": child.get_move_path(),
                "max_branches": max_branches,
                "max_depth": self.max_depth,
            }
            evaluation = self.llm_caller(branch_context)
        else:
            # Use fallback evaluation if no LLM caller provided
            evaluation = evaluate_position(child.board)
            
        # Cache and add evaluation to the node
        if evaluation:
            self.evaluation_cache[board_hash] = evaluation
            child.add_evaluation(evaluation)
            
        return evaluation
    
    def spawn_sub_branch(self, moves: List[str], next_move: str, 
                         mcts_params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Spawn a sub-branch exploration from an existing branch.
        
        Args:
            moves: Previous moves leading to the branching point
            next_move: Next move to explore
            mcts_params: MCTS parameters (C, max_branches)
            
        Returns:
            Evaluation dictionary
        """
        if mcts_params is None:
            mcts_params = {
                "C": self.exploration_weight,
                "max_branches": self.max_branches
            }
            
        # Navigate to the correct node
        node = self.root
        for move in moves:
            found = False
            for child in node.children:
                if child.move.uci() == move:
                    node = child
                    found = True
                    break
            
            if not found:
                # If intermediate node doesn't exist, create it
                new_node = node.add_child(move)
                if new_node is None:
                    return None
                node = new_node
                
        # Add the next move as a child
        child = node.add_child(next_move)
        if child is None:
            return None
            
        # Use cached evaluation if available
        board_hash = str(child.board)
        if board_hash in self.evaluation_cache:
            evaluation = self.evaluation_cache[board_hash]
            child.add_evaluation(evaluation)
            return evaluation
            
        # Request LLM evaluation of the branch
        evaluation = None
        if self.llm_caller:
            branch_context = {
                "board": child.board,
                "move_path": child.get_move_path(),
                "max_branches": mcts_params.get("max_branches", self.max_branches),
                "max_depth": self.max_depth,
            }
            evaluation = self.llm_caller(branch_context)
        else:
            # Use fallback evaluation if no LLM caller provided
            evaluation = evaluate_position(child.board)
            
        # Cache and add evaluation to the node
        if evaluation:
            self.evaluation_cache[board_hash] = evaluation
            child.add_evaluation(evaluation)
            
        return evaluation
    
    def select_moves_ucb(self, candidates: List[chess.Move], parent_visits: int,
                        C: float = None, max_branches: int = None) -> List[chess.Move]:
        """
        Select moves based on UCB formula for exploration/exploitation balance.
        
        Args:
            candidates: List of candidate moves
            parent_visits: Number of visits to parent node
            C: Exploration weight
            max_branches: Maximum number of branches to return
            
        Returns:
            List of selected moves
        """
        if C is None:
            C = self.exploration_weight
        if max_branches is None:
            max_branches = self.max_branches
            
        # Initialize visit counts and scores
        move_stats = []
        for move in candidates:
            # Look for existing child nodes
            child = None
            for existing_child in self.root.children:
                if existing_child.move == move:
                    child = existing_child
                    break
                    
            # Use stats from existing child or default values
            if child:
                N = child.visits if child.visits > 0 else 1
                Q = child.score
            else:
                N = 1  # Default visit count
                Q = 0.5  # Default neutral score
                
            # Calculate UCB
            ucb = Q / N + C * (parent_visits ** 0.5) / N
            move_stats.append((move, ucb))
            
        # Sort by UCB score descending
        sorted_moves = sorted(move_stats, key=lambda x: x[1], reverse=True)
        
        # Select top moves
        return [move for move, _ in sorted_moves[:max_branches]]
    
    def limit_branches_static(self, moves: List[chess.Move], max_branches: int = None) -> List[chess.Move]:
        """
        Limit candidate moves to a fixed number (static pruning).
        
        Args:
            moves: List of legal moves
            max_branches: Maximum number of moves to return
            
        Returns:
            Pruned list of moves
        """
        if max_branches is None:
            max_branches = self.max_branches
            
        # Ensure we don't exceed the number of available moves
        max_branches = min(max_branches, len(moves))
        
        # Shuffle moves for randomness when no visits are available
        shuffled_moves = list(moves)
        random.shuffle(shuffled_moves)
        
        return shuffled_moves[:max_branches]
    
    def evaluate_all_moves(self, max_branches: int = None, use_ucb: bool = True) -> Dict[str, Any]:
        """
        Evaluate all top candidate moves from the current position.
        
        Args:
            max_branches: Maximum number of moves to evaluate
            use_ucb: Whether to use UCB for move selection
            
        Returns:
            Dictionary with evaluation results
        """
        if max_branches is None:
            max_branches = self.max_branches
            
        # Get legal moves
        legal_moves = self.engine.get_legal_moves()
        
        # Select moves to explore
        selected_moves = []
        if use_ucb and self.root.visits > 0:
            selected_moves = self.select_moves_ucb(legal_moves, self.root.visits, max_branches=max_branches)
        else:
            selected_moves = self.limit_branches_static(legal_moves, max_branches=max_branches)
            
        # Explore each selected move
        for move in selected_moves:
            self.explore_branch(move, max_branches=max_branches)
            
        # Collect results
        results = []
        for child in self.root.children:
            if child.visits > 0:
                evaluation = child.get_aggregate_evaluation()
                evaluation['move'] = child.move.uci()
                results.append(evaluation)
                
        # Sort by visit count and position score
        sorted_results = sorted(results, key=lambda x: (x.get('visits', 0), x.get('position_score', 0)), reverse=True)
        
        return {
            "board": str(self.engine.get_board()),
            "evaluations": sorted_results,
            "best_move": sorted_results[0]['move'] if sorted_results else None,
        }
    
    def get_best_move(self) -> Optional[chess.Move]:
        """
        Get the best move based on current tree knowledge.
        
        Returns:
            Best move or None if no evaluated moves
        """
        return self.root.select_best_move()
    
    def commit_move(self, move: Union[str, chess.Move] = None) -> bool:
        """
        Execute a move on the main board.
        
        Args:
            move: Move to execute (if None, execute best move)
            
        Returns:
            True if move executed successfully
        """
        if move is None:
            move = self.get_best_move()
            if move is None:
                return False
                
        return self.apply_move(move)
    
    def backtrack(self, steps: int = 1) -> bool:
        """
        Move up the tree by undoing moves.
        
        Args:
            steps: Number of steps to backtrack
            
        Returns:
            True if backtracking was successful
        """
        if steps <= 0:
            return False
            
        # Attempt to undo moves
        for _ in range(steps):
            if self.engine.board.move_stack:
                self.engine.board.pop()
                
                # Update root to appropriate parent node if possible
                if self.root.parent:
                    self.root = self.root.parent
                else:
                    # Create a new root based on current board state
                    self.root = TreeNode(self.engine.get_board())
            else:
                # No more moves to undo
                return False
                
        return True