import chess
from typing import List, Dict, Optional, Any, Union
import math

class TreeNode:
    def __init__(self, board: chess.Board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children = []  # Child nodes
        self.visits = 0  # Number of times this node was visited
        self.score = 0.0  # Accumulated score
        self.evaluations = []  # List of evaluation dictionaries from LLM instances
        
    def add_child(self, move: Union[str, chess.Move]) -> 'TreeNode':
        """
        Add a child node by applying a move to the current board.
        
        Args:
            move: The move to apply
            
        Returns:
            New child node or None if the move is invalid
        """
        move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
        if move_obj in self.board.legal_moves:
            new_board = self.board.copy()
            new_board.push(move_obj)
            child = TreeNode(new_board, parent=self, move=move_obj)
            self.children.append(child)
            return child
        return None
    
    def update(self, score: float):
        """
        Update node statistics after a visit.
        
        Args:
            score: Evaluation score for this node
        """
        self.visits += 1
        self.score += score
    
    def get_ucb_score(self, exploration_weight: float = 1.4) -> float:
        """
        Calculate UCB score for node selection.
        
        Args:
            exploration_weight: Weight for the exploration term
            
        Returns:
            UCB score for this node
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        exploitation = self.score / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def get_average_score(self) -> float:
        """Get the average score for this node."""
        return self.score / self.visits if self.visits > 0 else 0.0
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal state (game over)."""
        return self.board.is_game_over()
    
    def get_move_path(self) -> List[str]:
        """Get the sequence of moves from root to this node."""
        moves = []
        node = self
        while node.parent is not None:
            moves.append(node.move.uci())
            node = node.parent
        return list(reversed(moves))
    
    def add_evaluation(self, evaluation: Dict[str, Any]):
        """
        Add an evaluation from an LLM instance.
        
        Args:
            evaluation: Dictionary with evaluation metrics
        """
        self.evaluations.append(evaluation)
        
        # Update score based on position score in the evaluation
        # Convert from [-1.0, 1.0] to [0.0, 1.0] for UCB
        if 'position_score' in evaluation:
            normalized_score = (evaluation['position_score'] + 1.0) / 2.0
            self.update(normalized_score)
    
    def get_aggregate_evaluation(self) -> Dict[str, Any]:
        """
        Aggregate all evaluations for this node.
        
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        if not self.evaluations:
            return {}
        
        # Simple average of numeric values and collection of plans/variations
        aggregate = {}
        numeric_keys = ['position_score', 'material_balance', 'confidence']
        list_keys = ['plans', 'key_variations']
        
        for key in numeric_keys:
            values = [eval.get(key, 0.0) for eval in self.evaluations if key in eval]
            if values:
                aggregate[key] = sum(values) / len(values)
        
        for key in list_keys:
            items = [item for eval in self.evaluations if key in eval for item in eval[key]]
            if items:
                aggregate[key] = items
        
        # Include metadata
        aggregate['visits'] = self.visits
        aggregate['move_path'] = self.get_move_path()
        
        return aggregate
    
    def select_best_child(self, exploration_weight: float = 1.4) -> Optional['TreeNode']:
        """
        Select the best child node according to UCB score.
        
        Args:
            exploration_weight: Weight for the exploration term
            
        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            ucb_score = child.get_ucb_score(exploration_weight)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def select_best_move(self) -> Optional[chess.Move]:
        """
        Select the best move based on visit count.
        
        Returns:
            Best move or None if no children
        """
        if not self.children:
            return None
        
        # Sort children by visit count (most visited first)
        sorted_children = sorted(self.children, key=lambda child: child.visits, reverse=True)
        return sorted_children[0].move
    
    def __str__(self) -> str:
        """String representation of the node."""
        move_str = f"Move: {self.move.uci()}" if self.move else "Root"
        score_str = f"Score: {self.get_average_score():.3f}"
        visits_str = f"Visits: {self.visits}"
        return f"{move_str} | {score_str} | {visits_str} | Children: {len(self.children)}"