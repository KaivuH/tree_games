from typing import Dict, List, Any, Optional, Callable
import chess
import time
import threading
import queue

from tree_games.tree_search.controller import TreeSearchController
from tree_games.llm_interface.agent import LLMAgent

class LLMOrchestrator:
    def __init__(self, llm_api_caller: Callable, model_name: str = "gpt-4"):
        """
        Initialize the LLM orchestrator for multi-agent chess analysis.
        
        Args:
            llm_api_caller: Function to call LLM API
            model_name: Name of the model to use
        """
        self.llm_api_caller = llm_api_caller
        self.model_name = model_name
        
        # Create root agent
        self.root_agent = LLMAgent(
            llm_api_caller=llm_api_caller,
            model_name=model_name,
            system_message=(
                "You are the Root LLM in a multi-agent chess analysis system. "
                "Your role is to coordinate the exploration of different chess variations "
                "by delegating to Branch Explorer LLMs. You will analyze a position, identify "
                "promising candidate moves, and call explore_branch() for each candidate to "
                "create parallel explorations. After receiving evaluations from all branches, "
                "you will aggregate the insights and make a final move decision."
            )
        )
        
        # Create tree search controller with LLM caller
        self.controller = TreeSearchController(llm_caller=self._llm_branch_caller)
        
        # Track branch explorer agents
        self.branch_agents = {}  # Maps branch ID to LLMAgent
        self.branch_results = {}  # Maps branch ID to evaluation results
        
        # Create request queue and worker threads
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.worker_threads = []
        self.max_workers = 4
        self.running = False
        
    def _llm_branch_caller(self, branch_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper function to call LLM via appropriate agent.
        
        Args:
            branch_context: Dictionary with branch context
            
        Returns:
            Evaluation dictionary
        """
        # Create branch ID from move path
        move_path = branch_context.get("move_path", [])
        branch_id = "_".join(move_path) if move_path else "root"
        
        # Create new branch agent if needed
        if branch_id not in self.branch_agents:
            branch_depth = len(move_path)
            system_message = (
                f"You are a Branch Explorer LLM at depth {branch_depth} in a multi-agent chess analysis system. "
                f"Your role is to analyze a specific variation after the moves {', '.join(move_path) if move_path else 'from initial position'}. "
                f"You will evaluate the position, consider responses, and can call spawn_sub_branch() "
                f"to explore deeper variations. Return a comprehensive evaluation of this position."
            )
            
            self.branch_agents[branch_id] = LLMAgent(
                llm_api_caller=self.llm_api_caller,
                model_name=self.model_name,
                system_message=system_message
            )
            
        # Get evaluation from agent
        agent = self.branch_agents[branch_id]
        evaluation = agent.generate_branch_exploration(branch_context)
        
        # Store result
        self.branch_results[branch_id] = evaluation
        
        return evaluation
    
    def _worker_thread(self):
        """Worker thread to process LLM requests in parallel."""
        while self.running:
            try:
                # Get request from queue
                task_id, task_type, args = self.request_queue.get(timeout=1)
                
                try:
                    # Process request based on type
                    if task_type == "explore_branch":
                        move, max_branches = args
                        result = self.controller.explore_branch(move, max_branches)
                    elif task_type == "spawn_sub_branch":
                        moves, next_move, mcts_params = args
                        result = self.controller.spawn_sub_branch(moves, next_move, mcts_params)
                    else:
                        result = {"error": f"Unknown task type: {task_type}"}
                        
                    # Put result in response queue
                    self.response_queue.put((task_id, result))
                    
                except Exception as e:
                    self.response_queue.put((task_id, {"error": str(e)}))
                    
                finally:
                    # Mark task as done
                    self.request_queue.task_done()
                    
            except queue.Empty:
                # Queue timeout, continue looping
                pass
                
            except Exception as e:
                print(f"Worker thread error: {str(e)}")
                
    def start_workers(self):
        """Start worker threads for parallel processing."""
        self.running = True
        
        # Create worker threads
        for _ in range(self.max_workers):
            thread = threading.Thread(target=self._worker_thread)
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
            
    def stop_workers(self):
        """Stop worker threads."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=2)
                
        self.worker_threads = []
        
    def explore_branch(self, move: str, max_branches: int = None) -> str:
        """
        Explore a branch starting with a specific move.
        
        Args:
            move: First move to explore
            max_branches: Maximum branching factor
            
        Returns:
            Branch ID
        """
        # Create branch ID
        branch_id = f"branch_{move}_{int(time.time())}"
        
        # Add task to request queue
        self.request_queue.put((branch_id, "explore_branch", (move, max_branches)))
        
        return branch_id
        
    def spawn_sub_branch(self, moves: List[str], next_move: str, 
                         mcts_params: Dict[str, Any] = None) -> str:
        """
        Spawn a sub-branch from an existing branch.
        
        Args:
            moves: Previous moves in the branch
            next_move: Next move to explore
            mcts_params: MCTS parameters
            
        Returns:
            Sub-branch ID
        """
        # Create sub-branch ID
        move_str = "_".join(moves + [next_move])
        sub_branch_id = f"subbranch_{move_str}_{int(time.time())}"
        
        # Add task to request queue
        self.request_queue.put((sub_branch_id, "spawn_sub_branch", (moves, next_move, mcts_params)))
        
        return sub_branch_id
        
    def collect_results(self, timeout: float = 0.1) -> Dict[str, Any]:
        """
        Collect available results from the response queue.
        
        Args:
            timeout: Timeout for queue.get
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        
        while True:
            try:
                # Get result from queue with timeout
                task_id, result = self.response_queue.get(timeout=timeout)
                results[task_id] = result
                self.response_queue.task_done()
                
            except queue.Empty:
                # No more results
                break
                
        return results
        
    def wait_for_branches(self, branch_ids: List[str], timeout: float = 30.0) -> Dict[str, Any]:
        """
        Wait for specific branches to complete.
        
        Args:
            branch_ids: List of branch IDs to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary mapping branch IDs to results
        """
        results = {}
        start_time = time.time()
        remaining_ids = set(branch_ids)
        
        while remaining_ids and time.time() - start_time < timeout:
            # Check for new results
            new_results = self.collect_results(timeout=0.1)
            
            # Add completed branches to results
            for branch_id in list(remaining_ids):
                if branch_id in new_results:
                    results[branch_id] = new_results[branch_id]
                    remaining_ids.remove(branch_id)
                    
            # Short sleep to avoid busy waiting
            if remaining_ids:
                time.sleep(0.1)
                
        # Return collected results
        return results
        
    def evaluate_all_moves(self, max_branches: int = None, use_ucb: bool = True) -> Dict[str, Any]:
        """
        Evaluate all top candidate moves from the current position.
        
        Args:
            max_branches: Maximum number of moves to evaluate
            use_ucb: Whether to use UCB for move selection
            
        Returns:
            Dictionary with evaluation results
        """
        return self.controller.evaluate_all_moves(max_branches, use_ucb)
        
    def get_best_move(self) -> Optional[str]:
        """
        Get the best move based on current tree knowledge.
        
        Returns:
            Best move in UCI format or None if no evaluated moves
        """
        move = self.controller.get_best_move()
        return move.uci() if move else None
        
    def commit_move(self, move: str = None) -> bool:
        """
        Execute a move on the main board.
        
        Args:
            move: Move to execute (if None, execute best move)
            
        Returns:
            True if move executed successfully
        """
        return self.controller.commit_move(move)
        
    def get_board_visual(self) -> str:
        """
        Get a text representation of the current board.
        
        Returns:
            Text representation of the board
        """
        return self.controller.engine.get_board_visual()
        
    def get_branch_results(self) -> Dict[str, Any]:
        """
        Get all branch results collected so far.
        
        Returns:
            Dictionary mapping branch IDs to results
        """
        return self.branch_results.copy()
        
    def clear_branches(self):
        """Clear branch agents and results."""
        self.branch_agents = {}
        self.branch_results = {}
        
    def reset(self):
        """Reset the controller to initial state."""
        self.controller.reset()
        self.clear_branches()