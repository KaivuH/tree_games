from typing import Dict, List, Any, Optional, Callable
import chess
import json
import time

class LLMAgent:
    def __init__(self, 
                 llm_api_caller: Callable,
                 model_name: str = "gpt-4",
                 system_message: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        """
        Initialize an LLM agent for chess analysis.
        
        Args:
            llm_api_caller: Function to call LLM API
            model_name: Name of the model to use
            system_message: System message for the LLM
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
        """
        self.llm_api_caller = llm_api_caller
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Default system message if none provided
        if system_message is None:
            self.system_message = (
                "You are an expert chess analyst. Your task is to analyze the chess position provided "
                "and evaluate it thoroughly. Return your evaluation in a structured JSON format "
                "including position score, material balance, development assessment, tactical opportunities, "
                "and strategic plans. Be concise but comprehensive in your analysis."
            )
        else:
            self.system_message = system_message
            
        self.history = []
        
    def _get_board_description(self, board: chess.Board) -> str:
        """
        Get a human-readable description of the board state.
        
        Args:
            board: Chess board
            
        Returns:
            String description of the board
        """
        fen = board.fen()
        ascii_board = str(board)
        
        turn = "White" if board.turn == chess.WHITE else "Black"
        castling_rights = ""
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_rights += "K"
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_rights += "Q"
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_rights += "k"
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_rights += "q"
        
        fullmove_number = board.fullmove_number
        halfmove_clock = board.halfmove_clock
        
        description = (
            f"Board State:\n{ascii_board}\n\n"
            f"FEN: {fen}\n"
            f"Turn: {turn}\n"
            f"Castling rights: {castling_rights or 'None'}\n"
            f"Fullmove number: {fullmove_number}\n"
            f"Halfmove clock: {halfmove_clock}\n"
        )
        
        return description
    
    def _format_move_history(self, move_path: List[str]) -> str:
        """
        Format move history for LLM context.
        
        Args:
            move_path: List of moves in UCI format
            
        Returns:
            Formatted move history
        """
        if not move_path:
            return "No moves played yet."
            
        move_pairs = []
        for i in range(0, len(move_path), 2):
            if i + 1 < len(move_path):
                move_pairs.append(f"{i//2 + 1}. {move_path[i]} {move_path[i+1]}")
            else:
                move_pairs.append(f"{i//2 + 1}. {move_path[i]}")
                
        return "Move history: " + " ".join(move_pairs)
    
    def analyze_position(self, board: chess.Board, move_path: List[str] = None, 
                         max_depth: int = 3, max_branches: int = 3) -> Dict[str, Any]:
        """
        Request LLM to analyze a chess position.
        
        Args:
            board: Chess board
            move_path: Path of moves that led to this position
            max_depth: Maximum depth to analyze
            max_branches: Maximum branching factor
            
        Returns:
            Analysis dictionary
        """
        # Prepare context for LLM
        board_description = self._get_board_description(board)
        move_history = self._format_move_history(move_path or [])
        
        # Construct prompt
        user_prompt = (
            f"{board_description}\n"
            f"{move_history}\n\n"
            f"Please analyze this position and provide a comprehensive evaluation. "
            f"Consider material balance, piece activity, king safety, and potential plans "
            f"for both sides. You may explore up to {max_depth} moves deep, considering "
            f"up to {max_branches} candidate moves at each position.\n\n"
            f"Return your analysis as a JSON object with the following structure:\n"
            f"```json\n"
            f"{{\n"
            f'  "position_score": float,  // between -1.0 and 1.0, positive favors White\n'
            f'  "confidence": float,      // between 0.0 and 1.0\n'
            f'  "material_balance": float,  // in pawns, positive favors White\n'
            f'  "development_score": float,  // between -1.0 and 1.0\n'
            f'  "tactical_score": float,  // between -1.0 and 1.0\n'
            f'  "plans": [string],  // List of strategic plans\n'
            f'  "key_variations": [string]  // Critical variations explored\n'
            f"}}\n```"
        )
        
        # Call LLM API
        try:
            response = self.llm_api_caller(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract JSON from response
            response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Store in history
            self.history.append({
                "board": str(board),
                "prompt": user_prompt,
                "response": response_text,
                "timestamp": time.time()
            })
            
            # Parse JSON
            try:
                # Extract JSON if wrapped in code fences
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].strip()
                else:
                    json_str = response_text
                    
                evaluation = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["position_score", "confidence", "material_balance", 
                                 "development_score", "tactical_score", "plans", "key_variations"]
                for field in required_fields:
                    if field not in evaluation:
                        evaluation[field] = 0.0 if field != "plans" and field != "key_variations" else []
                
                return evaluation
                
            except json.JSONDecodeError:
                # Fallback to simple evaluation if JSON parsing fails
                return {
                    "position_score": 0.0,
                    "confidence": 0.1,
                    "material_balance": 0.0,
                    "development_score": 0.0,
                    "tactical_score": 0.0,
                    "plans": ["Structured analysis unavailable"],
                    "key_variations": ["JSON parsing failed"],
                    "raw_response": response_text
                }
                
        except Exception as e:
            # Handle API errors
            return {
                "position_score": 0.0,
                "confidence": 0.0,
                "material_balance": 0.0,
                "development_score": 0.0,
                "tactical_score": 0.0,
                "plans": [f"Error: {str(e)}"],
                "key_variations": ["API call failed"],
                "error": str(e)
            }
    
    def generate_branch_exploration(self, branch_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a branch exploration based on context.
        
        Args:
            branch_context: Dictionary containing board, move_path, and parameters
            
        Returns:
            Evaluation dictionary
        """
        board = branch_context.get("board")
        move_path = branch_context.get("move_path", [])
        max_depth = branch_context.get("max_depth", 3)
        max_branches = branch_context.get("max_branches", 3)
        
        if not board:
            return {"error": "No board provided in context"}
            
        return self.analyze_position(board, move_path, max_depth, max_branches)
        
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of LLM interactions.
        
        Returns:
            List of interaction records
        """
        return self.history
        
    def clear_history(self):
        """Clear the history of LLM interactions."""
        self.history = []