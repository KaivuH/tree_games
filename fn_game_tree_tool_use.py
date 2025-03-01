import asyncio
import os
from typing import Optional, List, Dict, Any, Tuple, Union, Type
import logging
from pydantic import BaseModel
from openai import AsyncOpenAI
import json
import math

class ModelInterface:
    """
    A unified, asynchronous interface for calling language models.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None, max_tokens: int = 16000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY in environment.")
        # Assume AsyncOpenAI is set up correctly.
        self.client = AsyncOpenAI()

    def _format_messages(self, system_prompt: str, user_messages: List[dict]) -> List[dict]:
        return [{"role": "system", "content": system_prompt}] + user_messages

    async def call(
        self,
        messages: List[dict],
        system_prompt: str = "You are a helpful assistant.",
        output_type: Optional[Type[BaseModel]] = None,
        max_retries: int = 3,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Union[str, BaseModel]:
        """
        Calls the LLM using OpenAI's chat completion or HF, returning either raw text
        or a Pydantic-validated object.
        """
        # Otherwise, if using an OpenAI-based model:
        formatted_messages = self._format_messages(system_prompt, messages)
        # logging.info("\n=== Model Call (OpenAI) ===")
        # logging.info("System:", system_prompt)
        # for msg in messages:
        #     logging.info(f"{msg['role'].upper()}:", msg['content'])
        # logging.info("=" * 80)

        last_exception = None
        for attempt in range(max_retries):
            try:
                if output_type is None:
                    response = await self.client.chat.completions.create(
                        messages=formatted_messages,
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        tools=tools
                    )
                    if tools:
                        print(response)
                        result = response.choices[0].message
                    else:
                        result = response.choices[0].message.content
                    # logging.info("\nRESPONSE:", result)
                    # logging.info("=" * 80)
                    return result
                else:
                    response = await self.client.beta.chat.completions.parse(
                        messages=formatted_messages,
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        response_format=output_type,
                    )
                    parsed_obj = response.choices[0].message.parsed
                    result = output_type.model_validate(parsed_obj)
                    # logging.info("\nRESPONSE:", result)
                    # logging.info("=" * 80)
                    return result

            except Exception as e:
                logging.info(
                    f"[DEBUG] Attempt {attempt+1}/{max_retries} failed with exception: {e}"
                )
                logging.info("[DEBUG] System prompt (verbatim):")
                logging.info(system_prompt)
                logging.info("[DEBUG] User messages (verbatim):")
                for idx, msg in enumerate(messages):
                    logging.info(f"  Message {idx+1} - role='{msg['role']}':")
                    logging.info(msg["content"])
                last_exception = e
                await asyncio.sleep(1.0 * (attempt + 1))

        raise last_exception



class Move(BaseModel):
    action: str
    desc: str
    score: Optional[float] = None


class MoveChoices(BaseModel):
    choices: List[Move]


class Score(BaseModel):
    score: float
    explanation: str


class Game:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        # Define chess analysis tools
        self.chess_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_hanging_pieces",
                    "description": "Check the board for any hanging (undefended) pieces",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fork_candidates",
                    "description": "Check the board for potential fork opportunities",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_material_balance",
                    "description": "Get the current material balance (positive means white advantage)",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pins",
                    "description": "Find all pins on the current board",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            }
        ]

    async def play_full_game(self):
        """Play a full game with tree search and tool use for black player."""
        move_count = 1
        while not self.env.is_game_over():
            current_player = self.env.get_turn()
            print(f"\nMove {move_count}: {current_player.upper()} to play")
            print(str(self.env))
            print("-" * 40)
            
            # Set compute budget - use tree search with depth 2 for black
            budget = 2 if current_player == "black" else 0
            
            # Use tree search with tools for black, direct evaluation for white
            result = await self.play_with_state(self.env.copy(), [], budget, start=True)
            best_move, score = result
            print(f"Decision score for {current_player} is {score}")
            
            # Apply the final chosen move
            action = best_move.action if best_move and hasattr(best_move, 'action') else None

            if action:
                try:
                    valid = self.env.take_action(action)
                except Exception as e:
                    print(f"Error with move format: {action}. Error: {e}")
                    valid = False
                    
                if not valid:
                    # Handle invalid move with one retry
                    legal_moves = self.env.get_legal_moves()
                    legal_moves_uci = [move.uci() for move in legal_moves]
                    legal_moves_str = ", ".join(legal_moves_uci[:10])
                    if len(legal_moves) > 10:
                        legal_moves_str += "..."
                    
                    print(f"Move {action} was invalid. Valid moves include: {legal_moves_str}")
                    
                    # One more chance with valid moves list
                    messages = [{
                        "role": "user",
                        "content": (
                            f"Board state:\n{str(self.env)}\n"
                            f"Your move '{action}' was INVALID. Choose a VALID move from this list: "
                            f"{', '.join(legal_moves_uci)}\n"
                            "Return a valid UCI move (e.g., 'e2e4')."
                        )
                    }]
                    
                    try:
                        retry_move = await self.model.call(
                            messages, 
                            system_prompt=f"You are playing chess as {current_player}. You must choose a valid move.",
                            output_type=Move
                        )
                        action = retry_move.action
                        valid = self.env.take_action(action)
                        
                        if not valid:
                            print(f"Second move {action} was also invalid. Player loses by forfeit.")
                            return f"Game over. {current_player} loses by forfeit (two invalid moves)"
                    except Exception as e:
                        print(f"Error getting second move: {e}")
                        return f"Game over. {current_player} loses by forfeit (error getting move)"
                
                print(f"Move taken: {action}")
                move_count += 1
            else:
                print("No valid action found.")
                break
                
        # Game is over
        result = self.env.get_result()
        print(f"\nGame over. Result: {result}")
        return result

    async def play_with_state(
        self, state, history: List[dict], budget: int, 
        root_move: Optional[Move] = None, start: bool = False
    ) -> Tuple[Optional[Move], Union[Score, float]]:
        """
        Recursive tree search that incorporates tool use for black player.
        When not at the root, returns a list of evaluation results.
        At the root, aggregates evaluations and returns the final Move.
        """
        current_player = state.get_turn()
        
        # Base case: no compute budget left, evaluate the state directly
        if budget <= 0:
            messages = [{
                "role": "user",
                "content": (
                    f"Board state:\n{str(state)}\n"
                    "Given the current position, return the best move in UCI format like 'e2e4' "
                    "and evaluate how good the position is for you on a scale from -100 to +100. "
                    "Use UCI format where the first two characters are the source square and the next two are the destination square."
                )
            }]
            
            try:
                if start and current_player == "black":
                    # For black at the starting position (budget=0), use tools
                    move = await self.evaluate_with_tools(state)
                else:
                    # Otherwise direct evaluation
                    system_prompt = f"You are a chess AI providing a move in UCI format. You are {current_player}."
                    move = await self.model.call(messages, system_prompt=system_prompt, output_type=Move)
                
                # Return the move and its score
                return (move, move.score or 0)
            except Exception as e:
                print(f"Error evaluating position: {e}")
                # Fallback move
                default_move = Move(action="e2e4" if current_player == "white" else "e7e5", desc="Default move", score=0)
                return (default_move, 0)
        
        # Get candidate moves to explore
        messages = [{
            "role": "user",
            "content": (
                f"Board state:\n{str(state)}\n"
                "Given the current position, list up to 2 candidate moves in UCI format with descriptions. "
                "UCI format must be like 'e2e4' where the first two characters are the source square and "
                "the next two are the destination square. Each move must be valid."
            )
        }]
        
        try:
            options = (await self.model.call(
                messages, 
                system_prompt=f"You are a chess AI providing candidate moves in UCI format. You are {current_player}.",
                output_type=MoveChoices
            )).choices
        except Exception as e:
            print(f"Error getting candidate moves: {e}")
            # Fallback with a single move
            options = [Move(action="e2e4" if current_player == "white" else "e7e5", desc="Default move", score=0)]
        
        print(f"Candidate moves at depth {budget}: {[opt.action for opt in options]}")
        
        if not options:
            # If no options, evaluate the current state
            evaluation = 0  # Default neutral score
            # If no options, just return None for the move and the evaluation score
            return (None, evaluation)
        
        async def simulate_option(option: Move) -> Tuple[Move, Union[float, Score]]:
            # Set this move as the root of the branch if not already set
            branch_root = root_move if root_move is not None else option
            
            # Copy the state to avoid modifying the original
            sim_state = state.copy()
            
            # For black at depth 2, analyze the move with tools before making it
            if current_player == "black" and budget == 2:
                print(f"\nAnalyzing black's move {option.action} with tools before applying")
                await self.analyze_move_with_tools(sim_state, option.action)
            
            # Apply the move
            if not sim_state.take_action(option.action) or sim_state.is_game_over():
                # If the move is invalid or game is over, perform a terminal evaluation
                if sim_state.is_game_over():
                    result = sim_state.get_result()
                    # Convert result to score: 1-0 (white wins) = -1 for black, 0-1 (black wins) = 1 for black
                    score_value = -1 if result == "1-0" else 1 if result == "0-1" else 0
                else:
                    # Invalid move penalty
                    score_value = -999 if current_player == "white" else 999
                return option, score_value
            
            # Add the move to history
            new_history = history + [{"action": option.action, "desc": option.desc}]
            
            # After black's move and before white's response at depth 1, analyze the resulting position
            if current_player == "black" and budget == 2:
                print(f"\nAnalyzing position after black's move {option.action} with tools")
                await self.analyze_position_with_tools(sim_state)
            
            # Recurse with a reduced budget
            return option, await self.play_with_state(sim_state, new_history, budget - 1, root_move=branch_root)
        
        # Launch simulations in parallel for each candidate move
        tasks = [simulate_option(option) for option in options]
        results = await asyncio.gather(*tasks)
        
        # Select the best move based on minimax logic
        best_move = None
        best_score = None
        
        if current_player == self.env.get_turn():  # Maximizing player
            best_score = -math.inf
            for move, score in results:
                # Extract score value properly based on type
                if isinstance(score, tuple):
                    score_value = score[1]  # Get the second element from tuple
                elif isinstance(score, Score):
                    score_value = score.score
                else:
                    score_value = float(score) if score is not None else 0.0
                
                # Handle recursive case
                if isinstance(score_value, Score):
                    score_value = score_value.score
                elif isinstance(score_value, tuple):
                    score_value = score_value[1]
                
                if score_value > best_score:
                    best_score = score_value
                    best_move = move
        else:  # Minimizing player
            best_score = math.inf
            for move, score in results:
                # Extract score value properly
                if isinstance(score, tuple):
                    score_value = score[1]
                elif isinstance(score, Score):
                    score_value = score.score
                else:
                    score_value = float(score) if score is not None else 0.0
                
                # Handle recursive case
                if isinstance(score_value, Score):
                    score_value = score_value.score
                elif isinstance(score_value, tuple):
                    score_value = score_value[1]
                
                if score_value < best_score:
                    best_score = score_value
                    best_move = move
        
        print(f"Selected move at depth {budget}: {best_move.action if best_move else 'None'} with score {best_score}")
        return (best_move, best_score)

    async def evaluate_with_tools(self, state) -> Move:
        """Evaluate a position using tools and return a move."""
        # Direct analysis without API call
        analysis_results = []
        
        try:
            # Get hanging pieces
            hanging = state.get_hanging_pieces()
            white_hanging = len(hanging['white'])
            black_hanging = len(hanging['black'])
            analysis_results.append(f"Hanging pieces: White: {white_hanging}, Black: {black_hanging}")
            
            # Get fork candidates
            forks = state.get_fork_candidates()
            analysis_results.append(f"Fork opportunities: {len(forks)} potential forks")
            
            # Get material balance
            balance = state.get_material_balance()
            advantage = "White" if balance > 0 else "Black" if balance < 0 else "Even"
            analysis_results.append(f"Material balance: {balance} centipawns ({advantage} advantage)")
            
            # Get pins
            pins = state.get_pins()
            analysis_results.append(f"Pins: {len(pins)} pins found")
        except Exception as e:
            print(f"Error during evaluation analysis: {e}")
        
        print("\nPOSITION ANALYSIS:")
        for result in analysis_results:
            print(f"- {result}")
        
        # Build analysis summary for the model
        analysis_summary = "\n".join(analysis_results)
        
        # Request move based on analysis
        move_request = [{
            "role": "user", 
            "content": f"Based on this chess analysis:\n{analysis_summary}\n\nBoard state:\n{str(state)}\n\nWhat's your best move for black? Return a Move object with the best UCI move and a score from -100 to +100 indicating how good the position is for black."
        }]
        
        # Get move recommendation
        move = await self.model.call(
            move_request,
            system_prompt="You are playing as black. Choose your best move based on the analysis. Return a Move object with action in UCI format (e.g., 'e7e5') and score.",
            output_type=Move
        )
        
        print(f"Selected move with tools: {move.action} - {move.desc} (score: {move.score})")
        return move

    async def analyze_position_with_tools(self, state):
        """Analyze a position using tools and print results."""
        # Direct analysis without API call
        analysis_results = []
        
        try:
            # Get hanging pieces
            hanging = state.get_hanging_pieces()
            white_hanging = len(hanging['white'])
            black_hanging = len(hanging['black'])
            analysis_results.append(f"Hanging pieces: White: {white_hanging}, Black: {black_hanging}")
            
            # Get fork candidates
            forks = state.get_fork_candidates()
            analysis_results.append(f"Fork opportunities: {len(forks)} potential forks")
            
            # Get material balance
            balance = state.get_material_balance()
            advantage = "White" if balance > 0 else "Black" if balance < 0 else "Even"
            analysis_results.append(f"Material balance: {balance} centipawns ({advantage} advantage)")
            
            # Get pins
            pins = state.get_pins()
            analysis_results.append(f"Pins: {len(pins)} pins found")
        except Exception as e:
            print(f"Error during position analysis: {e}")
        
        print("\nPOST-MOVE ANALYSIS:")
        for result in analysis_results:
            print(f"- {result}")

    async def analyze_move_with_tools(self, state, move):
        """Analyze a potential move before making it."""
        # Make a copy of the state to check what happens after the move
        sim_state = state.copy()
        valid = sim_state.take_action(move)
        
        if not valid:
            print(f"Move {move} is invalid and cannot be analyzed.")
            return
        
        # Direct analysis without API call
        analysis_results = []
        
        try:
            # Get hanging pieces
            hanging = sim_state.get_hanging_pieces()
            white_hanging = len(hanging['white'])
            black_hanging = len(hanging['black'])
            analysis_results.append(f"Hanging pieces after move: White: {white_hanging}, Black: {black_hanging}")
            
            # Get fork candidates
            forks = sim_state.get_fork_candidates()
            analysis_results.append(f"Fork opportunities after move: {len(forks)} potential forks")
            
            # Get material balance
            balance = sim_state.get_material_balance()
            advantage = "White" if balance > 0 else "Black" if balance < 0 else "Even"
            analysis_results.append(f"Material balance after move: {balance} centipawns ({advantage} advantage)")
            
            # Get pins
            pins = sim_state.get_pins()
            analysis_results.append(f"Pins after move: {len(pins)} pins found")
        except Exception as e:
            print(f"Error during analysis: {e}")
        
        print(f"\nPRE-MOVE ANALYSIS FOR {move}:")
        for result in analysis_results:
            print(f"- {result}")


# Example main game loop
async def main():
    from chess_engine import core
    
    # Initialize chess engine and model interface
    board = core.ChessEngine()
    model_interface = ModelInterface(model_name="o3-mini")
    
    # Create game instance and play
    game = Game(board, model_interface)
    result = await game.play_full_game()
    print("Game result:", result)

if __name__ == "__main__":
    asyncio.run(main())