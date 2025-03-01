import asyncio
import json
import os
import random
from typing import Optional, List, Dict, Any, Tuple, Union, Type
import logging
from pydantic import BaseModel
from openai import AsyncOpenAI
import math
# Import for Stockfish evaluation and opening positions
from chess_engine.analysis import stockfish_evaluate
from chess_engine.opening_positions import OPENING_POSITIONS, get_random_opening

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="metagame_test.log",
    filemode="w"
)
logger = logging.getLogger(__name__)



# (Assume openai and chess_engine imports and initialization as before)

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
                        tools=tools
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

class MoveChoices(BaseModel):
    choices: List[Move]

class BranchResult(BaseModel):
    root_move: Move  # The anchored candidate move from the starting state.
    evaluation: Move  # The evaluated outcome for that branch (with description, and optionally a score).

class Move(BaseModel):
    action: str
    desc: str
    score: Optional[float] = None  # e.g. from -1 (losing) to 1 (winning)
# Extend your Move model to include a numerical score.
# New model for aggregating simulation outcomes.


class Score(BaseModel):
    score: float
    explanation: str

class Game:
    def __init__(self, env, model, action_signature: Dict[str, Any], opening_name: Optional[str] = None):
        self.env = env
        self.model = model
        self.action_signature = action_signature
        self.env.comp_budget = 0  # Initialize comp_budget
        self.stockfish_imbalance_count = 0  # Track consecutive imbalanced positions
        self.previous_evaluations = []  # Keep track of position evaluations
        
        # Set up from opening position if specified
        if opening_name:
            if opening_name.lower() == "random":
                name, fen = get_random_opening()
                print(f"Selected random opening: {name}")
            else:
                name = opening_name
                fen = OPENING_POSITIONS.get(opening_name)
                if not fen:
                    print(f"Opening '{opening_name}' not found. Using starting position.")
                    fen = OPENING_POSITIONS["Starting Position"]
                    name = "Starting Position"
                    
            print(f"Setting up board from opening: {name}")
            print(f"FEN: {fen}")
            self.env.board = self.env.create_board_from_fen(fen)
            print(self.env.get_board_visual())

    async def play_full_game(self, comp_budget: int, use_stockfish_early_stopping: bool = True):
        """
        Play a full game by evaluating moves in parallel until the game is over.
        
        Args:
            comp_budget: Computational budget for tree search
            use_stockfish_early_stopping: Whether to use Stockfish for early stopping when game is imbalanced
        """
        move_count = 1
        while not self.env.is_game_over():
            current_player = self.env.get_turn()
            print(f"\nMove {move_count}: {current_player.upper()} to play")
            print("Current board:")
            print(str(self.env))  # Print the current board state
            print("-" * 40)
            
            # Evaluate position with Stockfish if early stopping is enabled
            if use_stockfish_early_stopping and move_count > 10:  # Skip early game positions
                try:
                    # Evaluate current position with Stockfish
                    stockfish_result = stockfish_evaluate(self.env.board, depth=15)
                    if 'stockfish_eval' in stockfish_result and stockfish_result['stockfish_eval']:
                        eval_data = stockfish_result['stockfish_eval']
                        
                        # Print Stockfish evaluation
                        print(f"Stockfish evaluation: {eval_data}")
                        
                        # Check for overwhelming advantage (more than 3 points)
                        advantage = 0
                        
                        # Extract advantage value from evaluation
                        if 'type' in eval_data:
                            if eval_data['type'] == 'cp' and 'value' in eval_data:
                                advantage = abs(eval_data['value']) / 100.0  # Convert centipawns to pawns
                            elif eval_data['type'] == 'mate':
                                advantage = 10.0  # Mate is definitely overwhelming
                        
                        # Track imbalance
                        if advantage > 3.0:
                            self.stockfish_imbalance_count += 1
                            print(f"Position is imbalanced ({advantage} pawns). Imbalance count: {self.stockfish_imbalance_count}/4")
                            
                            # If imbalanced for 4 consecutive positions, end the game
                            if self.stockfish_imbalance_count >= 4:
                                # Determine winner based on evaluation
                                winner = "white" if (eval_data.get('type') == 'cp' and eval_data.get('value', 0) > 0) else "black"
                                print(f"Game stopped early: {winner.upper()} has overwhelming advantage.")
                                return f"Game stopped early. {winner.upper()} wins by overwhelming advantage."
                        else:
                            # Reset counter if position is balanced
                            self.stockfish_imbalance_count = 0
                except Exception as e:
                    print(f"Error with Stockfish evaluation: {e}")
                    # Continue with the game despite Stockfish error
            
            # Set compute budget - only black uses tree search, white uses direct evaluation
            current_budget = comp_budget if current_player == "black" else 0
            self.env.comp_budget = current_budget  # Store for reference in other methods
            
            # Use the recursive parallel search method for black, direct evaluation for white
            result = await self.play_with_state(self.env.copy(), [], current_budget, start=True)
            best_move, score = result
            print(f"Decision score for {current_player} is {score}")
            # Apply the final chosen move on the actual environment.
            action = best_move.action if best_move and hasattr(best_move, 'action') else None

            if action:
                try:
                    valid = self.env.take_action(action)
                except Exception as e:
                    print(f"Error with move format: {action}. Error: {e}")
                    valid = False
                if not valid:
                    # Get legal moves
                    legal_moves = self.env.get_legal_moves()
                    legal_moves_uci = [move.uci() for move in legal_moves]  # Convert to UCI strings
                    legal_moves_str = ", ".join(legal_moves_uci[:10])  # Show first 10 moves
                    
                    print(f"Move {action} was invalid. Valid moves include: {legal_moves_str}" + 
                          ("..." if len(legal_moves) > 10 else ""))
                    
                    # One more chance with the valid moves list
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
                        move = await self.model.call(
                            messages, 
                            system_prompt=f"You are playing chess as {current_player}. You must choose a valid move.",
                            output_type=Move
                        )
                        action = move.action
                        valid = self.env.take_action(action)
                        
                        if not valid:
                            print(f"Second move {action} was also invalid. Player loses by forfeit.")
                            return f"Game over. {current_player} loses by forfeit (two invalid moves)"
                        
                        print(f"Final action taken: {action}")
                        move_count += 1
                    except Exception as e:
                        print(f"Error getting second move: {e}")
                        return f"Game over. {current_player} loses by forfeit (error getting move)"
                else:
                    print(f"Final action taken: {action}")
                    move_count += 1
            else:
                print("No valid action found.")
                break
        return self.env.get_result()

    async def play_full_with_tools(self, tool_budget: int=2, use_stockfish_early_stopping=False) -> str:
        """
        Plays a full chess game using tool-augmented LLM moves.
        Returns the game result as a string.
        """
        move_count = 0
        TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": "get_hanging_pieces",
                    "description": "Find all hanging (undefended) pieces on the current chess board",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                },
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_legal_moves", 
                    "description": "Get all legal moves in the current position",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "play_move",
                    "description": "Play a move on the chess board",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "move": {
                                "type": "string",
                                "description": "The move in UCI format (e.g. e2e4)"
                            }
                        },
                        "required": ["move"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_fork_candidates",
                    "description": "Get candidate forks on the board",
                    "parameters": {
                        "type": "object",
                        "properties": {
                        },
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]

        tool_store = {
            "get_hanging_pieces": self.env.get_hanging_pieces,
            "get_legal_moves": self.env.get_legal_moves,
            "get_fork_candidates": self.env.get_fork_candidates
        }

        while not self.env.is_game_over() and move_count < 200:  # Prevent infinite games
            current_player = self.env.get_turn()
            board_state = str(self.env)

            if use_stockfish_early_stopping:  # Skip early game positions
                try:
                    # Evaluate current position with Stockfish
                    stockfish_result = stockfish_evaluate(self.env.board, depth=15)
                    if 'stockfish_eval' in stockfish_result and stockfish_result['stockfish_eval']:
                        eval_data = stockfish_result['stockfish_eval']
                        
                        # Print Stockfish evaluation
                        logging.info(f"Stockfish evaluation: {eval_data}")
                        
                        # Check for overwhelming advantage (more than 3 points)
                        advantage = 0
                        
                        # Extract advantage value from evaluation
                        if 'type' in eval_data:
                            if eval_data['type'] == 'cp' and 'value' in eval_data:
                                advantage = abs(eval_data['value']) / 100.0  # Convert centipawns to pawns
                            elif eval_data['type'] == 'mate':
                                advantage = 10.0  # Mate is definitely overwhelming
                        
                        # Track imbalance
                        logging.info(f"Stockfish advantage: {advantage}")
                        if advantage > 3.0:
                            self.stockfish_imbalance_count += 1
                            logging.info(f"Position is imbalanced ({advantage} pawns). Imbalance count: {self.stockfish_imbalance_count}/4")
                            
                            # If imbalanced for 4 consecutive positions, end the game
                            if self.stockfish_imbalance_count >= 4:
                                # Determine winner based on evaluation
                                winner = "white" if (eval_data.get('type') == 'cp' and eval_data.get('value', 0) > 0) else "black"
                                logging.info(f"Game stopped early: {winner.upper()} has overwhelming advantage.")
                                return winner
                        else:
                            # Reset counter if position is balanced
                            self.stockfish_imbalance_count = 0
                except Exception as e:
                    print(f"Error with Stockfish evaluation: {e}")
                    # Continue with the game despite Stockfish error

            logging.info(board_state)
            fen = self.env.board.fen()
            messages = [{
                "role": "user",
                "content": (
                    f"Board state:\n{board_state}\n"
                    f"FEN: {fen}\n"
                    f"You are playing as {current_player}. Make your next move."
                )
            }]
            c_tool_budget = tool_budget
            if current_player == "white":
                c_tool_budget = 1

            for i in range(tool_budget):

                system_prompt=f"You are playing chess as {current_player}. Use the provided tools to analyze the position to make strong moves. Use the non move tools before you do your output to gather useful information. You have {c_tool_budget - i - 1} non move taking action calls before you need to take an action"
                logging.info(f"Prompt to model:\n{system_prompt}\n{messages}")
                response = await self.model.call(
                    messages,
                    system_prompt=system_prompt,
                    tools=TOOLS
                )

                if response.tool_calls:
                    tool_call = response.tool_calls[0]
                    arguments = json.loads(tool_call.function.arguments)
                    logging.info(tool_call)

                    if tool_call.function.name == "play_move":
                        action = arguments["move"]
                        valid = self.env.take_action(action)

                        if not valid:
                            logging.info(f"Move {action} was invalid. Player {current_player} loses by forfeit.")
                            continue
                            logging.info("Invalid move")
                            
                        logging.info(f"Move played: {action}")
                        move_count += 1
                        break
                    else:
                        fn = tool_store[tool_call.function.name]
                        result = fn()
                        messages.append({
                            "role": "user",
                            "content": f"Model called function {tool_call.function.name}. Output was:\n:{result}"
                        })
                else:
                    continue
                    action = response.action
                    
                
            #except Exception as e:
             #   print(f"Error getting move: {e}")
              #  return f"Game over. {current_player} loses by forfeit (error)"
        loggging.info(f"Winner: {self.env.get_result()}")
                
        return self.env.get_result()

    async def play_with_state(
        self, state, history: List[dict], budget: int, 
        root_move: Optional[Move] = None, start: bool = False
    ) -> Tuple[Optional[Move], Union[Score, float]]:
        """
        Recursive tree search that performs MCTS-like rollouts.
        When not at the root (start=False), returns a list of BranchResult objects.
        At the root (start=True), aggregates branch evaluations and returns the final Move.
        """
        # Base case: no compute budget left. Evaluate the state.
        if budget <= 0:
            if start:
                player = state.get_turn()
                messages = [{
                    "role": "user",
                    "content": (
                        f"Board state:\n{str(state)}\n"
                        f"IMPORTANT: You are playing as {player}. "
                        "Given the objective (win the game), return the best move and evaluate the board position. "
                        "You MUST provide your move in UCI format like 'e2e4', where the first two characters are the source square and the next two are the destination square. "
                        "Do NOT use notation like 'Nf3' or 'Qd1b3'. ONLY use simple source-destination format like 'g1f3' for a knight move. "
                        f"For the score, use a centipawn-like scale where POSITIVE means GOOD FOR {player.upper()} and NEGATIVE means BAD FOR {player.upper()}. "
                        f"For example, if you're playing as {player} and have a queen advantage, the score should be around +900. "
                        f"If you're playing as {player} and are down a queen, the score should be around -900."
                    )
                }]
                print("Requesting a move with zero budget (direct evaluation)")
                try:
                    move = (await self.model.call(
                        messages, 
                        system_prompt=f"You are a chess AI providing candidate moves in UCI format (e.g., 'e2e4', 'g1f3'). You are {player}. Positive scores mean good for {player}, negative means bad for {player}.",
                        output_type=Move
                    ))
                    print(f"Got move from model: {move}")
                    # Return the move object and its score
                    return (move, move.score or 0)
                except Exception as e:
                    print(f"Error getting move: {e}")
                    # If there's an error, create a basic move with e2e4 (standard white opening)
                    default_move = Move(action="e2e4", desc="Standard opening move", score=0)
                    return (default_move, 0)

            else:
                score = await self.evaluate_position_only(state)
            
            return (None, score)
        
        # Request candidate moves from the model.
        player = state.get_turn()
        messages = [{
            "role": "user",
            "content": (
                f"History: {history}\n"
                f"Board state:\n{str(state)}\n"
                f"IMPORTANT: You are playing as {player}. "
                "Given the objective (win the game), list up to 2 candidate moves in UCI format with descriptions. "
                "UCI must be in the format like 'e2e4' or 'g1f3' where the first two characters are the source square and "
                "the next two are the destination square. Do NOT use notation like 'Nf3'. Each move MUST be valid. "
                "Focus on moves that gain material or improve your position. Be especially alert for opportunities to capture valuable pieces like queens."
            )
        }]
        options = (await self.model.call(
            messages, 
            system_prompt=f"You are a chess AI providing candidate moves in UCI format (e.g., 'e2e4', 'g1f3'). You are {player}.",
            output_type=MoveChoices
        )).choices
        print("Candidate options:", options)
    
        if not options:
            evaluation = await self.evaluate_state(state, history, root_move)
            # If no options, just return None for the move and the evaluation score
            return (None, evaluation)
    
        async def simulate_option(option: Move) -> List[BranchResult]:
            # Anchor the branch on the candidate move if not already set.
            branch_root = root_move if root_move is not None else option
            sim_state = state.copy()  # Assumes a proper deep copy.
            # If the move is invalid or the game is over, perform a terminal evaluation.
            if not sim_state.take_action(option.action) or sim_state.is_game_over():
                #evaluation = await self.evaluate_state(sim_state, history + [option.model_dump()], branch_root)
                evaluation = await self.evaluate_position_only(sim_state)
                # Return the option and the Score object (not just the score field)
                return option, evaluation
            
            # Print the board after applying this candidate move
            if budget == self.env.comp_budget and start:  # Only at the top level of search for the original player
                print(f"\n===== Evaluating candidate: {option.action} =====")
                print(f"Board after {state.get_turn().upper()} plays {option.action}:")
                print(str(sim_state))
                
                # Get separate evaluation for this position
                position_eval = await self.evaluate_position_only(sim_state)
                print(f"Direct position evaluation after {option.action}: {position_eval.score}")
            
            new_history = history + [option.model_dump()]  # Updated to use model_dump() for Pydantic v2
            # Recurse with a reduced budget.
            result = await self.play_with_state(sim_state, new_history, budget - 1, root_move=branch_root)
            
            # Show opponent's best response (only at the top level)
            if budget == self.env.comp_budget and start and isinstance(result, tuple) and len(result) == 2:
                opponent_move, opponent_score = result
                if opponent_move:
                    print(f"Best response by {sim_state.get_turn().upper()}: {opponent_move.action}")
                    resp_state = sim_state.copy()
                    if resp_state.take_action(opponent_move.action):
                        print(f"Board after response {opponent_move.action}:")
                        print(str(resp_state))
                    print(f"Score after response: {opponent_score}")
                    print("=" * 40)
            
            return option, result
    
        # Launch simulations in parallel for each candidate.
        tasks = [simulate_option(option) for option in options]
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists.
        # Aggregate results by candidate move (using its action as key).
        b_move = None
        curr_score = None
        if state.get_turn() == self.env.get_turn():
            curr_score = -math.inf
            for move, score in results:
                # Extract score value properly based on type
                if isinstance(score, tuple):
                    score_value = score[1]  # Get the second element from tuple
                elif isinstance(score, Score):
                    score_value = score.score  # Get the score field from Score object
                else:
                    score_value = float(score) if score is not None else 0.0  # Convert to float or default to 0.0
                
                # Handle case where score_value is still a Score object
                if isinstance(score_value, Score):
                    score_value = score_value.score
                
                if score_value > curr_score:
                    curr_score = score_value
                    b_move = move
        else:
            curr_score = math.inf
            for move, score in results:
                # Extract score value properly based on type
                if isinstance(score, tuple):
                    score_value = score[1]  # Get the second element from tuple
                elif isinstance(score, Score):
                    score_value = score.score  # Get the score field from Score object
                else:
                    score_value = float(score) if score is not None else 0.0  # Convert to float or default to 0.0
                
                # Handle case where score_value is still a Score object
                if isinstance(score_value, Score):
                    score_value = score_value.score
                
                if score_value < curr_score:
                    curr_score = score_value
                    b_move = move
    
        print(curr_score, b_move)
        return (b_move, curr_score)

    async def evaluate_position_only(self, state) -> Score:
        """
        Evaluates just the current position without considering history or future moves.
        Returns a Score object with numerical score and explanation.
        """
        player = state.get_turn()
        messages = [{
            "role": "user",
            "content": (
                f"Board state:\n{str(state)}\n"
                f"IMPORTANT: You are playing as {player}. "
                f"Provide ONLY a position evaluation score in centipawns where POSITIVE means GOOD FOR {player.upper()} and NEGATIVE means BAD FOR {player.upper()}. "
                f"For example, if you're playing as {player} and have a queen advantage, the score should be around +900. "
                f"If you're playing as {player} and are down a queen, the score should be around -900. "
                f"Consider material, piece activity, king safety, and pawn structure. "
                f"Focus ONLY on the current position, not potential future moves."
            )
        }]
        evaluation = await self.model.call(
            messages,
            system_prompt=f"Position evaluation only. You are {player}. Positive scores mean good for {player}, negative means bad for {player}.",
            output_type=Score
        )
        return evaluation
        
    async def evaluate_state(self, state, history: List[dict], root_move: Optional[Move]) -> Score:
        """
        Terminal evaluation when no compute budget is left.
        Asks the model to evaluate the board state and recommend a move, including a score.
        Returns a Score object with numerical score and explanation.
        """
        player = state.get_turn()
        messages = [{
            "role": "user",
            "content": (
                f"Final board state:\n{str(state)}\n"
                f"History: {history}\n"
                f"IMPORTANT: You are playing as {player}. "
                f"Provide a terminal evaluation and include an evaluation score in centipawns where POSITIVE means GOOD FOR {player.upper()} and NEGATIVE means BAD FOR {player.upper()}. "
                f"For example, if you're playing as {player} and have a queen advantage, the score should be around +900. "
                f"If you're playing as {player} and are down a queen, the score should be around -900. "
                "Consider material, piece activity, king safety, and other relevant factors."
            )
        }]
        evaluation = await self.model.call(
            messages,
            system_prompt=f"Terminal state evaluation. You are {player}. Positive scores mean good for {player}, negative means bad for {player}.",
            output_type=Score
        )
        return evaluation


# Example main game loop
async def main():
    from chess_engine import core
    import chess
    
    # Configuration options (modify these directly)
    opening_name = "random"  # Use "random" or any opening name from OPENING_POSITIONS
    comp_budget = 3  # Computational budget for tree search
    use_stockfish_early_stopping = True  # Enable early stopping with Stockfish
    
    # Initialize your chess engine/environment
    board = core.ChessEngine()
    model_interface = ModelInterface(model_name="gpt-4o")
    
    # Define the action signature for moves (for documentation purposes)
    action_sig = {
        "action": {
            "type": "string",
            "description": "UCI formatted move (e.g., 'f3e5')"
        }
    }
    
    # Initialize game with chosen opening
    game = Game(board, model_interface, action_sig, opening_name=opening_name)
    
    # Play with specified computational budget
    print(f"Starting game with computational budget: {comp_budget}")
    print(f"Stockfish early stopping: {'Enabled' if use_stockfish_early_stopping else 'Disabled'}")
    print("=" * 50)
    
    # Run N games and collect statistics
    N = 10  # Number of games to play
    stats = {"white": 0, "black": 0, "draw": 0}
    
    # Create tasks for all games
    tasks = []
    for i in range(N):
        # Create a new board and game instance for each parallel task
        game_board = core.ChessEngine()
        game_instance = Game(game_board, model_interface, action_sig, opening_name=opening_name)
        
        print(f"\nStarting game {i+1}/{N}")
        tasks.append(game_instance.play_full_with_tools(tool_budget=3, use_stockfish_early_stopping=True))
    
    # Run all games in parallel and wait for results
    results = await asyncio.gather(*tasks)
    
    # Process results
    for i, result in enumerate(results):
        print(f"\nGame {i+1} complete")
        # Parse result to determine winner
        if "WHITE" in result.upper():
            stats["white"] += 1
        elif "BLACK" in result.upper():
            stats["black"] += 1
        else:
            stats["draw"] += 1
            
        print(f"Current stats - White: {stats['white']}, Black: {stats['black']}, Draw: {stats['draw']}")
    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total games: {N}")
    print(f"White wins: {stats['white']} ({stats['white']/N*100:.1f}%)")
    print(f"Black wins: {stats['black']} ({stats['black']/N*100:.1f}%)")
    print(f"Draws: {stats['draw']} ({stats['draw']/N*100:.1f}%)")
   # game = Game(board, model_interface, action_sig)
    # Play with a computational budget of 3 for black's tree search (white uses direct evaluation)
    print("Game result:", result)

if __name__ == "__main__":
    asyncio.run(main())