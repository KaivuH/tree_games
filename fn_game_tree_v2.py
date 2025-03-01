import asyncio
import json
import os
from typing import Optional, List, Dict, Any, Tuple, Union, Type
import logging
from pydantic import BaseModel
from openai import AsyncOpenAI
import math

# (Assume openai and chess_engine imports and initialization as before)

class ModelInterface:
    """
    A unified, asynchronous interface for calling language models.
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None, max_tokens: int = 16000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
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
    score: int
    explanation: str

class Game:
    def __init__(self, env, model, action_signature: Dict[str, Any]):
        self.env = env
        self.model = model
        self.action_signature = action_signature

    async def play_full_game(self, comp_budget: int):
        """Play a full game by evaluating moves in parallel until the game is over."""
        while not self.env.is_game_over():
            current_player = self.env.get_turn()
            print("Turn:", current_player)
            # Set compute budget (could be adapted per player)
            current_budget = comp_budget if current_player == "black" else 0
            # Use the new recursive parallel search method
            result = await self.play_with_state(self.env.copy(), [], current_budget)
            print(f"Decision score for {current_player} is {result.score}")
            # Apply the final chosen move on the actual environment.
            action = result[1]

            if action:
                valid = self.env.take_action(action)
                if not valid:
                    print("Final move was invalid. Retrying...")
                    continue
                print(f"Final action taken: {action}")
            else:
                print("No valid action found.")
                break
        return self.env.get_result()

    async def play_with_state(
        self, state, history: List[dict], budget: int, 
        root_move: Optional[Move] = None, start: bool = False
    ) -> Tuple[Move, Score]:
        """
        Recursive tree search that performs MCTS-like rollouts.
        When not at the root (start=False), returns a list of BranchResult objects.
        At the root (start=True), aggregates branch evaluations and returns the final Move.
        """
        # Base case: no compute budget left. Evaluate the state.
        if budget <= 0:
            if start:
                messages = [{
                    "role": "user",
                    "content": (
                        f"Board state:\n{str(state)}\n"
                        "Given the objective (win the game), return the best move and evaluate the board based on how good the position is for you"
                        "UCI is for example 'f3e5a'. No Nb stuff."
                    )
                }]
                move = (await self.model.call(
                    messages, 
                    system_prompt=f"You are a chess AI providing candidate moves that will help you win. You are {state.get_turn()}",
                    output_type=Move
                ))
                return (move.action, move.score)

            else:
                score = await self.evaluate_state(state, history, root_move)
            
            return (None, score)
        
        # Request candidate moves from the model.
        messages = [{
            "role": "user",
            "content": (
                f"History: {history}\n"
                f"Board state:\n{str(state)}\n"
                "Given the objective (win the game), list up to 2 candidate moves in UCI format with descriptions. "
                "UCI is for example 'f3e5a'. No Nb stuff."
            )
        }]
        options = (await self.model.call(
            messages, 
            system_prompt=f"You are a chess AI providing candidate moves that will help you win. You are {state.get_turn()}",
            output_type=MoveChoices
        )).choices
        print("Candidate options:", options)
    
        if not options:
            eval_move = await self.evaluate_state(state, history, root_move)
            anchored = root_move if root_move is not None else eval_move
            return [BranchResult(root_move=anchored, total_score=eval_move.score or 0, visits=1)]
    
        async def simulate_option(option: Move) -> List[BranchResult]:
            # Anchor the branch on the candidate move if not already set.
            branch_root = root_move if root_move is not None else option
            sim_state = state.copy()  # Assumes a proper deep copy.
            # If the move is invalid or the game is over, perform a terminal evaluation.
            if not sim_state.take_action(option.action) or sim_state.is_game_over():
                score = await self.evaluate_state(sim_state, history + [option.dict()], branch_root)
                return option, score
            new_history = history + [option.dict()]
            # Recurse with a reduced budget.
            return option, self.play_with_state(sim_state, new_history, budget - 1, root_move=branch_root)
    
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
                if score > curr_score:
                    curr_score = score
                    b_move = move
        else:
            curr_score = math.inf
            for move, score in results:
                if score < curr_score:
                    curr_score = score
                    b_move = move
    
        print(curr_score, b_move)
        return (b_move, curr_score)

    async def evaluate_state(self, state, history: List[dict], root_move: Optional[Move]) -> Move:
        """
        Terminal evaluation when no compute budget is left.
        Asks the model to evaluate the board state and recommend a move, including a score.
        """
        messages = [{
            "role": "user",
            "content": (
                f"Final board state:\n{str(state)}\n"
                f"History: {history}\n"
                "Provide a terminal evaluation and recommend and include an evaluation score between -1 (losing) and 1 (winning). "
                f"You are {state.get_turn()}"
            )
        }]
        evaluation = await self.model.call(
            messages,
            system_prompt="Terminal state evaluation.",
            output_type=Score
        )
        return evaluation


# Example main game loop
async def main():
    from chess_engine import core
    import chess
    # Initialize your chess engine/environment
    board = core.ChessEngine()
    model_interface = ModelInterface(model_name="o3-mini")
    # Define the action signature for moves (for documentation purposes)
    action_sig = {
        "action": {
            "type": "string",
            "description": "UCI formatted move (e.g., 'f3e5')"
        }
    }
    game = Game(board, model_interface, action_sig)
    result = await game.play_full_game(comp_budget=3)
    print("Game result:", result)

if __name__ == "__main__":
    asyncio.run(main())