import asyncio
import json
import os
from typing import Optional, List, Dict, Any, Tuple, Union, Type
import logging
from pydantic import BaseModel
from openai import AsyncOpenAI

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


class Game:
    def __init__(self, env, model, action_signature: Dict[str, Any]):
        self.env = env
        self.model = model
        self.action_signature = action_signature
        # Remove the extra tool definitions – we now rely on a single model call.

    async def play_full_game(self, comp_budget: int):
        """Play a full game by evaluating moves in parallel until the game is over."""
        while not self.env.is_game_over():
            current_player = self.env.get_turn()
            print("Turn:", current_player)
            # Set compute budget (could be adapted per player)
            current_budget = comp_budget if current_player == "white" else 1
            # Use the new recursive parallel search method
            decision = await self.play_with_state(self.env.copy(), [], current_budget)
            # Apply the final chosen move on the actual environment.
            action = decision.action
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

    async def play_with_state(self, state, history: List[dict], budget: int, root_move: Optional[Move] = None) -> Move:
        """
        Recursive, parallel tree search from a given state with the remaining compute budget.
        Anchors each branch to the candidate move taken at the starting state.
        
        Returns a Move (with 'action' and 'desc') that is the recommendation at the root.
        """
        # Base case: if no compute budget left, evaluate the current state.
        if budget <= 0:
            return await self.evaluate_state(state, history, root_move)
    
        # Request candidate moves from the model.
        messages = [{
            "role": "user",
            "content": (
                f"History: {history}\n"
                f"Board state:\n{str(state)}\n"
                "Given the objective (win the game), list up to 3 candidate moves in UCI format with descriptions. "
                "UCI is for example 'f3e5a'. No Nb stuff."
            )
        }]
        options = (await self.model.call(
            messages, 
            system_prompt=f"You are a chess AI providing candidate moves that will help you win. You are {state.get_turn()}",
            output_type=MoveChoices
        )).choices
        print(options)
    
        if not options:
            # Fallback: if no options, evaluate the state.
            return await self.evaluate_state(state, history, root_move)
    
        async def simulate_option(option: Move) -> Tuple[Move, Move]:
            # If no root_move is set for this branch, anchor it to the current candidate.
            branch_root = root_move if root_move is not None else option
            sim_state = state.copy()  # Assumes a proper copy method.
            if not sim_state.take_action(option.action):
                # If the move is invalid, return an anchored evaluation.
                return branch_root, Move(action=option.action, desc="Invalid move")
            new_history = history + [option.dict()]
            # Recurse with a reduced budget, while propagating the anchored (root) move.
            subtree_decision = await self.play_with_state(sim_state, new_history, budget - 1, root_move=branch_root)
            return branch_root, subtree_decision
    
        # Launch simulations in parallel for each candidate option.
        tasks = [simulate_option(option) for option in options]
        simulated_results = await asyncio.gather(*tasks)
        print(simulated_results)
    
        # Synthesize evaluations: force the synthesis to pick among the anchored moves.
        synthesis_input = "Based on the following simulated branches from the original board state, decide which candidate move is best. " \
                          "ONLY consider the moves proposed at the starting state (the anchored moves) and provide a brief justification.\n"
        for branch_root, sim_eval in simulated_results:
            synthesis_input += (
                f"Candidate move: {branch_root.action}, initial description: {branch_root.desc}; "
                f"Simulation evaluation: {sim_eval.desc}\n"
            )
        synthesis_input += f"Board state was:\n{str(state)}\nHistory: {history}\n"
    
        messages = [{
            "role": "user",
            "content": synthesis_input
        }]
        final_decision = await self.model.call(
            messages,
            system_prompt=f"Synthesize branch results into one move decision in the chess game in UCI format. " \
                          f"Only consider the anchored candidate moves from the starting state. You are {state.get_turn()}",
            output_type=Move
        )
        print(final_decision)
        return final_decision
    
    
    async def evaluate_state(self, state, history: List[dict], root_move: Optional[Move]) -> Move:
        """
        Terminal evaluation when no compute budget is left.
        Returns an evaluation anchored to the branch's candidate move from the starting state.
        """
        messages = [{
            "role": "user",
            "content": (
                f"Final board state:\n{str(state)}\n"
                f"History: {history}\n"
                "Provide a terminal evaluation and recommend a final move (in UCI format, e.g., f3e5) with a brief explanation. "
                "ONLY consider the move proposed at the starting state of this branch. No Nb stuff."
            )
        }]
        evaluation = await self.model.call(
            messages,
            system_prompt="Terminal state evaluation.",
            output_type=Move
        )
        # Anchor the evaluation to the root move if one exists.
        if root_move is not None:
            evaluation.action = root_move.action
            evaluation.desc = f"Anchored evaluation: {evaluation.desc}"
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