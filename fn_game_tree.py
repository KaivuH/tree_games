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

class BranchResult(BaseModel):
    root_move: Move  # The anchored candidate move from the starting state.
    evaluation: Move  # The evaluated outcome for that branch (with description, and optionally a score).


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
            current_budget = comp_budget if current_player == "black" else 0
            # Use the new recursive parallel search method
            decision = await self.play_with_state(self.env.copy(), [], current_budget, start=True)
            print(decision)
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

    async def play_with_state(
        self, state, history: List[dict], budget: int, root_move: Optional[Move] = None, start: bool = False
    ) -> Union[Move, List[BranchResult]]:
        """
        Recursive tree search that, when not at the root, returns a list of BranchResult objects.
        At the root (start=True), the method synthesizes the aggregated branch evaluations into one final move.
        """
        # Base case: if no compute budget left, do a terminal evaluation.
        if budget <= 0:
            eval_move = await self.evaluate_state(state, history, root_move)
            # Return a list with one BranchResult. If no root_move was set, anchor on the evaluated move.
            anchored = root_move if root_move is not None else eval_move
            if start:
                return eval_move
            else:
                return [BranchResult(root_move=anchored, evaluation=eval_move)]
    
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
            # Fallback: if no options, evaluate the state.
            eval_move = await self.evaluate_state(state, history, root_move)
            anchored = root_move if root_move is not None else eval_move
            return [BranchResult(root_move=anchored, evaluation=eval_move)]
    
        async def simulate_option(option: Move) -> List[BranchResult]:
            # Anchor the branch on the candidate move if not already set.
            branch_root = root_move if root_move is not None else option
            sim_state = state.copy()  # Assumes a proper deep copy.
            if not sim_state.take_action(option.action) or sim_state.is_game_over():
                # If the move is invalid, return an immediate branch result.
                return [BranchResult(root_move=branch_root, evaluation=Move(action=option.action, desc="Invalid move"))]
            new_history = history + [option.dict()]
            # Recurse with a reduced budget.
            return await self.play_with_state(sim_state, new_history, budget - 1, root_move=branch_root)
    
        # Launch simulations in parallel for each candidate option.
        tasks = [simulate_option(option) for option in options]
        branches_results_lists = await asyncio.gather(*tasks)
        # Flatten the list of lists into a single list of BranchResult objects.
        all_branch_results = [br for sublist in branches_results_lists for br in sublist]
    
        # At a non-root level, simply return the aggregated branch results.
        if not start:
            return all_branch_results
    
        # At the root, synthesize a final decision from all branch results.
        synthesis_input = "Based on the following branch evaluations from the original board state, " \
                          "choose the best candidate move (from the starting state) and justify your choice briefly.\n"
        for branch in all_branch_results:
            synthesis_input += (
                f"Candidate move: {branch.root_move.action}, initial description: {branch.root_move.desc}, "
                f"Simulation evaluation: {branch.evaluation.desc}\n"
            )
        synthesis_input += f"Board state was:\n{str(state)}\nHistory: {history}\n"
    
        messages = [{"role": "user", "content": synthesis_input}]
        final_decision = await self.model.call(
            messages,
            system_prompt=f"Synthesize branch evaluations into one final move decision in UCI format. " \
                          f"Only consider the anchored candidate moves from the starting state. You are {self.env.get_turn()}",
            output_type=Move
        )
        print("Final decision:", final_decision)
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
                "ONLY consider the move proposed at the starting state of this branch. No Nb stuff. "
                f"You are {state.get_turn()}" # POTENTIAL SOURCE OF DANGER
            )
        }]
        evaluation = await self.model.call(
            messages,
            system_prompt="Terminal state evaluation.",
            output_type=Move
        )
        # If a branch anchor exists, force the evaluation to refer to that move.
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