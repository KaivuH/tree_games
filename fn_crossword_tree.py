import asyncio
import json
import os
from typing import Optional, List, Dict, Any, Union, Type
import logging
from pydantic import BaseModel
from openai import AsyncOpenAI

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
        formatted_messages = self._format_messages(system_prompt, messages)
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
                        result = response.choices[0].message
                    else:
                        result = response.choices[0].message.content
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
                    return result
            except Exception as e:
                logging.info(f"[DEBUG] Attempt {attempt+1}/{max_retries} failed with exception: {e}")
                logging.info("[DEBUG] System prompt (verbatim):")
                logging.info(system_prompt)
                logging.info("[DEBUG] User messages (verbatim):")
                for idx, msg in enumerate(messages):
                    logging.info(f"  Message {idx+1} - role='{msg['role']}':")
                    logging.info(msg["content"])
                last_exception = e
                await asyncio.sleep(1.0 * (attempt + 1))
        raise last_exception

# Define a model for candidate crossword moves.
class CrosswordMove(BaseModel):
    row: int
    col: int
    direction: str  # "across" or "down"
    word: str
    desc: str     # a brief description or justification

class MoveChoices(BaseModel):
    choices: List[CrosswordMove]

class BranchResult(BaseModel):
    root_move: CrosswordMove  # The candidate move proposed at the starting grid.
    evaluation: CrosswordMove  # The evaluation of that branch (with description).

class Game:
    def __init__(self, env, model, action_signature: Dict[str, Any]):
        self.env = env
        self.model = model
        self.action_signature = action_signature

    async def play_full_game(self, comp_budget: int):
        """Continue applying moves until the crossword is solved."""
        while not self.env.is_solved():
            print("Current grid:")
            print(str(self.env))
            decision = await self.play_with_state(self.env.copy(), [], comp_budget, start=True)
            print("Decision:", decision)
            # Apply the final chosen move on the actual environment.
            move = decision
            if move:
                valid = self.env.take_action(move.dict(exclude={"desc"}))
                if not valid:
                    print("Final move was invalid. Retrying...")
                    continue
                print(f"Final action taken: {move.dict()}")
            else:
                print("No valid action found.")
                break
        return self.env.get_result()

    async def play_with_state(
        self, state, history: List[dict], budget: int, root_move: Optional[CrosswordMove] = None, start: bool = False
    ) -> Union[CrosswordMove, List[BranchResult]]:
        # Terminal case: no remaining compute budget.
        if budget <= 0:
            eval_move = await self.evaluate_state(state, history, root_move)
            anchored = root_move if root_move is not None else eval_move
            if start:
                return eval_move
            else:
                return [BranchResult(root_move=anchored, evaluation=eval_move)]
    
        messages = [{
            "role": "user",
            "content": (
                f"History: {history}\n"
                f"Grid state:\n{str(state)}\n"
                "Given the objective of solving the crossword, list up to 2 candidate moves in JSON format with keys: row, col, direction, word, and a brief description. "
                "Ensure the move fits within the grid and does not fill a black cell."
            )
        }]
        options = (await self.model.call(
            messages, 
            system_prompt="You are a crossword solving AI. Evaluate the grid and propose candidate moves.",
            output_type=MoveChoices
        )).choices
        print("Candidate options:", options)
    
        if not options:
            eval_move = await self.evaluate_state(state, history, root_move)
            anchored = root_move if root_move is not None else eval_move
            return [BranchResult(root_move=anchored, evaluation=eval_move)]
    
        async def simulate_option(option: CrosswordMove) -> List[BranchResult]:
            branch_root = root_move if root_move is not None else option
            sim_state = state.copy()  # assumes deep copy of the grid state
            if not sim_state.take_action(option.dict(exclude={"desc"})):
                return [BranchResult(
                    root_move=branch_root, 
                    evaluation=CrosswordMove(
                        row=option.row, col=option.col, direction=option.direction, 
                        word=option.word, desc="Invalid move"
                    )
                )]
            new_history = history + [option.dict()]
            return await self.play_with_state(sim_state, new_history, budget - 1, root_move=branch_root)
    
        tasks = [simulate_option(option) for option in options]
        branches_results_lists = await asyncio.gather(*tasks)
        all_branch_results = [br for sublist in branches_results_lists for br in sublist]
    
        if not start:
            return all_branch_results
    
        # At the root, synthesize a final decision.
        synthesis_input = "Based on the following branch evaluations from the original grid state, choose the best candidate move and justify your choice briefly.\n"
        for branch in all_branch_results:
            synthesis_input += (
                f"Candidate move: row {branch.root_move.row}, col {branch.root_move.col}, direction {branch.root_move.direction}, "
                f"word '{branch.root_move.word}', description: {branch.root_move.desc}; "
                f"Simulation evaluation: {branch.evaluation.desc}\n"
            )
        synthesis_input += f"Grid state was:\n{str(state)}\nHistory: {history}\n"
    
        messages = [{"role": "user", "content": synthesis_input}]
        final_decision = await self.model.call(
            messages,
            system_prompt=(
                "Synthesize branch evaluations into one final move decision in JSON format with keys: "
                "row, col, direction, word, and a brief explanation."
            ),
            output_type=CrosswordMove
        )
        print("Final decision:", final_decision)
        return final_decision

    async def evaluate_state(self, state, history: List[dict], root_move: Optional[CrosswordMove]) -> CrosswordMove:
        messages = [{
            "role": "user",
            "content": (
                f"History: {history}\n"
                f"Grid state:\n{str(state)}\n"
                f"Across clues:\n{state.across_clues}\n"
                f"Down clues:\n{state.down_clues}\n"
                "Using these clues as guidance, list up to 2 candidate moves in JSON format with keys: row, col, direction, word, and a brief description. "
                "Ensure the move fits within the grid, matches the clues, and does not fill a black cell."
            )
        }]
        evaluation = await self.model.call(
            messages,
            system_prompt="Terminal grid state evaluation.",
            output_type=CrosswordMove
        )
        if root_move is not None:
            evaluation.row = root_move.row
            evaluation.col = root_move.col
            evaluation.direction = root_move.direction
            evaluation.word = root_move.word
            evaluation.desc = f"Anchored evaluation: {evaluation.desc}"
        return evaluation


def create_unsolved_grid(puzzle_json):
    solved = puzzle_json["grid"]  # This is the solved grid in list form.
    rows = puzzle_json["size"]["rows"]
    cols = puzzle_json["size"]["cols"]
    clues = puzzle_json["clues"]
    across_clues = "\n".join(f"{i+1}. {clue}" for i, clue in enumerate(clues["across"]))
    down_clues = "\n".join(f"{i+1}. {clue}" for i, clue in enumerate(clues["down"]))

    unsolved = []
    solution_grid = []
    for r in range(rows):
        unsolved_row = []
        solution_row = []
        for c in range(cols):
            cell = solved[r * cols + c]
            solution_row.append(cell)
            # Replace fillable letters with None, keep black cells as "#"
            unsolved_row.append("#" if cell == "." else None)
        unsolved.append(tuple(unsolved_row))
        solution_grid.append(tuple(solution_row))
    return tuple(unsolved), across_clues, down_clues, tuple(solution_grid)

async def main():
    from crosswords_env import CrosswordEnv  
    with open("/Users/kushalt/cryptics/tree_games/evals/crosswords/nyt_crosswords/1978/01/01.json", "r") as f:
        puzzle_json = json.load(f)
    unsolved_grid, across_clues, down_clues, solution_grid = create_unsolved_grid(puzzle_json)
    env = CrosswordEnv(unsolved_grid, across_clues, down_clues, solution_grid)
    
    model_interface = ModelInterface(model_name="o3-mini")
    action_sig = {
        "row": {"type": "integer", "description": "Starting row index for the word."},
        "col": {"type": "integer", "description": "Starting column index for the word."},
        "direction": {"type": "string", "description": "Direction ('across' or 'down')."},
        "word": {"type": "string", "description": "The word to fill in."}
    }
    game = Game(env, model_interface, action_sig)
    result = await game.play_full_game(comp_budget=3)
    print("Final result:", result)

if __name__ == "__main__":
    asyncio.run(main())
