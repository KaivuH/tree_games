import openai
from openai import AsyncOpenAI
from typing import Optional, List, Any, Tuple, Type, Union, Dict
import os
from pydantic import BaseModel
import logging
import asyncio
import json
import httpx


class ModelInterface:
    """
    A unified, asynchronous interface for calling language models with the
    structured parse method. If the user does not provide an output_type,
    we can parse into a trivial wrapper with a single text field.
    """

    def __init__(
        self, model_name: str, api_key: Optional[str] = None, max_tokens: int = 16000
    ):
        """
        Initialize the OpenRouter interface.

        Args:
            model_name: Model name with provider prefix 
                (e.g., 'anthropic/claude-3-sonnet-20240229')
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY in environment."
            )

        #self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.client = AsyncOpenAI()


    
    def _format_messages(
        self, system_prompt: str, user_messages: List[dict]
    ) -> List[dict]:
        """
        Utility to prepend a system message.
        user_messages is typically: [{'role': 'user', 'content': ...}, ...]
        """
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


GAME_PROMPT = """
You are a chess AI. You are in a chess environment and you are given affordances to plan before you take your move.

Please use them to explore the space and play against yourself in the simulation before you play against your real opponent. Use them when you need to think about the env.
Only communicate via tool calls. ALSO simulate and use the simulation tools before you do take_action. This is a way for you to acquire info.

This is how you should approach the problem. 
Take steps and learn the situation of the game/explore outcomes using take_simulated action and only after you have gained enough info go back and take_action. Explore several different moves before you finish.
Remember you can backtrack to explore beyond the current search path. Please backtrack if it's useful. before you run out you need to call take_action, maybe in the last budget.

Simulate the environment using the functions BEFORE you overthink. it's fine to simulate before you overthink.
"""

class Game:
    def __init__(self, env, model, action_signature):
        self.env = env
        self.model = model
        self.leaves = set()
        self.action_signature = action_signature
        self.tools = [
        {
            "type": "function",
            "function": {
                "name": "take_simulated_action",
                "description": f"Jump into a simulated inner model of the action and what its effect would be on the environment.",
                "parameters": {
                    "type": "object",
                    "properties": action_signature,
                    "required": ["uci_action"]
                },
            },
        },
        {
            "type": "function", 
            "function": {
                "name": "create_new_tool",
                "description": "Create a new tool/abstraction to interact with the environment. Given your knowledge of the environment you can create tools that allow you to save on general computations/things to do in the env",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "name of the tool you will use"
                        },
                        "function_code": {
                            "type": "string",
                            "description": "Code of a python function that computes some property of the environment without modifying the environment. takes in a GameState as argument."
                        }
                      },
                    }
                }
            }
        ,
        {
            "type": "function",
            "function": {
                "name": "backtrack",
                "description": "Go back to the previous state in the environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n_moves": {
                            "type": "integer",
                            "description": "How many moves you want to backtrack."
                        }
                    }
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "take_action", 
                "description": "Take a real action in the environment (not simulated)",
                "parameters": {
                    "type": "object",
                    "properties": action_signature
                },
            }
        }
        ]

        self.added_functions = {}

    async def play_full_game(self, comp_budget: int):
        """Play a full game with one player using comp_budget=1 and the other using the specified comp_budget."""
        while not self.env.is_game_over():
            # Get current player
            current_player = self.env.get_turn()

            print("turn", current_player)
            
            # Set compute budget based on player
            if current_player == "white":
                current_budget = comp_budget
            else:
                current_budget = 2
                
            # Make move for current player
            await self.play(current_budget)
            
        # Return game result
        return self.env.get_result()


    async def play(self, comp_budget: int):
        # todo: for now, naive understanding of compute
        self.comp_budget = comp_budget
        # Create a deep copy of the environment by serializing and deserializing its state
        c_state = type(self.env)()  # Create new instance of same class
        c_state.__dict__.update(self.env.__dict__.copy())  # Copy all attributes
        messages = [{
            "role": "user", 
            "content": f"You are playing a game of chess as {c_state.get_turn()}. Current board state:\n{str(c_state)}\n\nAs {c_state.get_turn()}, you need to analyze the position by making some exploratory moves using take_simulate_action and back_track before committing to your final move. You have {comp_budget} compute budget to explore variations. Use the provided functions to analyze and find the best move."
        }]

        i = 0
        while i < self.comp_budget:

            response = await self.model.call(messages, GAME_PROMPT, tools=self.tools)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                arguments = json.loads(tool_call.function.arguments)
                
                if tool_call.function.name == "create_new_tool":
                    self.added_functions[arguments["tool_name"]] = arguments["function_code"]
                    # todo 
                    self.tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": arguments["tool_name"],
                                "description": f"", #todo put desc
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        self.action_signature
                                    }
                                },
                                "strict": True
                            },
                        },
                    )
                
                elif tool_call.function.name in self.added_functions:
                    result = eval(self.added_functions[tool_call.function.name])(*arguments)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(result)
                    })
                
                elif tool_call.function.name == "take_simulated_action":
                    print(arguments['action'])
                    result = c_state.take_action(arguments["action"])

                    if not result:
                        print("Invalid move")
                        continue

                    messages.append({
                        "role": "user",
                        "content": f"Took action {arguments} on game state. New env:\n{str(c_state)}. You are now playing as {c_state.get_turn()}"
                    })

                    #result = self.play_from_state(, new_messages)
                elif tool_call.function.name == "backtrack":
                    # Backtrack to previous state
                    print("Backtracking...")
                    c_state.backtrack(arguments["n_moves"])
                    messages = messages + [{
                        "role": "user", 
                        "content": f"Backtracked to previous state. Current env:\n{str(c_state)}."
                    }]
                
                elif tool_call.function.name == "take_action":
                    result = self.env.take_action(arguments["action"])

                    if not result:
                        print("invalid move")
                        continue
                    print("Final action", arguments["action"])
                    return arguments
            else:
                raise "Did not use tools"

            messages.append({
                "role": "user",
                "content": f"You have {self.comp_budget - i} actions left."
            })
            if i == self.comp_budget - 2:
                print("ad")
                messages.append({
                    "role": "user",
                    "content": f"Now call take_action with the final action based on what you've learned. Take this action on the original board state: {str(self.env)}. Remember that this is not the last state of the simulated game that was to learn. now take an action on the original starting game just given. You are {c_state.get_turn()}"
                })
            i += 1

    

async def main():
    from chess_engine import core
    import chess
    board = core.ChessEngine()
    #board.board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    model_interface = ModelInterface(model_name="o3-mini")
    action_sig = {
        "action": {
            "type": "string",
            "description": "UCI formatted action on board. No Nf3e5 for eg, jut f3e5"
        }
    }
    game = Game(board, model_interface, action_sig)

    #await game.play(20)
    await game.play_full_game(10)

if __name__ == "__main__":
    asyncio.run(main())

# f7e8
# 4o f3d5