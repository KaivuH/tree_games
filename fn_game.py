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
        self, model_name: str, api_key: Optional[str] = None, max_tokens: int = 1024
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
        messages: List[Dict[str, Any]],
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        output_type: Optional[Type[BaseModel]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Make an API call to OpenRouter.

        Args:
            messages: List of message objects
            system_prompt: System prompt message
            temperature: Sampling temperature
            tools: List of tool definitions
            output_type: Currently not supported for OpenRouter
            max_retries: Number of retries on failure

        Returns:
            API response
        """
        # Prepare messages with system prompt
        formatted_messages = self._format_messages(system_prompt, messages)

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "tree_games_chess_demo",  # App identifier
            "X-Title": "Chess Tree Demo",  # App title
        }

        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            payload["tools"] = tools

        # Make API request with retry logic
        last_exception = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=120.0,  # Extended timeout for LLM responses
                    )

                    if response.status_code != 200:
                        logging.error(
                            f"OpenRouter API error: {response.status_code} - {response.text}"
                        )
                        raise Exception(
                            f"OpenRouter API error: {response.status_code} - {response.text}"
                        )

                    result = response.json()
                    
                    # Format the response to match the expected format in the Game class
                    if tools and "choices" in result and len(result["choices"]) > 0:
                        if "tool_calls" in result["choices"][0]["message"]:
                            return result["choices"][0]["message"]
                        else:
                            return result["choices"][0]["message"]["content"]
                    else:
                        return result["choices"][0]["message"]["content"]
                
            except Exception as e:
                logging.error(f"Error calling OpenRouter API: {str(e)}")
                last_exception = e
                await asyncio.sleep(1.0 * (attempt + 1))

        raise last_exception


    # def _format_messages(
    #     self, system_prompt: str, user_messages: List[dict]
    # ) -> List[dict]:
    #     """
    #     Utility to prepend a system message.
    #     user_messages is typically: [{'role': 'user', 'content': ...}, ...]
    #     """
    #     return [{"role": "system", "content": system_prompt}] + user_messages

    # async def call(
    #     self,
    #     messages: List[dict],
    #     system_prompt: str = "You are a helpful assistant.",
    #     output_type: Optional[Type[BaseModel]] = None,
    #     max_retries: int = 3,
    #     tools: Optional[Dict[str, Any]] = None,
    # ) -> Union[str, BaseModel]:
    #     """
    #     Calls the LLM using OpenAI's chat completion or HF, returning either raw text
    #     or a Pydantic-validated object.
    #     """
    #     if not self.uses_openai:
    #         # For HF models, we simply flatten messages into a single string prompt:
    #         formatted_prompt = f"{system_prompt}\n"
    #         for m in messages:
    #             formatted_prompt += f"{m['role'].upper()}:\n{m['content']}\n\n"
    #         logging.info("\n=== Model Call (HuggingFace) ===")
    #         logging.info(formatted_prompt)
    #         logging.info("=" * 80)
    #         return self.call_hf(formatted_prompt, max_length=self.max_tokens)

    #     # Otherwise, if using an OpenAI-based model:
    #     formatted_messages = self._format_messages(system_prompt, messages)
    #     # logging.info("\n=== Model Call (OpenAI) ===")
    #     # logging.info("System:", system_prompt)
    #     # for msg in messages:
    #     #     logging.info(f"{msg['role'].upper()}:", msg['content'])
    #     # logging.info("=" * 80)

    #     last_exception = None
    #     for attempt in range(max_retries):
    #         try:
    #             if output_type is None:
    #                 response = await self.client.chat.completions.create(
    #                     messages=formatted_messages,
    #                     model=self.model_name,
    #                     max_completion_tokens=self.max_tokens,
    #                     tools=tools
    #                 )
    #                 print(response)
    #                 if tools:
    #                     result = response.choices[0].message
    #                 else:
    #                     result = response.choices[0].message.content
    #                 # logging.info("\nRESPONSE:", result)
    #                 # logging.info("=" * 80)
    #                 return result
    #             else:
    #                 response = await self.client.beta.chat.completions.parse(
    #                     messages=formatted_messages,
    #                     model=self.model_name,
    #                     max_completion_tokens=self.max_tokens,
    #                     response_format=output_type,
    #                 )
    #                 parsed_obj = response.choices[0].message.parsed
    #                 result = output_type.model_validate(parsed_obj)
    #                 # logging.info("\nRESPONSE:", result)
    #                 # logging.info("=" * 80)
    #                 return result

    #         except Exception as e:
    #             logging.info(
    #                 f"[DEBUG] Attempt {attempt+1}/{max_retries} failed with exception: {e}"
    #             )
    #             logging.info("[DEBUG] System prompt (verbatim):")
    #             logging.info(system_prompt)
    #             logging.info("[DEBUG] User messages (verbatim):")
    #             for idx, msg in enumerate(messages):
    #                 logging.info(f"  Message {idx+1} - role='{msg['role']}':")
    #                 logging.info(msg["content"])
    #             last_exception = e
    #             await asyncio.sleep(1.0 * (attempt + 1))

    #     raise last_exception


GAME_PROMPT = ""

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
                    "properties": action_signature
                },
                "strict": True
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
                        "properties": {
                            "tool_name": {
                                "type": "str",
                                "description": "name of the tool you will use"
                            },
                            "function_code": {
                                "type": "str",
                                "description": "Code of a python function that computes some property of the environment without modifying the environment. takes in a GameState as argument."
                            }
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
                    "properties": {}
                },
                "strict": True
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
                "strict": True
            }
        }
        ]

        self.added_functions = {}


    def play(self, comp_budget: int):
        # todo: for now, naive understanding of compute
        self.compute_budget = comp_budget

        c_state = self.env.copy()
        for i in range(self.comp_budget):

            response = self.model.call(messages, GAME_PROMPT, tools=self.tools)

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
                    result = eval(self.added_functions[tool_call.function.name])(arguments)
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
                    c_state.take_action(arguments)
                    messages.append({
                        "role": "user",
                        "content": f"Took action {arguments} on game state. New env:\n{str(c_state)}."
                    })

                    #result = self.play_from_state(, new_messages)
                elif tool_call.function.name == "backtrack":
                    # Backtrack to previous state
                    c_state.backtrack()
                    messages = messages + [{
                        "role": "user", 
                        "content": f"Backtracked to previous state. Current env:\n{str(c_state)}."
                    }]
                
                elif tool_call.function.name == "take_action":
                    self.env.take_action(arguments)
                    return arguments
            else:
                raise "Did not use tools"

            messages.append({
                "role": "user",
                "content": f"You have {self.comp_budget - i} actions left."
            })

if __name__ == "__main__":
    from chess_engine import core

    board = core.ChessEngine()
    model_interface = ModelInterface(model_name="o3-mini")
    action_sig = {
        "uci_action": {
            "type": "str",
            "description": "UCI formatted action on board"
        }
    }
    game = Game(board, model_interface, action_sig)
    game.play(1)
    

