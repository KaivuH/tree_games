import requests
import json
import os
from copy import deepcopy
from dotenv import load_dotenv
import asyncio

load_dotenv()  # This loads the .env file into the environment
ant_key = os.getenv("ANT_KEY")

import anthropic
import openai

client = anthropic.AsyncAnthropic()
# client = openai.AsyncOpenAI()

from typing import Optional

# async def call_openai(system, messages, tools=[], )

async def call_claude(system, messages, tools=[], think_budget:Optional[int]=None, max_tokens=1000, temp=1):
    """Thinking requires a minimum budget of 1,024 tokens and counts towards your max_tokens limit."""
    if think_budget is None:
        message = await client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            tools=tools,
            temperature=temp,
            system=system,
            messages=messages
        )
    else:
        message = await client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=max_tokens,
            thinking={
                "type": "enabled",
                "budget_tokens": think_budget,
            },
            tools=tools,
            temperature=temp,
            system=system,
            messages=messages
        )
    return message.content

import random
import re
import json

def extract_json_blocks(text):
    json_blocks = []
    
    # Pattern to match JSON blocks enclosed in triple backticks with "json"
    pattern_code = r"```json\s*([\s\S]*?)\s*```"
    code_matches = re.findall(pattern_code, text)
    
    for block in code_matches:
        try:
            parsed = json.loads(block)
            json_blocks.append(parsed)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON block from code block: {e}")
    
    # Remove the triple-backtick JSON sections from the text so we don't double-match.
    text_without_code = re.sub(r"```json\s*[\s\S]*?\s*```", "", text)
    
    # Pattern to match plain JSON objects (assumes no nested braces)
    pattern_plain = r"(\{[^{}]*\})"
    plain_matches = re.findall(pattern_plain, text_without_code)
    
    for block in plain_matches:
        try:
            parsed = json.loads(block)
            json_blocks.append(parsed)
        except json.JSONDecodeError as e:
            print(f"Error parsing plain JSON block: {e}")
    
    return json_blocks

def extract_python_blocks(text):
    # Find all text between triple backticks with python
    python_blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', text)
    
    # If none found, try without language specifier
    if not python_blocks:
        python_blocks = re.findall(r'```\s*([\s\S]*?)\s*```', text)
    
    return python_blocks

def execute_function_from_string(function_string, input_data):
    """
    Safely execute a function defined in a string with the given input data.
    
    Args:
        function_string (str): String containing Python function definition
        input_data: The input to pass to the function
        
    Returns:
        tuple: (success, result_or_error)
            - success: Boolean indicating if execution was successful
            - result_or_error: Either the function result or an error message
    """
    # Create an isolated namespace to avoid polluting the global namespace
    local_namespace = {}
    
    try:
        # First try to compile the code to catch syntax errors early
        compiled_code = compile(function_string, '<string>', 'exec')
        
        # Execute the compiled code in the isolated namespace
        exec(compiled_code, {}, local_namespace)
        
        # Find a callable function in the local namespace
        function = None
        for name, obj in local_namespace.items():
            if callable(obj) and not name.startswith('__'):
                function = obj
                break
        
        if not function:
            return False, "No callable function found in the provided code"
        
        # Call the function with the input data
        try:
            result = function(input_data)
            return True, result
        except Exception as e:
            return False, f"Error during function execution: {str(e)}"
        
    except SyntaxError as e:
        return False, f"Syntax error in function code: {str(e)}"
    except Exception as e:
        return False, f"Error during code compilation or execution: {str(e)}"


from anthropic.types import (
    ToolUseBlock,
    TextBlock,
    ThinkingBlock,
)

propose_new_tool = {
    "name": "propose_new_tool",
    "description": "Propose the function specs for <=3 new python functions you can call anytime as new tool calls later on. Brainstorm some functions that do specific algorithmic computations about the board that helps your decision, and write detailed specs of those.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_name_1": {
                "type": "string",
                "description": "Name of the tool you will use"
            },
            "tool_name_2": {
                "type": "string",
                "description": "Name of the tool you will use"
            },
            "tool_name_3": {
                "type": "string",
                "description": "Name of the tool you will use"
            },
            "function_spec_1": {
                "type": "string",
                "description": "A detailed description of the function you want to propose, as well as the signature of its outputs. The function should only receive an input which is a 2D array representing the current visible board state, where in the array X represents a hit, O represents a miss, and ~ represents unknown cells."  
            },
            "function_spec_2": {
                "type": "string",
                "description": "A detailed description of the function you want to propose, as well as the signature of its outputs. The function should only receive an input which is a 2D array representing the current visible board state, where in the array X represents a hit, O represents a miss, and ~ represents unknown cells."  
            },
            "function_spec_3": {
                "type": "string",
                "description": "A detailed description of the function you want to propose, as well as the signature of its outputs. The function should only receive an input which is a 2D array representing the current visible board state, where in the array X represents a hit, O represents a miss, and ~ represents unknown cells."  
            }
        },
        "required": ["tool_name_1", "function_spec_1"]
    }
}

class BattleshipGame:
    def __init__(self, board_size=8):
        self.board_size = board_size
        # Define ship names and their lengths.
        self.ship_info = {
            "Carrier": 5,
            "Battleship": 4,
            "Cruiser": 3,
            # "Submarine": 3,
            # "Destroyer": 2
        }
        # Initialize two players with their boards and ship placements.
        self.players = [self._initialize_player_board(), self._initialize_player_board()]
        self.current_player = 1  # Index of the player whose turn it is
        self.tools = [[propose_new_tool], []] # tools available for each player
        self.extra_fns = [{}, {}] # code for extra functions added by each player

    def _initialize_player_board(self):
        """Creates an empty board and randomly places ships on it."""
        # Create an empty board (using '~' for water)
        board = [['~' for _ in range(self.board_size)] for _ in range(self.board_size)]
        ships = {}
        # Place each ship randomly
        for ship, size in self.ship_info.items():
            placed = False
            while not placed:
                orientation = random.choice(['H', 'V'])
                if orientation == 'H':
                    row = random.randint(0, self.board_size - 1)
                    col = random.randint(0, self.board_size - size)
                    # Check if space is free horizontally
                    if all(board[row][col + i] == '~' for i in range(size)):
                        for i in range(size):
                            board[row][col + i] = ship[0]  # Mark with first letter of the ship
                        ships[ship] = {'coords': [(row, col + i) for i in range(size)], 'hits': set()}
                        placed = True
                else:  # Vertical orientation
                    row = random.randint(0, self.board_size - size)
                    col = random.randint(0, self.board_size - 1)
                    # Check if space is free vertically
                    if all(board[row + i][col] == '~' for i in range(size)):
                        for i in range(size):
                            board[row + i][col] = ship[0]
                        ships[ship] = {'coords': [(row + i, col) for i in range(size)], 'hits': set()}
                        placed = True
        return {'board': board, 'ships': ships, 'guesses': set()}

    def print_board(self, player_index, print_out=True, reveal_ships=False):
        """
        Prints the board for the given player.
        If reveal_ships is False, all ship markers are hidden (displayed as water).
        """
        board = self.players[player_index]['board']
        display_board = []
        for row in board:
            display_row = []
            for cell in row:
                # Hide ship placements if not revealing
                if cell != '~' and not reveal_ships and cell not in ['X', 'O']:
                    display_row.append('~')
                else:
                    display_row.append(cell)
            display_board.append(display_row)
        # Print header with column numbers
        board_str = "  " + " ".join(str(i) for i in range(self.board_size)) + "\n"
        for idx, row in enumerate(display_board):
            board_str += f"{idx} " + " ".join(row) + "\n"
        
        if print_out:
            print(board_str)
        return display_board, board_str
    
    def make_prompt(self, player_index):
        """Returns a system prompt and a conversation dictionary"""
        _, board_str = self.print_board(1-player_index, print_out=False, reveal_ships=False)
        system = f"You are playing the game of Battleship against an opponent. Your goal is to sink all of your opponent's ships in the fewest moves possible. Your opponent has {len(self.ship_info)} ships of varying lengths."
        prompt = "Here is your observation of your opponent's board. X represents a hit, O represents a miss, and ~ represents unknown cells.\n"
        prompt += board_str
        prompt += """Use the tools at your disposal to make the best move possible. Only output a JSON block with keys: "row" (integer) and "col" (integer), representing the row and column of the cell you want to fire at."""
        message = [{"role": "user", "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]}]
        return system, message
    

    def take_turn(self, attacker_index, target_index, row, col):
        """
        The attacker fires at (row, col) on the target's board.
        Returns True if it was a hit, False if a miss.
        """
        print(f"Player {attacker_index + 1} fires at position ({row}, {col}):")
        target = self.players[target_index]
        # Prevent firing at the same cell twice.
        if (row, col) in target['guesses']:
            print("You have already fired at that position. Try again.")
            return None  # Indicate invalid move
        target['guesses'].add((row, col))
        cell = target['board'][row][col]
        if cell != '~' and cell not in ['X', 'O']:
            # It’s a hit.
            print("Hit!")
            # Determine which ship was hit.
            for ship, data in target['ships'].items():
                if (row, col) in data['coords']:
                    data['hits'].add((row, col))
                    if len(data['hits']) == len(data['coords']):
                        print(f"You sunk the {ship}!")
                    break
            # Mark the hit on the board.
            target['board'][row][col] = 'X'
            return (True, "That was a hit!")
        else:
            print("Miss!")
            target['board'][row][col] = 'O'
            return (False, "That was a miss!")

    def check_winner(self):
        """
        Checks both players' boards to determine if one has lost all ships.
        Returns the index of the defeated player if found, or None otherwise.
        """
        for idx, player in enumerate(self.players):
            all_sunk = True
            for ship, data in player['ships'].items():
                if len(data['hits']) != len(data['coords']):
                    all_sunk = False
                    break
            if all_sunk:
                return idx  # This player's ships are all sunk.
        return None

    async def play(self):
        """Main loop to run the game until one player wins."""
        print("Starting Battleship Game!\n")
        messages = [[], []]
        while True:    
            attacker = self.current_player
            defender = 1 - self.current_player

            if len(messages[attacker]) >= 10:
                messages[attacker] = messages[attacker][-10:]
            print(f"Player {attacker + 1}'s turn to attack Player {defender + 1}'s board.")
            # Show the attacker's view of the defender's board (ships hidden)
            self.print_board(defender, reveal_ships=False)
            
            # Get valid input for row and column.
            while True:
                try:
                    system, new_message = self.make_prompt(attacker)
                    if messages[attacker]: 
                        if new_message != messages[attacker][-1]:
                            messages[attacker] += new_message
                    else:
                        messages[attacker] += new_message
                    print("length of messages:", len(messages[attacker]))
                    print("full message history:", messages[attacker])
                    print("number of tools available:", len(self.tools[attacker]))

                    if attacker == 0:
                        response = await call_claude(system, messages[attacker], tools=self.tools[attacker], think_budget=1024, max_tokens=2000)
                    else:
                        response = await call_claude(system, messages[attacker], tools=self.tools[attacker], think_budget=5000, max_tokens=6000)
                    print(response)

                    if isinstance(response[-1], ToolUseBlock):
                        tool_call = response[-1]
                        # print(tool_call)
                        arguments = tool_call.input
                        if tool_call.name == "propose_new_tool":
                            for i in range(1, 4):
                                if f"tool_name_{i}" in arguments:
                                    tool_name = arguments[f"tool_name_{i}"]
                                    if tool_name not in self.extra_fns[attacker]:
                                        function_spec = arguments[f"function_spec_{i}"]
                                        code = await call_claude(
                                            system="You are an expert programmer who is very smart and careful.",
                                            messages=[{"role": "user", "content": [{"type": "text", "text": function_spec + "Put the code for the function in a python block."}]}], think_budget=1024, max_tokens=5000
                                        )
                                        code_formatted = extract_python_blocks(code[1].text)[0]
                                        print(code_formatted)

                                        self.tools[attacker].append(
                                            {
                                                "name": tool_name,
                                                "description": function_spec,
                                                "input_schema": {
                                                    "type": "object",
                                                    "properties": {
                                                    }
                                                },
                                            }
                                        )
                                        self.extra_fns[attacker][tool_name] = code_formatted

                                    continue

                        else:
                            name = tool_call.name

                            fn = self.extra_fns[attacker][name]
                            visible_board, _ = self.print_board(1-attacker, print_out=False, reveal_ships=False)

                            is_successful, result = execute_function_from_string(fn, visible_board)
                            if is_successful:
                                messages[attacker].append({
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": name}],
                                })
                                messages[attacker].append({
                                    "role": "user", 
                                    "content": [{"type": "text", "text": str(result)}]
                                })
                                continue
                            else:
                                self.extra_fns[attacker].pop(name)
                                self.tools[attacker] = [tool for tool in self.tools[attacker] if tool["name"] != name]
                                continue
                    
                    elif isinstance(response[-1], TextBlock):
                        try:
                            response_formatted = extract_json_blocks(response[-1].text)[0]
                            print(response_formatted)
                            row = int(response_formatted["row"])
                            col = int(response_formatted["col"])
                            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                                break
                            else:
                                print("Coordinates out of range. Please try again.")
                        except:
                            print("Error parsing JSON response, try again.")
                except ValueError:
                    print("Invalid input. Please enter integer values.")
            
            # Process the move; if the cell was already guessed, ask for another input.
            result = self.take_turn(attacker, defender, row, col)
            if result is None:
                messages[attacker].append({"role": "user", "content": [{"type": "text", "text": "You have already fired at that position. Try again."}]})
                continue  # Invalid move; let the same player try again.
            else:
                messages[attacker].append({"role": "user", "content": [{"type": "text", "text": result[1]}]})

            # Check if the defender has lost all ships.
            loser = self.check_winner()
            if loser is not None:
                winner = 1 - loser
                print(f"\nPlayer {winner + 1} wins! All of Player {loser + 1}'s ships have been sunk.")
                print("Final boards:")
                print("Player 1's board:")
                self.print_board(0, reveal_ships=True)
                print("Player 2's board:")
                self.print_board(1, reveal_ships=True)
                return winner
            
            # Switch turns.
            self.current_player = defender


async def main(n):
    tasks = []
    for _ in range(n):
        game = BattleshipGame()
        tasks.append(game.play())
    results = await asyncio.gather(*tasks)
    print(results)

if __name__ == "__main__":
    asyncio.run(main(7))