# Recursive LLM Chess Tree Search

This project implements a novel approach to chess reasoning using recursive LLM calls in a tree search framework. Unlike traditional approaches where a single LLM makes a move decision, this system:

1. Spawns multiple LLM instances to explore different branches of the game tree
2. Allows each branch to recursively call new LLM instances for deeper exploration
3. Aggregates insights from parallel explorations to inform the final move decision
4. Creates a multi-agent simulation environment where different instances of the model play against each other

## Features

- **Multi-Agent Tree Search**: Distributes reasoning across multiple LLM instances
- **MCTS-Inspired Selection**: Balances exploration and exploitation in move selection
- **Position Evaluation**: Comprehensive evaluation including material, development, and tactics
- **Branch Pruning**: Efficiently allocates compute to promising variations
- **CLI Interface**: Interactive command-line interface for playing and analysis
- **Visualization Tools**: Tools for visualizing the search tree and evaluations
- **Jupyter Integration**: Compatible with Jupyter notebooks for interactive analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tree_games.git
cd tree_games

# Install the package
pip install -e .

# For visualization capabilities
pip install -e ".[visualization]"

# For OpenAI API integration
pip install -e ".[api]"

# For Jupyter notebook support
pip install -e ".[jupyter]"
```

## Usage

### Command-Line Interface

```bash
# Start with default settings
python chess_cli.py

# Use mock LLM API (no API key needed)
python chess_cli.py --mock

# Start from a specific position
python chess_cli.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

### Basic CLI Commands

- `move <index or uci>` - Make a move by index or UCI notation
- `analyze [depth]` - Analyze current position with optional depth
- `explore <move> [n]` - Explore a branch starting with move, with n variations
- `undo` - Undo the last move
- `display` - Display the current board
- `eval` - Show the current position evaluation
- `best` - Show the best move
- `new` - Start a new game
- `quit` - Exit the application
- `help` - Show the help message

### API Usage

```python
from tree_games.tree_search.controller import TreeSearchController
from tree_games.llm_interface.agent import LLMAgent

# Create a simple controller with built-in evaluation
controller = TreeSearchController()

# Or create a controller with LLM evaluation
def llm_caller(branch_context):
    # Your LLM API call here
    pass

controller = TreeSearchController(llm_caller=llm_caller)

# Explore a branch
controller.explore_branch("e4")

# Get the best move
best_move = controller.get_best_move()

# Make a move
controller.commit_move(best_move)
```

## System Architecture

The system is organized into several components:

1. **Chess Engine**: Core chess logic using python-chess
2. **Tree Search**: MCTS-inspired tree search algorithm
3. **LLM Interface**: Communication with language models
4. **CLI**: Command-line interface for interaction
5. **Visualization**: Tools for visualizing the search process

## Project Structure

```
tree_games/
├── chess_engine/       # Chess board and rules
│   ├── core.py         # Core chess functionality
│   └── analysis.py     # Position analysis
├── tree_search/        # Tree search algorithm
│   ├── node.py         # Tree node implementation
│   └── controller.py   # Tree search controller
├── llm_interface/      # LLM interaction
│   ├── agent.py        # LLM agent implementation
│   └── orchestrator.py # Multi-agent orchestration
├── utils/              # Utility functions
│   └── visualization.py # Visualization tools
└── chess_cli.py        # Command-line interface
```

## Requirements

- Python 3.8+
- python-chess
- pandas
- matplotlib
- networkx
- openai (optional)
- graphviz (optional)
- jupyter (optional)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project was inspired by research on multi-agent systems, tree search algorithms, and language model capabilities.