# CLAUDE.md - Guidelines for tree_games

## Build, Test & Lint Commands
- Install package: `pip install -e .`
- Install with extras: `pip install -e ".[api,visualization,jupyter]"`
- Run CLI: `python chess_cli.py --mock`
- Set OpenAI API Key: `export OPENAI_API_KEY="your_key_here"`
- Run CLI with API: `python chess_cli.py`

## Code Style Guidelines
- **Formatting**: 4-space indentation, 88 character line limit
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Imports**: Group imports (standard lib, external libs, internal modules)
- **Types**: Use typing module for type hints
- **Documentation**: Google-style docstrings
- **Error Handling**: Use specific exceptions, document raised exceptions
- **Tests**: Write unit tests for each function, use pytest
- **Visualization**: Support both CLI text output and Jupyter visualization