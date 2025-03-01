#!/usr/bin/env python3
import asyncio
import argparse
import berserk
import chess
import chess.pgn
import logging
import os
import time
import threading
import json
from typing import Dict, Any, Optional

# Import tree search v2 and chess engine
from fn_game_tree_v2 import ModelInterface, Game
from chess_engine.core import ChessEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="lichess_bot.log",
    filemode="w"
)
logger = logging.getLogger(__name__)

class LichessGame:
    """Handles a single game on Lichess"""
    
    def __init__(self, client, game_id, color, **kwargs):
        self.client = client
        self.game_id = game_id
        self.is_white = color == "white"
        self.my_color = color
        self.board = chess.Board()
        self.opponent_name = kwargs.get('opponent', {}).get('username', 'Opponent')
        self.engine = ChessEngine()
        self.is_game_over = False
        self.move_overhead = 1.0  # Extra time in seconds to account for network delays
        
        # Configure tree search parameters
        self.comp_budget = kwargs.get('comp_budget', 3)
        self.model_name = kwargs.get('model_name', 'gpt-4o')
        self.api_key = kwargs.get('api_key', os.environ.get("OPENAI_API_KEY"))
        
        logger.info(f"Game {game_id} started against {self.opponent_name}")
        logger.info(f"Playing as {color} with computational budget: {self.comp_budget}")
        
        # Initialize the model interface for tree search
        self.model_interface = ModelInterface(model_name=self.model_name, api_key=self.api_key)
        
        # Action signature for the game
        self.action_sig = {
            "action": {
                "type": "string",
                "description": "UCI formatted move (e.g., 'f3e5')"
            }
        }
        
        # Initialize the tree search game
        self.game = Game(self.engine, self.model_interface, self.action_sig)

    def handle_state_change(self, game_state):
        """Process game state updates from Lichess"""
        if game_state.get('status') == 'started':
            moves = game_state.get('moves', '')
            self.update_board(moves)
            
            if self.should_make_move():
                # Start move calculation in a background thread
                threading.Thread(target=self.make_move_background).start()
    
    def update_board(self, moves_string: str):
        """Update the local board state with moves from Lichess"""
        self.board = chess.Board()
        self.engine.board = chess.Board()
        
        if not moves_string:
            return
        
        # Apply all moves to both boards
        moves = moves_string.split()
        for move in moves:
            try:
                self.board.push_uci(move)
                self.engine.take_action(move)
            except ValueError:
                logger.error(f"Invalid move: {move}")
                break
                
        logger.info(f"Current board state:\n{self.board}")
        
    def should_make_move(self) -> bool:
        """Determine if it's our turn to move"""
        is_white_turn = self.board.turn
        should_move = (is_white_turn and self.is_white) or (not is_white_turn and not self.is_white)
        return should_move and not self.board.is_game_over()
    
    def get_clock_info(self):
        """Get remaining time and increment from game state"""
        try:
            game_info = self.client.games.get_ongoing()[0]  # Assuming this is the current game
            my_clock = game_info['clock']['white'] if self.is_white else game_info['clock']['black']
            return my_clock.get('remaining', 60), my_clock.get('increment', 0)
        except Exception as e:
            logger.error(f"Error getting clock info: {e}")
            return 60, 0  # Default to 60 seconds remaining, 0 increment
    
    def make_move_background(self):
        """Calculate and make a move using tree search in a background thread"""
        try:
            # Get time control information
            remaining_time, increment = self.get_clock_info()
            logger.info(f"Time remaining: {remaining_time}s, Increment: {increment}s")
            
            # Adjust comp_budget based on time remaining
            if remaining_time < 10:
                adjusted_budget = 1  # Very limited time, minimal search
            elif remaining_time < 30:
                adjusted_budget = 2  # Low time, reduced search
            else:
                adjusted_budget = self.comp_budget  # Normal search
                
            logger.info(f"Using computational budget: {adjusted_budget}")
            
            # Calculate the best move using tree search
            event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(event_loop)
            
            best_move, score = event_loop.run_until_complete(
                self.game.play_with_state(self.engine.copy(), [], adjusted_budget, start=True)
            )
            
            # Get the move string
            move_uci = best_move.action if best_move and hasattr(best_move, 'action') else None
            
            if move_uci and self.is_valid_move(move_uci):
                logger.info(f"Making move: {move_uci} with evaluation: {score}")
                self.client.board.make_move(self.game_id, move_uci)
                
                # Update our local board state
                self.board.push_uci(move_uci)
                self.engine.take_action(move_uci)
            else:
                # Fallback: If tree search fails, make a random legal move
                logger.warning(f"Tree search returned invalid move: {move_uci}. Using fallback strategy.")
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    fallback_move = legal_moves[0].uci()  # Just take the first legal move
                    logger.info(f"Making fallback move: {fallback_move}")
                    self.client.board.make_move(self.game_id, fallback_move)
                    
                    # Update our local board state
                    self.board.push_uci(fallback_move)
                    self.engine.take_action(fallback_move)
                else:
                    logger.error("No legal moves available")
            
            event_loop.close()
            
        except Exception as e:
            logger.error(f"Error making move: {e}")
            # Try to make a random legal move as fallback
            try:
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    fallback_move = legal_moves[0].uci()
                    logger.info(f"Making emergency fallback move: {fallback_move}")
                    self.client.board.make_move(self.game_id, fallback_move)
            except Exception as fallback_error:
                logger.error(f"Fallback move also failed: {fallback_error}")
    
    def is_valid_move(self, move_uci: str) -> bool:
        """Check if a move is valid"""
        try:
            move = chess.Move.from_uci(move_uci)
            return move in self.board.legal_moves
        except ValueError:
            return False


class LichessBot:
    """Main Lichess bot class that handles multiple games"""
    
    def __init__(self, token: str, **kwargs):
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        self.user = self.client.account.get()
        self.active_games = {}
        
        # Bot configuration
        self.config = {
            'comp_budget': kwargs.get('comp_budget', 3),
            'model_name': kwargs.get('model_name', 'gpt-4o'),
            'api_key': kwargs.get('api_key', os.environ.get("OPENAI_API_KEY")),
            'challenge': {
                'variants': kwargs.get('variants', ['standard']),
                'time_controls': kwargs.get('time_controls', ['blitz', 'rapid']),
                'modes': kwargs.get('modes', ['casual', 'rated']),
                'min_rating': kwargs.get('min_rating', 0),
                'max_rating': kwargs.get('max_rating', 3000),
            }
        }
        
        logger.info(f"Bot initialized. Username: {self.user['username']}")
        logger.info(f"Computational budget: {self.config['comp_budget']}")
        logger.info(f"Model: {self.config['model_name']}")
        
    def start(self):
        """Start the bot and listen for events"""
        logger.info("Starting Lichess bot...")
        
        try:
            # Handle already ongoing games
            self.handle_ongoing_games()
            
            # Listen for new events
            for event in self.client.board.stream_incoming_events():
                if event['type'] == 'challenge':
                    self.handle_challenge(event['challenge'])
                elif event['type'] == 'gameStart':
                    game_id = event['game']['id']
                    self.handle_game_start(game_id)
                    
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Lichess API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            
    def handle_ongoing_games(self):
        """Handle games that were already in progress"""
        ongoing_games = self.client.games.get_ongoing()
        for game in ongoing_games:
            game_id = game['gameId']
            logger.info(f"Joining ongoing game: {game_id}")
            self.handle_game_start(game_id)
    
    def handle_challenge(self, challenge: Dict[str, Any]):
        """Accept or decline a challenge based on configured criteria"""
        challenger = challenge['challenger']['name']
        variant = challenge['variant']['key']
        time_control = challenge.get('speed', 'unknown')
        rated = challenge['rated']
        mode = 'rated' if rated else 'casual'
        
        logger.info(f"Received challenge from {challenger} ({variant}, {time_control}, {mode})")
        
        # Check if the challenge meets our criteria
        if (
            variant in self.config['challenge']['variants'] and
            time_control in self.config['challenge']['time_controls'] and
            mode in self.config['challenge']['modes'] and
            self.config['challenge']['min_rating'] <= challenge['challenger'].get('rating', 0) <= self.config['challenge']['max_rating']
        ):
            logger.info(f"Accepting challenge from {challenger}")
            self.client.challenges.accept(challenge['id'])
        else:
            logger.info(f"Declining challenge from {challenger} (doesn't meet criteria)")
            self.client.challenges.decline(challenge['id'])
    
    def handle_game_start(self, game_id: str):
        """Initialize and start handling a game"""
        try:
            # Get game info
            game_info = self.client.games.export(game_id)
            color = 'white' if game_info['players']['white']['user']['id'] == self.user['id'] else 'black'
            opponent = game_info['players']['black' if color == 'white' else 'white']['user']['username']
            
            # Create new game handler
            game = LichessGame(
                self.client, 
                game_id, 
                color, 
                opponent={'username': opponent},
                comp_budget=self.config['comp_budget'],
                model_name=self.config['model_name'],
                api_key=self.config['api_key']
            )
            
            # Store in active games
            self.active_games[game_id] = game
            
            # Start listening for game state updates
            game_stream = self.client.board.stream_game_state(game_id)
            for state in game_stream:
                if state.get('status') == 'aborted':
                    logger.info(f"Game {game_id} aborted")
                    break
                elif state.get('status') == 'resign':
                    winner = state.get('winner')
                    logger.info(f"Game {game_id} resigned. Winner: {winner}")
                    break
                elif state.get('status') in ['mate', 'stalemate', 'draw']:
                    result = state.get('winner', 'draw')
                    logger.info(f"Game {game_id} over. Result: {result}")
                    break
                else:
                    game.handle_state_change(state)
            
            # Clean up when game is finished
            if game_id in self.active_games:
                del self.active_games[game_id]
                
        except Exception as e:
            logger.error(f"Error handling game {game_id}: {e}")
            if game_id in self.active_games:
                del self.active_games[game_id]


def main():
    """Parse command line arguments and start the bot"""
    parser = argparse.ArgumentParser(description='Lichess bot using tree search v2')
    parser.add_argument('--token', type=str, help='Lichess API token')
    parser.add_argument('--budget', type=int, default=3, help='Computational budget for tree search')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model name for LLM calls')
    parser.add_argument('--variants', type=str, default='standard', help='Comma-separated list of chess variants to accept')
    parser.add_argument('--time-controls', type=str, default='blitz,rapid', help='Comma-separated list of time controls to accept')
    parser.add_argument('--modes', type=str, default='casual,rated', help='Game modes to accept (casual, rated)')
    parser.add_argument('--min-rating', type=int, default=0, help='Minimum opponent rating')
    parser.add_argument('--max-rating', type=int, default=3000, help='Maximum opponent rating')
    
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get('LICHESS_TOKEN')
    if not token:
        parser.error("Lichess API token required. Set LICHESS_TOKEN environment variable or use --token")
    
    # Start the bot
    bot = LichessBot(
        token=token,
        comp_budget=args.budget,
        model_name=args.model,
        variants=args.variants.split(','),
        time_controls=args.time_controls.split(','),
        modes=args.modes.split(','),
        min_rating=args.min_rating,
        max_rating=args.max_rating
    )
    
    logger.info("Bot configuration complete. Starting...")
    bot.start()


if __name__ == "__main__":
    main()