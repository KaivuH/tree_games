# Recursive LLM Chess Tree Search Specification

## Core Concept: Recursive Multi-Agent Tree Search

This project implements a novel approach to chess reasoning using recursive LLM calls in a tree search framework. Unlike traditional approaches where a single LLM makes a move decision, this system:

1. Spawns multiple LLM instances to explore different branches of the game tree
2. Allows each branch to recursively call new LLM instances for deeper exploration
3. Aggregates insights from parallel explorations to inform the final move decision
4. Creates a multi-agent simulation environment where different instances of the model play against each other

The key innovation is treating the LLM as a recursive function that can call itself to delegate exploration tasks, creating a distributed reasoning system.

## System Architecture

```
                          Root LLM
                             |
                     Evaluate Position
                             |
                 +-----------+-----------+
                 |           |           |
              Branch A    Branch B    Branch C
                 |           |           |
         +-------+-----+ +---+---+ +-----+-------+
         |       |     | |       | |     |       |
     Sub-Branch Sub-Branch Sub-Branch Sub-Branch Sub-Branch
         |       |     | |       | |     |       |
    [LLM Call] [LLM Call] ... [LLM Call] [LLM Call]
```

### 1. Primary Game Controller
- Maintains the official game state
- Orchestrates the tree exploration process
- Manages communication between LLM instances
- Collects and aggregates results from different branches

### 2. Branch Explorer LLMs
- Specialized LLM instances spawned to explore specific move sequences
- Each instance can:
  - Evaluate positions
  - Make candidate moves
  - Simulate opponent responses
  - Spawn additional LLM instances for deeper exploration
  - Return evaluations to the parent LLM

### 3. Chess Environment
- Provides board state information to all LLM instances
- Validates moves and enforces chess rules
- Creates "phantom" boards for each simulation branch
- Tracks the history of moves in each branch

### 4. Function Framework
- Deterministic chess analysis functions
- Tree navigation commands
- LLM instantiation and communication protocols
- Evaluation aggregation utilities

## Detailed Workflow

### Initial Setup and Root LLM
```
SYSTEM [to Root LLM]: You are analyzing a chess position as White. Current board state:
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

You will explore multiple candidate moves by spawning Branch Explorer LLMs.
For each promising move, call explore_branch() to create a new simulation.
```

### Branch Exploration
```
ROOT LLM: I'll analyze three candidate first moves: e4, d4, and Nf3.

SYSTEM: Command: explore_branch("e4")
Creating Branch Explorer LLM for move sequence starting with e4...

SYSTEM [to Branch Explorer LLM 1]: You are analyzing a chess position after e4.
Current board state:
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R

Simulate how Black would respond, then explore White's options from there.
You can call spawn_sub_branch() to create additional LLMs for deeper exploration.
```

### Recursive Exploration
```
BRANCH EXPLORER LLM 1: As Black, the most common response to e4 is e5. Let me explore that.
After e5, White has several strong options including Nf3 and the Italian Game with Bc4.
I'll spawn sub-branches to explore these lines.

SYSTEM: Command: spawn_sub_branch("e5", "Nf3")
Creating Sub-Branch Explorer LLM for move sequence e4, e5, Nf3...

SYSTEM [to Sub-Branch Explorer LLM 1.1]: You are analyzing a chess position after e4, e5, Nf3.
Current board state:
r n b q k b n r
p p p p . p p p
. . . . . . . .
. . . . p . . .
. . . . P . . .
. . . . . N . .
P P P P . P P P
R N B Q K B . R

Simulate how Black would respond, explore White's follow-ups, and evaluate the position.
```

### Parallel Exploration of Different Branches
```
SYSTEM [to Root LLM]: Branch Explorer LLM 1 (e4) is processing...

SYSTEM: Command: explore_branch("d4")
Creating Branch Explorer LLM for move sequence starting with d4...

SYSTEM [to Branch Explorer LLM 2]: You are analyzing a chess position after d4.
Current board state:
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . P . . . .
. . . . . . . .
P P P . P P P P
R N B Q K B N R

Simulate how Black would respond, then explore White's options from there.
```

### Branch Evaluation and Reporting
```
SUB-BRANCH EXPLORER LLM 1.1: After analyzing the position following e4, e5, Nf3, I've explored Black's main responses:
1. Nc6 leading to the Two Knights Defense or Italian Game
2. Nf6 leading to the Petrov Defense

My evaluation: This position is equal with dynamic possibilities for both sides. White maintains a small initiative and has comfortable development prospects. Mainline theory is well-established here with numerous options.

Position score: +0.3 (slight advantage to White)
Developmental lead: 1 tempo
Key tactical motifs: fork potential on d5, possible kingside attack

SYSTEM [to Branch Explorer LLM 1]: Received evaluation from Sub-Branch Explorer LLM 1.1
```

### Aggregation and Decision Making
```
SYSTEM [to Root LLM]: All Branch Explorer LLMs have completed their analysis.

Branch 1 (e4) evaluation summary:
- Position score: +0.2 (slight advantage to White)
- Main variations explored: e5, c5 (Sicilian), e6 (French)
- Development tempo: Good
- Tactical opportunities: Moderate
- Strategic plans: Kingside attack, central control

Branch 2 (d4) evaluation summary:
- Position score: +0.3 (slight advantage to White)
- Main variations explored: d5, Nf6 (Indian defenses)
- Development tempo: Excellent
- Tactical opportunities: Limited
- Strategic plans: Queenside expansion, central dominance

Branch 3 (Nf3) evaluation summary:
- Position score: +0.1 (approximately equal)
- Main variations explored: d5, c5, Nf6
- Development tempo: Moderate
- Tactical opportunities: Few
- Strategic plans: Flexible development, delayed commitment

Based on these evaluations, what move do you choose to play?

ROOT LLM: After reviewing the branch explorations, I'll play d4. The analysis shows it offers the best combination of positional advantage, development prospects, and strategic opportunities. The branch explorer found particularly promising lines against both d5 and the Indian defenses.

SYSTEM: Move d4 executed on the main board.
```

## Implementation Details

### 1. Python Chess Integration

The system will be built on [python-chess](https://python-chess.readthedocs.io/) to provide a robust foundation:

```python
import chess

# Core board management
def create_board():
    return chess.Board()

def apply_move(board, move):
    move_obj = chess.Move.from_uci(move) if isinstance(move, str) else move
    if move_obj in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move_obj)
        return board_copy
    return None

# Board analysis functions
def get_legal_moves(board):
    return list(board.legal_moves)

def material_count(board, color):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    count = sum(values[piece.piece_type] for piece in board.piece_map().values()
                if piece.color == (color == 'white'))
    return count

def center_control(board, color):
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    control_count = 0
    for square in center_squares:
        if board.is_attacked_by(color == 'white', square):
            control_count += 1
    return control_count
```

### 2. Branching Factor Control

The system implements both static and MCTS-inspired dynamic branching controls:

```python
import math

# Static branching limitation
def limit_branches_static(moves, max_branches=3):
    """Limit candidate moves to a fixed number"""
    return moves[:max_branches] if max_branches else moves

# MCTS-inspired selection policy
def select_moves_ucb(candidates, parent_visits, C=1.4, max_branches=3):
    """Select moves based on UCB formula to balance exploration/exploitation"""
    ucb_scores = []
    for candidate in candidates:
        N = candidate['visits'] if candidate['visits'] > 0 else 1
        Q = candidate['score']
        ucb = Q / N + C * math.sqrt(math.log(parent_visits) / N)
        ucb_scores.append((ucb, candidate))
    
    # Sort candidates by UCB score descending
    sorted_candidates = sorted(ucb_scores, key=lambda x: x[0], reverse=True)
    # Select up to max_branches candidates
    return [cand for (_, cand) in sorted_candidates[:max_branches]]
```

### 3. Tree Navigation Commands

Commands have been enhanced to incorporate branching control:

- `explore_branch(move, max_branches=3)`: Spawn an LLM to explore a specific first move, limiting responses to `max_branches`
- `spawn_sub_branch(previous_moves, next_move, mcts_params={'C':1.4, 'max_branches':3})`: Create deeper exploration with MCTS parameters
- `evaluate_position(board)`: Generate a structured assessment including:
  - Material balance
  - Position score
  - Development assessment
  - Tactical opportunities
  - Strategic plans
- `backtrack(steps=1)`: Move up the tree `steps` levels
- `commit_move(move)`: Execute the chosen move on the main board

### 4. LLM Instance Management

- Each branch creates a new LLM instance with appropriate context
- Instances are tracked in a tree structure matching the game exploration
- Visit counts and evaluation scores are maintained for MCTS selection
- Results are cached to avoid duplicate analysis

### 5. Information Sharing Protocol

- Standardized evaluation metrics for comparing branches:
  ```python
  class BranchEvaluation:
      def __init__(self):
          self.position_score = 0.0      # -∞ to +∞, positive favors White
          self.confidence = 0.0          # 0.0 to 1.0
          self.visits = 0                # Number of times this branch was explored
          self.material_balance = 0      # Material difference in pawns
          self.development_score = 0.0   # -1.0 to 1.0
          self.tactical_score = 0.0      # -1.0 to 1.0
          self.plans = []                # List of strategic plans
          self.key_variations = []       # Critical variations explored
  ```
- Clear taxonomy of chess concepts (e.g., development, king safety)
- Confidence scores for evaluations

## Enhanced Features

### 1. Self-Play Training
- Use recursive LLM instances to play complete games against themselves
- Generate training data from these self-play games
- Identify patterns of successful/unsuccessful reasoning

### 2. Focused Exploration
- Dynamically allocate more compute to promising branches
- Allow deeper exploration of tactically complex positions
- Implement pruning for clearly inferior variations

### 3. Collective Intelligence
- Pool insights across multiple branches
- Allow branch explorers to suggest new branches to examine
- Implement voting mechanisms for move selection

### 4. Explainable Reasoning
- Generate narrative explanations of decision process
- Visualize the explored tree with evaluations
- Highlight critical decision points and key insights

## Evaluation Methods

### 1. Decision Quality
- Compare move quality with/without recursive exploration
- Measure tactical accuracy on benchmark positions
- Evaluate strategic coherence across complete games

### 2. Computational Efficiency
- Analyze token usage relative to exploration depth
- Measure exploration breadth under resource constraints
- Optimize branch allocation strategies

### 3. Reasoning Transparency
- Assess quality of position evaluations
- Measure correlation between branch exploration and final decisions
- Evaluate human interpretability of decision explanations

## Project Milestones

### Phase 1: Core Framework
- Implement basic LLM spawning mechanism
- Create the tree data structure
- Develop simple branch exploration

### Phase 2: Recursive Exploration
- Implement sub-branch spawning
- Create protocols for evaluation reporting
- Develop aggregation mechanisms

### Phase 3: Optimization
- Implement adaptive exploration depth
- Create branch pruning mechanisms
- Optimize resource allocation

### Phase 4: Advanced Features
- Implement collective reasoning capabilities
- Create visualization system for exploration tree
- Develop self-play training framework

## Conclusion

This project creates a novel approach to chess reasoning by leveraging recursive LLM calls in a multi-agent tree search framework. By allowing the model to spawn versions of itself to explore different possibilities, we create a distributed reasoning system that more closely mimics human thought processes while enhancing planning capabilities through environmental feedback.

The system demonstrates how inference-time compute can dramatically improve decision quality by enabling broader and deeper exploration of possibility spaces before commitment to action.