import chess
import chess.svg
import pandas as pd
from typing import Dict, List, Any, Optional
import json
import os
from IPython.display import display, SVG, HTML

def display_board(board: chess.Board, size: int = 400):
    """
    Display a chess board in a Jupyter notebook.
    
    Args:
        board: Chess board
        size: Size of the board in pixels
    """
    svg_content = chess.svg.board(board, size=size)
    return SVG(svg_content)

def display_evaluation(evaluation: Dict[str, Any]):
    """
    Display an evaluation in a formatted way.
    
    Args:
        evaluation: Evaluation dictionary
    """
    if not evaluation:
        return HTML("<p>No evaluation available</p>")
        
    # Format numeric values
    numeric_fields = ["position_score", "confidence", "material_balance", 
                     "development_score", "tactical_score"]
    
    table_rows = []
    for field in numeric_fields:
        if field in evaluation:
            value = evaluation[field]
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)
            table_rows.append(f"<tr><td><b>{field}</b></td><td>{formatted_value}</td></tr>")
    
    # Format lists
    list_fields = ["plans", "key_variations"]
    list_html = ""
    
    for field in list_fields:
        if field in evaluation and evaluation[field]:
            list_html += f"<h4>{field.replace('_', ' ').title()}</h4><ul>"
            for item in evaluation[field]:
                list_html += f"<li>{item}</li>"
            list_html += "</ul>"
    
    # Construct table
    table_html = f"""
    <table>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
    {list_html}
    """
    
    return HTML(table_html)

def plot_tree(controller, max_depth: int = 3):
    """
    Visualize the exploration tree.
    
    Args:
        controller: TreeSearchController object
        max_depth: Maximum depth to display
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_agraph import graphviz_layout
    except ImportError:
        return HTML("<p>Error: Required libraries (networkx, matplotlib) not installed</p>")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges starting from root
    def add_nodes_recursive(node, depth=0):
        if depth > max_depth or not node:
            return
            
        node_id = id(node)
        label = str(node.move.uci() if node.move else "Root")
        score = node.get_average_score()
        visits = node.visits
        
        # Add node with attributes
        G.add_node(node_id, label=label, score=score, visits=visits)
        
        # Add child nodes and edges
        for child in node.children:
            child_id = id(child)
            add_nodes_recursive(child, depth+1)
            G.add_edge(node_id, child_id)
    
    add_nodes_recursive(controller.root)
    
    # Prepare node labels, colors, and sizes
    labels = {}
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        attrs = G.nodes[node]
        label = attrs['label']
        score = attrs['score']
        visits = attrs['visits']
        
        # Create label with score and visits
        labels[node] = f"{label}\n{score:.2f} ({visits})"
        
        # Node color based on score (green for high scores, red for low)
        node_colors.append((1.0 - score, score, 0.0))
        
        # Node size based on visits
        node_sizes.append(100 + 20 * visits)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, labels=labels, node_color=node_colors, 
           node_size=node_sizes, font_size=8, arrows=True, 
           arrowsize=10, width=1.5, with_labels=True)
    plt.title("Exploration Tree")
    plt.axis('off')
    plt.tight_layout()
    
    return plt

def format_branch_results(branch_results: Dict[str, Any]) -> str:
    """
    Format branch exploration results for display.
    
    Args:
        branch_results: Dictionary of branch evaluation results
        
    Returns:
        Formatted HTML string
    """
    if not branch_results:
        return "<p>No branch results available</p>"
        
    html = "<div style='max-height: 500px; overflow-y: auto;'>"
    html += "<h3>Branch Exploration Results</h3>"
    
    for branch_id, evaluation in branch_results.items():
        # Skip branches with errors
        if "error" in evaluation:
            html += f"<h4>Branch: {branch_id}</h4><p>Error: {evaluation['error']}</p>"
            continue
            
        html += f"<h4>Branch: {branch_id}</h4>"
        
        # Format numeric values
        numeric_fields = ["position_score", "confidence", "material_balance", 
                         "development_score", "tactical_score"]
        
        table_rows = []
        for field in numeric_fields:
            if field in evaluation:
                value = evaluation[field]
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                table_rows.append(f"<tr><td><b>{field}</b></td><td>{formatted_value}</td></tr>")
        
        # Format lists
        list_fields = ["plans", "key_variations"]
        list_html = ""
        
        for field in list_fields:
            if field in evaluation and evaluation[field]:
                list_html += f"<h5>{field.replace('_', ' ').title()}</h5><ul>"
                for item in evaluation[field]:
                    list_html += f"<li>{item}</li>"
                list_html += "</ul>"
        
        # Construct table
        html += f"""
        <table>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        {list_html}
        <hr/>
        """
    
    html += "</div>"
    return HTML(html)

def export_analysis_to_pgn(controller, filename: str, analysis_depth: int = 2):
    """
    Export analysis to PGN format.
    
    Args:
        controller: TreeSearchController object
        filename: Output filename
        analysis_depth: Depth of analysis to include
    """
    try:
        import chess.pgn
    except ImportError:
        return "Error: python-chess library not installed"
    
    # Create a game object
    game = chess.pgn.Game()
    
    # Add headers
    game.headers["Event"] = "Recursive LLM Analysis"
    game.headers["Site"] = "Tree Search Analysis"
    game.headers["Date"] = game.headers["Date"]
    game.headers["Round"] = "1"
    game.headers["White"] = "LLM Analysis"
    game.headers["Black"] = "LLM Analysis"
    game.headers["Result"] = "*"
    
    # Add initial position if not standard
    if controller.engine.board.fen() != chess.STARTING_FEN:
        game.setup(controller.engine.board)
        node = game
    else:
        node = game
    
    # Add moves from move stack
    for move in controller.engine.board.move_stack:
        node = node.add_variation(move)
    
    # Recursively add evaluations as comments
    def add_evaluations_recursive(tree_node, pgn_node, depth=0):
        if depth > analysis_depth or not tree_node or tree_node.visits == 0:
            return
            
        # Add evaluation as comment if available
        evaluation = tree_node.get_aggregate_evaluation()
        if evaluation:
            comment_parts = []
            
            # Add score
            if "position_score" in evaluation:
                comment_parts.append(f"Score: {evaluation['position_score']:.2f}")
                
            # Add material balance
            if "material_balance" in evaluation:
                comment_parts.append(f"Material: {evaluation['material_balance']:.1f}")
                
            # Add plans
            if "plans" in evaluation and evaluation["plans"]:
                plans_str = "; ".join(evaluation["plans"][:2])  # Limit to first 2 plans
                comment_parts.append(f"Plans: {plans_str}")
                
            pgn_node.comment = " | ".join(comment_parts)
            
        # Add child variations
        for child in tree_node.children:
            if child.visits > 0:
                child_node = pgn_node.add_variation(child.move)
                add_evaluations_recursive(child, child_node, depth+1)
    
    # Add evaluations starting from the root
    add_evaluations_recursive(controller.root, node)
    
    # Write to PGN file
    with open(filename, "w") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
    
    return f"Analysis exported to {filename}"

def json_to_dataframe(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of evaluation dictionaries to a pandas DataFrame.
    
    Args:
        evaluations: List of evaluation dictionaries
        
    Returns:
        Pandas DataFrame
    """
    # Extract and flatten relevant fields
    flattened_data = []
    
    for eval_dict in evaluations:
        row = {
            "move": eval_dict.get("move", ""),
            "position_score": eval_dict.get("position_score", 0.0),
            "confidence": eval_dict.get("confidence", 0.0),
            "material_balance": eval_dict.get("material_balance", 0.0),
            "development_score": eval_dict.get("development_score", 0.0),
            "tactical_score": eval_dict.get("tactical_score", 0.0),
            "visits": eval_dict.get("visits", 0)
        }
        
        # Add plans as comma-separated string
        plans = eval_dict.get("plans", [])
        row["plans"] = ", ".join(plans[:3]) if plans else ""  # Limit to first 3 plans
        
        # Add variations as comma-separated string
        variations = eval_dict.get("key_variations", [])
        row["key_variations"] = ", ".join(variations[:3]) if variations else ""  # Limit to first 3 variations
        
        flattened_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Reorder columns
    column_order = ["move", "position_score", "confidence", "material_balance", 
                   "development_score", "tactical_score", "visits", 
                   "plans", "key_variations"]
    
    return df[column_order]

def display_comparison_table(evaluations: List[Dict[str, Any]]):
    """
    Display a comparison table of move evaluations.
    
    Args:
        evaluations: List of evaluation dictionaries
        
    Returns:
        HTML table
    """
    if not evaluations:
        return HTML("<p>No evaluations available</p>")
        
    # Convert to DataFrame
    df = json_to_dataframe(evaluations)
    
    # Style DataFrame
    styled_df = df.style.format({
        "position_score": "{:.3f}",
        "confidence": "{:.2f}",
        "material_balance": "{:.2f}",
        "development_score": "{:.2f}",
        "tactical_score": "{:.2f}"
    })
    
    # Add background color based on position score
    styled_df = styled_df.background_gradient(cmap="RdYlGn", subset=["position_score"])
    
    # Display as HTML
    return styled_df