import json
import os
import glob
from openai import OpenAI

import json
import os
import glob

def generate_jsonl(output_file, system_prompt, user_files_pattern, assistant_files_pattern):
    """
    Generate a JSONL file with the specified format.
    
    Args:
        output_file: Path to the output JSONL file
        system_prompt: The system prompt to use for all entries
        user_files_pattern: Glob pattern for user prompt files (e.g., "user_*.json")
        assistant_files_pattern: Glob pattern for assistant response files (e.g., "assistant_*.txt")
    """
    # Get lists of files
    def extract_number(filename):
        # Extract just the numeric part from the filename
        basename = os.path.splitext(os.path.basename(filename))[0]
        # Find all digits in the filename
        import re
        match = re.search(r'(\d+)', basename)
        if match:
            return int(match.group(1))
        return 0  # Default if no number found
        
    user_files = sorted(glob.glob(user_files_pattern), key=extract_number)
    assistant_files = sorted(glob.glob(assistant_files_pattern), key=extract_number)
    
    # Ensure we have matching files
    if len(user_files) != len(assistant_files):
        raise ValueError(f"Number of user files ({len(user_files)}) does not match number of assistant files ({len(assistant_files)})")
    
    with open(output_file, 'w') as f_out:
        for user_file, assistant_file in zip(user_files, assistant_files):
            # Read user prompt from JSON file
            with open(user_file, 'r') as f_user:
                user_data = json.load(f_user)
                # Extract only the "clues" and "gridnums" fields
                clues = user_data.get('clues', {})
                gridnums = user_data.get('gridnums', [])
                user_content = {
                    "clues": clues,
                    "gridnums": gridnums
                }
                user_content = json.dumps(user_content)  # Convert to string for the JSONL
            
            # Read assistant response from TXT file
            with open(assistant_file, 'r') as f_assistant:
                assistant_content = f_assistant.read().strip()
            
            # Create the JSONL entry
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            
            # Write to output file
            f_out.write(json.dumps(entry) + '\n')
    
    print(f"Successfully generated {output_file} with {len(user_files)} entries")

# Example usage
if __name__ == "__main__":
    # OpenAI default system prompt (you can replace this with the actual default)
    from dotenv import load_dotenv
    load_dotenv()
    system_prompt = "You are trying to solve a crossword puzzle by careful reasoning."
    
    # File patterns - adjust these to match your actual file naming convention
    user_files_pattern = "nyt*.json"  # e.g., "nyt1.json", "nyt2.json", etc.
    assistant_files_pattern = "output*.txt"  # e.g., "output1.txt", "output2.txt", etc.
    
    generate_jsonl("output.jsonl", system_prompt, user_files_pattern, assistant_files_pattern)

    client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

    job = client.fine_tuning.jobs.create(
        training_file="output.jsonl",
        model="gpt-4o-2024-08-06",
    )


