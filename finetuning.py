# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
# {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}

def purge(input_file, output_file, target_string):
    """
    Reads a text file and writes a new file with all lines containing the target string
    and the line immediately following it removed.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output text file
        target_string (str): The string to search for in each line
        
    Returns:
        int: Number of lines removed
    """
    removed_count = 0
    
    try:
        # Read all lines from the input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        # Open output file for writing
        with open(output_file, 'w', encoding='utf-8') as outfile:
            skip_next = False
            
            for i, line in enumerate(lines):
                if i == 0:
                    removed_count += 1
                
                else:
                    # If we need to skip this line because previous line had the target
                    if skip_next:
                        skip_next = False
                        removed_count += 1
                        continue
                    
                    # If current line has the target string
                    if target_string in line:
                        removed_count += 1
                        # Set flag to skip the next line if it exists
                        if i < len(lines) - 1:
                            skip_next = True
                    else:
                        outfile.write(line)
                    
        print(f"Successfully processed file. Removed {removed_count} lines (including following lines).")
        return removed_count
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return -1
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return -1

if __name__ == "__main__":
    purge("crosswords_file.txt", "crosswords_clean.txt", "kushalt/")


# from openai import OpenAI

# client = OpenAI()

# job = client.fine_tuning.jobs.create(
#     training_file="crosswords_file",
#     model="gpt-4o-2024-08-06",
# )


