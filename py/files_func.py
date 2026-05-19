
import os
import json
import folder_paths
from typing import List, Tuple

#=================================================================================== 
def get_input_files_by_extensions(directory: str, extensions: Tuple[str, ...]) -> List[str]:
    """
    Recursively scan the directory and return a sorted list of relative paths
    to all files whose extensions match any in the provided tuple.
    
    Args:
        directory: The root directory to scan.
        extensions: Tuple of extensions including the dot, e.g., ('.txt', '.json', '.md').
    
    Returns:
        List of relative file paths (relative to directory), sorted case-insensitively.
    """
    result = []
    if not os.path.isdir(directory):
        return result
    
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            # Check if file ends with any of the extensions (case-insensitive)
            if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, directory)
                result.append(rel_path)
    
    result.sort(key=str.lower)
    return result

#=================================================================================== 
def load_line(file: str, line: int) -> Tuple[str]:
    """
    Load the specified line from the selected text file.
    Returns the line content as a string (without trailing newline).
    Always returns a valid line:
      - Wraps around when line number > number of lines
      - Handles negative/overflowed line numbers via modulo
      - Returns empty string for empty files
    """
    input_dir = folder_paths.get_input_directory()
    full_path = os.path.join(input_dir, file)
    
    if not os.path.exists(full_path):
        return (f"ERROR: File not found: {full_path}",)
    
    try:
        with open(full_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        
        num_lines = len(lines)
        if num_lines == 0:
            return ("",)  # empty file → valid empty line
        
        # Convert 1‑indexed to 0‑indexed, then modulo to always stay in range
        # Python's % works correctly for negative numbers as well.
        line_idx = (line - 1) % num_lines
        
        return (lines[line_idx].rstrip("\r\n"),)
    
    except Exception as e:
        return (f"ERROR: Could not read file - {str(e)}",)
    
#=================================================================================== 
def load_json_value(FileName: str, name: str) -> Tuple[str]:
    """
    Load the JSON file, look up the key 'name', and return the associated string.
    If the key is missing or the value is not a string, return an error message.
    """
    input_dir = folder_paths.get_input_directory()
    full_path = os.path.join(input_dir, FileName)
    
    # Check file existence
    if not os.path.exists(full_path):
        return (f"ERROR: JSON file not found: {full_path}",)
    
    try:
        with open(full_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return (f"ERROR: Invalid JSON - {str(e)}",)
    except Exception as e:
        return (f"ERROR: Could not read file - {str(e)}",)
    
    # Check if key exists
    if name not in data:
        available_keys = list(data.keys())
        return (f"ERROR: Key '{name}' not found. Available keys: {available_keys}",)
    
    value = data[name]
    
    # Ensure value is a string (convert if needed? spec says return string associate with key)
    if not isinstance(value, str):
        # If it's not a string, convert to string (maybe user expects prompt as string)
        # But to be safe, we can return an error or convert. Let's convert for flexibility.
        # However, the spec says "return the string associate with the key" -> implies it should be a string.
        # We'll warn and convert.
        return (f"WARNING: Value for '{name}' is not a string (type: {type(value).__name__}). Converting to string: {str(value)}",)
    
    return (value,)
#===================================================================================
