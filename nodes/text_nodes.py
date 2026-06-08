

import folder_paths
from typing import Tuple

from ..py.files_func import get_input_files_by_extensions, load_line, load_json_value

module_cat = "md/files"

#=================================================================================== 
class mdLoadTextFileLine:
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        input_dir = folder_paths.get_input_directory()
        txt_files = get_input_files_by_extensions(input_dir, (".txt",))
        return {
            "required": {
                "file": (txt_files,),
                "line": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "exec"
    CATEGORY = module_cat
    
    def exec(self, file: str, line: int) -> Tuple[str]:
       return load_line(file, line)
       
#=================================================================================== 
class mdLoadJSONKeyValue:
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        input_dir = folder_paths.get_input_directory()
        json_files = get_input_files_by_extensions(input_dir, (".json",))
        return {
            "required": {
                "FileName": (json_files or ["placeholder.json"],),
                "name": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "exec"
    CATEGORY = module_cat
    
    def exec(self, FileName: str, name: str) -> Tuple[str]:
        if not FileName or not name:
            return ("ERROR: FileName and key name are required",)
        return load_json_value(FileName, name)
    
#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdLoadTextFileLine": mdLoadTextFileLine,
    "mdLoadJSONKeyValue": mdLoadJSONKeyValue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdLoadTextFileLine": "Load Text File Line (MD)",
    "mdLoadJSONKeyValue": "Load JSON Key Value (MD)",
}