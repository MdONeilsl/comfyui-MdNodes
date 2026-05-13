
from ..py.cui_type import ANY
from ..py.string_func import get_date, format_string

from typing import Any, Tuple 

module_cat = "md/string"

#===================================================================================
class mdTimeString:
    """
    Outputs a formatted date/time string.
    The format field accepts simple tokens:
        yyyy -> year (4 digits)
        yy   -> year (2 digits)
        MM   -> month (01-12)
        dd   -> day (01-31)
        HH   -> hour (00-23)
        hh   -> hour (01-12)
        mm   -> minute (00-59)
        ss   -> second (00-59)
        a    -> AM/PM
    Other characters are passed through unchanged.
    Example: "yyyy-MM-dd HH:mm:ss" -> "2025-03-14 15:30:45"
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "format": ("STRING", {"default": "yyyy-MM-dd", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("date",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, format: str) -> Tuple[str]:
        return (get_date(format),)
    
#===================================================================================
class mdStringFormat:
    """Formats a string using printf‑style placeholders (e.g., %s, %d)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "str": ("STRING",),
                "args": (ANY, {"default": []}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "exec"          # Fixed: matches the method name
    CATEGORY = module_cat

    def exec(self, str: str, args: Any) -> Tuple[str]:
        return (format_string(str, args),)
    
#===================================================================================
NODE_CLASS_MAPPINGS = {
    "mdTimeString": mdTimeString,
    "mdStringFormat": mdStringFormat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdTimeString": "Get Time String (MD)",
    "mdStringFormat": "Format String (MD)",
}
