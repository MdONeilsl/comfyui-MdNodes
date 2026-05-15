
import datetime
from typing import Any

#===================================================================================
def _translate_format(fmt: str) -> str:
    """Convert simplified tokens to Python strftime codes."""
    # Order matters: replace longer tokens first to avoid partial matches.
    replacements = [
        ("yyyy", "%Y"),
        ("yy", "%y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("hh", "%I"),
        ("mm", "%M"),
        ("ss", "%S"),
        ("a", "%p"),
    ]
    for old, new in replacements:
        fmt = fmt.replace(old, new)
    return fmt
#===================================================================================
def get_date(format: str) -> str:
    py_format = _translate_format(format)
    now = datetime.datetime.now()
    return now.strftime(py_format)
#===================================================================================
def format_string(format_string: str, args: Any) -> str:
    """Apply `format_string % args` safely.

    Args:
        format_string: A printf‑style format string.
        args: A list/tuple of values to substitute.

    Returns:
        A tuple containing the formatted string.

    Raises:
        ValueError: If formatting fails (catches TypeError, KeyError, etc.).
    """
    # Ensure args is a tuple for the % operator
    if not isinstance(args, (list, tuple)):
        args = (args,)
    try:
        result = format_string % tuple(args)
    except Exception as e:
        raise ValueError(f"String formatting failed: {e}") from e
    return result
#===================================================================================
