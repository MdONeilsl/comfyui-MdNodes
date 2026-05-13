import importlib
import pkgutil
import os

# Tells ComfyUI to load the JS files from the /web folder
WEB_DIRECTORY = "./web"

# Dynamically collect all node mappings from the nodes/ folder
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_current_dir = os.path.dirname(__file__)
_nodes_dir = os.path.join(_current_dir, "nodes")

for module_info in pkgutil.iter_modules([_nodes_dir]):
    module_name = module_info.name
    module = importlib.import_module(f".nodes.{module_name}", package=__package__)

    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        for key, value in module.NODE_CLASS_MAPPINGS.items():
            if key in NODE_CLASS_MAPPINGS:
                print(f"Warning: duplicate node key '{key}' from {module_name}")
            NODE_CLASS_MAPPINGS[key] = value

    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

# Clean up helper variables (optional, but keeps namespace tidy)
del _current_dir, _nodes_dir, module_info, module_name, module

# Public API – yes, you should keep __all__ so users know what's available
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 
