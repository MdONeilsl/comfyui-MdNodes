
import logging
import json
import os
import folder_paths
from typing import Dict, Tuple, List, Set, Any

from ..py.cui_type import ANY

module_cat = "md/control"

def load_json_file(base_dir: str, filename: str, input_dir: str) -> Dict:
    """
    Attempt to load a JSON file.
    First tries `base_dir/filename`.
    If that fails, tries `input_dir/filename` (global input folder).
    Raises FileNotFoundError only if both fail.
    """
    # Try relative to base_dir
    path = os.path.join(base_dir, filename)
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Try relative to global input directory
    alt_path = os.path.join(input_dir, filename)
    if os.path.isfile(alt_path):
        logging.warning(f"Loaded sub-file from fallback location: {alt_path}")
        with open(alt_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"Sub-file not found: tried {path} and {alt_path}")

#===================================================================================
class mdSwitchControl:
    MAX_OUTPUTS = 64   # fixed number of outputs

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "switches_state": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = tuple(["STRING"] * 64)
    RETURN_NAMES = ("." * 64,)
    FUNCTION = "execute"
    CATEGORY = module_cat   # replace with your category

    def execute(self, switches_state: str = "[]") -> Tuple[str, ...]:
        """Parse the switch config and return a JSON string for each of the 64 outputs."""
        try:
            switches = json.loads(switches_state) if switches_state else []
        except json.JSONDecodeError:
            logging.error("mdSwitchControl: invalid switches_state JSON")
            switches = []

        # Build an array of exactly MAX_OUTPUTS JSON strings
        outputs = []
        for i in range(self.MAX_OUTPUTS):
            if i < len(switches):
                sw = switches[i]
                name = sw.get("label", f"Switch_{i}")
                value = bool(sw.get("value", False))
            else:
                name = ""
                value = False
            outputs.append(json.dumps({"name": name, "value": value}))

        return tuple(outputs)
    
#===================================================================================
class mdMergeControls:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "input_0": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execute"
    CATEGORY = module_cat

    def execute(self, **kwargs) -> Tuple[str]:
        items = []

        for key in sorted(kwargs.keys()):
            if not key.startswith("input_"):
                continue

            value = kwargs[key]
            if not isinstance(value, str):
                logging.warning(f"mdMergeControls: input {key} is not a string, skipping")
                continue

            # Skip empty strings
            if not value.strip():
                continue

            try:
                parsed = json.loads(value)

                if isinstance(parsed, dict):
                    # Single object -> append
                    items.append(parsed)

                elif isinstance(parsed, list):
                    # Array of objects -> extend (flatten one level)
                    for elem in parsed:
                        if isinstance(elem, dict):
                            items.append(elem)
                        else:
                            logging.warning(
                                f"mdMergeControls: non‑dict element in array from {key}: {type(elem)}"
                            )
                else:
                    logging.warning(
                        f"mdMergeControls: {key} is neither a dict nor a list, got {type(parsed)}"
                    )

            except json.JSONDecodeError:
                logging.error(f"mdMergeControls: invalid JSON in {key}: {value[:100]}")

        # Return as a JSON string (always a list, even if empty)
        return (json.dumps(items),)
    
#===================================================================================
class mdTagsThreeLoader:

    # ---------------------------------------------------------
    # ComfyUI
    # ---------------------------------------------------------

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()

        files = []

        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.lower().endswith(".json"):
                    full_path = os.path.join(root, filename)

                    # Store relative path for ComfyUI dropdown
                    rel_path = os.path.relpath(full_path, input_dir)
                    rel_path = rel_path.replace("\\", "/")

                    files.append(rel_path)

        return {
            "required": {
                "file_name": (sorted(files),),
                "json_array": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[]"
                    }
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "hierarchy")

    FUNCTION = "extract_tags"

    CATEGORY = module_cat

    # ---------------------------------------------------------
    # JSON Loader
    # ---------------------------------------------------------

    @staticmethod
    def load_json_file(base_dir: str, file_name: str, input_dir: str) -> Dict:

        # Resolve path relative to CURRENT file
        full_path = os.path.abspath(
            os.path.join(base_dir, file_name)
        )

        # Sandbox protection
        input_dir_abs = os.path.abspath(input_dir)

        if not full_path.startswith(input_dir_abs):
            raise ValueError(
                f"Path traversal detected: {file_name}"
            )

        if not os.path.isfile(full_path):
            raise FileNotFoundError(
                f"File not found: {full_path}"
            )

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"JSON root must be an object: {full_path}"
            )

        return data

    # ---------------------------------------------------------
    # Active Map
    # ---------------------------------------------------------

    @staticmethod
    def parse_active_map(json_array: str) -> Dict[str, bool]:

        try:
            parsed = json.loads(json_array)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid json_array JSON: {e}"
            )

        if not isinstance(parsed, list):
            raise ValueError(
                "json_array must be a JSON array"
            )

        active = {}

        for item in parsed:

            if not isinstance(item, dict):
                continue

            name = item.get("name")

            if not isinstance(name, str):
                continue

            active[name] = bool(
                item.get("value", False)
            )

        return active

    # ---------------------------------------------------------
    # Weighted Tag Formatting
    # ---------------------------------------------------------

    @staticmethod
    def normalize_tag(tag: Any) -> str | None:
        # Supported:
        # "masterpiece"
        # ["masterpiece", 1.2]

        # Simple tag
        if isinstance(tag, str):
            return tag.strip()

        # Weighted tag
        if (
            isinstance(tag, list)
            and len(tag) == 2
            and isinstance(tag[0], str)
            and isinstance(tag[1], (int, float))
        ):
            name = tag[0].strip()
            weight = float(tag[1])

            return f"({name}:{weight:g})"

        return None

    # ---------------------------------------------------------
    # Main
    # ---------------------------------------------------------

    def extract_tags(self, file_name: str, json_array: str) -> Tuple[str, str]:

        input_dir = folder_paths.get_input_directory()

        # -----------------------------------------------------
        # Parse active map
        # -----------------------------------------------------

        active = self.parse_active_map(json_array)

        # -----------------------------------------------------
        # Load main file
        # -----------------------------------------------------

        main_full_path = os.path.abspath(
            os.path.join(input_dir, file_name)
        )

        main_dir = os.path.dirname(main_full_path)

        main_data = self.load_json_file(
            input_dir,
            file_name,
            input_dir
        )

        root_name = main_data.get("root")

        if not isinstance(root_name, str):
            raise ValueError(
                "JSON must contain a valid 'root'"
            )

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

        collected_tags: List[str] = []
        collected_hierarchy: List[str] = []

        seen_tags: Set[str] = set()
        visited_nodes: Set[Tuple[str, str]] = set()

        # -----------------------------------------------------
        # Add tags helper
        # -----------------------------------------------------

        def add_tags(tags: List[Any]):

            if not isinstance(tags, list):
                return

            for raw_tag in tags:

                tag = self.normalize_tag(raw_tag)

                if not tag:
                    continue

                if tag in seen_tags:
                    continue

                seen_tags.add(tag)
                collected_tags.append(tag)

        # -----------------------------------------------------
        # Recursive traversal
        # -----------------------------------------------------

        def traverse(
            node_name: str,
            data: Dict,
            current_dir: str
        ):

            # ---------------------------------------------
            # Prevent recursion loops
            # ---------------------------------------------

            visit_key = (
                current_dir,
                node_name
            )

            if visit_key in visited_nodes:
                return

            visited_nodes.add(visit_key)

            # ---------------------------------------------
            # Active state
            # ---------------------------------------------

            is_active = active.get(
                node_name,
                False
            )

            # Entire subtree disabled
            if not is_active:
                return

            # ---------------------------------------------
            # Get node
            # ---------------------------------------------

            node = data.get(node_name)

            if not isinstance(node, dict):

                logging.warning(
                    f"Node '{node_name}' "
                    f"not found or invalid"
                )

                return

            # ---------------------------------------------
            # Hierarchy
            # ---------------------------------------------

            collected_hierarchy.append(
                node_name
            )

            # ---------------------------------------------
            # Tags
            # ---------------------------------------------

            tags = node.get("tags", [])
            add_tags(tags)

            # ---------------------------------------------
            # Subs
            # ---------------------------------------------

            subs = node.get("subs", [])

            # Allow:
            # "subs": "test.json"
            # or
            # "subs": ["a", "b"]

            if isinstance(subs, str):
                subs = [subs]

            if not isinstance(subs, list):
                return

            # ---------------------------------------------
            # Traverse subs
            # ---------------------------------------------

            for sub in subs:

                if not isinstance(sub, str):
                    continue

                sub = sub.strip()

                if not sub:
                    continue

                # -----------------------------------------
                # External file
                # -----------------------------------------

                if sub.lower().endswith(".json"):

                    try:

                        sub_data = self.load_json_file(
                            current_dir,
                            sub,
                            input_dir
                        )

                        sub_root = sub_data.get("root")

                        if not isinstance(sub_root, str):

                            logging.warning(
                                f"Sub-file '{sub}' "
                                f"has invalid root"
                            )

                            continue

                        sub_full_path = os.path.abspath(
                            os.path.join(
                                current_dir,
                                sub
                            )
                        )

                        sub_dir = os.path.dirname(
                            sub_full_path
                        )

                        traverse(
                            sub_root,
                            sub_data,
                            sub_dir
                        )

                    except Exception as e:

                        logging.warning(
                            f"Failed loading "
                            f"sub-file '{sub}': {e}"
                        )

                # -----------------------------------------
                # Local node
                # -----------------------------------------

                else:

                    if sub not in data:

                        logging.warning(
                            f"Sub-node '{sub}' "
                            f"not found"
                        )

                        continue

                    traverse(
                        sub,
                        data,
                        current_dir
                    )

        # -----------------------------------------------------
        # Start traversal
        # -----------------------------------------------------

        traverse(
            root_name,
            main_data,
            main_dir
        )

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

        tags_str = ", ".join(collected_tags)
        hierarchy_str = ", ".join(collected_hierarchy)

        return (
            tags_str,
            hierarchy_str
        )

#===================================================================================
NODE_CLASS_MAPPINGS = {
    "mdSwitchControl": mdSwitchControl,
    "mdMergeControls": mdMergeControls,
    "mdTagsThreeLoader": mdTagsThreeLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdSwitchControl": "Switch Control (MD)",
    "mdMergeControls": "Merge Controls (MD)",
    "mdTagsThreeLoader": "Tags Three Loader (MD)",
}

