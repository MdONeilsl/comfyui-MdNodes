import re

module_cat = "md/tags"

#===================================================================================
class mdTagUnderscoreOperator:

    DESCRIPTION = "Transforms tags by replacing spaces with underscores (add) or underscores with spaces (remove)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags_in": ("STRING", {"forceInput": True, "tooltip": "Comma-separated list of tags to process."}),
                "operator": (["add", "remove"], {"default": "add", "tooltip": "Choose to add underscores or remove them from tags."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_out",)
    OUTPUT_TOOLTIPS = ["Comma-separated list of processed tags."]

    FUNCTION = "process_tags"
    CATEGORY = module_cat

    def process_tags(self, tags_in: str, operator: str) -> tuple:
        # Split by commas, strip whitespace, and filter out empty strings
        tags = [tag.strip() for tag in tags_in.split(",") if tag.strip()]

        if operator == "add":
            # Replace any run of whitespace (spaces, tabs, etc.) with a single underscore
            processed = [re.sub(r"\s+", "_", tag) for tag in tags]
        else:  # operator == "remove"
            # Replace any run of underscores with a single space, then strip
            processed = [re.sub(r"_+", " ", tag).strip() for tag in tags]

        # Join back with comma + space
        result = ", ".join(processed)
        return (result,)

#===================================================================================
# Node registration metadata (optional, but good practice)
NODE_CLASS_MAPPINGS = {
    "mdTagUnderscoreOperator": mdTagUnderscoreOperator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdTagUnderscoreOperator": "Tag Underscore Operator",
}

