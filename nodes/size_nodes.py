from ..py.size_func import STANDARD_SIZES, scale_mat_dimensions

from typing import Tuple

module_cat = "md/size"

#===================================================================================
class mdStandardMapSizePrimitive:
    
    DESCRIPTION = "Selects a standard map size from predefined options."
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "select": (STANDARD_SIZES, {"tooltip": "Choose a standard map size from the predefined list."}),
            },
        } 
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("size",)
    FUNCTION = "exec"
    CATEGORY = module_cat
    OUTPUT_TOOLTIPS = ["The selected standard map size value."]

    def exec(self, select: int) -> Tuple[int]:
        return (select,)
    
#===================================================================================
class mdPow2ScallingType:
    
    DESCRIPTION = "Defines the rounding behavior for power-of-two scaling."
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "round":(["closest", "ceilling", "flooring"], {"tooltip": "Choose the rounding method for scaling operations."}),
            },
        } 
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("type",)
    FUNCTION = "exec"
    CATEGORY = module_cat
    OUTPUT_TOOLTIPS = ["The selected rounding method for power-of-two scaling."]

    def exec(self, round: str) -> Tuple[str]:
        return (round,)
    
#===================================================================================
class mdStandardMapSize:
    
    DESCRIPTION = "Selects standard width and height values with a rounding option."
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width": (STANDARD_SIZES, {"tooltip": "Select the standard width size from predefined options."}),
                "height": (STANDARD_SIZES, {"tooltip": "Select the standard height size from predefined options."}),
                "round":(["closest", "ceilling", "flooring"], {"tooltip": "Choose the rounding method for the selected dimensions."}),
            },
        } 
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "round")
    FUNCTION = "exec"
    CATEGORY = module_cat
    OUTPUT_TOOLTIPS = ["The selected standard width value.", "The selected standard height value.", "The selected rounding method."]

    def exec(self, width: int, height: int, round: str) -> Tuple[int, int, str]:
        return (width, height, round)
    
#===================================================================================
class mdScaleToStandardMaterialSize:

    DESCRIPTION = "Scales input material dimensions to a standard size using a specified scale factor and rounding method."
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width": ("INT", {"forceInput": True, "tooltip": "The input width value to scale."}),
                "height": ("INT", {"forceInput": True, "tooltip": "The input height value to scale."}),
                "scale": (["1", "½", "¼", "⅛"], {"default": "½", "tooltip": "Select the scale factor to apply to the material dimensions."}),
                "round":(["closest", "ceilling", "flooring"], {"tooltip": "Choose the rounding method for the scaled dimensions."}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "exec"
    CATEGORY = module_cat
    OUTPUT_TOOLTIPS = ["The scaled width value.", "The scaled height value."]

    def exec(self, width: int, height: int, scale: str, round: str) -> Tuple[int, int]:
        return scale_mat_dimensions(width, height, scale, round)
    
#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdStandardMapSizePrimitive": mdStandardMapSizePrimitive,
    "mdPow2ScallingType": mdPow2ScallingType,
    "mdStandardMapSize": mdStandardMapSize,
    "mdScaleToStandardMaterialSize": mdScaleToStandardMaterialSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdStandardMapSizePrimitive": "Standard Map Size Primitive (MD)",
    "mdPow2ScallingType": "Pow2 Scalling Type (MD)",
    "mdStandardMapSize": "Standard Map Size W/H (MD)",
    "mdScaleToStandardMaterialSize": "Sub Mat Scale (MD)",
}

