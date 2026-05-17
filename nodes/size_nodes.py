
from ..py.size_func import STANDARD_SIZES, scale_mat_dimensions

from typing import Tuple

module_cat = "md/size"

#===================================================================================
class mdStandardMapSizePrimitive:
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "select": (STANDARD_SIZES,),
            },
        } 
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("size",)
    FUNCTION = "exec"
    CATEGORY = module_cat
    
    def exec(self, select: int) -> Tuple[int]:
        return (select,)
    
#===================================================================================
class mdPow2ScallingType:
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "round":(["closest", "ceilling", "flooring"],),
            },
        } 
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("type",)
    FUNCTION = "exec"
    CATEGORY = module_cat
    
    def exec(self, round: str) -> Tuple[str]:
        return (round,)
    
#===================================================================================
class mdStandardMapSize:
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width": (STANDARD_SIZES,),
                "height": (STANDARD_SIZES, ),
                "round":(["closest", "ceilling", "flooring"],),
            },
        } 
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "round")
    FUNCTION = "exec"
    CATEGORY = module_cat
    
    def exec(self, width: int, height: int, round: str) -> Tuple[int, int, str]:
        return (width, height, round)
    
#===================================================================================
class mdScaleToStandardMaterialSize:

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width": ("INT",{"forceInput":True}),
                "height": ("INT",{"forceInput":True}),
                "scale": (["1", "½", "¼", "⅛"], {"default": "½"}),
                "round":(["closest", "ceilling", "flooring"],),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "exec"
    CATEGORY = module_cat

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