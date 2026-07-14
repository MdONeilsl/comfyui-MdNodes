import torch

from ..py.map_func import scale_normal_map, merge_normal_maps, height_to_normal, smooth_rough_convert, mix_channel, split_channel, split_alpha_chan
from ..py.image_func import apply_alpha

from typing import Tuple, Any

from ..py.cui_type import ANY

module_cat = "md/map"

#===================================================================================
class mdNormalMapScaler:

    DESCRIPTION = "Scales a normal map to a specified width and height while respecting the chosen convention (OpenGL or DirectX)."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input normal map to scale."}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "tooltip": "Target width of the output normal map."}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "tooltip": "Target height of the output normal map."}),
                "convention": (["OpenGL", "DirectX"], {"tooltip": "The normal map convention to use (OpenGL or DirectX)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = ["The scaled normal map."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(
        self, 
        image: torch.Tensor, 
        width: int, 
        height: int, 
        convention: str
    ) -> Tuple[torch.Tensor]:
        return (scale_normal_map(image, width, height, convention),)
    
#===================================================================================
class mdNormalMapMerger:
    """
    Merges a base normal map with a detail normal map using a mask.
    The add map is interpreted as a perturbation from the flat surface
    (0,0,1). The base is rotated around an axis derived from the add
    normal's XY components by an angle = acos(add_z) * mask.
    """

    DESCRIPTION = "Merges a base normal map with a detail normal map using a mask to create a composite normal map."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "base": ("IMAGE", {"tooltip": "The base normal map to be merged."}),
                "add": ("IMAGE", {"tooltip": "The detail normal map to add to the base."}),
                "mask": ("MASK", {"tooltip": "The mask to control the blending intensity of the add map."}),
                "convention": (["OpenGL", "DirectX"], {"tooltip": "The normal map convention to use (OpenGL or DirectX)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = ["The merged normal map."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, base: torch.Tensor, add: torch.Tensor, mask: torch.Tensor, convention: str) -> Tuple[torch.Tensor]:
        return (merge_normal_maps(base, add, mask, convention),)
    
#===================================================================================
class mdHeightToNormalMap:

    DESCRIPTION = "Converts a height map into a normal map using configurable parameters like strength, smoothing, and channel inversion."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input height map to convert."}),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.01, "tooltip": "The strength of the normal map conversion."}),
                "invert_red": ("BOOLEAN", {"default": False, "tooltip": "Invert the red channel of the output normal map."}),
                "invert_green": ("BOOLEAN", {"default": True, "tooltip": "Invert the green channel of the output normal map."}),
                "smoothing": ("INT", {"default": 1, "min": 0, "max": 2, "tooltip": "The smoothing level applied to the normal map (0-2)."}),
                "use_scharr": ("BOOLEAN", {"default": True, "tooltip": "Use Scharr operator for gradient calculation (recommended for sharp edges)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    OUTPUT_TOOLTIPS = ["The generated normal map from the height map."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor, strength: float, invert_red: bool, invert_green: bool, smoothing: int, use_scharr: bool,) -> Tuple[torch.Tensor]:
        return (height_to_normal(image, strength, invert_red, invert_green, smoothing, use_scharr),)

#===================================================================================
class mdApplySpec2NormalMap:
    
    DESCRIPTION = "Applies a specular mask to a normal map to simulate lighting effects."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input normal map to which the mask will be applied."}),
                "mask": ("MASK", {"tooltip": "The specular mask to apply to the normal map."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ["The normal map with the specular mask applied."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor]:
        return (apply_alpha(image, mask),)
    
#===================================================================================          
class mdSmoothRoughConvert:
    """
    Converts smoothness ↔ roughness. Accepts either an IMAGE (4D: B,H,W,C) or a MASK (3D: B,H,W).
    Outputs both an image (4D, channels preserved) and a mask (3D, single channel).
    """

    DESCRIPTION = "Converts smoothness values to roughness values or vice versa. Accepts image or mask input and outputs both image and mask."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "input_data": (ANY, {"tooltip": "The input data, either an IMAGE or a MASK, to convert smoothness to roughness or vice versa."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    OUTPUT_TOOLTIPS = ["The converted image (smoothness/roughness values preserved).", "The converted mask (single channel)."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, input_data: Any) -> tuple[torch.Tensor]:
        return smooth_rough_convert(input_data)
    
#===================================================================================
class mdChannelMixer:
    """
    Combines separate Red, Green, Blue, and Alpha masks into a single RGB or RGBA image tensor.
    Missing inputs are treated as black (zero) channels. Output is an RGB image unless a valid
    Alpha mask is provided, in which case an RGBA image is returned.
    """
    DESCRIPTION = "Combines individual channel masks (R, G, B, A) into a single RGB or RGBA image."

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {},
            "optional": {
                "red_mask": ("MASK", {"tooltip": "Mask for the Red channel. Missing = black."}),
                "green_mask": ("MASK", {"tooltip": "Mask for the Green channel. Missing = black."}),
                "blue_mask": ("MASK", {"tooltip": "Mask for the Blue channel. Missing = black."}),
                "alpha_mask": ("MASK", {"tooltip": "Mask for the Alpha channel. Missing results in RGB output."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ["The combined RGB or RGBA image."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, red_mask=None, green_mask=None, blue_mask=None, alpha_mask=None)-> tuple[torch.Tensor]:
        return (mix_channel(red_mask, green_mask, blue_mask, alpha_mask),)
    
#===================================================================================
class mdChannelToMasks:
    """Splits an image into its individual channel masks.
    
    - For RGB images (3 channels): outputs R, G, B masks + a full white mask.
    - For RGBA images (4 channels): outputs R, G, B, A masks.
    - Images with >4 channels are rejected.
    """
    
    DESCRIPTION = "Splits an image into its individual channel masks (R, G, B, A) for further manipulation."

    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to split into channel masks."}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("red", "green", "blue", "alpha")
    OUTPUT_TOOLTIPS = ["Red channel mask.", "Green channel mask.", "Blue channel mask.", "Alpha channel mask."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor)-> tuple[torch.Tensor]:
        return split_channel(image)

#===================================================================================
class mdSplitAlphaChannle:
    
    DESCRIPTION = "Splits an RGBA image into its RGB and Alpha components for separate processing."

    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input RGBA image to split into RGB and Alpha."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgb", "alpha")
    OUTPUT_TOOLTIPS = ["The RGB portion of the image.", "The Alpha channel as a mask."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor)-> tuple[torch.Tensor]:
        return split_alpha_chan(image)
#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdNormalMapScaler": mdNormalMapScaler,
    "mdNormalMapMerger": mdNormalMapMerger,
    "mdHeightToNormalMap": mdHeightToNormalMap,
    "mdApplySpec2NormalMap": mdApplySpec2NormalMap,
    "mdSmoothRoughConvert": mdSmoothRoughConvert,
    "mdChannelMixer": mdChannelMixer,
    "mdChannelToMasks": mdChannelToMasks,
    "mdSplitAlphaChannle": mdSplitAlphaChannle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdNormalMapScaler": "Normal Map Scaler (MD)",
    "mdNormalMapMerger": "Normal Map Merger (MD)",
    "mdHeightToNormalMap": "Height To Normal Map (MD)",
    "mdApplySpec2NormalMap": "Apply Specular to Normal Map (MD)",
    "mdSmoothRoughConvert": "Smooth/Rough Convert (MD)",
    "mdChannelMixer": "Channel Mixer (MD)",
    "mdChannelToMasks": "Channel To Masks (MD)",
    "mdSplitAlphaChannle": "Split Alpha Channel (MD)",
}

