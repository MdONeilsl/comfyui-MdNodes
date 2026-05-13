
import torch

from ..py.map_func import scale_normal_map, merge_normal_maps, height_to_normal, smooth_rough_convert, mix_channel, split_channel
from ..py.image_func import apply_alpha

from typing import Tuple, Any

from ..py.cui_type import ANY

module_cat = "md/map"

#===================================================================================
class mdStandardMapSize:
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width": ([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],),
                "height": ([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], ),
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
class mdNormalMapScaler:

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "convention": (["OpenGL", "DirectX"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
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

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "base": ("IMAGE",),
                "add": ("IMAGE",),
                "mask": ("MASK",),
                "convention": (["OpenGL", "DirectX"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, base: torch.Tensor, add: torch.Tensor, mask: torch.Tensor, convention: str) -> Tuple[torch.Tensor]:
        return (merge_normal_maps(base, add, mask, convention),)
    
#===================================================================================
class mdHeightToNormalMap:

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.01}),
                "invert_red": ("BOOLEAN", {"default": False}),
                "invert_green": ("BOOLEAN", {"default": True}),
                "smoothing": ("INT", {"default": 1, "min": 0, "max": 2}),
                "use_scharr": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor, strength: float, invert_red: bool, invert_green: bool, smoothing: int, use_scharr: bool,) -> Tuple[torch.Tensor]:
        return (height_to_normal(image, strength, invert_red, invert_green, smoothing, use_scharr),)

#===================================================================================
class mdApplySpec2NormalMap:
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
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

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "input_data": (ANY,),  # Accepts any type (IMAGE or MASK)
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
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
    
    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {
                "image": ("IMAGE",),   # batch of images: (B, H, W, C)
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("red", "green", "blue", "alpha")
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor)-> tuple[torch.Tensor]:
        return split_channel(image)

#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdStandardMapSize": mdStandardMapSize,
    "mdNormalMapScaler": mdNormalMapScaler,
    "mdNormalMapMerger": mdNormalMapMerger,
    "mdHeightToNormalMap": mdHeightToNormalMap,
    "mdApplySpec2NormalMap": mdApplySpec2NormalMap,
    "mdSmoothRoughConvert": mdSmoothRoughConvert,
    "mdChannelMixer": mdChannelMixer,
    "mdChannelToMasks": mdChannelToMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdStandardMapSize": "Standard Map Size (MD)",
    "mdNormalMapScaler": "Normal Map Scaler (MD)",
    "mdNormalMapMerger": "Normal Map Merger (MD)",
    "mdHeightToNormalMap": "Height To Normal Map (MD)",
    "mdApplySpec2NormalMap": "Apply Specular to Normal Map (MD)",
    "mdSmoothRoughConvert": "Smooth/Rough Convert (MD)",
    "mdChannelMixer": "Channel Mixer (MD)",
    "mdChannelToMasks": "Channel To Masks (MD)",
}
