
import torch

module_cat = "md/mask"

from ..py.color_func import _COLORS
from ..py.mask_func import generate_blank, invert_mask, to_gray, trans_mask
from ..py.image_func import image_colorspace

_COLOR_NAMES = list(_COLORS.keys())

#===================================================================================
class mdBlankMask:
    """Creates a solid‑value MASK (single‑channel) with the given size and batch count.
       The chosen colour is converted to a grayscale value (mean of R, G, B)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width":       ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height":      ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "batch_size":  ("INT", {"default": 1, "min": 1, "max": 64}),
                "color":       (_COLOR_NAMES,),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, width: int, height: int, batch_size: int, color: str) -> tuple[torch.Tensor]:
        return (generate_blank(width, height, batch_size, color),)
    
#===================================================================================
class mdMaskInvert:
    """Inverts a mask batch (1.0 - mask). White becomes black, black becomes white."""
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, mask: torch.Tensor) -> tuple[torch.Tensor]:
        return (invert_mask(mask),)
    
#===================================================================================
class mdImageToGrayMask:
    """Extracts a grayscale mask using simple RGB average."""
    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "kind": (["red", "green", "blue", "alpha", "mean", "luminance"],{"default": "luminance"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor, kind: str, intensity: float) -> tuple[torch.Tensor]:
        return (to_gray(image, kind, intensity),)
    
#=================================================================================== 
class mdMaskColorSpace:
    """Convert a MASK batch between sRGB and Linear color space.
       (Applies the same gamma encoding/decoding as for grayscale images.)"""
    
    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {
                "mask": ("MASK",),
                "direction": (["sRGB to Linear", "Linear to sRGB"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "exec"
    CATEGORY = module_cat 

    def exec(self, mask: torch.Tensor, direction: str) -> tuple[torch.Tensor]:
        return (image_colorspace(mask, direction),)
    
#===================================================================================
class mdTransparencyToMask:
    """Converts the alpha channel of an RGBA image into a mask.
       Black = fully transparent, White = fully opaque."""
    
    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor) -> tuple[torch.Tensor]:
        return (trans_mask(image),)
    
#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdBlankMask": mdBlankMask,
    "mdMaskInvert": mdMaskInvert,
    "mdImageToGrayMask": mdImageToGrayMask,
    "mdMaskColorSpace": mdMaskColorSpace,
    "mdTransparencyToMask": mdTransparencyToMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdBlankMask": "Blank Mask (MD)",
    "mdMaskInvert": "Mask Invert (MD)",
    "mdImageToGrayMask": "Image To Gray Mask (MD)",
    "mdMaskColorSpace": "Mask Color Space (MD)",
    "mdTransparencyToMask": "Transparency To Mask (MD)",
}
