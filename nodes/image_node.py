
import torch
from typing import Tuple, Dict, Any

from ..py.color_func import _COLORS
from ..py.image_func import convert_to_base64, generate_blank, apply_alpha, image_colorspace

module_cat = "md/image"

_COLOR_NAMES = list(_COLORS.keys())

#===================================================================================
class mdBlankImage:
    """Creates a solid‑colour RGB image with the given size and batch count."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width":       ("INT", {"default": 512, "min": 2, "max": 8192, "step": 1}),
                "height":      ("INT", {"default": 512, "min": 2, "max": 8192, "step": 1}),
                "batch_size":  ("INT", {"default": 1, "min": 1, "max": 64}),
                "color":       (_COLOR_NAMES,),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, width: int, height: int, batch_size: int, color: str,) -> tuple[torch.Tensor]:
        return (generate_blank(width, height, batch_size, color),)

#===================================================================================
class mdImageToBase64:
    """
    Converts an image to a base64 data URI (usable directly in <img src="...">).
    - image: input image tensor
    - format: jpg / png / webp
    - quality: 0.01 – 1.0 (ignored for PNG)

    Outputs:
      1. IMAGE – the original image (pass‑through)
      2. STRING – the full data URI
    The URI is also shown on the node as a multiline display.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "format": (["jpg", "png", "webp"],),
                "quality": ("FLOAT", {"default": 0.92, "min": 0.01, "max": 1.0, "step": 0.01}),
                "info_display": ("STRING", { "multiline": True }),
            }
        }

    RETURN_TYPES: Tuple[str, str] = ("IMAGE", "STRING")
    RETURN_NAMES: Tuple[str, str] = ("image", "b64_uri")
    FUNCTION: str = "exec"
    CATEGORY: str = module_cat

    def exec(self, image: torch.Tensor, format: str, quality: float, info_display: str) -> Dict[str, Any]:
        rep = convert_to_base64(image, format, quality)
        return {"ui": {"info": [rep[1]]}, "result": rep} 

#===================================================================================
class mdApplyTransparency:
    """Applies a mask as the alpha channel of an image.
       RGB input becomes RGBA; existing alpha is replaced."""
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
class mdImageColorSpace:
    """Convert an IMAGE batch between sRGB and Linear color space."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["sRGB to Linear", "Linear to sRGB"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor, direction: str) -> tuple[torch.Tensor]:
        return (image_colorspace(image, direction),)
        
#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdImageToBase64": mdImageToBase64,
    "mdBlankImage": mdBlankImage,
    "mdApplyTransparency": mdApplyTransparency,
    "mdImageColorSpace": mdImageColorSpace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdImageToBase64": "Image To Base64 (MD)",
    "mdBlankImage": "Blank Image (MD)",
    "mdApplyTransparency": "Apply Transparency (MD)",
    "mdImageColorSpace": "Image Color Space (MD)",
}