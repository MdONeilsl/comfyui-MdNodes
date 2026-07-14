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

    DESCRIPTION = "Creates a solid-value mask with specified dimensions and color. Color is converted to grayscale using RGB average."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "width":       ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Width of the mask in pixels."}),
                "height":      ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, "tooltip": "Height of the mask in pixels."}),
                "batch_size":  ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Number of mask batches to generate."}),
                "color":       (_COLOR_NAMES, {"tooltip": "Color to use for the mask. Converted to grayscale using RGB average."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ["Generated mask with specified dimensions and color."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, width: int, height: int, batch_size: int, color: str) -> tuple[torch.Tensor]:
        return (generate_blank(width, height, batch_size, color),)
    
#===================================================================================
class mdMaskInvert:
    """Inverts a mask batch (1.0 - mask). White becomes black, black becomes white."""
    
    DESCRIPTION = "Inverts the mask values (1.0 - mask), turning white to black and black to white."

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Input mask to invert."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ["Inverted mask (white becomes black, black becomes white)."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, mask: torch.Tensor) -> tuple[torch.Tensor]:
        return (invert_mask(mask),)
    
#===================================================================================
class mdImageToGrayMask:
    """Extracts a grayscale mask using simple RGB average."""
    DESCRIPTION = "Converts an image to a grayscale mask using specified channel or luminance method."

    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to convert to grayscale mask."}),
                "kind": (["red", "green", "blue", "alpha", "mean", "luminance"], {"default": "luminance", "tooltip": "Channel or method to use for grayscale conversion."}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Intensity multiplier for the grayscale conversion (0.0 to 2.0)."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ["Grayscale mask generated from the input image using specified method."]

    FUNCTION = "exec"
    CATEGORY = module_cat

    def exec(self, image: torch.Tensor, kind: str, intensity: float) -> tuple[torch.Tensor]:
        return (to_gray(image, kind, intensity),)
    
#=================================================================================== 
class mdMaskColorSpace:
    """Convert a MASK batch between sRGB and Linear color space.
       (Applies the same gamma encoding/decoding as for grayscale images.)"""
    
    DESCRIPTION = "Converts mask color space between sRGB and Linear, applying gamma encoding/decoding."

    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Input mask to convert color space."}),
                "direction": (["sRGB to Linear", "Linear to sRGB"], {"tooltip": "Direction of color space conversion."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ["Mask converted to the specified color space (sRGB or Linear)."]

    FUNCTION = "exec"
    CATEGORY = module_cat 

    def exec(self, mask: torch.Tensor, direction: str) -> tuple[torch.Tensor]:
        return (image_colorspace(mask, direction),)
    
#===================================================================================
class mdTransparencyToMask:
    """Converts the alpha channel of an RGBA image into a mask.
       Black = fully transparent, White = fully opaque."""
    
    DESCRIPTION = "Converts the alpha channel of an RGBA image into a mask (black = transparent, white = opaque)."

    @classmethod
    def INPUT_TYPES(cls)-> dict:
        return {"required": {"image": ("IMAGE", {"tooltip": "Input RGBA image to extract alpha channel as mask."})}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    OUTPUT_TOOLTIPS = ["Mask generated from the alpha channel of the input image."]

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

