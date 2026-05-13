
import torch
from typing import Tuple

from ..py.cui_type import ANY
from ..py.tensor_func import tensor_shape_info

module_cat = "md/utility"

#===================================================================================
class mdShowTensorShape:
    """
    Debug node that displays the shape and statistics of an incoming image/mask batch.
    One line per sample, plus global info.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> dict: 
        return {
            "required": {
                "image": (ANY,),
                "info_display": ("STRING", { "multiline": True }),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "exec"
    CATEGORY = module_cat
    OUTPUT_NODE = True

    def exec(self, image: torch.Tensor, info_display: str) -> Tuple[str, str]:
        return {
            "ui": {"info": tensor_shape_info(image)},
            "result": (image,),
        }

#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdShowTensorShape": mdShowTensorShape,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdShowTensorShape": "Show Tensor Shape (MD)",
}
