
import torch

from .color_func import _COLORS

#===================================================================================
def generate_blank(width: int, height: int, batch_size: int, color: str) -> torch.Tensor:
    r, g, b = _COLORS[color]
    gray = (r + g + b) / 3
    mask = torch.full((batch_size, height, width), gray, dtype=torch.float32,)
    return mask
#===================================================================================
def invert_mask(mask: torch.Tensor) -> torch.Tensor:
    # Mask shape: [B, H, W]
    if mask.ndim != 3:
        raise ValueError(f"Expected mask with shape [B, H, W], got {mask.shape}")
    
    # Clamp to 0-1 just in case, then invert
    mask_clamped = mask.clamp(0.0, 1.0)
    inverted = 1.0 - mask_clamped
    
    return inverted
#===================================================================================
def to_gray(image: torch.Tensor, kind: str, intensity: float) -> tuple[torch.Tensor]:
    # average over the three colour channels
    mask = None

    if kind == "red":
        mask = image[..., 0]
        
    elif kind == "green":
        mask = image[..., 1] 
        
    elif kind == "blue":
        mask = image[..., 2] 
        
    elif kind == "alpha":
        B, H, W, C = image.shape
        if C >= 4:
            mask = image[..., 3]
        else:
            mask = torch.ones((B, H, W), device=image.device, dtype=image.dtype)
        
    elif kind == "mean":
        mask = image[..., :3].mean(dim=-1)
        
    elif kind == "luminance":
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        mask = 0.299 * r + 0.587 * g + 0.114 * b

    scaled = mask * intensity
    scaled = scaled.clamp(0.0, 1.0)

    return scaled
#===================================================================================
def trans_mask(image: torch.Tensor)-> torch.Tensor:
    # Expect shape [B, H, W, C]
    if image.ndim != 4:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    B, H, W, C = image.shape
    if C >= 4:
        # Extract alpha channel
        mask = image[..., 3]
    else:
        # No alpha → assume fully opaque
        mask = torch.ones((B, H, W), device=image.device, dtype=image.dtype)

    # Clamp just in case (ComfyUI expects 0–1)
    mask = mask.clamp(0.0, 1.0)

    return mask
#===================================================================================
