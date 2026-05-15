
import torch
import numpy as np
import base64
import io
from PIL import Image
import torch.nn.functional as F

from .color_func import _COLORS

from typing import Tuple

#===================================================================================
def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB values to linear (remove gamma)."""
    srgb = srgb.clamp(0.0, 1.0)
    linear = torch.where(
        srgb <= 0.04045,
        srgb / 12.92,
        torch.pow((srgb + 0.055) / 1.055, 2.4)
    )
    return linear

#===================================================================================
def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Convert linear values to sRGB (apply gamma)."""
    linear = linear.clamp(0.0, 1.0)
    srgb = torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * torch.pow(linear, 1.0 / 2.4) - 0.055
    )
    return srgb.clamp(0.0, 1.0)

#===================================================================================
def convert_to_base64(image: torch.Tensor, format: str, quality: float) -> Tuple[torch.Tensor, str]:
    # Use first image of the batch only
    img_tensor = image[0]
    img_array = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array, 'RGB')

    # Save to buffer with selected format & quality
    buffer = io.BytesIO()
    if format == "png":
        pil_img.save(buffer, format="PNG")
    elif format == "webp":
        pil_img.save(buffer, format="WEBP", quality=int(quality * 100))
    else:  # jpg
        pil_img.save(buffer, format="JPEG", quality=int(quality * 100))

    # Re‑open the compressed image from buffer → tensor
    buffer.seek(0)
    compressed_pil = Image.open(buffer).convert("RGB")
    compressed_array = np.array(compressed_pil).astype(np.float32) / 255.0
    compressed_tensor = torch.from_numpy(compressed_array).unsqueeze(0)  # (1, H, W, C)

    # Generate data URI from the same buffer
    b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    mime = "image/jpeg" if format == "jpg" else f"image/{format}"
    data_uri = f"data:{mime};base64,{b64_str}"

    # Show the URI in a multiline widget on the node (for the second output)
    return (compressed_tensor, data_uri)

#===================================================================================
def generate_blank(width: int, height: int, batch_size: int, color: str) -> torch.Tensor:
    r, g, b = _COLORS[color]
    color_tensor = torch.tensor([r, g, b], dtype=torch.float32)
    img = torch.ones(batch_size, height, width, 3, dtype=torch.float32) * color_tensor
    return img

#===================================================================================
def apply_alpha(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected 4D image tensor (B, H, W, C), got {image.ndim}D")
    c = image.shape[-1]
    if c not in (3, 4):
        raise ValueError(f"Expected image with 3 or 4 channels, got {c} channels")

    # --- Normalise mask shape to (B, H, W, 1) ---
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)          # (B, H, W) -> (B, H, W, 1)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        pass  # already correct
    else:
        raise ValueError(f"Mask must be (B, H, W) or (B, H, W, 1), got shape {mask.shape}")

    # --- Ensure mask has the same spatial size as the image ---
    img_h, img_w = image.shape[1], image.shape[2]
    mask_h, mask_w = mask.shape[1], mask.shape[2]
    if (mask_h != img_h) or (mask_w != img_w):
        # Resize mask to match image dimensions (bilinear interpolation is fine for alpha)
        # Change to (B, 1, H, W) for interpolate, then back to (B, H, W, 1)
        mask = mask.permute(0, 3, 1, 2)                 # (B, 1, H, W)
        mask = F.interpolate(mask, size=(img_h, img_w), mode='bilinear', align_corners=False)
        mask = mask.permute(0, 2, 3, 1)                 # (B, H, W, 1)

    # Extract RGB channels (works for both RGB and RGBA inputs)
    rgb = image[..., :3]

    # Clamp mask to [0,1] and match dtype/device
    alpha = mask.clamp(0.0, 1.0).to(dtype=image.dtype, device=image.device)

    return torch.cat([rgb, alpha], dim=-1)

#===================================================================================
def image_colorspace(image: torch.Tensor, direction: str) -> torch.Tensor:
    # Image shape: [B, H, W, C]
    if direction == "sRGB to Linear":
        return srgb_to_linear(image)
    else:  # "Linear to sRGB"
        return linear_to_srgb(image)
    