
import torch
import torch.nn.functional as F

from typing import Tuple, Any

#===================================================================================
def scale_normal_map(image: torch.Tensor, width: int, height: int, convention: str) -> torch.Tensor:
    if convention not in ("OpenGL", "DirectX"):
        raise ValueError(f"Invalid convention: {convention}")

    B, H, W, C = image.shape
    device = image.device
    dtype = image.dtype

    # Ensure RGBA
    if C == 3:
        alpha = torch.ones((B, H, W, 1), device=device, dtype=dtype)
        image = torch.cat([image, alpha], dim=-1)
    elif C != 4:
        raise ValueError(f"Expected 3 or 4 channels, got {C}")

    rgb = image[..., :3]
    alpha = image[..., 3]

    # --------------------------------------------------
    # 1. Build sampling coordinates (pixel-center aligned)
    # --------------------------------------------------
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height * H - 0.5
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width * W - 0.5

    fy = torch.floor(ys).long()
    fx = torch.floor(xs).long()

    dy = (ys - fy).unsqueeze(1)  # (H',1)
    dx = (xs - fx).unsqueeze(0)  # (1,W')

    # Clamp
    y0 = fy.clamp(0, H - 1)
    y1 = (fy + 1).clamp(0, H - 1)
    x0 = fx.clamp(0, W - 1)
    x1 = (fx + 1).clamp(0, W - 1)

    # Expand for broadcasting
    y0 = y0.view(-1, 1)
    y1 = y1.view(-1, 1)
    x0 = x0.view(1, -1)
    x1 = x1.view(1, -1)

    # --------------------------------------------------
    # 2. Gather neighbors (pure tensor indexing)
    # --------------------------------------------------
    def gather(img, yy, xx):
        return img[:, yy, xx]  # (B, H', W', C)

    v00 = gather(rgb, y0, x0)
    v10 = gather(rgb, y0, x1)
    v01 = gather(rgb, y1, x0)
    v11 = gather(rgb, y1, x1)

    a00 = gather(alpha.unsqueeze(-1), y0, x0)[..., 0]
    a10 = gather(alpha.unsqueeze(-1), y0, x1)[..., 0]
    a01 = gather(alpha.unsqueeze(-1), y1, x0)[..., 0]
    a11 = gather(alpha.unsqueeze(-1), y1, x1)[..., 0]

    # --------------------------------------------------
    # 3. Decode normals
    # --------------------------------------------------
    def decode(v):
        r = v[..., 0:1] * 2.0 - 1.0
        g = v[..., 1:2]
        b = v[..., 2:3] * 2.0 - 1.0

        if convention == "OpenGL":
            y = g * 2.0 - 1.0
        else:
            y = 1.0 - g * 2.0

        vec = torch.cat([r, y, b], dim=-1)

        length = torch.linalg.norm(vec, dim=-1, keepdim=True)
        mask = length > 1e-8

        fallback = torch.tensor([0, 0, 1], device=device, dtype=dtype)
        vec = torch.where(mask, vec / length, fallback)

        return vec

    n00 = decode(v00)
    n10 = decode(v10)
    n01 = decode(v01)
    n11 = decode(v11)

    # --------------------------------------------------
    # 4. Bilinear weights
    # --------------------------------------------------
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    w00 = w00.unsqueeze(-1)
    w10 = w10.unsqueeze(-1)
    w01 = w01.unsqueeze(-1)
    w11 = w11.unsqueeze(-1)

    # --------------------------------------------------
    # 5. Vector interpolation (correct domain)
    # --------------------------------------------------
    blended = (
        w00 * n00 +
        w10 * n10 +
        w01 * n01 +
        w11 * n11
    )

    # --------------------------------------------------
    # 6. Renormalize
    # --------------------------------------------------
    length2 = torch.linalg.norm(blended, dim=-1, keepdim=True)
    mask2 = length2 > 1e-8

    fallback = torch.tensor([0, 0, 1], device=device, dtype=dtype)
    blended = torch.where(mask2, blended / length2, fallback)

    # --------------------------------------------------
    # 7. Encode
    # --------------------------------------------------
    r_out = (blended[..., 0:1] + 1.0) * 0.5

    if convention == "OpenGL":
        g_out = (blended[..., 1:2] + 1.0) * 0.5
    else:
        g_out = (1.0 - blended[..., 1:2]) * 0.5

    b_out = (blended[..., 2:3] + 1.0) * 0.5

    out_rgb = torch.cat([r_out, g_out, b_out], dim=-1)
    out_rgb = torch.clamp(out_rgb, 0.0, 1.0)

    # --------------------------------------------------
    # 8. Alpha interpolation
    # --------------------------------------------------
    out_a = w00[..., 0] * a00 + w10[..., 0] * a10 + w01[..., 0] * a01 + w11[..., 0] * a11
    out_a = torch.nan_to_num(out_a, nan=1.0, posinf=1.0, neginf=1.0)

    out = torch.cat([out_rgb, out_a.unsqueeze(-1)], dim=-1)
    return out
#===================================================================================
def merge_normal_maps(base: torch.Tensor, add: torch.Tensor, mask: torch.Tensor, convention: str) -> torch.Tensor:
    if convention not in ("OpenGL", "DirectX"):
        raise ValueError(f"Invalid convention: {convention}")

    B, H, W, C = base.shape
    device = base.device
    dtype = base.dtype

    # ----- Validate & prepare mask -----
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)               # (1, H, W)
    if mask.ndim != 3:
        raise ValueError(f"Mask must be 2D or 3D, got shape {mask.shape}")
    if mask.shape[-2:] != (H, W):
        raise ValueError("Mask spatial size must match base")
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, H, W)
    elif mask.shape[0] != B:
        raise ValueError(f"Mask batch {mask.shape[0]} ≠ base batch {B}")

    # Force to exact 4D (B, H, W, 1)
    mask = mask.reshape(B, H, W, 1).to(dtype=dtype).clamp(0.0, 1.0)

    # ----- Ensure RGBA -----
    if C == 3:
        base = torch.cat([base, torch.ones((B, H, W, 1), device=device, dtype=dtype)], dim=-1)
    if add.shape[-1] == 3:
        add = torch.cat([add, torch.ones((B, H, W, 1), device=device, dtype=dtype)], dim=-1)

    base_rgb = base[..., :3]
    base_a   = base[..., 3:4]
    add_rgb  = add[..., :3]
    add_a    = add[..., 3:4]

    # ----- Decode & normalize -----
    def decode(rgb):
        r = rgb[..., 0:1] * 2.0 - 1.0
        g = rgb[..., 1:2]
        b = rgb[..., 2:3] * 2.0 - 1.0
        if convention == "OpenGL":
            y = g * 2.0 - 1.0
        else:
            y = 1.0 - g * 2.0
        return torch.cat([r, y, b], dim=-1)

    def normalize(v):
        length = torch.linalg.norm(v, dim=-1, keepdim=True)
        safe_len = length.clamp(min=1e-12)
        fallback = v.new_tensor([0.0, 0.0, 1.0])
        normed = v / safe_len
        return torch.where(length > 1e-8, normed, fallback.expand_as(v))

    base_vec = normalize(decode(base_rgb))   # (B, H, W, 3)
    add_vec  = normalize(decode(add_rgb))

    # ----- Axis & angle from add normal -----
    axis_x = -add_vec[..., 1:2]   # (B, H, W, 1)
    axis_y =  add_vec[..., 0:1]
    axis_z = torch.zeros_like(axis_x)
    axis = torch.cat([axis_x, axis_y, axis_z], dim=-1)  # (B, H, W, 3)

    axis_len = torch.linalg.norm(axis, dim=-1, keepdim=True)  # (B, H, W, 1)
    valid_axis = axis_len > 1e-12

    fallback_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    n_axis = torch.where(
        valid_axis,
        axis / axis_len.clamp(min=1e-12),
        fallback_axis.expand_as(axis),
    )

    dot_product = add_vec[..., 2:3].clamp(-1.0, 1.0)   # (B, H, W, 1)
    angle = torch.acos(dot_product) * mask               # (B, H, W, 1)

    # ----- Build active mask -----
    eps = 1e-6
    # Important: ensure valid_axis and angle comparison are both 4D (B,H,W,1)
    active = valid_axis & (angle > eps)   # (B, H, W, 1)

    # Start with base normal
    result_vec = base_vec.clone()

    if active.any():
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        k = 1.0 - cos_a

        ux = n_axis[..., 0:1]
        uy = n_axis[..., 1:2]

        x = base_vec[..., 0:1]
        y = base_vec[..., 1:2]
        z = base_vec[..., 2:3]

        rx = x * (cos_a + ux * ux * k) + y * (ux * uy * k)     + z * (uy * sin_a)
        ry = x * (ux * uy * k)     + y * (cos_a + uy * uy * k) + z * (-ux * sin_a)
        rz = x * (-uy * sin_a)     + y * (ux * sin_a)          + z * cos_a

        rotated = torch.cat([rx, ry, rz], dim=-1)
        rotated = normalize(rotated)

        # Expand active to (B, H, W, 3)
        active_3 = active.expand(-1, -1, -1, 3)   # safe: active is 4D (B,H,W,1) -> (B,H,W,3)
        result_vec = torch.where(active_3, rotated, result_vec)

    # ----- Encode -----
    r_out = (result_vec[..., 0:1] + 1.0) * 0.5
    if convention == "OpenGL":
        g_out = (result_vec[..., 1:2] + 1.0) * 0.5
    else:
        g_out = (1.0 - result_vec[..., 1:2]) * 0.5
    b_out = (result_vec[..., 2:3] + 1.0) * 0.5
    out_rgb = torch.cat([r_out, g_out, b_out], dim=-1).clamp(0.0, 1.0)

    # ----- Alpha -----
    out_a = (1.0 - mask) * base_a + mask * add_a
    out_a = torch.nan_to_num(out_a, nan=1.0, posinf=1.0, neginf=1.0)

    out = torch.cat([out_rgb, out_a], dim=-1)
    return out
#===================================================================================
def height_to_normal(image: torch.Tensor, strength: float, invert_red: bool, invert_green: bool, smoothing: int, use_scharr: bool) -> torch.Tensor:

    B, H, W, C = image.shape
    device = image.device
    dtype = image.dtype

    # --------------------------------------------------
    # 1. Height extraction (linear domain assumed)
    # --------------------------------------------------
    h = image[..., 0:1].permute(0, 3, 1, 2)  # (B,1,H,W)

    # --------------------------------------------------
    # 2. Optional Gaussian smoothing
    # --------------------------------------------------
    if smoothing == 1:
        k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], device=device, dtype=dtype) / 16.0
        k = k.view(1,1,3,3)
        h = F.conv2d(F.pad(h, (1,1,1,1), mode="reflect"), k)
    elif smoothing == 2:
        k = torch.tensor(
            [[1,4,6,4,1],
                [4,16,24,16,4],
                [6,24,36,24,6],
                [4,16,24,16,4],
                [1,4,6,4,1]],
            device=device, dtype=dtype
        ) / 256.0
        k = k.view(1,1,5,5)
        h = F.conv2d(F.pad(h, (2,2,2,2), mode="reflect"), k)

    # --------------------------------------------------
    # 3. Gradient kernels (normalized)
    # --------------------------------------------------
    if use_scharr:
        kx = torch.tensor([[-3,0,3],[-10,0,10],[-3,0,3]], device=device, dtype=dtype) / 32.0
        ky = torch.tensor([[-3,-10,-3],[0,0,0],[3,10,3]], device=device, dtype=dtype) / 32.0
    else:
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype) / 8.0
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype) / 8.0

    kx = kx.view(1,1,3,3)
    ky = ky.view(1,1,3,3)

    dx = F.conv2d(F.pad(h, (1,1,1,1), mode="reflect"), kx)
    dy = F.conv2d(F.pad(h, (1,1,1,1), mode="reflect"), ky)

    dx = dx.squeeze(1)  # (B,H,W)
    dy = dy.squeeze(1)

    # --------------------------------------------------
    # 4. Convert pixel gradients → UV gradients
    # --------------------------------------------------
    dx = dx * W
    dy = dy * H

    # --------------------------------------------------
    # 5. Apply strength to slope (physically meaningful)
    # --------------------------------------------------
    dx = dx * strength
    dy = dy * strength

    # --------------------------------------------------
    # 6. Construct normal: [-∂h/∂u, -∂h/∂v, 1]
    # --------------------------------------------------
    nx = -dx
    ny = -dy
    nz = torch.ones_like(nx)

    # Apply user inversion flags (after definition)
    if invert_red:
        nx = -nx
    if invert_green:
        ny = -ny

    # --------------------------------------------------
    # 7. Normalize safely
    # --------------------------------------------------
    n = torch.stack([nx, ny, nz], dim=-1)
    length = torch.linalg.norm(n, dim=-1, keepdim=True)

    fallback = n.new_tensor([0,0,1])
    n = torch.where(length > 1e-8, n / length, fallback)

    # --------------------------------------------------
    # 8. Encode to tangent-space normal map
    # --------------------------------------------------
    r = (n[..., 0:1] + 1.0) * 0.5

    # Convention: OpenGL (+Y up), DirectX (+Y down)
    if invert_green:
        g = (n[..., 1:2] + 1.0) * 0.5
    else:
        g = (1.0 - n[..., 1:2]) * 0.5

    b = (n[..., 2:3] + 1.0) * 0.5
    a = torch.ones_like(r)

    out = torch.cat([r, g, b, a], dim=-1).clamp(0.0, 1.0)
    return out
#===================================================================================
def smooth_rough_convert(input_data: Any) -> tuple[torch.Tensor]:
    """
    input_data: tensor from ComfyUI – could be IMAGE [B,H,W,C] or MASK [B,H,W].
    Returns (image_out, mask_out).
    """
    # Determine input type based on shape
    is_mask = (input_data.dim() == 3)  # [B, H, W]
    is_image = (input_data.dim() == 4) # [B, H, W, C]

    if not (is_mask or is_image):
        raise ValueError(f"Unsupported input shape: {input_data.shape}. Expected 3D (mask) or 4D (image).")

    # Clamp and invert values
    data_clamped = torch.clamp(input_data, 0.0, 1.0)
    inverted = 1.0 - data_clamped
    inverted = torch.clamp(inverted, 0.0, 1.0)

    # Produce both output formats
    if is_mask:
        # Input is a mask -> output mask (same 3D), and also convert to image (add channel dimension)
        mask_out = inverted                # [B, H, W]
        # Create a grayscale image: repeat the mask across 3 channels
        image_out = mask_out.unsqueeze(-1).repeat(1, 1, 1, 3)  # [B, H, W, 3]
    else:  # is_image
        # Input is an image -> output image (keep channels), and also extract luminance as mask
        image_out = inverted               # [B, H, W, C]
        # Convert to mask by taking the mean across channels (or use red channel? Usually luminance)
        if image_out.shape[-1] == 4:  # RGBA, use alpha or luminance? Luminance is safer
            mask_out = image_out[..., :3].mean(dim=-1)  # [B, H, W]
        else:
            mask_out = image_out.mean(dim=-1)            # [B, H, W]

    return (image_out, mask_out)
#===================================================================================
def mix_channel(red_mask=None, green_mask=None, blue_mask=None, alpha_mask=None)-> torch.Tensor:
    """
    Combines up to 4 channel masks into an image tensor.
    
    Args:
        red_mask: Optional mask tensor for the red channel.
        green_mask: Optional mask tensor for the green channel.
        blue_mask: Optional mask tensor for the blue channel.
        alpha_mask: Optional mask tensor for the alpha channel.
    
    Returns:
        Tuple containing the assembled image tensor (B, H, W, C).
    """
    # 1. Collect and filter valid masks
    mask_channels = {
        "red": red_mask,
        "green": green_mask,
        "blue": blue_mask,
        "alpha": alpha_mask
    }
    valid_masks = {channel: mask for channel, mask in mask_channels.items() if mask is not None}

    # 2. Validate: at least one mask must be provided
    if not valid_masks:
        raise ValueError("At least one mask (red, green, blue, or alpha) must be provided.")

    # 3. Determine reference dimensions from the first valid mask
    ref_mask = next(iter(valid_masks.values()))
    ref_shape = ref_mask.shape  # Expected: (B, H, W)
    if ref_mask.dim() == 2:
        ref_shape = (1, *ref_shape)  # Add batch dimension if missing
    batch_size, height, width = ref_shape

    # 4. Validate that all provided masks have matching dimensions
    for channel, mask in valid_masks.items():
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add batch dim for comparison
        if mask.shape != ref_shape:
            raise ValueError(
                f"Mask '{channel}' has shape {mask.shape}, but expected {ref_shape} "
                f"(matching the first valid mask). All masks must have identical (B, H, W) dimensions."
            )

    # 5. Determine output channels: RGBA if alpha is present, else RGB
    has_alpha = "alpha" in valid_masks
    output_channels = 4 if has_alpha else 3

    # 6. Create a black canvas (all zeros)
    output = torch.zeros((batch_size, height, width, output_channels),
                            dtype=torch.float32,
                            device=ref_mask.device)

    # 7. Fill channels using the provided masks (or keep them black)
    for i, channel_name in enumerate(["red", "green", "blue", "alpha"]):
        if channel_name in valid_masks:
            mask = valid_masks[channel_name]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # Ensure shape is (B, H, W)
            # Add channel dimension for assignment: (B, H, W) -> (B, H, W, 1)
            mask_4d = mask.unsqueeze(-1)
            output[:, :, :, i] = mask_4d.squeeze(-1)  # Align shapes correctly

    # Optional: If no alpha but an RGB image was assembled, clip values to [0,1]
    output = torch.clamp(output, 0.0, 1.0)

    return output
#===================================================================================  
def split_channel(image: torch.Tensor)-> tuple[torch.Tensor]:
    # Image tensor shape: (B, H, W, C)
    if image.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B, H, W, C), got {image.ndim}D")

    c = image.shape[-1]
    if c not in (3, 4):
        raise ValueError(
            f"Image must have 3 (RGB) or 4 (RGBA) channels, got {c} channels"
        )

    # Extract R, G, B (always present)
    r = image[..., 0]   # (B, H, W)
    g = image[..., 1]
    b = image[..., 2]

    # Alpha channel: use existing if available, otherwise create white mask
    if c == 4:
        a = image[..., 3]
    else:
        a = torch.ones_like(r)

    # Masks are expected to be float in [0,1]; image values are already in that range
    return (r, g, b, a)
#===================================================================================
def split_alpha_chan(image: torch.Tensor)-> tuple[torch.Tensor]:
    rgb = image[..., :3]
    
    B, H, W, C = image.shape
    if C >= 4:
        alpha = image[..., 3]
    else:
        alpha = torch.ones((B, H, W), device=image.device, dtype=image.dtype)
            
    return (rgb, alpha)