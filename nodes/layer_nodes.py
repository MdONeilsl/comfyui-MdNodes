import torch

module_cat = "md/layer"

#===================================================================================
class mdLayerStackNode:
    """
    Composites a layer over a background using a blend mode, a mask, and a global transparency.
    Handles RGBA images (4 channels) and respects the layer's own alpha channel.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "layer": ("IMAGE",),
                "blending_mode": (["Normal", "Screen", "Multiply", "Overlay", "Difference",
                                  "Dissolve", "Clear", "Add", "Subtract", "Replace", 
                                  "Darken Only", "Lighten Only", "Color Dodge", "Color Burn", 
                                  "Hard Light", "Soft Light", "Pin Light", "Linear Dodge (Add)", 
                                  "Linear Burn (Multiply)", "Vivid Light", "Halo", "Negate", 
                                  "Merge RGB", "Merge Cyan", "Merge Magenta", "Merge Yellow", 
                                  "Merge Black", "Fill with FG Color", "Fill with BG Color", 
                                  "Average", "Composite", "Behind", "Template"],),
                "transparency": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = module_cat

    # ========== BLENDING MODES (RGB only) ==========
    
    @staticmethod
    def blend_normal(bg_rgb, fg_rgb):
        return fg_rgb

    @staticmethod
    def blend_screen(bg_rgb, fg_rgb):
        return 1.0 - (1.0 - bg_rgb) * (1.0 - fg_rgb)

    @staticmethod
    def blend_multiply(bg_rgb, fg_rgb):
        return bg_rgb * fg_rgb

    @staticmethod
    def blend_overlay(bg_rgb, fg_rgb):
        result = torch.where(bg_rgb <= 0.5, 2.0 * bg_rgb * fg_rgb,
                             1.0 - 2.0 * (1.0 - bg_rgb) * (1.0 - fg_rgb))
        return result.clamp(0.0, 1.0)

    @staticmethod
    def blend_difference(bg_rgb, fg_rgb):
        return torch.abs(bg_rgb - fg_rgb)

    @staticmethod
    def blend_dissolve(bg_rgb, fg_rgb):
        # Stochastic: uses random noise per pixel
        noise = torch.rand_like(bg_rgb)
        return torch.where(noise < 0.5, fg_rgb, bg_rgb)

    @staticmethod
    def blend_clear(bg_rgb, fg_rgb):
        # Returns background (the layer becomes transparent)
        return bg_rgb

    @staticmethod
    def blend_add(bg_rgb, fg_rgb):
        return torch.clamp(bg_rgb + fg_rgb, 0.0, 1.0)

    @staticmethod
    def blend_subtract(bg_rgb, fg_rgb):
        return torch.clamp(bg_rgb - fg_rgb, 0.0, 1.0)

    @staticmethod
    def blend_replace(bg_rgb, fg_rgb):
        return fg_rgb

    @staticmethod
    def blend_darken_only(bg_rgb, fg_rgb):
        return torch.min(bg_rgb, fg_rgb)

    @staticmethod
    def blend_lighten_only(bg_rgb, fg_rgb):
        return torch.max(bg_rgb, fg_rgb)

    @staticmethod
    def blend_color_dodge(bg_rgb, fg_rgb):
        result = bg_rgb / (1.0 - fg_rgb + 1e-8)
        return result.clamp(0.0, 1.0)

    @staticmethod
    def blend_color_burn(bg_rgb, fg_rgb):
        result = 1.0 - (1.0 - bg_rgb) / (fg_rgb + 1e-8)
        return result.clamp(0.0, 1.0)

    @staticmethod
    def blend_hard_light(bg_rgb, fg_rgb):
        result = torch.where(fg_rgb <= 0.5, 2.0 * bg_rgb * fg_rgb,
                             1.0 - 2.0 * (1.0 - bg_rgb) * (1.0 - fg_rgb))
        return result.clamp(0.0, 1.0)

    @staticmethod
    def blend_soft_light(bg_rgb, fg_rgb):
        result = torch.where(fg_rgb <= 0.5,
                             (2.0 * bg_rgb * fg_rgb) + (bg_rgb * bg_rgb * (1.0 - 2.0 * fg_rgb)),
                             2.0 * bg_rgb * (1.0 - fg_rgb) + torch.sqrt(bg_rgb) * (2.0 * fg_rgb - 1.0))
        return result.clamp(0.0, 1.0)

    @staticmethod
    def blend_pin_light(bg_rgb, fg_rgb):
        result = torch.where(fg_rgb <= 0.5,
                             torch.min(bg_rgb, 2.0 * fg_rgb),
                             torch.max(bg_rgb, 2.0 * fg_rgb - 1.0))
        return result.clamp(0.0, 1.0)

    @staticmethod
    def blend_linear_dodge_add(bg_rgb, fg_rgb):
        return torch.clamp(bg_rgb + fg_rgb, 0.0, 1.0)

    @staticmethod
    def blend_linear_burn_multiply(bg_rgb, fg_rgb):
        return torch.clamp(bg_rgb + fg_rgb - 1.0, 0.0, 1.0)

    @staticmethod
    def blend_vivid_light(bg_rgb, fg_rgb):
        result = torch.where(fg_rgb <= 0.5,
                             1.0 - (1.0 - bg_rgb) / (2.0 * fg_rgb + 1e-8),
                             bg_rgb / (2.0 * (1.0 - fg_rgb) + 1e-8))
        return result.clamp(0.0, 1.0)

    @staticmethod
    def blend_halo(bg_rgb, fg_rgb):
        edge = torch.abs(bg_rgb - fg_rgb)
        glow = (bg_rgb + fg_rgb) * edge
        return glow.clamp(0.0, 1.0)

    @staticmethod
    def blend_negate(bg_rgb, fg_rgb):
        return 1.0 - torch.abs(bg_rgb - fg_rgb)

    @staticmethod
    def blend_merge_rgb(bg_rgb, fg_rgb):
        return fg_rgb

    @staticmethod
    def blend_merge_cyan(bg_rgb, fg_rgb):
        # Cyan = 1 - Red (simplified)
        cyan = 1.0 - fg_rgb[..., 0:1]
        return cyan.repeat(1, 1, 1, 3)

    @staticmethod
    def blend_merge_magenta(bg_rgb, fg_rgb):
        magenta = 1.0 - fg_rgb[..., 1:2]
        return magenta.repeat(1, 1, 1, 3)

    @staticmethod
    def blend_merge_yellow(bg_rgb, fg_rgb):
        yellow = 1.0 - fg_rgb[..., 2:3]
        return yellow.repeat(1, 1, 1, 3)

    @staticmethod
    def blend_merge_black(bg_rgb, fg_rgb):
        intensity = fg_rgb.mean(dim=-1, keepdim=True)
        return intensity.repeat(1, 1, 1, 3)

    @staticmethod
    def blend_fill_fg_color(bg_rgb, fg_rgb):
        return fg_rgb

    @staticmethod
    def blend_fill_bg_color(bg_rgb, fg_rgb):
        return bg_rgb

    @staticmethod
    def blend_average(bg_rgb, fg_rgb):
        return (bg_rgb + fg_rgb) * 0.5

    @staticmethod
    def blend_composite(bg_rgb, fg_rgb):
        # Standard alpha compositing would be done later, here just return fg
        return fg_rgb

    @staticmethod
    def blend_behind(bg_rgb, fg_rgb):
        # This mode will be handled specially in the main function (paint only on transparent areas)
        return fg_rgb

    @staticmethod
    def blend_template(bg_rgb, fg_rgb):
        return fg_rgb

    # ========== MAIN COMPOSITING ==========

    def composite(self, background, layer, blending_mode, transparency, mask=None):
        """
        background, layer: tensors of shape [B, H, W, C] where C=3 or 4
        mask: optional tensor [B, H, W] or [H, W]; values in [0,1]
        transparency: global opacity multiplier
        """
        device = background.device
        layer = layer.to(device)
        bg_h, bg_w = background.shape[1:3]

        # ---------- 1. Ensure background and layer have 4 channels (add alpha if missing) ----------
        bg = background.clone()
        if bg.shape[-1] == 3:
            alpha_bg = torch.ones((bg.shape[0], bg_h, bg_w, 1), device=device)
            bg = torch.cat([bg, alpha_bg], dim=-1)
        else:
            alpha_bg = bg[..., 3:4]

        fg = layer.clone()
        if fg.shape[-1] == 3:
            alpha_fg = torch.ones((fg.shape[0], fg.shape[1], fg.shape[2], 1), device=device)
            fg = torch.cat([fg, alpha_fg], dim=-1)
        else:
            alpha_fg = fg[..., 3:4]

        bg_rgb = bg[..., :3]
        fg_rgb = fg[..., :3]

        # ---------- 2. Resize layer to match background dimensions ----------
        if fg.shape[1:3] != (bg_h, bg_w):
            fg_rgb = torch.nn.functional.interpolate(
                fg_rgb.permute(0, 3, 1, 2), size=(bg_h, bg_w), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)
            alpha_fg = torch.nn.functional.interpolate(
                alpha_fg.permute(0, 3, 1, 2), size=(bg_h, bg_w), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)

        # ---------- 3. Prepare mask (if None, use 1.0) ----------
        if mask is None:
            mask = torch.ones((bg.shape[0], bg_h, bg_w, 1), device=device)
        else:
            mask = mask.to(device)
            # Resize mask to match background dimensions
            if mask.dim() == 2:          # (H, W)
                mask = mask.unsqueeze(0).unsqueeze(-1)
            elif mask.dim() == 3:        # (B, H, W) or (B, H, W, 1)
                if mask.shape[-1] != 1:
                    mask = mask.unsqueeze(-1)
            # Interpolate
            if mask.shape[1:3] != (bg_h, bg_w):
                mask = torch.nn.functional.interpolate(
                    mask.permute(0, 3, 1, 2), size=(bg_h, bg_w), mode='bilinear', align_corners=False
                ).permute(0, 2, 3, 1)
            # Expand batch dimension if needed
            if mask.shape[0] < bg.shape[0]:
                mask = mask.expand(bg.shape[0], -1, -1, -1)

        # ---------- 4. Compute final opacity factor ----------
        # opacity = layer_alpha * mask * transparency
        opacity = alpha_fg * mask * transparency   # shape [B, H, W, 1]

        # Special handling for "Clear" and "Behind" blending modes
        if blending_mode.lower() == "clear":
            # Clear makes the layer fully transparent, so opacity = 0
            opacity = torch.zeros_like(opacity)
        elif blending_mode.lower() == "behind":
            # Behind: only apply where background is transparent (alpha_bg < 1)
            # Multiply opacity by (1 - alpha_bg)
            opacity = opacity * (1.0 - alpha_bg)

        # ---------- 5. Apply RGB blend mode ----------
        blend_func_name = blending_mode.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        blend_func = getattr(self, f"blend_{blend_func_name}", None)
        if blend_func is None:
            print(f"Warning: Unknown blend mode '{blending_mode}', using Normal.")
            blend_func = self.blend_normal

        blended_rgb = blend_func(bg_rgb, fg_rgb)

        # ---------- 6. Composite RGB using opacity ----------
        result_rgb = bg_rgb * (1.0 - opacity) + blended_rgb * opacity

        # ---------- 7. Compute result alpha (standard over compositing) ----------
        result_alpha = alpha_bg + (1.0 - alpha_bg) * opacity

        # ---------- 8. Concatenate and clamp ----------
        result = torch.cat([result_rgb, result_alpha], dim=-1).clamp(0.0, 1.0)

        return (result,)
    
#===================================================================================
NODE_CLASS_MAPPINGS = {
    "mdLayerStackNode": mdLayerStackNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdLayerStackNode": "Layer Stack Node (MD)",
}

