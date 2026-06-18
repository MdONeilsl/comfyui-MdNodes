
STANDARD_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

#===================================================================================
def nearest_standard(value, rounding_type="closest"):
    """Return a standard size according to the specified rounding type.
    
    Args:
        value: The input size to snap.
        rounding_type: One of "closest", "ceiling", "flooring".
    
    Returns:
        The snapped standard size.
    """
    # Clamp to the valid range for consistent behavior across all types
    min_size = STANDARD_SIZES[0]
    max_size = STANDARD_SIZES[-1]
    value = max(min_size, min(max_size, value))

    if rounding_type == "closest":
        # Original behavior: minimum absolute difference
        return min(STANDARD_SIZES, key=lambda x: abs(x - value))
    
    elif rounding_type == "ceiling":
        # Smallest standard size >= value
        for size in STANDARD_SIZES:
            if size >= value:
                return size
        return max_size  # should not happen due to clamping
    
    elif rounding_type == "flooring":
        # Largest standard size <= value
        for size in reversed(STANDARD_SIZES):
            if size <= value:
                return size
        return min_size  # should not happen due to clamping
    
    else:
        raise ValueError(f"Invalid rounding_type: {rounding_type}. Choose from 'closest', 'ceiling', 'flooring'.")

#===================================================================================
def scale_mat_dimensions(width, height, scale, rounding_type="closest"):
    """Scale mat dimensions and snap to standard sizes using the specified rounding type.
    
    Args:
        width, height: Original dimensions.
        scale: One of "1", "½", "¼", "⅛".
        rounding_type: Passed to nearest_standard().
    
    Returns:
        Tuple of snapped (new_width, new_height).
    """
    factor_map = {"1": 1, "½": 0.5, "¼": 0.25, "⅛": 0.125}
    factor = factor_map[scale]
    
    # Scale using standard rounding (half up)
    scaled_w = int(round(width * factor))
    scaled_h = int(round(height * factor))
    
    # Snap each dimension using the chosen rounding type
    out_w = nearest_standard(scaled_w, rounding_type)
    out_h = nearest_standard(scaled_h, rounding_type)
    
    return (out_w, out_h)