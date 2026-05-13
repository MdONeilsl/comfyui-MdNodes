
import torch

from typing import Tuple
#===================================================================================


def tensor_shape_info(image: torch.Tensor) -> Tuple[str]:
    info_lines = []

    if image is None:
        info_lines.append("No input connected")
    elif not torch.is_tensor(image):
        info_lines.append(f"Input type: {type(image)} (not a tensor)")
    else:
        shape = image.shape
        dtype = image.dtype
        device = image.device

        info_lines.append(f"Batch shape: {list(shape)}")
        info_lines.append(f"Dtype: {dtype}")
        info_lines.append(f"Device: {device}")
        if image.numel() > 0:
            info_lines.append(
                f"Global min: {image.min().item():.4f}  max: {image.max().item():.4f}  mean: {image.mean().item():.4f}"
            )

        # Per‑sample information
        batch_size = shape[0]
        for i in range(batch_size):
            sample = image[i]
            if sample.numel() == 0:
                info_lines.append(f"Sample {i}: empty")
                continue
            s_shape = list(sample.shape)
            s_min = sample.min().item()
            s_max = sample.max().item()
            s_mean = sample.mean().item()
            info_lines.append(
                f"Sample {i}: shape {s_shape}  min {s_min:.4f}  max {s_max:.4f}  mean {s_mean:.4f}"
            )

    return info_lines 