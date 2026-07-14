
import os
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import folder_paths  # ComfyUI utility
import librosa
import soundfile as sf

#===================================================================================
def process_and_save_frames(images, video_info, max_image_size, max_frame_size,
                            fps_target, side_target, output_folder):
    """
    Standalone function that performs all the work:
    sampling, resizing, grid creation, saving, and returning results.
    """
    # 1. Extract frame sampling info from video_info
    source_fps = video_info.get('source_fps', 30.0)
    loaded_frame_count = video_info.get('loaded_frame_count', len(images))

    if fps_target < source_fps:
        sample_ratio = source_fps / fps_target
        sample_indices = [int(i * sample_ratio) for i in range(int(loaded_frame_count / sample_ratio))]
    else:
        sample_indices = list(range(loaded_frame_count))

    sample_indices = [i for i in sample_indices if i < loaded_frame_count]
    if len(sample_indices) == 0:
        sample_indices = list(range(min(loaded_frame_count, 1)))

    sampled_frames = [images[i].clone() for i in sample_indices]
    print(f"Sampled {len(sampled_frames)} frames from {loaded_frame_count} total at {fps_target} FPS")

    # 2. Resize frames to max_frame_size (nearest-neighbor)
    resized_frames = []
    for frame in sampled_frames:
        h, w = frame.shape[1], frame.shape[0]  # B,H,W,C order
        if max(h, w) > max_frame_size:
            if h > w:
                new_h = max_frame_size
                new_w = int(w * max_frame_size / h)
            else:
                new_w = max_frame_size
                new_h = int(h * max_frame_size / w)

            # Resize with nearest-exact interpolation
            frame_resized = torch.nn.functional.interpolate(
                frame.permute(2, 0, 1).unsqueeze(0),
                size=(new_h, new_w),
                mode='nearest-exact'
            ).squeeze(0).permute(1, 2, 0)
            resized_frames.append(frame_resized)
        else:
            resized_frames.append(frame)

    # 3. Group frames into grids
    grids = _create_grids(resized_frames, max_image_size, max_frame_size)

    # 4. Save grids
    output_dir = _get_output_path(output_folder)
    saved_paths = []

    for i, grid in enumerate(grids):
        grid_np = (grid.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(grid_np, 'RGB')

        rows, cols = _get_grid_dimensions(len(sampled_frames), max_image_size, max_frame_size)
        duration = len(sampled_frames) / fps_target if fps_target > 0 else 0

        filename = f"img_{i}_{side_target}_{fps_target}_{duration:.4f}_{len(sampled_frames)}_{rows}_{cols}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, 'PNG')
        saved_paths.append(filepath)
        print(f"Saved grid {i+1}/{len(grids)}: {filename}")

    if len(grids) > 0:
        return (torch.stack(grids), len(grids))
    else:
        return (torch.zeros((0, 64, 64, 3)), 0)

#===================================================================================
def _create_grids(frames, max_canvas_size, max_frame_size):
    """Grid creation helper – moved out of class."""
    total_frames = len(frames)
    if total_frames == 0:
        return []

    frame_h, frame_w = frames[0].shape[1], frames[0].shape[0]

    frames_per_row = max(1, min(total_frames, max_canvas_size // frame_w))
    frames_per_col = max(1, min((total_frames + frames_per_row - 1) // frames_per_row,
                                max_canvas_size // frame_h))

    if frames_per_row * frame_w > max_canvas_size:
        frames_per_row = max_canvas_size // frame_w
    if frames_per_col * frame_h > max_canvas_size:
        frames_per_col = max_canvas_size // frame_h

    canvas_width = frames_per_row * frame_w
    canvas_height = frames_per_col * frame_h
    frames_per_canvas = frames_per_row * frames_per_col
    num_canvases = (total_frames + frames_per_canvas - 1) // frames_per_canvas

    grids = []
    for canvas_idx in range(num_canvases):
        start_idx = canvas_idx * frames_per_canvas
        end_idx = min(start_idx + frames_per_canvas, total_frames)

        canvas = torch.zeros((canvas_height, canvas_width, 3), dtype=torch.float32)

        for i, frame_idx in enumerate(range(start_idx, end_idx)):
            row = i // frames_per_row
            col = i % frames_per_row
            y = row * frame_h
            x = col * frame_w
            if y + frame_h <= canvas_height and x + frame_w <= canvas_width:
                canvas[y:y+frame_h, x:x+frame_w, :] = frames[frame_idx]

        # Resize to power-of-two if it fits within max size
        canvas_pil = Image.fromarray((canvas.cpu().numpy() * 255).astype(np.uint8), 'RGB')
        power_w = 2 ** ((canvas_width - 1).bit_length())
        power_h = 2 ** ((canvas_height - 1).bit_length())

        if power_w <= max_canvas_size and power_h <= max_canvas_size:
            canvas_pil = canvas_pil.resize((power_w, power_h), Image.NEAREST)

        canvas_tensor = torch.from_numpy(np.array(canvas_pil).astype(np.float32) / 255.0)
        grids.append(canvas_tensor)

    return grids

#===================================================================================
def _get_grid_dimensions(total_frames, max_canvas_size, max_frame_size):
    """Simple rows/cols calculation."""
    frames_per_row = min(total_frames, max_canvas_size // max_frame_size)
    frames_per_col = (total_frames + frames_per_row - 1) // frames_per_row
    return frames_per_col, frames_per_row

#===================================================================================
def _get_output_path(output_folder):
    """Determine and create output directory."""
    if output_folder and output_folder.strip():
        output_dir = os.path.join(folder_paths.get_output_directory(), output_folder.strip())
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(folder_paths.get_output_directory(), f"media_to_prim_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

#===================================================================================
def segment_and_save_audio(
    audio,
    project_name,
    slice_length,
    mode,
    channel_mode,
    format,
    sample_rate,
    start_time=0.0,
    end_time=-1.0,
):
    """Extract, slice, resample, and save audio segments."""

    # --- Helper: extract waveform and sample rate from ComfyUI audio dict ---
    def _get_audio_data(audio_input):
        if isinstance(audio_input, dict):
            waveform = audio_input.get("waveform")
            sr = audio_input.get("sample_rate")
            if waveform is not None and sr is not None:
                waveform_np = waveform.squeeze(0).cpu().numpy()  # shape (C, N)
                return waveform_np, sr
        raise ValueError("Audio input must be a dict with 'waveform' and 'sample_rate' keys.")

    # --- Helper: resample 1D or 2D data ---
    def _resample(data, orig_sr, target_sr):
        if data.ndim == 1:
            return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        else:
            # data shape (channels, samples)
            new_len = int(data.shape[1] * target_sr / orig_sr)
            resampled = np.zeros((data.shape[0], new_len))
            for ch in range(data.shape[0]):
                resampled[ch] = librosa.resample(
                    data[ch], orig_sr=orig_sr, target_sr=target_sr
                )
            return resampled

    # 1. Extract waveform (channels × samples)
    waveform, original_sr = _get_audio_data(audio)
    duration_seconds = waveform.shape[1] / original_sr

    # 2. Convert to target channel configuration
    if mode == "mono":
        if waveform.shape[0] == 1:
            mono = waveform[0]
        else:
            if channel_mode == "average":
                mono = np.mean(waveform, axis=0)
            elif channel_mode == "left_only":
                mono = waveform[0]
            elif channel_mode == "right_only":
                mono = waveform[1] if waveform.shape[0] > 1 else waveform[0]
            else:
                mono = np.mean(waveform, axis=0)
        data = mono  # 1D
        is_stereo = False
    else:  # stereo
        if waveform.shape[0] == 1:
            data = np.vstack([waveform[0], waveform[0]])
        elif waveform.shape[0] >= 2:
            data = waveform[:2]
        else:
            data = waveform
        is_stereo = True

    # 3. Apply start/end time slicing (convert minutes to seconds)
    start_seconds = start_time * 60.0
    end_seconds = end_time * 60.0 if end_time > 0 else duration_seconds
    if start_seconds >= end_seconds or start_seconds >= duration_seconds:
        print("[mdAudioSegmenterSaver] Invalid time range. No segments saved.")
        return ()

    start_sample = int(start_seconds * original_sr)
    end_sample = int(end_seconds * original_sr)

    if is_stereo:
        sliced = data[:, start_sample:end_sample]
    else:
        sliced = data[start_sample:end_sample]

    # 4. Resample to target sample rate
    if original_sr != sample_rate:
        sliced = _resample(sliced, original_sr, sample_rate)
        final_sr = sample_rate
    else:
        final_sr = original_sr

    # 5. Calculate number of segments
    slice_samples = slice_length * final_sr
    total_samples = sliced.shape[-1] if is_stereo else len(sliced)
    num_segments = int(np.ceil(total_samples / slice_samples))

    if num_segments == 0:
        print("[mdAudioSegmenterSaver] No segments to save.")
        return ()

    # 6. Save segments
    base_output_dir = folder_paths.get_output_directory()
    save_dir = os.path.join(base_output_dir, "audio_segments", project_name)
    os.makedirs(save_dir, exist_ok=True)

    saved_files = []
    for i in range(num_segments):
        start_idx = i * slice_samples
        end_idx = min(start_idx + slice_samples, total_samples)

        if is_stereo:
            segment = sliced[:, start_idx:end_idx]  # shape (2, seg_len)
            if segment.shape[1] < slice_samples:
                padded = np.zeros((2, slice_samples))
                padded[:, : segment.shape[1]] = segment
                segment = padded
            # soundfile expects (frames, channels)
            segment = segment.T
        else:
            segment = sliced[start_idx:end_idx]
            if len(segment) < slice_samples:
                padded = np.zeros(slice_samples)
                padded[: len(segment)] = segment
                segment = padded

        filename = f"segment_{i+1:03d}.{format}"
        filepath = os.path.join(save_dir, filename)

        try:
            sf.write(filepath, segment, final_sr, format=format.upper())
            saved_files.append(filepath)
            print(f"[mdAudioSegmenterSaver] Saved: {filepath}")
        except Exception as e:
            print(f"[mdAudioSegmenterSaver] Error saving segment {i+1}: {e}")

    print(
        f"[mdAudioSegmenterSaver] Successfully saved {len(saved_files)} segments to: {save_dir}"
    )
    return ()
























