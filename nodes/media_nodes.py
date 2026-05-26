import torch
import numpy as np
import folder_paths
import os
from PIL import Image
from datetime import datetime
import librosa
import soundfile as sf

module_cat = "md/media"

#===================================================================================
class mdMediaToPrimGrid:
    """
    ComfyUI custom node that processes video frames into grid images for Second Life.
    Replicates the functionality of the original Media to Prim web application.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "video_info": ("VHS_VIDEOINFO",),
                "max_image_size": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 64}),
                "max_frame_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "fps_target": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1}),
                "side_target": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "output_folder": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("grid_images", "image_count")
    FUNCTION = "process_and_save"
    CATEGORY = module_cat
    
    def process_and_save(self, images, video_info, max_image_size, max_frame_size, fps_target, side_target, output_folder):
        """
        Main processing function that samples frames, creates grids, and saves images.
        """
        # 1. Extract frame sampling info from video_info
        source_fps = video_info.get('source_fps', 30.0)
        loaded_frame_count = video_info.get('loaded_frame_count', len(images))
        
        # Calculate frame sampling indices based on target FPS
        if fps_target < source_fps:
            # Sample every Nth frame to achieve target FPS
            sample_ratio = source_fps / fps_target
            sample_indices = [int(i * sample_ratio) for i in range(int(loaded_frame_count / sample_ratio))]
        else:
            # Use all frames (FPS target higher than source)
            sample_indices = list(range(loaded_frame_count))
        
        # Ensure we don't exceed available frames
        sample_indices = [i for i in sample_indices if i < loaded_frame_count]
        
        if len(sample_indices) == 0:
            sample_indices = list(range(min(loaded_frame_count, 1)))
        
        # Extract sampled frames
        sampled_frames = [images[i].clone() for i in sample_indices]
        print(f"Sampled {len(sampled_frames)} frames from {loaded_frame_count} total at {fps_target} FPS")
        
        # 2. Resize frames to max_frame_size using nearest-neighbor
        resized_frames = []
        for frame in sampled_frames:
            h, w = frame.shape[1], frame.shape[0]  # ComfyUI tensor is B,H,W,C
            if max(h, w) > max_frame_size:
                # Calculate new dimensions preserving aspect ratio
                if h > w:
                    new_h = max_frame_size
                    new_w = int(w * max_frame_size / h)
                else:
                    new_w = max_frame_size
                    new_h = int(h * max_frame_size / w)
                
                # Resize with nearest-neighbor interpolation
                frame_resized = torch.nn.functional.interpolate(
                    frame.permute(2, 0, 1).unsqueeze(0),
                    size=(new_h, new_w),
                    mode='nearest-exact'
                ).squeeze(0).permute(1, 2, 0)
                resized_frames.append(frame_resized)
            else:
                resized_frames.append(frame)
        
        # 3. Group frames into grids
        grids = self._create_grids(resized_frames, max_image_size, max_frame_size)
        
        # 4. Save grids with naming convention
        output_dir = self._get_output_path(output_folder)
        saved_paths = []
        
        for i, grid in enumerate(grids):
            # Generate filename following original app convention
            # Format: img_{index}_{side}_{fps}_{duration}_{frames}_{rows}_{cols}.png
            grid_np = (grid.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(grid_np, 'RGB')
            
            # Calculate grid dimensions for filename
            rows, cols = self._get_grid_dimensions(len(sampled_frames), max_image_size, max_frame_size)
            duration = len(sampled_frames) / fps_target if fps_target > 0 else 0
            
            filename = f"img_{i}_{side_target}_{fps_target}_{duration:.4f}_{len(sampled_frames)}_{rows}_{cols}.png"
            filepath = os.path.join(output_dir, filename)
            
            img.save(filepath, 'PNG')
            saved_paths.append(filepath)
            print(f"Saved grid {i+1}/{len(grids)}: {filename}")
        
        # Return as batch for potential workflow continuation
        if len(grids) > 0:
            return (torch.stack(grids), len(grids))
        else:
            return (torch.zeros((0, 64, 64, 3)), 0)
    
    def _create_grids(self, frames, max_canvas_size, max_frame_size):
        """
        Create grid layouts similar to original app's draw_frames function.
        """
        total_frames = len(frames)
        if total_frames == 0:
            return []
        
        # Get frame dimensions (assuming all frames same size after resizing)
        frame_h, frame_w = frames[0].shape[1], frames[0].shape[0]
        
        # Calculate frames per row/col based on max_canvas_size
        frames_per_row = max(1, min(total_frames, max_canvas_size // frame_w))
        frames_per_col = max(1, min((total_frames + frames_per_row - 1) // frames_per_row, 
                                    max_canvas_size // frame_h))
        
        # Adjust if canvas would exceed max size
        if frames_per_row * frame_w > max_canvas_size:
            frames_per_row = max_canvas_size // frame_w
        if frames_per_col * frame_h > max_canvas_size:
            frames_per_col = max_canvas_size // frame_h
        
        # Calculate actual canvas dimensions
        canvas_width = frames_per_row * frame_w
        canvas_height = frames_per_col * frame_h
        
        frames_per_canvas = frames_per_row * frames_per_col
        num_canvases = (total_frames + frames_per_canvas - 1) // frames_per_canvas
        
        grids = []
        
        for canvas_idx in range(num_canvases):
            start_idx = canvas_idx * frames_per_canvas
            end_idx = min(start_idx + frames_per_canvas, total_frames)
            frames_in_canvas = end_idx - start_idx
            
            # Create canvas
            canvas = torch.zeros((canvas_height, canvas_width, 3), dtype=torch.float32)
            
            # Place frames on canvas
            for i, frame_idx in enumerate(range(start_idx, end_idx)):
                row = i // frames_per_row
                col = i % frames_per_row
                y = row * frame_h
                x = col * frame_w
                
                # Ensure frame fits within canvas bounds
                if y + frame_h <= canvas_height and x + frame_w <= canvas_width:
                    canvas[y:y+frame_h, x:x+frame_w, :] = frames[frame_idx]
            
            # Resize canvas to nearest power of two (if needed for SL compatibility)
            canvas_pil = Image.fromarray((canvas.cpu().numpy() * 255).astype(np.uint8), 'RGB')
            
            # Calculate nearest power of two dimensions
            power_w = 2 ** ((canvas_width - 1).bit_length())
            power_h = 2 ** ((canvas_height - 1).bit_length())
            
            if power_w <= max_canvas_size and power_h <= max_canvas_size:
                canvas_resized = canvas_pil.resize((power_w, power_h), Image.NEAREST)
            else:
                canvas_resized = canvas_pil
            
            # Convert back to tensor
            canvas_tensor = torch.from_numpy(np.array(canvas_resized).astype(np.float32) / 255.0)
            grids.append(canvas_tensor)
        
        return grids
    
    def _get_grid_dimensions(self, total_frames, max_canvas_size, max_frame_size):
        """
        Calculate optimal rows and columns for grid layout.
        """
        # Simplified calculation (actual implementation would match original app's logic)
        frames_per_row = min(total_frames, max_canvas_size // max_frame_size)
        frames_per_col = (total_frames + frames_per_row - 1) // frames_per_row
        return frames_per_col, frames_per_row
    
    def _get_output_path(self, output_folder):
        """
        Get output directory path, creating if necessary.
        """
        if output_folder and output_folder.strip():
            # Use custom folder path
            output_dir = os.path.join(folder_paths.get_output_directory(), output_folder.strip())
        else:
            # Use default output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(folder_paths.get_output_directory(), f"media_to_prim_{timestamp}")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

#===================================================================================
class mdAudioSegmenterSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "project_name": ("STRING", {"default": "audio_segments", "multiline": False}),
                "slice_length": ("INT", {"default": 30, "min": 1, "max": 3600, "step": 1}),
                "mode": (["mono", "stereo"], {"default": "mono"}),
                "channel_mode": (["average", "left_only", "right_only"], {"default": "average"}),
                "format": (["wav", "mp3", "flac"], {"default": "wav"}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1000}),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": -1.0, "min": -1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "segment_and_save"
    CATEGORY = module_cat
    OUTPUT_NODE = True

    def get_audio_data(self, audio):
        """
        Extracts audio waveform (numpy, channels × samples) and sample rate.
        """
        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            if waveform is not None and sample_rate is not None:
                # Convert torch tensor to numpy, remove batch dimension if present
                waveform_np = waveform.squeeze(0).cpu().numpy()
                # waveform_np shape is (channels, samples)
                return waveform_np, sample_rate
        raise ValueError("Audio input must be a dict with 'waveform' and 'sample_rate' keys.")

    def resample_audio(self, data, orig_sr, target_sr):
        """Resample mono (1D) or stereo (2D) audio."""
        if data.ndim == 1:
            return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        else:
            # data shape: (channels, samples)
            resampled = np.zeros((data.shape[0], int(data.shape[1] * target_sr / orig_sr)))
            for ch in range(data.shape[0]):
                resampled[ch] = librosa.resample(data[ch], orig_sr=orig_sr, target_sr=target_sr)
            return resampled

    def segment_and_save(self, audio, project_name, slice_length, mode, channel_mode,
                         format, sample_rate, start_time=0.0, end_time=-1.0):
        # --- 1. Extract waveform (channels × samples) ---
        waveform, original_sr = self.get_audio_data(audio)  # shape (C, N)
        duration_seconds = waveform.shape[1] / original_sr

        # --- 2. Convert to target channel configuration ---
        if mode == "mono":
            # Downmix to mono using channel_mode
            if waveform.shape[0] == 1:
                mono = waveform[0]  # already mono
            else:
                if channel_mode == "average":
                    mono = np.mean(waveform, axis=0)
                elif channel_mode == "left_only":
                    mono = waveform[0]
                elif channel_mode == "right_only":
                    mono = waveform[1] if waveform.shape[0] > 1 else waveform[0]
                else:
                    mono = np.mean(waveform, axis=0)
            data = mono  # 1D array
            is_stereo = False
        else:  # stereo
            # Ensure exactly 2 channels
            if waveform.shape[0] == 1:
                # duplicate mono to both channels
                data = np.vstack([waveform[0], waveform[0]])  # (2, N)
            elif waveform.shape[0] >= 2:
                data = waveform[:2]  # take first two channels
            else:
                data = waveform  # should not happen
            is_stereo = True

        # --- 3. Apply start/end time slicing (along time axis) ---
        start_seconds = start_time * 60.0
        end_seconds = end_time * 60.0 if end_time > 0 else duration_seconds
        if start_seconds >= end_seconds or start_seconds >= duration_seconds:
            print("[mdAudioSegmenterSaver] Invalid time range. No segments saved.")
            return ()

        start_sample = int(start_seconds * original_sr)
        end_sample = int(end_seconds * original_sr)

        if is_stereo:
            sliced = data[:, start_sample:end_sample]  # (2, samples)
        else:
            sliced = data[start_sample:end_sample]     # (samples,)

        # --- 4. Resample to target sample rate ---
        if original_sr != sample_rate:
            sliced = self.resample_audio(sliced, original_sr, sample_rate)
            final_sr = sample_rate
        else:
            final_sr = original_sr

        # --- 5. Calculate segments ---
        slice_samples = slice_length * final_sr
        total_samples = sliced.shape[-1] if is_stereo else len(sliced)
        num_segments = int(np.ceil(total_samples / slice_samples))

        if num_segments == 0:
            print("[mdAudioSegmenterSaver] No segments to save.")
            return ()

        # --- 6. Save segments ---
        base_output_dir = folder_paths.get_output_directory()
        save_dir = os.path.join(base_output_dir, "audio_segments", project_name)
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []
        for i in range(num_segments):
            start_idx = i * slice_samples
            end_idx = min(start_idx + slice_samples, total_samples)

            if is_stereo:
                segment = sliced[:, start_idx:end_idx]  # (2, seg_len)
                # Pad if needed
                if segment.shape[1] < slice_samples:
                    padded = np.zeros((2, slice_samples))
                    padded[:, :segment.shape[1]] = segment
                    segment = padded
                # soundfile expects (frames, channels) -> transpose
                segment = segment.T  # (slice_samples, 2)
            else:
                segment = sliced[start_idx:end_idx]    # (seg_len,)
                if len(segment) < slice_samples:
                    padded = np.zeros(slice_samples)
                    padded[:len(segment)] = segment
                    segment = padded
                # mono remains 1D

            # Generate filename
            filename = f"segment_{i+1:03d}.{format}"
            filepath = os.path.join(save_dir, filename)

            try:
                sf.write(filepath, segment, final_sr, format=format.upper())
                saved_files.append(filepath)
                print(f"[mdAudioSegmenterSaver] Saved: {filepath}")
            except Exception as e:
                print(f"[mdAudioSegmenterSaver] Error saving segment {i+1}: {e}")

        print(f"[mdAudioSegmenterSaver] Successfully saved {len(saved_files)} segments to: {save_dir}")
        return ()

#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdMediaToPrimGrid": mdMediaToPrimGrid,
    "mdAudioSegmenterSaver": mdAudioSegmenterSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdMediaToPrimGrid": "Media to Prim Grid Generator (MD)",
    "mdAudioSegmenterSaver": "Audio Segmenter (MD)",
}
