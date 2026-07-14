import numpy as np
import folder_paths
import os
import librosa
import soundfile as sf

from ..py.media_func import process_and_save_frames, segment_and_save_audio

module_cat = "md/media"

#===================================================================================
class mdMediaToPrimGrid:
    """
    ComfyUI custom node that processes video frames into grid images for Second Life.
    Replicates the functionality of the original Media to Prim web application.
    """

    DESCRIPTION = "Converts video frames into a grid of images suitable for Second Life prim rendering."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Input video frames as images."}),
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Video information object containing metadata like FPS, resolution, etc."}),
                "max_image_size": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 64, "tooltip": "Maximum dimension (width/height) for each output image in pixels."}),
                "max_frame_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64, "tooltip": "Maximum size for each frame segment (in pixels) before tiling."}),
                "fps_target": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1, "tooltip": "Target frames per second for the output grid animation."}),
                "side_target": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1, "tooltip": "Target number of sides for the grid layout (0 = auto, 1-8 = fixed grid)."}),
                "output_folder": ("STRING", {"default": "", "multiline": False, "tooltip": "Output folder path to save the grid images. Leave empty to use default."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("grid_images", "image_count")
    OUTPUT_TOOLTIPS = ["Grid of processed images for Second Life prim rendering.", "Total number of images generated in the grid."]

    FUNCTION = "process_and_save"
    CATEGORY = module_cat  # must be defined elsewhere or replaced

    def process_and_save(self, images, video_info, max_image_size, max_frame_size,
                         fps_target, side_target, output_folder):
        """Delegate all work to the external function."""
        return process_and_save_frames(
            images, video_info, max_image_size, max_frame_size,
            fps_target, side_target, output_folder
        )
        
#===================================================================================
class mdAudioSegmenterSaver:
    DESCRIPTION = "Splits an audio file into segments and saves them as individual files."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Input audio file to be segmented."}),
                "project_name": ("STRING", {"default": "audio_segments", "multiline": False, "tooltip": "Base name for output files."}),
                "slice_length": ("INT", {"default": 30, "min": 1, "max": 3600, "step": 1, "tooltip": "Length of each audio segment in seconds."}),
                "mode": (["mono", "stereo"], {"default": "mono", "tooltip": "Audio channel mode for output segments."}),
                "channel_mode": (["average", "left_only", "right_only"], {"default": "average", "tooltip": "How to handle multi-channel audio (average, left-only, right-only)."}),
                "format": (["wav", "mp3", "flac"], {"default": "wav", "tooltip": "Output audio file format."}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1000, "tooltip": "Sample rate for output audio files."}),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1, "tooltip": "Start time (in seconds) for segmenting audio (default: 0)."}),
                "end_time": ("FLOAT", {"default": -1.0, "min": -1.0, "step": 0.1, "tooltip": "End time (in seconds) for segmenting audio. Use -1 for the end of the audio."}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_TOOLTIPS = []

    FUNCTION = "segment_and_save"
    CATEGORY = module_cat  # assumes module_cat is defined elsewhere
    OUTPUT_NODE = True

    def segment_and_save(self, **kwargs):
        # Delegate to the pure function
        return segment_and_save_audio(**kwargs)
    
#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdMediaToPrimGrid": mdMediaToPrimGrid,
    "mdAudioSegmenterSaver": mdAudioSegmenterSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdMediaToPrimGrid": "Media to Prim Grid Generator (MD)",
    "mdAudioSegmenterSaver": "Audio Segmenter (MD)",
}

