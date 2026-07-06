import os
import folder_paths
from ..py.gltf_func import perform_save_gltf, perform_load_gltf

module_cat = "md/GLTF"

#===================================================================================
class mdSavePBRGLTF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "color": ("IMAGE",),
                "metal": ("IMAGE",),   # ORM
                "emissive": ("IMAGE",),
                "normal": ("IMAGE",),
            },
            "required": {
                "double_size": ("BOOLEAN", {"default": False}),
                "color_tint": ("INT", {"default": 16777215, "min": 0, "max": 16777215, "display": "color"}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "alpha_mode": (["OPAQUE", "BLEND", "MASK"],),
                "alpha_cut": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "metallic_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "roughness_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emissive_color": ("INT", {"default": 0, "min": 0, "max": 16777215, "display": "color"}),
                "filename_prefix": ("STRING", {"default": "pbr_material"}),
            },
        }

    RETURN_TYPES = ()
    CATEGORY = module_cat
    FUNCTION = "exec"
    OUTPUT_NODE = True

    def exec(self, double_size, color_tint, alpha, alpha_mode, alpha_cut,
                  metallic_factor, roughness_factor, emissive_color,
                  filename_prefix, **kwargs):
        perform_save_gltf(
            double_size=double_size,
            color_tint=color_tint,
            alpha=alpha,
            alpha_mode=alpha_mode,
            alpha_cut=alpha_cut,
            metallic_factor=metallic_factor,
            roughness_factor=roughness_factor,
            emissive_color=emissive_color,
            filename_prefix=filename_prefix,
            **kwargs
        )
        return ()

#===================================================================================
class mdLoadPBRGLTF:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        supported_ext = (".glb", ".gltf")
        files = []
        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                if not filename.lower().endswith(supported_ext):
                    continue
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, input_dir).replace("\\", "/")
                files.append(relative_path)
        files.sort()
        return {
            "required": {
                "gltf_filename": (files,),
            }
        }

    RETURN_TYPES = (
        "IMAGE",   # color_map
        "IMAGE",   # metal_map
        "IMAGE",   # emissive_map
        "IMAGE",   # normal_map
        "MASK",    # occlusion_map
        "BOOLEAN", # double_side
        "INT",     # color_tint
        "FLOAT",   # alpha
        "STRING",  # alpha_mode
        "FLOAT",   # alpha_cut
        "FLOAT",   # metallic_factor
        "FLOAT",   # roughness_factor
        "INT",     # emissive_color
    )
    
    RETURN_NAMES = (
        "color_map", "metal_map", "emissive_map", "normal_map",
        "occlusion_map", "double_side", "color_tint", "alpha",
        "alpha_mode", "alpha_cut", "metallic_factor", "roughness_factor",
        "emissive_color",
    )
    
    CATEGORY = module_cat
    FUNCTION = "exec"

    def exec(self, gltf_filename):
        return perform_load_gltf(gltf_filename)

#===================================================================================
NODE_CLASS_MAPPINGS = {
    "mdSavePBRGLTF": mdSavePBRGLTF,
    "mdLoadPBRGLTF": mdLoadPBRGLTF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdSavePBRGLTF": "Save PBR GLTF (MD-SL)",
    "mdLoadPBRGLTF": "Load PBR GLTF (MD-SL)",
}