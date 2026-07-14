import os
import folder_paths
from ..py.gltf_func import perform_save_gltf, perform_load_gltf

module_cat = "md/GLTF"

#===================================================================================
class mdSavePBRGLTF:
    DESCRIPTION = "Saves a PBR material as a GLTF file with configurable properties such as color tint, alpha mode, and metallic/roughness factors."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "color": ("IMAGE", {"tooltip": "The base color map for the material."}),
                "metal": ("IMAGE", {"tooltip": "The metalness map for the material (part of ORM)."}),
                "emissive": ("IMAGE", {"tooltip": "The emissive map for the material."}),
                "normal": ("IMAGE", {"tooltip": "The normal map for the material."}),
            },
            "required": {
                "double_size": ("BOOLEAN", {"default": False, "tooltip": "If true, the texture will be saved at double the size."}),
                "color_tint": ("INT", {"default": 16777215, "min": 0, "max": 16777215, "display": "color", "tooltip": "The color tint value for the material, represented as an integer."}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The alpha value for transparency."}),
                "alpha_mode": (["OPAQUE", "BLEND", "MASK"], {"tooltip": "The alpha mode for the material (OPAQUE, BLEND, or MASK)."}),
                "alpha_cut": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The alpha cutoff value for transparency."}),
                "metallic_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The metallic factor for the material."}),
                "roughness_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The roughness factor for the material."}),
                "emissive_color": ("INT", {"default": 0, "min": 0, "max": 16777215, "display": "color", "tooltip": "The emissive color value for the material, represented as an integer."}),
                "filename_prefix": ("STRING", {"default": "pbr_material", "tooltip": "The prefix for the output GLTF filename."}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_TOOLTIPS = []
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
    DESCRIPTION = "Loads a PBR material from a GLTF file and returns its maps and properties."

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
                "gltf_filename": (files, {"tooltip": "The GLTF file to load."}),
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
    
    OUTPUT_TOOLTIPS = [
        "The base color map for the material.",
        "The metalness map for the material (part of ORM).",
        "The emissive map for the material.",
        "The normal map for the material.",
        "The occlusion map for the material.",
        "Whether the material is double-sided.",
        "The color tint value for the material, represented as an integer.",
        "The alpha value for transparency.",
        "The alpha mode for the material (OPAQUE, BLEND, or MASK).",
        "The alpha cutoff value for transparency.",
        "The metallic factor for the material.",
        "The roughness factor for the material.",
        "The emissive color value for the material, represented as an integer."
    ]
    
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

