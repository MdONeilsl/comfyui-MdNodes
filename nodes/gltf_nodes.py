
import os
import json
import struct
import io
import torch
import base64
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import folder_paths

# Prevent decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 16384 * 16384

module_cat = "md/GLTF"

# ============================================================
# Utilities
# ============================================================

def _pad4(data: bytes) -> bytes:
    return data + b"\x00" * ((4 - len(data) % 4) % 4)

def _tensor_to_numpy(t):
    arr = t.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return np.clip(arr, 0.0, 1.0)

def _to_uint8(arr):
    return (arr * 255.0 + 0.5).astype(np.uint8)

def _ensure_rgb(arr):
    if arr.ndim != 3:
        raise ValueError("Expected HWC image")

    c = arr.shape[-1]
    if c == 1:
        return np.repeat(arr, 3, axis=-1)
    if c >= 3:
        return arr[..., :3]
    raise ValueError("Unsupported channel count")

def _pack_glb(json_dict, bin_blob):
    json_bytes = _pad4(json.dumps(json_dict, separators=(",", ":")).encode("utf-8"))
    bin_blob = _pad4(bin_blob)

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)

    return (
        struct.pack("<4sII", b"glTF", 2, total_len)
        + struct.pack("<I4s", len(json_bytes), b"JSON")
        + json_bytes
        + struct.pack("<I4s", len(bin_blob), b"BIN\0")
        + bin_blob
    )

def _ensure_rgba(arr):
    if arr.ndim != 3:
        raise ValueError("Expected HWC image")

    c = arr.shape[-1]

    if c == 4:
        return arr
    elif c == 3:
        alpha = np.ones((*arr.shape[:2], 1), dtype=arr.dtype)
        return np.concatenate([arr, alpha], axis=-1)
    elif c == 1:
        rgb = np.repeat(arr, 3, axis=-1)
        alpha = np.ones((*arr.shape[:2], 1), dtype=arr.dtype)
        return np.concatenate([rgb, alpha], axis=-1)

    raise ValueError("Unsupported channel count")

# ============================================================
# Tangent Generation (MikkTSpace-style)
# ============================================================
def _compute_tangents(pos, uv, indices):
    tan1 = np.zeros_like(pos)
    tan2 = np.zeros_like(pos)

    for i in range(0, len(indices), 3):
        i1, i2, i3 = indices[i:i+3]

        v1, v2, v3 = pos[i1], pos[i2], pos[i3]
        w1, w2, w3 = uv[i1], uv[i2], uv[i3]

        x1 = v2 - v1
        x2 = v3 - v1

        s1 = w2[0] - w1[0]
        s2 = w3[0] - w1[0]
        t1 = w2[1] - w1[1]
        t2 = w3[1] - w1[1]

        denom = (s1 * t2 - s2 * t1)
        r = 1.0 / denom if abs(denom) > 1e-8 else 0.0

        sdir = (t2 * x1 - t1 * x2) * r
        tdir = (s1 * x2 - s2 * x1) * r

        tan1[i1] += sdir
        tan1[i2] += sdir
        tan1[i3] += sdir

        tan2[i1] += tdir
        tan2[i2] += tdir
        tan2[i3] += tdir

    tangents = []
    n = np.array([0, 0, 1], dtype=np.float32)

    for i in range(len(pos)):
        t = tan1[i]
        t = t - n * np.dot(n, t)

        norm = np.linalg.norm(t)
        if norm > 1e-8:
            t = t / norm
        else:
            t = np.array([1, 0, 0], dtype=np.float32)

        w = 1.0 if np.dot(np.cross(n, t), tan2[i]) > 0 else -1.0
        tangents.append([t[0], t[1], t[2], w])

    return np.array(tangents, dtype=np.float32)

# ============================================================
# Node
# ============================================================
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
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "alpha_mode": (["OPAQUE", "BLEND", "MASK"],),
                "alpha_cut": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "metallic_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "roughness_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "emissive_color": ("INT", {"default": 0, "min": 0, "max": 16777215, "display": "color"}),
                "filename_prefix": ("STRING", {"default": "pbr_material"}),
            },
        }

    RETURN_TYPES = ()
    CATEGORY = module_cat
    FUNCTION = "save_gltf"
    OUTPUT_NODE = True

    def save_gltf(self, double_size, color_tint, alpha, alpha_mode, alpha_cut,
                  metallic_factor, roughness_factor, emissive_color,
                  filename_prefix, **kwargs):

        output_dir = os.path.join(folder_paths.get_output_directory(), "pbr_gltf")
        os.makedirs(output_dir, exist_ok=True)

        base = filename_prefix.strip() or "pbr_material"
        scale = 2 if double_size else 1

        images, textures, bufferViews = [], [], []
        samplers = [{"magFilter":9729,"minFilter":9729,"wrapS":10497,"wrapT":10497}]
        bin_blob = b""

        def save_and_embed(arr, name, force_rgba=False):
            nonlocal bin_blob

            if force_rgba:
                arr = _ensure_rgba(arr)
                mode = "RGBA"
            else:
                arr = _ensure_rgb(arr)
                mode = "RGB"

            img = Image.fromarray(_to_uint8(arr), mode)

            if scale != 1:
                img = img.resize(
                    (img.width * scale, img.height * scale),
                    Image.Resampling.LANCZOS
                )

            # Save to disk (preserves alpha if present)
            
            disk_path = os.path.join(output_dir, f"{base}_{name}.png")
            os.makedirs(os.path.dirname(disk_path), exist_ok=True)
            img.save(disk_path)

            # Embed in GLB
            
            buf = io.BytesIO()
            img.save(buf, format="PNG", save_all=False, exif=b"")
            data = _pad4(buf.getvalue())

            offset = len(bin_blob)
            bin_blob += data

            bufferViews.append({
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(data)
            })

            images.append({
                "bufferView": len(bufferViews) - 1,
                "mimeType": "image/png"
            })

            textures.append({
                "source": len(images) - 1,
                "sampler": 0
            })

            return len(textures) - 1

        # =========================
        # TEXTURES
        # =========================
        base_idx = None
        orm_idx = None
        ao_idx = None
        normal_idx = None
        emissive_idx = None

        if kwargs.get("color") is not None:
            base_idx = save_and_embed(
                _tensor_to_numpy(kwargs["color"]),
                "color",
                force_rgba=True
            )

        if kwargs.get("metal") is not None:
            orm = _tensor_to_numpy(kwargs["metal"])
            orm_idx = save_and_embed(orm, "metal_orm")

            ao = np.repeat(orm[..., 0:1], 3, axis=-1)
            ao_idx = save_and_embed(ao, "ao")

        if kwargs.get("normal") is not None:
            normal_idx = save_and_embed(_tensor_to_numpy(kwargs["normal"]), "normal")

        if kwargs.get("emissive") is not None:
            emissive_idx = save_and_embed(_tensor_to_numpy(kwargs["emissive"]), "emissive")

        # =========================
        # MATERIAL
        # =========================
        def unpack_rgb(i):
            return [((i>>16)&255)/255, ((i>>8)&255)/255, (i&255)/255]

        pbr = {
            "baseColorFactor": unpack_rgb(color_tint) + [alpha],
            "metallicFactor": metallic_factor,
            "roughnessFactor": roughness_factor
        }

        if base_idx is not None:
            pbr["baseColorTexture"] = {"index": base_idx}

        if orm_idx is not None:
            pbr["metallicRoughnessTexture"] = {"index": orm_idx}

        material = {
            "pbrMetallicRoughness": pbr,
            "alphaMode": alpha_mode,
            "emissiveFactor": unpack_rgb(emissive_color)
        }

        if alpha_mode == "MASK":
            material["alphaCutoff"] = alpha_cut

        if ao_idx is not None:
            material["occlusionTexture"] = {"index": ao_idx, "strength": 1.0}

        if normal_idx is not None:
            material["normalTexture"] = {"index": normal_idx, "scale": 1.0}

        if emissive_idx is not None:
            material["emissiveTexture"] = {"index": emissive_idx}

        # =========================
        # GEOMETRY + TANGENTS
        # =========================
        pos = np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]],np.float32)
        uv  = np.array([[0,1],[1,1],[1,0],[0,0]],np.float32)
        nrm = np.array([[0,0,1]]*4,np.float32)
        idx = np.array([0,1,2,0,2,3],np.uint16)
        tan = _compute_tangents(pos, uv, idx)

        def add_buf(arr):
            nonlocal bin_blob
            raw = arr.tobytes()
            offset = len(bin_blob)
            bin_blob += _pad4(raw)
            return offset, len(raw)

        offs = [add_buf(a) for a in (pos,nrm,uv,tan,idx)]

        bufferViews += [
            {"buffer":0,"byteOffset":offs[0][0],"byteLength":offs[0][1],"target":34962},
            {"buffer":0,"byteOffset":offs[1][0],"byteLength":offs[1][1],"target":34962},
            {"buffer":0,"byteOffset":offs[2][0],"byteLength":offs[2][1],"target":34962},
            {"buffer":0,"byteOffset":offs[3][0],"byteLength":offs[3][1],"target":34962},
            {"buffer":0,"byteOffset":offs[4][0],"byteLength":offs[4][1],"target":34963},
        ]

        accessors = [
            {"bufferView":len(bufferViews)-5,"componentType":5126,"count":4,"type":"VEC3"},
            {"bufferView":len(bufferViews)-4,"componentType":5126,"count":4,"type":"VEC3"},
            {"bufferView":len(bufferViews)-3,"componentType":5126,"count":4,"type":"VEC2"},
            {"bufferView":len(bufferViews)-2,"componentType":5126,"count":4,"type":"VEC4"},
            {"bufferView":len(bufferViews)-1,"componentType":5123,"count":6,"type":"SCALAR"},
        ]

        gltf = {
            "asset":{"version":"2.0"},
            "scene":0,
            "scenes":[{"nodes":[0]}],
            "nodes":[{"mesh":0}],
            "meshes":[{"primitives":[{
                "attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2,"TANGENT":3},
                "indices":4,
                "material":0
            }]}],
            "materials":[material],
            "textures":textures,
            "images":images,
            "samplers":samplers,
            "accessors":accessors,
            "bufferViews":bufferViews,
            "buffers":[{"byteLength":len(bin_blob)}]
        }

        out_path = os.path.join(output_dir, base + ".glb")

        with open(out_path, "wb") as f:
            f.write(_pack_glb(gltf, bin_blob))

        print(f"[GLB] Saved → {out_path}")

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

                relative_path = os.path.relpath(
                    full_path,
                    input_dir
                )

                relative_path = relative_path.replace("\\", "/")

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
        "INT",  # color_tint
        "FLOAT",   # alpha
        "STRING",  # alpha_mode
        "FLOAT",   # alpha_cut
        "FLOAT",   # metallic_factor
        "FLOAT",   # roughness_factor
        "INT",  # emissive_color
    )

    RETURN_NAMES = (
        "color_map",
        "metal_map",
        "emissive_map",
        "normal_map",
        "occlusion_map",
        "double_side",
        "color_tint",
        "alpha",
        "alpha_mode",
        "alpha_cut",
        "metallic_factor",
        "roughness_factor",
        "emissive_color",
    )

    CATEGORY = module_cat
    FUNCTION = "load_gltf"

    # =========================================================
    # Empty Tensor
    # =========================================================

    def _empty_tensor(self, channels=3):

        return torch.zeros(
            (1, 64, 64, channels),
            dtype=torch.float32
        )

    # =========================================================
    # Safe Image Open
    # =========================================================

    def _safe_open_image(self, img_bytes):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.load()
            img = ImageOps.exif_transpose(img)
            if img.width <= 0 or img.height <= 0:
                raise ValueError("Invalid image dimensions")
            return img
        
        except UnidentifiedImageError:
            raise ValueError("Unsupported image format")

    # =========================================================
    # Load Buffers
    # =========================================================

    def _load_buffers(
        self,
        gltf_json,
        gltf_dir,
        glb_bin_chunk
    ):

        buffers = []

        for i, buffer_info in enumerate(
            gltf_json.get("buffers", [])
        ):

            # Embedded GLB BIN chunk
            if i == 0 and glb_bin_chunk is not None:
                buffers.append(glb_bin_chunk)
                continue

            uri = buffer_info.get("uri")

            if not uri:
                buffers.append(b"")
                continue

            # Base64 embedded buffer
            if uri.startswith("data:"):

                _, encoded = uri.split(",", 1)

                buffers.append(
                    base64.b64decode(encoded)
                )

                continue

            # External .bin
            path = os.path.join(gltf_dir, uri)

            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Missing buffer file: {path}"
                )

            with open(path, "rb") as f:
                buffers.append(f.read())

        return buffers

    # =========================================================
    # Extract BufferView
    # =========================================================

    def _extract_buffer_view(
        self,
        gltf_json,
        buffers,
        buffer_view_index
    ):

        buffer_views = gltf_json.get(
            "bufferViews",
            []
        )

        if buffer_view_index >= len(buffer_views):
            raise ValueError(
                "Invalid bufferView index"
            )

        bv = buffer_views[buffer_view_index]

        buffer_index = bv["buffer"]

        if buffer_index >= len(buffers):
            raise ValueError(
                "Invalid buffer index"
            )

        data = buffers[buffer_index]

        byte_offset = bv.get("byteOffset", 0)
        byte_length = bv["byteLength"]

        end = byte_offset + byte_length

        if end > len(data):
            raise ValueError(
                "BufferView exceeds buffer size"
            )

        return data[byte_offset:end]

    # =========================================================
    # Load Image Bytes
    # =========================================================

    def _load_image_bytes(
        self,
        gltf_json,
        buffers,
        image_info,
        gltf_dir
    ):

        # BufferView image
        if "bufferView" in image_info:

            return self._extract_buffer_view(
                gltf_json,
                buffers,
                image_info["bufferView"]
            )

        # URI image
        uri = image_info.get("uri")

        if not uri:
            raise ValueError(
                "Image has no URI or bufferView"
            )

        # Embedded base64 image
        if uri.startswith("data:"):

            _, encoded = uri.split(",", 1)

            return base64.b64decode(encoded)

        # External file image
        path = os.path.join(gltf_dir, uri)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing image file: {path}"
            )

        with open(path, "rb") as f:
            return f.read()

    # =========================================================
    # Load Texture
    # =========================================================

    def _load_texture_tensor(
        self,
        gltf_json,
        buffers,
        tex_index,
        gltf_dir,
        channels=3
    ):

        if tex_index is None:
            return self._empty_tensor(channels)

        textures = gltf_json.get("textures", [])
        images = gltf_json.get("images", [])

        if tex_index >= len(textures):
            return self._empty_tensor(channels)

        texture = textures[tex_index]

        source_index = texture.get("source")

        if source_index is None:
            return self._empty_tensor(channels)

        if source_index >= len(images):
            return self._empty_tensor(channels)

        try:

            image_info = images[source_index]

            img_bytes = self._load_image_bytes(
                gltf_json,
                buffers,
                image_info,
                gltf_dir
            )

            img = self._safe_open_image(img_bytes)

            if channels == 4:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")

            img_np = (
                np.array(img)
                .astype(np.float32)
                / 255.0
            )

            return torch.from_numpy(img_np).unsqueeze(0)

        except Exception as e:

            print(f"[GLTF Texture Error] {e}")

            return self._empty_tensor(channels)

    # =========================================================
    # Extract Single Channel
    # =========================================================

    def _extract_channel(
        self,
        tensor,
        channel_index
    ):

        if tensor.shape[-1] <= channel_index:
            return self._empty_tensor(1)

        return tensor[
            ...,
            channel_index:channel_index + 1
        ].clone()

    # =========================================================
    # Main Loader
    # =========================================================

    def load_gltf(self, gltf_filename):

        filepath = os.path.join(
            folder_paths.get_input_directory(),
            gltf_filename
        )

        gltf_dir = os.path.dirname(filepath)

        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)

        gltf_json = {}
        glb_bin_chunk = None

        # =====================================================
        # Parse GLB / GLTF
        # =====================================================

        with open(filepath, "rb") as f:

            magic = f.read(4)

            # -------------------------------------------------
            # GLB
            # -------------------------------------------------

            if magic == b"glTF":

                version, length = struct.unpack(
                    "<II",
                    f.read(8)
                )

                if version != 2:
                    raise ValueError(
                        f"Unsupported GLB version: {version}"
                    )

                while f.tell() < length:

                    chunk_header = f.read(8)

                    if len(chunk_header) < 8:
                        break

                    chunk_len, chunk_type = struct.unpack(
                        "<I4s",
                        chunk_header
                    )

                    chunk_data = f.read(chunk_len)

                    # JSON chunk
                    if chunk_type == b"JSON":

                        try:

                            json_text = chunk_data.decode(
                                "utf-8"
                            )

                            # Remove GLB padding
                            json_text = json_text.rstrip(
                                "\x00 \t\r\n"
                            )

                            gltf_json = json.loads(
                                json_text
                            )

                        except json.JSONDecodeError as e:

                            raise ValueError(
                                f"Invalid GLB JSON chunk: {e}"
                            )

                    # BIN chunk
                    elif chunk_type == b"BIN\x00":

                        glb_bin_chunk = chunk_data

            # -------------------------------------------------
            # GLTF
            # -------------------------------------------------

            else:

                f.seek(0)

                try:
                    gltf_json = json.load(f)

                except json.JSONDecodeError as e:

                    raise ValueError(
                        f"Invalid GLTF JSON: {e}"
                    )

        # =====================================================
        # Load Buffers
        # =====================================================

        buffers = self._load_buffers(
            gltf_json,
            gltf_dir,
            glb_bin_chunk
        )

        # =====================================================
        # Material
        # =====================================================

        materials = gltf_json.get("materials", [])

        material = materials[0] if materials else {}

        pbr = material.get(
            "pbrMetallicRoughness",
            {}
        )

        # =====================================================
        # Factors
        # =====================================================

        base_color = pbr.get(
            "baseColorFactor",
            [1.0, 1.0, 1.0, 1.0]
        )

        to_byte = lambda x: int(round(max(0.0, min(1.0, x)) * 255))
        
        r, g, b, a = base_color
        red   = to_byte(r)
        green = to_byte(g)
        blue  = to_byte(b)

        color_tint = (red << 16) | (green << 8) | blue 
        alpha = float(a)

        emissive_factor = material.get(
            "emissiveFactor",
            [0.0, 0.0, 0.0]
        )

        er, eg, eb = emissive_factor
        red   = to_byte(er)
        green = to_byte(eg)
        blue  = to_byte(eb)

        emissive_color = (red << 16) | (green << 8) | blue

        metallic_factor = float(
            pbr.get("metallicFactor", 1.0)
        )

        roughness_factor = float(
            pbr.get("roughnessFactor", 1.0)
        )

        alpha_mode = material.get(
            "alphaMode",
            "OPAQUE"
        )

        alpha_cut = float(
            material.get("alphaCutoff", 0.5)
        )

        double_side = bool(
            material.get("doubleSided", False)
        )

        # =====================================================
        # Texture Indices
        # =====================================================

        color_idx = pbr.get(
            "baseColorTexture",
            {}
        ).get("index")

        metal_idx = pbr.get(
            "metallicRoughnessTexture",
            {}
        ).get("index")

        emissive_idx = material.get(
            "emissiveTexture",
            {}
        ).get("index")

        normal_idx = material.get(
            "normalTexture",
            {}
        ).get("index")

        occlusion_idx = material.get(
            "occlusionTexture",
            {}
        ).get("index")

        # =====================================================
        # Load Maps
        # =====================================================

        color_map = self._load_texture_tensor(
            gltf_json,
            buffers,
            color_idx,
            gltf_dir,
            channels=4
        )

        metal_map = self._load_texture_tensor(
            gltf_json,
            buffers,
            metal_idx,
            gltf_dir,
            channels=3
        )

        emissive_map = self._load_texture_tensor(
            gltf_json,
            buffers,
            emissive_idx,
            gltf_dir,
            channels=3
        )

        normal_map = self._load_texture_tensor(
            gltf_json,
            buffers,
            normal_idx,
            gltf_dir,
            channels=3
        )

        # =====================================================
        # Occlusion Map
        # =====================================================

        # Native occlusion texture
        if occlusion_idx is not None:

            occlusion_rgb = self._load_texture_tensor(
                gltf_json,
                buffers,
                occlusion_idx,
                gltf_dir,
                channels=3
            )

            occlusion_map = self._extract_channel(
                occlusion_rgb,
                0
            )

        # Fallback:
        # use RED channel from metal map
        else:

            occlusion_map = self._extract_channel(
                metal_map,
                0
            )

        # =====================================================
        # Return
        # =====================================================

        return (
            color_map,
            metal_map,
            emissive_map,
            normal_map,
            occlusion_map,
            double_side,
            color_tint,
            alpha,
            alpha_mode,
            alpha_cut,
            metallic_factor,
            roughness_factor,
            emissive_color,
        )

#===================================================================================

NODE_CLASS_MAPPINGS = {
    "mdSavePBRGLTF": mdSavePBRGLTF,
    "mdLoadPBRGLTF": mdLoadPBRGLTF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "mdSavePBRGLTF": "Save PBR GLTF (MD-SL)",
    "mdLoadPBRGLTF": "Load PBR GLTF (MD-SL)",
}
