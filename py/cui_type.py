
import numpy as np

class AnyType(str):
    """A special string subclass that always returns False for `!=` comparisons.
    This allows ComfyUI to treat `ANY` as a wildcard that matches any type.
    """
    def __ne__(self, __value: object) -> bool:
        return False


    
COMBO = "COMBO"
INT = "INT"
FLOAT = "FLOAT"
STRING = "STRING"
BOOLEAN = "BOOLEAN"
IMAGE = "IMAGE"
LATENT = "LATENT"
MASK = "MASK"
AUDIO = "AUDIO"
SAMPLER = "SAMPLER"
SIGMAS = "SIGMAS"
GUIDER = "GUIDER"

ANY = AnyType("*")
RGBA = "RGBA"
