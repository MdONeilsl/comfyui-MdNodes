# ------------------------------------------------------------------------------
# ComfyUI wildcard type – used to accept any input connection.
# ------------------------------------------------------------------------------
class AnyType(str):
    """A special string subclass that always returns False for `!=` comparisons.
    This allows ComfyUI to treat `ANY` as a wildcard that matches any type.
    """
    def __ne__(self, __value: object) -> bool:
        return False

ANY = AnyType("*")