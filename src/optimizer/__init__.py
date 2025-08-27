import warnings
import importlib

warnings.warn(
    "The 'optimizer' namespace is deprecated and will be removed in a future release. "
    "Please use 'patatune' instead.",
    FutureWarning,
    stacklevel=2,
)

_pat = importlib.import_module("patatune")
globals().update({k: getattr(_pat, k) for k in getattr(_pat, "__all__", dir(_pat))})
__all__ = getattr(_pat, "__all__", dir(_pat))
