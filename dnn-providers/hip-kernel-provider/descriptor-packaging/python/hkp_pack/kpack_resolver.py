import importlib
import sys
from pathlib import Path

from .errors import HkpPackError


def load_kpack(python_dir=None):
    """Import and return the rocm_kpack kpack + compression modules.

    Returns the (kpack, compression) module pair, both resolved from the same
    location so callers never rely on an implicit sys.path side effect to reach
    compression after importing kpack.

    python_dir, when given, is prepended to sys.path so a checkout or fetched
    clone that CMake resolved is used before any system install. If rocm_kpack
    still cannot be imported, raise a HkpPackError with an actionable message
    rather than a bare ImportError.
    """
    if python_dir is not None:
        p = Path(python_dir).resolve()
        if not p.is_dir():
            raise HkpPackError(f"rocm_kpack python dir does not exist: {p}")
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    try:
        kpack = importlib.import_module("rocm_kpack.kpack")
        compression = importlib.import_module("rocm_kpack.compression")
    except ImportError as exc:
        raise HkpPackError(
            "unable to import rocm_kpack; pass --kpack-python-dir pointing at "
            f"the rocm-kpack 'python' directory (import error: {exc})"
        ) from exc
    return kpack, compression
