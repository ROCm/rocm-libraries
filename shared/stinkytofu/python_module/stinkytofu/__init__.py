"""StinkyTofu: High-Level IR for AMDGPU Assembly Generation"""

import sys, os, glob, importlib.util

# Load C++ module (_stinkytofu.cpython-*.so)
_dir = os.path.dirname(__file__)
_so = glob.glob(os.path.join(_dir, "_stinkytofu.cpython-*.so"))
if not _so:
    raise ImportError("StinkyTofu C++ module not found")

_spec = importlib.util.spec_from_file_location("_stinkytofu", _so[0])
_cpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpp)

# Export ALL C++ symbols immediately
for _n in dir(_cpp):
    if not _n.startswith("_"):
        exec(f"{_n} = _cpp.{_n}")

# Runtime intrinsic data
_sigs = {}
_lib = None
_init = False

# Load intrinsics
try:
    reg = _cpp.IntrinsicRegistry.instance()
    if reg.is_initialized():
        _init = True
        _lib = reg.get_library()
        for name in _lib.get_intrinsic_names():
            _sigs[name] = [arg.name for arg in _lib.get_arguments(name)]
except:
    pass


# Python wrapper functions
def list_intrinsics():
    return list(_sigs.keys()) if _init else []


def get_intrinsic_signature(name):
    return _sigs.get(name)


def get_intrinsic_info(name):
    if not _init or name not in _sigs:
        return None
    return {"signature": _sigs[name], "comment": _lib.get_comment(name) if _lib else ""}


def Intrinsic(name, **kwargs):
    """Create intrinsic with validation and auto-reordering (ORDER DOESN'T MATTER!)"""
    if not _init:
        raise RuntimeError("Intrinsics not loaded")
    if name not in _sigs:
        raise ValueError(
            f"Unknown: '{name}' (available: {', '.join(sorted(_sigs.keys()))})"
        )

    expected = _sigs[name]
    provided = set(kwargs.keys())
    missing = set(expected) - provided
    extra = provided - set(expected)

    if missing:
        raise ValueError(f"'{name}' missing: {sorted(missing)} (expected: {expected})")
    if extra:
        raise ValueError(f"'{name}' unexpected: {sorted(extra)} (expected: {expected})")

    # Build kwargs in signature order (C++ Intrinsic iterates dict order)
    ordered_kwargs = {arg: kwargs[arg] for arg in expected}
    return _cpp.Intrinsic(name, **ordered_kwargs)
