# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# online.py -- in-process binding for the ONLINE portable-IR path. The Python
# builder serializes a kernel (concrete portable-IR JSON, or a CBOR recipe /
# bundle) and hands it to the pure-C backend, which expands + lowers it to
# AMDGPU LLVM IR -- all in this process, no subprocess, no pybind build step.
#
# Transport is ctypes over the C ABI (every entry point is extern "C"), so the
# only requirement is a shared librocke. Point ROCKE_ONLINE_LIB at it, or pass a
# path to load(); build_lib() will compile one from the rocke sources on demand.
#
# This is the same C core the OFFLINE provider uses (rocke_lower + recipe VM); the
# online and offline paths therefore produce byte-identical output.
import ctypes
import os
import subprocess
import tempfile
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_long,
    c_size_t,
    c_ubyte,
)
from typing import Dict, List, Optional, Tuple

_ERR_CAP = 256


class _SpecInt(Structure):
    _fields_ = [("name", c_char_p), ("value", c_long)]


class _SpecStr(Structure):
    _fields_ = [("name", c_char_p), ("value", c_char_p)]


_lib = None


def _platform_root() -> str:
    """rocke/platform/ -- the CMake source dir for the engine. Derived from
    __file__ so the tree stays relocatable (python/rocke/portable_ir/src)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", ".."))


def _default_cache_lib() -> str:
    return os.path.join(tempfile.gettempdir(), "rocke_online", "librocke.so")


def _default_lib_paths() -> List[str]:
    paths = []
    env = os.environ.get("ROCKE_ONLINE_LIB")
    if env:
        paths.append(env)
    paths += [
        _default_cache_lib(),
        os.path.join(_platform_root(), "build", "librocke.so"),
    ]
    return paths


def build_lib(out_path: Optional[str] = None) -> str:
    """Build a shared librocke.so from the C++ engine core (librocke_core.a, via
    CMake). The portable-IR replay tooling (json/cbor DOM, importer, recipe VM,
    online wrappers) is C++20 under cpp/portable_ir/ and is part of the core
    archive, so a single CMake build + whole-archive link produces a complete
    shared library — no separate tooling compile."""
    out_path = out_path or _default_cache_lib()
    coredir = os.path.join(os.path.dirname(out_path), "core")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            _platform_root(),
            "-B",
            coredir,
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DROCKE_BUILD_PYENV=OFF",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            coredir,
            "--target",
            "rocke_core",
            "-j",
            str(os.cpu_count() or 8),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    core = os.path.join(coredir, "librocke_core.a")
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-Wl,--whole-archive",
            core,
            "-Wl,--no-whole-archive",
            "-lm",
            "-o",
            out_path,
        ],
        check=True,
    )
    return out_path


def _bind(lib: ctypes.CDLL) -> None:
    spec_i = POINTER(_SpecInt)
    spec_s = POINTER(_SpecStr)
    dbl = POINTER(c_double)
    for name, argtypes in (
        (
            "rocke_online_recipe_cbor_to_llvm",
            [
                POINTER(c_ubyte),
                c_size_t,
                spec_i,
                c_int,
                spec_s,
                c_int,
                c_char_p,
                POINTER(c_char_p),
                dbl,
                dbl,
                c_char_p,
                c_size_t,
            ],
        ),
        (
            "rocke_online_bundle_cbor_to_llvm",
            [
                POINTER(c_ubyte),
                c_size_t,
                c_char_p,
                c_char_p,
                spec_i,
                c_int,
                spec_s,
                c_int,
                POINTER(c_char_p),
                dbl,
                dbl,
                c_char_p,
                c_size_t,
            ],
        ),
        (
            "rocke_online_ir_json_to_llvm",
            [c_char_p, c_char_p, POINTER(c_char_p), dbl, dbl, c_char_p, c_size_t],
        ),
    ):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = c_int
    lib.rocke_online_free.argtypes = [c_char_p]
    lib.rocke_online_free.restype = None


def load(path: Optional[str] = None) -> ctypes.CDLL:
    global _lib
    if _lib is not None and path is None:
        return _lib

    # A named library is a demand, not a hint. Skipping a missing ROCKE_ONLINE_LIB
    # and quietly loading the next candidate -- a cached /tmp build, or a fresh
    # compile -- means the caller verifies an engine it did not build, and the
    # substitution is invisible. That is fatal for a byte-identity gate, whose
    # entire claim is about the engine at this commit, so say so instead.
    explicit = path or os.environ.get("ROCKE_ONLINE_LIB")
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(
                f"librocke not found at {explicit!r} (from "
                f"{'load(path)' if path else 'ROCKE_ONLINE_LIB'}). Refusing to "
                f"fall back to another library: build it with `cmake --build "
                f"<dir> --target rocke_shared` or online.build_lib()"
            )
        lib = ctypes.CDLL(explicit)
        _bind(lib)
        _lib = lib
        return lib

    for p in _default_lib_paths():
        if p and os.path.exists(p):
            lib = ctypes.CDLL(p)
            _bind(lib)
            _lib = lib
            return lib
    # nothing prebuilt -> compile one
    p = build_lib()
    lib = ctypes.CDLL(p)
    _bind(lib)
    _lib = lib
    return lib


def _mk_specs(ints: Optional[Dict[str, int]], strs: Optional[Dict[str, str]]):
    ints = ints or {}
    strs = strs or {}
    ia = (
        (_SpecInt * len(ints))(*[_SpecInt(k.encode(), int(v)) for k, v in ints.items()])
        if ints
        else None
    )
    sa = (
        (_SpecStr * len(strs))(
            *[_SpecStr(k.encode(), str(v).encode()) for k, v in strs.items()]
        )
        if strs
        else None
    )
    return ia, len(ints), sa, len(strs)


def _take_ll(lib, out_ll: c_char_p) -> str:
    s = ctypes.cast(out_ll, c_char_p).value
    text = s.decode() if s else ""
    lib.rocke_online_free(out_ll)
    return text


def recipe_cbor_to_llvm(
    cbor: bytes,
    *,
    arch: str = "gfx950",
    ints: Optional[Dict[str, int]] = None,
    strs: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, float]]:
    """Expand+lower a CBOR recipe in C. Returns (.ll, {build_ms, lower_ms})."""
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    ia, ni, sa, ns = _mk_specs(ints, strs)
    out_ll = c_char_p()
    bms, lms = c_double(0), c_double(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.rocke_online_recipe_cbor_to_llvm(
        buf,
        len(cbor),
        ia,
        ni,
        sa,
        ns,
        arch.encode(),
        byref(out_ll),
        byref(bms),
        byref(lms),
        err,
        _ERR_CAP,
    )
    if st != 0:
        raise RuntimeError(f"online recipe lower failed ({st}): {err.value.decode()}")
    return _take_ll(lib, out_ll), {"build_ms": bms.value, "lower_ms": lms.value}


def bundle_cbor_to_llvm(
    cbor: bytes,
    key: str,
    *,
    arch: str = "gfx950",
    ints: Optional[Dict[str, int]] = None,
    strs: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, float]]:
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    ia, ni, sa, ns = _mk_specs(ints, strs)
    out_ll = c_char_p()
    bms, lms = c_double(0), c_double(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.rocke_online_bundle_cbor_to_llvm(
        buf,
        len(cbor),
        key.encode(),
        arch.encode(),
        ia,
        ni,
        sa,
        ns,
        byref(out_ll),
        byref(bms),
        byref(lms),
        err,
        _ERR_CAP,
    )
    if st != 0:
        raise RuntimeError(f"online bundle lower failed ({st}): {err.value.decode()}")
    return _take_ll(lib, out_ll), {"build_ms": bms.value, "lower_ms": lms.value}


def ir_json_to_llvm(text: str, *, arch: str = "gfx950") -> Tuple[str, Dict[str, float]]:
    lib = load()
    out_ll = c_char_p()
    bms, lms = c_double(0), c_double(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.rocke_online_ir_json_to_llvm(
        text.encode(),
        arch.encode(),
        byref(out_ll),
        byref(bms),
        byref(lms),
        err,
        _ERR_CAP,
    )
    if st != 0:
        raise RuntimeError(f"online ir-json lower failed ({st}): {err.value.decode()}")
    return _take_ll(lib, out_ll), {"build_ms": bms.value, "lower_ms": lms.value}


if __name__ == "__main__":
    # smoke test: build lib, expand the toy recipe at D=128, print a few lines.
    from rocke.portable_ir.src import recipe_bundle
    from rocke.portable_ir.examples import recipe_toy
    import json as _json

    cbor = recipe_bundle.cbor_encode(recipe_toy.make_recipe())
    ll, t = recipe_cbor_to_llvm(
        cbor, arch="gfx950", ints={"D": 128}, strs={"dtype": "f32"}
    )
    print(
        f"online D=128: build={t['build_ms']:.3f}ms lower={t['lower_ms']:.3f}ms "
        f"({ll.count(chr(10)) + 1} ll lines)"
    )
    print("\n".join(ll.splitlines()[:6]))
