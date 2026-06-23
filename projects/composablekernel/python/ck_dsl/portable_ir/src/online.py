# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# online.py -- in-process binding for the ONLINE portable-IR path. The Python
# builder serializes a kernel (concrete portable-IR JSON, or a CBOR recipe /
# bundle) and hands it to the pure-C backend, which expands + lowers it to
# AMDGPU LLVM IR -- all in this process, no subprocess, no pybind build step.
#
# Transport is ctypes over the C ABI (every entry point is extern "C"), so the
# only requirement is a shared libckc. Point CKC_LIB at it, or pass a path to
# load(); build_lib() will compile one from the ckc sources on demand.
#
# This is the same C core the OFFLINE provider uses (ckc_lower + recipe VM); the
# online and offline paths therefore produce byte-identical output.
import ctypes
import os
import subprocess
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


def _default_lib_paths() -> List[str]:
    paths = []
    env = os.environ.get("CKC_LIB")
    if env:
        paths.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(os.environ.get("TMPDIR", "/tmp"), "ckc_online", "libckc.so")
    paths += [cache, os.path.join(here, "..", "..", "..", "ck_dsl_c", "build", "libckc.so")]
    return paths


def build_lib(out_path: Optional[str] = None) -> str:
    """Build a shared libckc.so: the C++ engine core (libckc_core.a, via CMake)
    plus the flat-C portable-IR tooling (json/cbor DOM, importer, recipe VM,
    online wrappers) linked together. The engine core moved to src/core/**/*.cpp,
    so the tooling can no longer self-link from src/*.c alone."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    ckc = os.path.normpath(os.path.join(here, "..", "..", "..", "ck_dsl_c"))
    out_path = out_path or os.path.join(os.environ.get("TMPDIR", "/tmp"), "ckc_online", "libckc.so")
    base = os.path.dirname(out_path)
    coredir, objdir = os.path.join(base, "core"), os.path.join(base, "obj")
    os.makedirs(objdir, exist_ok=True)
    inc = os.path.join(ckc, "include")
    # 1) C++ engine core archive
    subprocess.run(["cmake", "-S", ckc, "-B", coredir, "-DCMAKE_BUILD_TYPE=Debug"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run(["cmake", "--build", coredir, "--target", "ckc_core",
                    "-j", str(os.cpu_count() or 8)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    core = os.path.join(coredir, "libckc_core.a")
    # 2) flat-C portable-IR tooling
    srcs = sorted(glob.glob(os.path.join(ckc, "src", "*.c")))
    subprocess.run(["cc", "-std=c99", "-O2", "-fPIC", "-I", inc, "-c", *srcs],
                   cwd=objdir, check=True)
    objs = sorted(glob.glob(os.path.join(objdir, "*.o")))
    # 3) link into one shared lib (whole-archive so all engine symbols export)
    subprocess.run(["c++", "-shared", "-fPIC", *objs,
                    "-Wl,--whole-archive", core, "-Wl,--no-whole-archive", "-lm",
                    "-o", out_path], check=True)
    return out_path


def _bind(lib: ctypes.CDLL) -> None:
    spec_i = POINTER(_SpecInt)
    spec_s = POINTER(_SpecStr)
    dbl = POINTER(c_double)
    for name, argtypes in (
        ("ckc_online_recipe_cbor_to_llvm",
         [POINTER(c_ubyte), c_size_t, spec_i, c_int, spec_s, c_int, c_char_p,
          POINTER(c_char_p), dbl, dbl, c_char_p, c_size_t]),
        ("ckc_online_bundle_cbor_to_llvm",
         [POINTER(c_ubyte), c_size_t, c_char_p, c_char_p, spec_i, c_int, spec_s, c_int,
          POINTER(c_char_p), dbl, dbl, c_char_p, c_size_t]),
        ("ckc_online_ir_json_to_llvm",
         [c_char_p, c_char_p, POINTER(c_char_p), dbl, dbl, c_char_p, c_size_t]),
    ):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = c_int
    lib.ckc_online_free.argtypes = [c_char_p]
    lib.ckc_online_free.restype = None


def load(path: Optional[str] = None) -> ctypes.CDLL:
    global _lib
    if _lib is not None and path is None:
        return _lib
    candidates = [path] if path else _default_lib_paths()
    last = None
    for p in candidates:
        if p and os.path.exists(p):
            lib = ctypes.CDLL(p)
            _bind(lib)
            _lib = lib
            return lib
        last = p
    # nothing prebuilt -> compile one
    p = build_lib()
    lib = ctypes.CDLL(p)
    _bind(lib)
    _lib = lib
    return lib


def _mk_specs(ints: Optional[Dict[str, int]], strs: Optional[Dict[str, str]]):
    ints = ints or {}
    strs = strs or {}
    ia = (_SpecInt * len(ints))(*[
        _SpecInt(k.encode(), int(v)) for k, v in ints.items()]) if ints else None
    sa = (_SpecStr * len(strs))(*[
        _SpecStr(k.encode(), str(v).encode()) for k, v in strs.items()]) if strs else None
    return ia, len(ints), sa, len(strs)


def _take_ll(lib, out_ll: c_char_p) -> str:
    s = ctypes.cast(out_ll, c_char_p).value
    text = s.decode() if s else ""
    lib.ckc_online_free(out_ll)
    return text


def recipe_cbor_to_llvm(cbor: bytes, *, arch: str = "gfx950",
                        ints: Optional[Dict[str, int]] = None,
                        strs: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, float]]:
    """Expand+lower a CBOR recipe in C. Returns (.ll, {build_ms, lower_ms})."""
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    ia, ni, sa, ns = _mk_specs(ints, strs)
    out_ll = c_char_p()
    bms, lms = c_double(0), c_double(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.ckc_online_recipe_cbor_to_llvm(buf, len(cbor), ia, ni, sa, ns, arch.encode(),
                                            byref(out_ll), byref(bms), byref(lms), err, _ERR_CAP)
    if st != 0:
        raise RuntimeError(f"online recipe lower failed ({st}): {err.value.decode()}")
    return _take_ll(lib, out_ll), {"build_ms": bms.value, "lower_ms": lms.value}


def bundle_cbor_to_llvm(cbor: bytes, key: str, *, arch: str = "gfx950",
                        ints: Optional[Dict[str, int]] = None,
                        strs: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, float]]:
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    ia, ni, sa, ns = _mk_specs(ints, strs)
    out_ll = c_char_p()
    bms, lms = c_double(0), c_double(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.ckc_online_bundle_cbor_to_llvm(buf, len(cbor), key.encode(), arch.encode(),
                                            ia, ni, sa, ns, byref(out_ll), byref(bms), byref(lms),
                                            err, _ERR_CAP)
    if st != 0:
        raise RuntimeError(f"online bundle lower failed ({st}): {err.value.decode()}")
    return _take_ll(lib, out_ll), {"build_ms": bms.value, "lower_ms": lms.value}


def ir_json_to_llvm(text: str, *, arch: str = "gfx950") -> Tuple[str, Dict[str, float]]:
    lib = load()
    out_ll = c_char_p()
    bms, lms = c_double(0), c_double(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.ckc_online_ir_json_to_llvm(text.encode(), arch.encode(), byref(out_ll),
                                        byref(bms), byref(lms), err, _ERR_CAP)
    if st != 0:
        raise RuntimeError(f"online ir-json lower failed ({st}): {err.value.decode()}")
    return _take_ll(lib, out_ll), {"build_ms": bms.value, "lower_ms": lms.value}


if __name__ == "__main__":
    # smoke test: build lib, expand the toy recipe at D=128, print a few lines.
    from ck_dsl.portable_ir.src import recipe_bundle
    from ck_dsl.portable_ir.examples import recipe_toy
    import json as _json
    cbor = recipe_bundle.cbor_encode(recipe_toy.make_recipe())
    ll, t = recipe_cbor_to_llvm(cbor, arch="gfx950", ints={"D": 128}, strs={"dtype": "f32"})
    print(f"online D=128: build={t['build_ms']:.3f}ms lower={t['lower_ms']:.3f}ms "
          f"({ll.count(chr(10)) + 1} ll lines)")
    print("\n".join(ll.splitlines()[:6]))
