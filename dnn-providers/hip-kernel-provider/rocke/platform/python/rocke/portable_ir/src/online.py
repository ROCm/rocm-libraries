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
    c_bool,
    c_char_p,
    c_double,
    c_int,
    c_long,
    c_size_t,
    c_ubyte,
    c_uint,
)
from typing import Dict, List, Optional, Tuple

from rocke.portable_ir.src import abi as _abi

_ERR_CAP = 256

# rocke_status_t (rocke/ir.h) and rocke_guard_verdict_t (rocke/recipe_guard.h).
_ROCKE_OK, _ROCKE_ERR_KEY = 0, 3
GUARD_ADMITTED, GUARD_REFUSED, GUARD_ABSENT = "admitted", "refused", "absent"
_VERDICTS = (GUARD_ADMITTED, GUARD_REFUSED, GUARD_ABSENT)

#: Demand a generator-verified point rather than merely a rule-legal one.
GUARD_REQUIRE_VERIFIED = 0x1


#: rocke_arg_kind_t (rocke/recipe_launch.h), by ordinal.
ARG_KINDS = ("pointer", "i32", "i64", "f32")


class _SpecInt(Structure):
    _fields_ = [("name", c_char_p), ("value", c_long)]


class _LaunchDims(Structure):
    _fields_ = [("x", c_uint), ("y", c_uint), ("z", c_uint)]


class _ArgDesc(Structure):
    """Mirror of rocke_arg_desc_t. Field order and types must match the header
    exactly; ctypes cannot check this, which is what the binary ABI version in
    _check_abi is for."""

    _fields_ = [
        ("name", c_char_p),
        ("type_name", c_char_p),
        ("kind", c_int),
        ("size", c_uint),
        ("offset", c_uint),
    ]


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
        (
            "rocke_recipe_check_guard_cbor",
            [
                POINTER(c_ubyte),
                c_size_t,
                spec_i,
                c_int,
                spec_s,
                c_int,
                c_uint,
                POINTER(c_int),
                c_char_p,
                c_size_t,
            ],
        ),
        (
            "rocke_bundle_check_guard_cbor",
            [
                POINTER(c_ubyte),
                c_size_t,
                c_char_p,
                c_char_p,
                spec_i,
                c_int,
                spec_s,
                c_int,
                c_uint,
                POINTER(c_int),
                c_char_p,
                c_size_t,
            ],
        ),
    ):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = c_int
    lib.rocke_online_free.argtypes = [c_char_p]
    lib.rocke_online_free.restype = None
    lib.rocke_bundle_contains.argtypes = [
        POINTER(c_ubyte),
        c_size_t,
        c_char_p,
        c_char_p,
    ]
    # Required, not cosmetic. ctypes defaults an unset restype to c_int and
    # reads four bytes of a one-byte `bool`; an optimized build leaves the upper
    # three as whatever was in the register, so bool(...) reads true for a
    # function that returned false. An -O0 build happens to zero them, which is
    # why this reads as correct locally and fails only under the release build
    # the gates use.
    lib.rocke_bundle_contains.restype = c_bool
    for name in ("rocke_recipe_plan_launch_cbor", "rocke_bundle_plan_launch_cbor"):
        fn = getattr(lib, name)
        bundle = name.startswith("rocke_bundle")
        fn.argtypes = (
            [POINTER(c_ubyte), c_size_t]
            + ([c_char_p, c_char_p] if bundle else [])
            + [
                POINTER(_SpecInt),
                c_int,
                POINTER(_SpecStr),
                c_int,
                POINTER(ctypes.c_void_p),
                c_char_p,
                c_size_t,
            ]
        )
        fn.restype = c_int
    lib.rocke_launch_plan_kernel_name.argtypes = [ctypes.c_void_p]
    lib.rocke_launch_plan_kernel_name.restype = c_char_p
    lib.rocke_launch_plan_geometry.argtypes = [
        ctypes.c_void_p,
        POINTER(_LaunchDims),
        POINTER(_LaunchDims),
        POINTER(c_uint),
    ]
    lib.rocke_launch_plan_geometry.restype = c_bool
    lib.rocke_launch_plan_num_args.argtypes = [ctypes.c_void_p]
    lib.rocke_launch_plan_num_args.restype = c_int
    lib.rocke_launch_plan_arg.argtypes = [ctypes.c_void_p, c_int]
    lib.rocke_launch_plan_arg.restype = POINTER(_ArgDesc)
    lib.rocke_launch_plan_kernarg_size.argtypes = [ctypes.c_void_p]
    lib.rocke_launch_plan_kernarg_size.restype = c_uint
    lib.rocke_launch_plan_free.argtypes = [ctypes.c_void_p]
    lib.rocke_launch_plan_free.restype = None
    # Provenance (rocke/rocke_build_id.h). These need an explicit restype:
    # ctypes resolves the symbol either way and defaults the return to c_int,
    # which silently truncates the pointer into a plausible-looking integer
    # rather than failing, so an unbound string getter reads as garbage.
    for name in ("rocke_engine_version", "rocke_build_id"):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = c_char_p


def provenance() -> Tuple[str, str]:
    """(engine version, build id) of the loaded engine. Debugging aid only."""
    lib = load()
    return (
        (lib.rocke_engine_version() or b"").decode(),
        (lib.rocke_build_id() or b"").decode(),
    )


def _check_abi(lib: ctypes.CDLL, path: str) -> None:
    """Refuse a library whose binary ABI differs from what these bindings assume.

    Everything above is a hand-written mirror of the C signatures in
    cpp/include/rocke/*.h, resolved by symbol name only -- ctypes cannot see
    that a struct gained a field or that an argument changed type, and calling
    through a stale mirror corrupts memory rather than failing. That makes this
    check different in kind from the wire-format one: a bad answer there is a
    wrong kernel, here it is undefined behaviour.

    Runs BEFORE the rest of the binding, so that a library from the wrong build
    is diagnosed as such instead of surfacing as a ctypes traceback about
    whichever symbol happened to be looked up first. A library predating the ABI
    export is itself the mismatch, so a missing symbol is reported the same way
    rather than skipped."""
    try:
        fn = lib.rocke_abi_version
        fn.argtypes = []
        fn.restype = c_int
        got = fn()
    except AttributeError:
        raise RuntimeError(
            f"librocke at {path!r} exports no rocke_abi_version, so it predates "
            f"the ABI contract these bindings require (v{_abi.BINARY_ABI}). "
            f"Rebuild it: online.build_lib()"
        ) from None
    if got != _abi.BINARY_ABI:
        raise RuntimeError(
            f"librocke at {path!r} has binary ABI v{got}, these bindings are "
            f"written against v{_abi.BINARY_ABI}. The library and this checkout "
            f"are from different builds; rebuild it with online.build_lib()"
        )


def _adopt(lib: ctypes.CDLL, path: str) -> ctypes.CDLL:
    """Verify the ABI, bind the entry points, and cache the library."""
    global _lib
    _check_abi(lib, path)
    _bind(lib)
    lib.rocke_recipe_abi_level.argtypes = []
    lib.rocke_recipe_abi_level.restype = c_int
    _lib = lib
    return lib


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
        return _adopt(ctypes.CDLL(explicit), explicit)

    for p in _default_lib_paths():
        if p and os.path.exists(p):
            return _adopt(ctypes.CDLL(p), p)
    # nothing prebuilt -> compile one
    p = build_lib()
    return _adopt(ctypes.CDLL(p), p)


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


def check_recipe_guard(
    cbor: bytes,
    *,
    ints: Optional[Dict[str, int]] = None,
    strs: Optional[Dict[str, str]] = None,
    require_verified: bool = False,
) -> Tuple[str, str]:
    """Ask the C engine whether a recipe serves this shape, without lowering.

    Returns `(verdict, reason)` where verdict is 'admitted', 'refused' or
    'absent' (no guard on the recipe). This is the same check the VM runs
    internally before replaying, exposed so a caller can route on the answer
    instead of paying for a failed compile -- and so the Python and C guard
    evaluators can be tested against each other."""
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    ia, ni, sa, ns = _mk_specs(ints, strs)
    verdict = c_int(1)
    reason = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.rocke_recipe_check_guard_cbor(
        buf,
        len(cbor),
        ia,
        ni,
        sa,
        ns,
        GUARD_REQUIRE_VERIFIED if require_verified else 0,
        byref(verdict),
        reason,
        _ERR_CAP,
    )
    if st != _ROCKE_OK:
        raise RuntimeError(f"guard check failed ({st}): {reason.value.decode()}")
    return _VERDICTS[verdict.value], reason.value.decode()


def check_bundle_guard(
    cbor: bytes,
    key: str,
    *,
    arch: str = "gfx950",
    ints: Optional[Dict[str, int]] = None,
    strs: Optional[Dict[str, str]] = None,
    require_verified: bool = False,
) -> Tuple[str, str]:
    """check_recipe_guard for one recipe inside a bundle.

    Raises KeyError when the bundle holds no such (key, arch). For a pruned
    bundle that is itself a rejection -- the generator did not build this -- so a
    caller usually treats it the same way it treats 'refused'."""
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    ia, ni, sa, ns = _mk_specs(ints, strs)
    verdict = c_int(1)
    reason = ctypes.create_string_buffer(_ERR_CAP)
    st = lib.rocke_bundle_check_guard_cbor(
        buf,
        len(cbor),
        key.encode(),
        arch.encode() if arch else None,
        ia,
        ni,
        sa,
        ns,
        GUARD_REQUIRE_VERIFIED if require_verified else 0,
        byref(verdict),
        reason,
        _ERR_CAP,
    )
    if st == _ROCKE_ERR_KEY:
        raise KeyError(reason.value.decode())
    if st != _ROCKE_OK:
        raise RuntimeError(f"guard check failed ({st}): {reason.value.decode()}")
    return _VERDICTS[verdict.value], reason.value.decode()


def _read_plan(lib, handle) -> Dict:
    """Copy an opaque rocke_launch_plan_t into plain Python and free it.

    Copied rather than wrapped because every string the plan hands out is owned
    by it and dies with it; a lazy accessor would hand back a c_char_p into
    freed memory the moment the caller let the handle go."""
    try:
        grid, block, lds = _LaunchDims(), _LaunchDims(), c_uint(0)
        has_geom = lib.rocke_launch_plan_geometry(
            handle, byref(grid), byref(block), byref(lds)
        )
        args = []
        for i in range(lib.rocke_launch_plan_num_args(handle)):
            a = lib.rocke_launch_plan_arg(handle, i).contents
            args.append(
                {
                    "name": a.name.decode(),
                    "type": a.type_name.decode(),
                    "kind": ARG_KINDS[a.kind],
                    "size": int(a.size),
                    "offset": int(a.offset),
                }
            )
        return {
            "kernel_name": lib.rocke_launch_plan_kernel_name(handle).decode(),
            "args": args,
            "kernarg_size": int(lib.rocke_launch_plan_kernarg_size(handle)),
            "geometry": (
                {
                    "grid": (grid.x, grid.y, grid.z),
                    "block": (block.x, block.y, block.z),
                    "lds_bytes": int(lds.value),
                }
                if has_geom
                else None
            ),
        }
    finally:
        lib.rocke_launch_plan_free(handle)


def plan_launch(
    cbor: bytes,
    key: Optional[str] = None,
    *,
    arch: str = "gfx950",
    ints: Optional[Dict[str, int]] = None,
    strs: Optional[Dict[str, str]] = None,
) -> Dict:
    """How to launch what this recipe builds: name, args, grid/block/LDS.

    `key` selects out of a bundle; omit it for a standalone recipe. Returns the
    same shape as src/launch.py::plan, which is what lets a test pin the two
    engines against each other. `geometry` is None when the recipe carries no
    launch block -- see rocke_launch_plan_geometry on why that is reported
    rather than defaulted."""
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    ia, ni, sa, ns = _mk_specs(ints, strs)
    handle = ctypes.c_void_p()
    err = ctypes.create_string_buffer(_ERR_CAP)
    if key is None:
        st = lib.rocke_recipe_plan_launch_cbor(
            buf, len(cbor), ia, ni, sa, ns, byref(handle), err, _ERR_CAP
        )
    else:
        st = lib.rocke_bundle_plan_launch_cbor(
            buf,
            len(cbor),
            key.encode(),
            arch.encode() if arch else None,
            ia,
            ni,
            sa,
            ns,
            byref(handle),
            err,
            _ERR_CAP,
        )
    if st == _ROCKE_ERR_KEY:
        raise KeyError(err.value.decode())
    if st != _ROCKE_OK:
        raise RuntimeError(f"launch plan failed ({st}): {err.value.decode()}")
    return _read_plan(lib, handle)


def bundle_contains(cbor: bytes, key: str, *, arch: Optional[str] = "gfx950") -> bool:
    """Is (key, arch) in this bundle? `arch=None` matches any arch."""
    lib = load()
    buf = (c_ubyte * len(cbor)).from_buffer_copy(cbor)
    return bool(
        lib.rocke_bundle_contains(
            buf, len(cbor), key.encode(), arch.encode() if arch else None
        )
    )


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
