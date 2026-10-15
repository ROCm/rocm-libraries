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
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
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
_lib_path: str | None = None


def _platform_root() -> str:
    """rocke/platform/ -- the CMake source dir for the engine. Derived from
    __file__ so the tree stays relocatable (python/rocke/portable_ir/src)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", ".."))


def source_build_id(platform_root: str | Path | None = None) -> str | None:
    """Return the CMake-compatible ID for the portable-IR native bundle.

    The ID covers repository-owned inputs that determine the shared engine and
    replay CLI: engine sources/headers, the CLI source, and their CMake recipes.
    It is a source/build-recipe ID, not a compiler or binary identity. Installed
    Python packages carry no engine source surface; ``None`` means freshness
    cannot be established locally.
    """
    root = Path(platform_root or _platform_root())
    bindings = root / "cpp" / "bindings"
    files = list((root / "cpp").rglob("*.cpp"))
    files += list((root / "cpp").rglob("*.h"))
    files += list((root / "cpp").rglob("*.hpp"))
    files = [path for path in files if bindings not in path.parents]
    if not files:
        return None
    for relative in (
        "tests/portable_ir/replay_cli.cpp",
        "CMakeLists.txt",
        "tests/CMakeLists.txt",
        "cmake/rocke_build_id.cmake",
        "cmake/generate_rocke_build_id.cmake",
    ):
        path = root / relative
        if path.is_file():
            files.append(path)
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{path.relative_to(root).as_posix()}:{digest}")
    blob = "\n".join(sorted(lines)).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _current_source_build_id() -> str | None:
    # Keep artifact selection/build boundaries sensitive to edits made by a
    # long-lived watcher. The loaded-engine hot path does not call this helper.
    return source_build_id()


def provenance_manifest(path: str | Path | None = None) -> tuple[str, str] | None:
    """Read the expected ``(version, build ID)`` for an installed artifact.

    A source checkout computes its expectation directly. Relocated tests have no
    native sources, so their build installs this independent manifest beside the
    prebuilt replay CLI and selects it through ``ROCKE_PROVENANCE_MANIFEST``.
    """
    selected = path or os.environ.get("ROCKE_PROVENANCE_MANIFEST")
    if selected is None:
        return None
    manifest = Path(selected)
    if not manifest.is_file():
        raise FileNotFoundError(
            f"portable-IR provenance manifest not found at {str(manifest)!r}"
        )
    parts = manifest.read_text().strip().split()
    if len(parts) != 2:
        raise RuntimeError(
            f"invalid portable-IR provenance manifest at {str(manifest)!r}"
        )
    return parts[0], parts[1]


def _expected_build_id() -> str | None:
    """Return the checkout or installed-manifest artifact expectation."""
    source_id = _current_source_build_id()
    installed = provenance_manifest()
    if installed is not None and source_id is not None and installed[1] != source_id:
        raise RuntimeError(
            "portable-IR provenance manifest does not match the current source: "
            f"manifest={installed[1]}, source={source_id}"
        )
    return source_id or (installed[1] if installed is not None else None)


def _shared_library_name() -> str:
    if sys.platform == "win32":
        return "rocke.dll"
    if sys.platform == "darwin":
        return "librocke.dylib"
    return "librocke.so"


def _executable_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def _default_cache_root() -> Path:
    build_id = _current_source_build_id() or "unknown-source"
    return Path(tempfile.gettempdir()) / "rocke_online" / build_id


def _default_cache_lib() -> str:
    return str(_default_cache_root() / _shared_library_name())


def _default_cache_cli() -> str:
    return str(_default_cache_root() / _executable_name("rocke_portable_ir_replay_cli"))


def _complete_marker(path: str | Path) -> Path:
    return Path(f"{path}.complete")


def _published(path: str | Path) -> bool:
    artifact = Path(path)
    marker = _complete_marker(artifact)
    if not artifact.is_file() or not marker.is_file():
        return False
    try:
        return marker.read_text().strip() == (
            _current_source_build_id() or "unknown-source"
        )
    except OSError:
        return False


@contextmanager
def _cache_lock(root: Path):
    """Serialize builders with an OS lock released automatically on process exit."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".build.lock"
    lock_file = lock_path.open("a+b")
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"0")
        lock_file.flush()
    deadline = time.monotonic() + 1200
    acquired = False
    try:
        while not acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for rocKE cache lock {lock_path}"
                    )
                time.sleep(0.1)
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _default_lib_paths() -> List[str]:
    return [
        os.path.join(_platform_root(), "build", _shared_library_name()),
    ]


def resolve_library_path(path: str | None = None, *, build: bool = True) -> str | None:
    """Resolve the exact shared engine used by the ctypes replay path.

    An explicit path is a demand and never falls back. Without one, the local
    cache is keyed by the current portable-IR bundle ID, so changing checkouts
    cannot silently reuse an older engine. When ``build`` is false, ``None``
    reports that no matching implicit artifact exists.
    """
    explicit = path or os.environ.get("ROCKE_ONLINE_LIB")
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(
                f"librocke not found at {explicit!r} (from "
                f"{'resolve_library_path(path)' if path else 'ROCKE_ONLINE_LIB'}). "
                "Refusing to fall back to another library"
            )
        return explicit
    if _current_source_build_id() is None:
        if not build:
            return None
        raise RuntimeError(
            "cannot discover or build librocke implicitly because this installed "
            "package has no portable-IR native source tree; set "
            "ROCKE_ONLINE_LIB explicitly"
        )
    cached = _default_cache_lib()
    if _published(cached):
        return cached
    for candidate in _default_lib_paths():
        if os.path.exists(candidate):
            return candidate
    return build_lib() if build else None


def _configure_cache_build(root: Path, output: Path) -> Path:
    build_dir = root / "core"
    args = [
        "cmake",
        "-S",
        _platform_root(),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DROCKE_BUILD_PYENV=OFF",
        "-DROCKE_BUILD_SHARED_ENGINE=ON",
        "-DBUILD_TESTING=ON",
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output}",
    ]
    for config in ("DEBUG", "RELEASE", "RELWITHDEBINFO", "MINSIZEREL"):
        args += [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config}={output}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{config}={output}",
        ]
    subprocess.run(
        args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    return build_dir


def _build_cached_target(target: str, destination: Path, produced_name: str) -> str:
    root = destination.parent
    with _cache_lock(root):
        if _published(destination):
            return str(destination)
        stage = root / f".stage-{os.getpid()}"
        stage.mkdir(parents=True, exist_ok=True)
        build_dir = _configure_cache_build(root, stage)
        try:
            subprocess.run(
                [
                    "cmake",
                    "--build",
                    str(build_dir),
                    "--config",
                    "Debug",
                    "--target",
                    target,
                    "-j",
                    str(os.cpu_count() or 8),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            produced = stage / produced_name
            if not produced.is_file():
                raise FileNotFoundError(
                    f"CMake target {target} did not produce {produced}"
                )
            os.replace(produced, destination)
            marker = _complete_marker(destination)
            marker.write_text(_current_source_build_id() or "unknown-source")
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return str(destination)


def build_lib(out_path: Optional[str] = None) -> str:
    """Build the portable shared engine through its cross-platform CMake target."""
    destination = Path(out_path or _default_cache_lib())
    destination.parent.mkdir(parents=True, exist_ok=True)
    built = _build_cached_target("rocke_shared", destination, _shared_library_name())
    return built


def build_replay_cli(out_path: str | None = None) -> str:
    """Build the standalone replay CLI in the source-keyed artifact cache."""
    destination = Path(out_path or _default_cache_cli())
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _build_cached_target(
        "rocke_portable_ir_replay_cli",
        destination,
        _executable_name("rocke_portable_ir_replay_cli"),
    )


def replay_cli_provenance(path: str) -> tuple[str, str]:
    """Return and validate the standalone replay CLI's version and build ID."""
    proc = subprocess.run(
        [path, "--provenance"], capture_output=True, text=True, check=True
    )
    parts = proc.stdout.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"invalid provenance from replay CLI at {path!r}")
    version, build_id = parts
    expected = _expected_build_id()
    if expected is not None and build_id != expected:
        raise RuntimeError(
            "stale/mixed rocKE replay CLI: "
            f"{path!r} has build-id {build_id}, but the selected environment "
            f"expects {expected}. Rebuild or reinstall the replay CLI and its "
            "provenance manifest from the same source snapshot"
        )
    return version, build_id


def resolve_replay_cli_path(
    path: str | None = None, *, build: bool = True
) -> str | None:
    """Resolve a source-matched standalone replay CLI, building when requested."""
    explicit = path or os.environ.get("ROCKE_REPLAY_CLI")
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(
                f"replay CLI not found at {explicit!r}; refusing to fall back"
            )
        replay_cli_provenance(explicit)
        return explicit
    if _current_source_build_id() is None:
        if not build:
            return None
        raise RuntimeError(
            "cannot discover or build the replay CLI implicitly because this "
            "installed package has no portable-IR native source tree; set "
            "ROCKE_REPLAY_CLI explicitly"
        )
    cached = _default_cache_cli()
    if _published(cached):
        replay_cli_provenance(cached)
        return cached
    if not build:
        return None
    built = build_replay_cli()
    replay_cli_provenance(built)
    return built


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
    lib.rocke_launch_plan_cluster.argtypes = [ctypes.c_void_p, POINTER(_LaunchDims)]
    lib.rocke_launch_plan_cluster.restype = c_bool
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


def library_provenance(path: str) -> tuple[str, str]:
    """Return ``(engine version, build ID)`` after validating a library."""
    lib = ctypes.CDLL(path)
    _check_source_freshness(lib, path)
    _check_abi(lib, path)
    for name in ("rocke_engine_version", "rocke_build_id"):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = c_char_p
    return (
        (lib.rocke_engine_version() or b"").decode(),
        (lib.rocke_build_id() or b"").decode(),
    )


def validate_artifact_bundle(
    library_path: str, replay_cli_path: str
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Validate a library/CLI pair and return their provenance records.

    The common source/build-recipe ID is the compatibility key. Engine versions
    include a configure date and are descriptive, so independently built
    artifacts may have different versions while still belonging to one bundle.
    This comparison also works for source-less installations, where no checkout
    exists to provide an expected ID.
    """
    library = library_provenance(library_path)
    replay_cli = replay_cli_provenance(replay_cli_path)
    if library[1] != replay_cli[1]:
        raise RuntimeError(
            "portable-IR artifacts are from mixed bundles: "
            f"library={library[0]}/{library[1]}, "
            f"CLI={replay_cli[0]}/{replay_cli[1]}"
        )
    return library, replay_cli


def provenance() -> Tuple[str, str]:
    """Return ``(engine version, build ID)`` for the loaded engine."""
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


def _check_source_freshness(lib: ctypes.CDLL, path: str) -> None:
    """Reject an engine that differs from its source or installed manifest."""
    expected = _expected_build_id()
    if expected is None:
        return
    try:
        fn = lib.rocke_build_id
        fn.argtypes = []
        fn.restype = c_char_p
        got = (fn() or b"").decode()
    except AttributeError:
        got = "<missing>"
    if got != expected:
        raise RuntimeError(
            "stale/mixed rocKE engine: "
            f"librocke at {path!r} has build-id {got}, but the selected "
            f"environment expects {expected}. Rebuild or reinstall the library "
            "and its provenance manifest from the same source snapshot"
        )


def _adopt(lib: ctypes.CDLL, path: str) -> ctypes.CDLL:
    """Verify source freshness and ABI, then cache the library."""
    global _lib, _lib_path
    _check_source_freshness(lib, path)
    _check_abi(lib, path)
    _bind(lib)
    lib.rocke_recipe_abi_level.argtypes = []
    lib.rocke_recipe_abi_level.restype = c_int
    _lib = lib
    _lib_path = str(Path(path).resolve())
    return lib


def load(path: Optional[str] = None) -> ctypes.CDLL:
    global _lib
    explicit = path or os.environ.get("ROCKE_ONLINE_LIB")
    if _lib is not None:
        if explicit and str(Path(explicit).resolve()) != _lib_path:
            raise RuntimeError(
                "rocKE engine already loaded from "
                f"{_lib_path!r}; refusing to switch this process to {explicit!r}"
            )
        # _adopt() validated this immutable native image before publishing it to
        # the process. Re-hashing the complete source tree here would put that
        # filesystem cost on every online API call without making the already-
        # loaded image replaceable; source edits require a new process/build.
        return _lib

    # A named library is a demand, not a hint. Skipping a missing ROCKE_ONLINE_LIB
    # and quietly loading the next candidate -- a cached /tmp build, or a fresh
    # compile -- means the caller verifies an engine it did not build, and the
    # substitution is invisible. That is fatal for a byte-identity gate, whose
    # entire claim is about the engine at this commit, so say so instead.
    resolved = resolve_library_path(path)
    assert resolved is not None
    return _adopt(ctypes.CDLL(resolved), resolved)


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
        cluster = _LaunchDims()
        has_cluster = lib.rocke_launch_plan_cluster(handle, byref(cluster))
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
            "cluster": ((cluster.x, cluster.y, cluster.z) if has_cluster else None),
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
    """How to launch what this recipe builds: name, args, grid/block/LDS, cluster.

    `key` selects out of a bundle; omit it for a standalone recipe. Returns the
    same shape as src/launch.py::plan, which is what lets a test pin the two
    engines against each other. `geometry` is None when the recipe carries no
    launch block -- see rocke_launch_plan_geometry on why that is reported
    rather than defaulted. `cluster` is None unless the kernel has a
    cluster_dims attr (see rocke_launch_plan_cluster)."""
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
