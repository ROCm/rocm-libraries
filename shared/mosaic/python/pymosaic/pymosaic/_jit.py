# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""JIT orchestrator for the Python layout bindings.

Given a canonicalized (shape, stride) plus optional algebra metadata,
``get_module(...)`` returns a Python extension module specialised for
that exact layout type. The module is loaded from disk if previously
compiled, or rendered + built on first request via ``ninja``.

The cache directory for a single entry contains, after a successful
compile:

    <hash>/
    ├── module.cpp              the rendered C++ source
    ├── build.ninja             the ninja build script
    ├── compile_commands.json   for clangd / editor integration
    ├── compile.log             captured stdout + stderr from ninja
    ├── manifest.json           structure, example shape/stride, version, host
    ├── python_repr.txt         the originating Python call
    ├── mosaic_jit_<hash>.so    the compiled extension
    └── debug/                  iff MOSAIC_JIT_DEBUG=1
        ├── module.E.cpp        preprocessed source (g++ -E)
        └── module.s            generated assembly

``manifest.json`` is written last; its absence indicates the entry was
never fully populated (e.g., killed mid-build) and forces a rebuild.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import shutil
import string
import subprocess
import sys
import sysconfig
import warnings
from typing import Any

from . import _render

# Bump VERSION to invalidate the entire cache after schema-incompatible
# changes (new template, new bindings, new debug artifacts).
# A version mismatch in manifest.json forces a rebuild.
#
# Version history:
#   "1" — initial: size/cosize/rank/eval/eval_coord/enumerate/repr.
#   "2" — added descriptor protocol: descriptor() + coalesce_descriptor()
#         + write_top_leg helper that lifts scalar legs into singleton
#         tuples to match the Python canonical form.
#   "3" — added right_inverse_descriptor().
#   "4" — added idx2crd / crd2idx / complement_descriptor bindings,
#         plus to_py / from_py helpers for tuple <-> nb::object.
#   "5" — DYNAMIC ROUTE: leaves are runtime std::size_t, modules keyed on
#         structure (one .so per nesting profile), values supplied at call
#         time via a LayoutInstance class. Dropped the descriptor protocol
#         and all algebra (coalesce / complement / right_inverse).
#   "6" — link the C++ runtime statically into each module by default
#         (-static-libstdc++ -static-libgcc) so the .so loads regardless of
#         the host interpreter's libstdc++ (conda-proof). Changes the link
#         line, so old entries must rebuild.
#   "7" — add -Wl,--exclude-libs,ALL to the static link. Localises the
#         embedded libstdc++/libgcc symbols so loading many structure modules
#         into one process doesn't interpose runtimes (which segfaulted).
#   "8" — add coalesce() binding (runtime fold on the flat leaf vectors).
VERSION = "8"

_TEMPLATE_PATH = pathlib.Path(__file__).parent / "_template.cpp.in"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class JitCompileError(RuntimeError):
    """Raised when ninja exits non-zero compiling a generated module.

    The instance preserves the cache directory, the ninja exit code,
    and the last few hundred lines of stderr. ``compile.log`` next to
    the module's ``module.cpp`` contains the full output.
    """

    def __init__(
        self, *, key: str, cache_dir: pathlib.Path, returncode: int, stderr: str
    ):
        self.key = key
        self.cache_dir = cache_dir
        self.returncode = returncode
        self.stderr_tail = "\n".join(stderr.splitlines()[-200:])
        super().__init__(
            f"mosaic JIT compile failed for key={key} (ninja exit={returncode})\n"
            f"  cache dir : {cache_dir}\n"
            f"  full log  : {cache_dir / 'compile.log'}\n"
            f"  source    : {cache_dir / 'module.cpp'}\n"
            f"  rebuild   : ninja -C {cache_dir}\n"
            f"--- last 200 lines of stderr ---\n{self.stderr_tail}\n"
            f"--------------------------------"
        )


# ---------------------------------------------------------------------------
# Cache root discovery
# ---------------------------------------------------------------------------


def _default_cache_root() -> pathlib.Path:
    """Resolve the JIT cache root.

    Order:
      1. $MOSAIC_JIT_CACHE
      2. $XDG_CACHE_HOME/mosaic/jit
      3. ~/.cache/mosaic/jit
      4. /tmp/mosaic-jit-$UID  (with a RuntimeWarning — non-persistent)
    """
    env = os.environ.get("MOSAIC_JIT_CACHE")
    if env:
        return pathlib.Path(env).expanduser()

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return pathlib.Path(xdg) / "mosaic" / "jit"

    home_cache = pathlib.Path.home() / ".cache"
    if _is_writable(home_cache.parent):
        # The directory itself may not exist yet; check the parent.
        return home_cache / "mosaic" / "jit"

    fallback = pathlib.Path(f"/tmp/mosaic-jit-{os.getuid()}")
    warnings.warn(
        f"mosaic JIT cache falling back to {fallback} (non-persistent across reboots); "
        f"set $MOSAIC_JIT_CACHE to a writable location to silence this.",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback


def _is_writable(path: pathlib.Path) -> bool:
    """True if ``path`` exists and is writable (or its parent is)."""
    p: pathlib.Path | None = path
    while p is not None and not p.exists():
        p = p.parent if p.parent != p else None
    return p is not None and os.access(p, os.W_OK)


def cache_root() -> pathlib.Path:
    """Public accessor for the resolved cache root (used by ``pymosaic.jit``)."""
    root = _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Path discovery — mosaic / nanobind / Python headers
# ---------------------------------------------------------------------------


def _find_mosaic_include() -> pathlib.Path:
    """Locate the directory that contains ``mosaic/mosaic.hpp``.

    Resolution order:

      1. ``$MOSAIC_INCLUDE_DIR`` — explicit override (must contain a
         ``mosaic/`` subdir).
      2. ``pymosaic/_include`` — headers vendored into the package at
         build time (the path used after a plain ``pip install``).
      3. A walk up from this file looking for ``include/mosaic/`` — covers
         editable installs and running straight from a repo checkout,
         where the headers live at ``<repo>/include/mosaic``.
    """
    env = os.environ.get("MOSAIC_INCLUDE_DIR")
    if env:
        env_path = pathlib.Path(env).expanduser()
        if (env_path / "mosaic").is_dir():
            return env_path
        raise RuntimeError(
            f"MOSAIC_INCLUDE_DIR={env!r} does not contain a 'mosaic/' "
            "subdirectory (expected mosaic/mosaic.hpp beneath it)."
        )

    package_dir = pathlib.Path(__file__).resolve().parent

    vendored = package_dir / "_include"
    if (vendored / "mosaic").is_dir():
        return vendored

    for candidate_parent in package_dir.parents:
        candidate = candidate_parent / "include" / "mosaic"
        if candidate.is_dir():
            return candidate_parent / "include"

    raise RuntimeError(
        "could not locate the mosaic C++ headers; searched "
        f"$MOSAIC_INCLUDE_DIR, {vendored}, and parents of {package_dir} "
        "for an 'include/mosaic' directory. Reinstall pymosaic, or set "
        "MOSAIC_INCLUDE_DIR to the path containing mosaic/mosaic.hpp."
    )


def _find_nanobind() -> tuple[list[pathlib.Path], pathlib.Path]:
    """Return ``(include_dirs, src_dir)`` for nanobind.

    Both are required: ``include_dirs`` are header search paths
    (nanobind's own headers PLUS the vendored ``tsl::robin_map``
    dependency under ``ext/robin_map/include``), and ``src_dir``
    contains ``nb_combined.cpp`` which we compile alongside the
    generated module to provide nanobind's runtime.
    """
    try:
        import nanobind
    except ImportError as exc:
        raise RuntimeError(
            "nanobind is required for the mosaic JIT. Install with `pip install nanobind`."
        ) from exc

    nanobind_root = pathlib.Path(nanobind.__file__).parent
    include = pathlib.Path(nanobind.include_dir())

    # nanobind's runtime depends on tsl::robin_map, which is vendored as
    # a submodule at <nanobind>/ext/robin_map/. Add its include dir
    # explicitly — nanobind's cmake module does the same.
    robin_map = nanobind_root / "ext" / "robin_map" / "include"
    if not robin_map.is_dir():
        raise RuntimeError(
            f"nanobind at {nanobind_root} is missing the vendored "
            f"tsl::robin_map dependency (expected at {robin_map}). "
            "This is unexpected for a standard pip install; reinstall "
            "nanobind to repair."
        )

    src = nanobind_root / "src"
    if not (src / "nb_combined.cpp").is_file():
        raise RuntimeError(
            f"nanobind installation at {nanobind_root} is missing "
            "nb_combined.cpp; this file ships with nanobind 2.x and is "
            "required for the JIT runtime."
        )

    return [include, robin_map], src


def _python_include() -> pathlib.Path:
    """Return the Python C-API header directory (e.g. ``Python.h``'s parent)."""
    paths = sysconfig.get_paths()
    return pathlib.Path(paths["include"])


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def cache_key(shape: _render.IntTuple, stride: _render.IntTuple) -> str:
    """Derive the 16-hex-char cache key for a layout's *structure*.

    The dynamic route keys on the nesting profile only (see
    ``_render.cache_key_payload``), so every layout sharing a structure
    maps to one key. Uses SHA-256 truncated to 64 bits.
    """
    payload = _render.cache_key_payload(shape, stride, version=VERSION)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def get_module(shape: _render.IntTuple, stride: _render.IntTuple) -> Any:
    """Return a loaded extension module for the given layout's structure.

    Cache hit: dlopen the existing ``.so`` (no compile).
    Cache miss: render template + build.ninja + run ``ninja`` + dlopen.

    The returned module exposes a ``Layout`` class; the caller constructs
    an instance from this layout's flat shape/stride values.
    """
    key = cache_key(shape, stride)
    root = cache_root()
    cache_dir = root / key
    so_name = f"mosaic_jit_{key}.so"
    so_path = cache_dir / so_name

    if not _is_complete_entry(cache_dir, so_path):
        _compile(cache_dir=cache_dir, key=key, shape=shape, stride=stride)

    return _load_module(so_path, key)


def _is_complete_entry(cache_dir: pathlib.Path, so_path: pathlib.Path) -> bool:
    """An entry counts as complete iff both the ``.so`` and
    ``manifest.json`` exist. Manifest is written last so its presence
    is the success sentinel — half-written entries (process killed,
    disk full, ninja crashed) force a rebuild instead of loading a
    stale or partial binary.
    """
    if not so_path.is_file():
        return False
    if not (cache_dir / "manifest.json").is_file():
        return False
    return True


# ---------------------------------------------------------------------------
# Compile pipeline
# ---------------------------------------------------------------------------


def _compile(
    *,
    cache_dir: pathlib.Path,
    key: str,
    shape: _render.IntTuple,
    stride: _render.IntTuple,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    layout_type = _render.render_layout_type(shape, stride)
    module_cpp = _render_module_cpp(key=key, layout_type=layout_type)
    (cache_dir / "module.cpp").write_text(module_cpp)

    mosaic_include = _find_mosaic_include()
    nb_includes, nb_src = _find_nanobind()
    py_include = _python_include()

    cxx_flags = _cxx_flags(
        mosaic_include=mosaic_include,
        nb_includes=nb_includes,
        py_include=py_include,
    )

    build_ninja = _render_build_ninja(
        key=key,
        cxx_flags=cxx_flags,
        nb_combined_src=nb_src / "nb_combined.cpp",
    )
    (cache_dir / "build.ninja").write_text(build_ninja)

    (cache_dir / "compile_commands.json").write_text(
        _render_compile_commands(cache_dir=cache_dir, cxx_flags=cxx_flags)
    )

    (cache_dir / "python_repr.txt").write_text(
        _render.render_python_repr(shape, stride) + "\n"
    )

    proc = subprocess.run(
        ["ninja", "-C", str(cache_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    # Always preserve the build log — captures warnings on success, full
    # context on failure. ~few KB; cheap.
    (cache_dir / "compile.log").write_text(
        f"--- command: ninja -C {cache_dir} ---\n"
        f"--- exit code: {proc.returncode} ---\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n"
    )

    if proc.returncode != 0:
        raise JitCompileError(
            key=key,
            cache_dir=cache_dir,
            returncode=proc.returncode,
            stderr=proc.stderr,
        )

    if os.environ.get("MOSAIC_JIT_DEBUG") == "1":
        _emit_debug_artifacts(cache_dir=cache_dir, cxx_flags=cxx_flags)

    # manifest.json is written LAST — its presence is the success sentinel.
    # The module is keyed on structure, so shape/stride here are recorded as
    # the *representative* layout that first triggered the compile.
    manifest = {
        "structure": _render.structure_signature(shape, stride),
        "example_shape": _render._to_json(shape),
        "example_stride": _render._to_json(stride),
        "version": VERSION,
        "key": key,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": _host_fingerprint(),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def _render_module_cpp(*, key: str, layout_type: str) -> str:
    """Substitute the template's $-placeholders.

    Using ``string.Template`` instead of ``str.format`` so the C++
    braces don't need doubling — generated source stays readable.
    """
    template = string.Template(_TEMPLATE_PATH.read_text())
    return template.substitute(key=key, layout_type=layout_type)


def _cxx_flags(
    *,
    mosaic_include: pathlib.Path,
    nb_includes: list[pathlib.Path],
    py_include: pathlib.Path,
) -> list[str]:
    """Compile flags shared by both compile rules in the ninja file."""
    flags = [
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-fvisibility=hidden",
        "-fno-strict-aliasing",  # recommended by nanobind
        "-Wno-unused-function",  # template instantiations regularly trigger this
        f"-I{mosaic_include}",
        f"-I{py_include}",
    ]
    for inc in nb_includes:
        flags.append(f"-I{inc}")
    return flags


def _render_build_ninja(
    *,
    key: str,
    cxx_flags: list[str],
    nb_combined_src: pathlib.Path,
) -> str:
    """Render a minimal build.ninja for the per-entry cache directory.

    Two compile units (module.cpp + nb_combined.cpp) → link to a
    single shared object. We rebuild nb_combined.cpp per-entry for
    now; sharing one .o across the whole cache is a future
    optimisation (needs careful invalidation if the nanobind / gcc /
    Python version changes).
    """
    cxx = os.environ.get("CXX", "g++")
    cxxflags = " ".join(cxx_flags)
    so_name = f"mosaic_jit_{key}.so"

    # Statically link the C++ runtime into the module by default. This makes
    # the .so independent of whatever libstdc++ the host interpreter ships,
    # which is what makes conda/miniconda environments work: conda's bundled
    # libstdc++ is frequently OLDER than the system g++ used to build here, so
    # a dynamically-linked module fails to load with "GLIBCXX_3.4.xx not
    # found".
    #
    # CRITICAL: pair the static link with `--exclude-libs,ALL`, which localises
    # every symbol pulled from a static archive (libstdc++.a, libgcc.a). Many
    # distinct structure modules get dlopen'd into ONE process; without
    # localisation each module would export its own libstdc++ symbols, the
    # loader would interpose one module's runtime onto another's, and the
    # resulting two-runtimes-in-one-process state segfaults. Localising keeps
    # each module's C++ runtime fully private. `PyInit_*` lives in our own
    # object files (not an archive) so it stays exported.
    #
    # Safe overall because each JIT module is a leaf that only exchanges plain
    # Python objects through nanobind — no C++ std types cross the .so
    # boundary. Opt out with MOSAIC_JIT_STATIC_CXX=0.
    ldflags = "-shared"
    if os.environ.get("MOSAIC_JIT_STATIC_CXX", "1") != "0":
        ldflags += " -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL"

    return f"""\
# Auto-generated by pymosaic._jit.
# Regenerate by clearing this cache entry and re-running the Python call.

cxx      = {cxx}
cxxflags = {cxxflags}
ldflags  = {ldflags}

rule cxx
  command = $cxx -MMD -MF $out.d $cxxflags -c $in -o $out
  description = CXX $in
  depfile = $out.d
  deps = gcc

rule link
  command = $cxx $ldflags $in -o $out
  description = LINK $out

build module.o: cxx module.cpp
build nb_combined.o: cxx {nb_combined_src}

build {so_name}: link module.o nb_combined.o

default {so_name}
"""


def _render_compile_commands(*, cache_dir: pathlib.Path, cxx_flags: list[str]) -> str:
    """Render a one-entry compile_commands.json for clangd / VSCode.

    Lets editors resolve mosaic / nanobind / Python includes when you
    open module.cpp directly. Tiny file (<1 KB).
    """
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, *cxx_flags, "-c", "module.cpp", "-o", "module.o"]
    entry = {
        "directory": str(cache_dir),
        "file": str(cache_dir / "module.cpp"),
        "output": str(cache_dir / "module.o"),
        "arguments": cmd,
    }
    return json.dumps([entry], indent=2)


def _emit_debug_artifacts(*, cache_dir: pathlib.Path, cxx_flags: list[str]) -> None:
    """Emit ``debug/module.E.cpp`` and ``debug/module.s`` for inspection.

    Off by default. Enable with ``MOSAIC_JIT_DEBUG=1``. Roughly doubles
    per-entry on-disk footprint but is the right tool when chasing
    template instantiation errors or verifying constant-folding.
    """
    debug = cache_dir / "debug"
    debug.mkdir(exist_ok=True)
    cxx = os.environ.get("CXX", "g++")

    # Preprocessed source — shows what the compiler actually sees after
    # macro and template expansion.
    subprocess.run(
        [cxx, *cxx_flags, "-E", "module.cpp", "-o", str(debug / "module.E.cpp")],
        cwd=cache_dir,
        capture_output=True,
        check=False,
    )

    # Generated assembly — shows what got constant-folded.
    subprocess.run(
        [cxx, *cxx_flags, "-S", "module.cpp", "-o", str(debug / "module.s")],
        cwd=cache_dir,
        capture_output=True,
        check=False,
    )


def _host_fingerprint() -> dict[str, str]:
    """Coarse host identity stamped into manifest.json. Useful when
    debugging "why did this cache entry build fine on machine A but
    fail on machine B" without committing to a precise dependency
    graph.
    """
    try:
        gcc_version = subprocess.check_output(
            [os.environ.get("CXX", "g++"), "-dumpfullversion"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        gcc_version = "unknown"

    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split(maxsplit=1)[0],
        "cxx": os.environ.get("CXX", "g++"),
        "cxx_version": gcc_version,
    }


# ---------------------------------------------------------------------------
# dlopen
# ---------------------------------------------------------------------------


def _load_module(so_path: pathlib.Path, key: str) -> Any:
    """Load a compiled ``.so`` via ``importlib.util.spec_from_file_location``.

    This is the same idiom PyTorch's ``cpp_extension.load`` uses.
    """
    module_name = f"mosaic_jit_{key}"

    # If we already loaded this key in the current process, return the
    # cached module — multiple Layout objects with the same hash share
    # one module instance. (Re-importing a .so twice with different
    # names in the same process is undefined behaviour for some
    # extension types.)
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to construct ImportSpec for {so_path}; this should never happen unless the .so is corrupted."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        sys.modules.pop(module_name, None)
        msg = str(exc)
        if any(tok in msg for tok in ("GLIBCXX", "CXXABI", "libstdc++")):
            raise ImportError(
                f"{msg}\n\n"
                "The JIT module compiled fine but could not load: this is a C++ "
                "runtime (libstdc++) version mismatch. The module was built with a "
                "newer g++ than the libstdc++ shipped with the Python you're running "
                "— most often a conda/miniconda environment whose bundled libstdc++ "
                "is older than your system compiler.\n"
                "Fixes (in order of preference):\n"
                "  1. Rebuild with the C++ runtime statically linked (the default). "
                "Ensure MOSAIC_JIT_STATIC_CXX is not '0', then clear the stale cache:\n"
                "       import pymosaic.jit as j; j.clear_cache()\n"
                "  2. Upgrade the environment's runtime:\n"
                "       conda install -c conda-forge libstdcxx-ng\n"
                "  3. Match the toolchain to the env:\n"
                "       conda install gxx && export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++\n"
                "  4. Stopgap preload:\n"
                "       LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python ..."
            ) from exc
        raise
    return module


# ---------------------------------------------------------------------------
# Cache management (re-exported by the public pymosaic.jit module)
# ---------------------------------------------------------------------------


def cache_info() -> dict[str, Any]:
    """Summary stats for the JIT cache. Used by ``pymosaic.jit.cache_info()``."""
    root = cache_root()
    entries = 0
    total_bytes = 0
    if root.is_dir():
        for entry in root.iterdir():
            if entry.is_dir() and (entry / "manifest.json").is_file():
                entries += 1
                for p in entry.rglob("*"):
                    if p.is_file():
                        total_bytes += p.stat().st_size
    return {
        "entries": entries,
        "size_bytes": total_bytes,
        "path": str(root),
        "version": VERSION,
    }


def clear_cache(*, key: str | None = None) -> int:
    """Remove cache entries. With ``key=None`` (default), remove every
    entry. With ``key=...``, remove only the matching entry.

    Returns the number of entries removed.
    """
    root = cache_root()
    if not root.is_dir():
        return 0

    if key is not None:
        target = root / key
        if target.is_dir():
            shutil.rmtree(target)
            return 1
        return 0

    removed = 0
    for entry in root.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
            removed += 1
    return removed
