# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pinned build environment for the Triton AOT generator.

* Environment: ``TRITON_DISABLE_LINE_INFO=1``, a fresh ``TRITON_CACHE_DIR``,
  every overriding knob cleared, interpreter/override modes refused.
* Version/shadow guards on the imported ``triton``.
* The gfx950 erratum guard: a wrapper over the wheel's
  ``triton.backends.amd.compiler.llvm`` that appends ``-packed-fp32-ops`` to the
  features string for gfx950 when the LLVM IR holds an MFMA/WMMA intrinsic.
* ``compile_instance``: one ``triton.compile`` with explicit ``HIPOptions``.

``triton`` is imported lazily, so the pure helpers import without it.
"""

from __future__ import annotations

import atexit
import dataclasses
import importlib.util
import os
import re
import shutil
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _contract as C
from ._codeobject import GeneratorError

GENERATORS_DIR = Path(__file__).resolve().parent
TRITON_DIR = GENERATORS_DIR.parent
HKP_DIR = TRITON_DIR.parent
REQUIREMENTS = TRITON_DIR / "requirements.txt"
KERNEL_SOURCE = TRITON_DIR / C.KERNEL_SOURCE_FILE
CONTENT_DIR = (
    HKP_DIR
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / C.FAMILY_FOLDER
)

# --- Environment knobs ------------------------------------------------------------------

REFUSED_KNOBS = ("TRITON_INTERPRET", "TRITON_KERNEL_OVERRIDE")
CLEARED_EXACT = ("TRITON_OVERRIDE_ARCH", "TRITON_ALWAYS_COMPILE")
CLEARED_PREFIXES = ("AMDGCN_", "TRITON_HIP_")
CLEARED_SUFFIXES = ("_FP_FUSION",)
# Recorded in the manifest as a fixed list derived from the predicates above (not
# the names that happened to be set), so the manifest does not depend on the
# invoking environment.
CLEARED_RECORD = tuple(
    sorted(
        (
            *CLEARED_EXACT,
            *(f"{prefix}*" for prefix in CLEARED_PREFIXES),
            *(f"*{suffix}" for suffix in CLEARED_SUFFIXES),
        )
    )
)

_prepared = False
_cache_dirs: list[str] = []


def _env_error(message: str) -> GeneratorError:
    return GeneratorError(f"{C.ERR_ENV}: {message}")


def _is_cleared(name: str) -> bool:
    return (
        name in CLEARED_EXACT
        or name.startswith(CLEARED_PREFIXES)
        or name.endswith(CLEARED_SUFFIXES)
    )


def _remove_cache_dirs() -> None:
    for path in _cache_dirs:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_remove_cache_dirs)


def new_cache_dir() -> str:
    """Point ``TRITON_CACHE_DIR`` at a fresh empty directory (removed at exit).

    A fresh directory per compile also guarantees no cache hit can bypass the
    erratum guard (the Triton cache key does not know about it).
    """
    path = tempfile.mkdtemp(prefix="tritongen-cache-")
    _cache_dirs.append(path)
    os.environ["TRITON_CACHE_DIR"] = path
    return path


def prepare() -> None:
    """Apply the environment scrub to this process. Idempotent; must precede ``import triton``."""
    global _prepared
    if _prepared:
        return
    for name in REFUSED_KNOBS:
        if os.environ.get(name):
            raise _env_error(f"{name} is set; refusing to generate under it")
    for name in [n for n in os.environ if _is_cleared(n)]:
        del os.environ[name]
    os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
    new_cache_dir()
    _prepared = True


def env_record() -> dict:
    """The environment-knob snapshot for the manifest."""
    f32 = os.environ.get("TRITON_F32_DEFAULT")
    return {
        "TRITON_DISABLE_LINE_INFO": os.environ.get("TRITON_DISABLE_LINE_INFO"),
        "TRITON_CACHE_DIR": "<fresh per run>",
        "TRITON_F32_DEFAULT": "unset" if f32 is None else f32,
        "cleared": list(CLEARED_RECORD),
    }


def wheel_sha256() -> str:
    """The wheel hash from the ``triton==`` line of ``requirements.txt``."""
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if line.startswith("triton=="):
            match = re.search(r"--hash=sha256:([0-9a-f]{64})", line)
            if match is None:
                raise _env_error(f"{REQUIREMENTS}: triton line carries no sha256 hash")
            sha = match.group(1)
            if sha != C.TRITON_WHEEL_SHA256:
                raise _env_error(
                    f"{REQUIREMENTS} hash {sha} differs from _contract {C.TRITON_WHEEL_SHA256}"
                )
            return sha
    raise _env_error(f"{REQUIREMENTS}: no triton== line")


# --- triton import + guards ------------------------------------------------------

_triton = None


def import_triton():
    """Import ``triton`` after ``prepare()``; assert version, API, and no shadowing."""
    global _triton
    if _triton is not None:
        return _triton
    prepare()
    import triton  # noqa: PLC0415

    if triton.__version__ != C.TRITON_VERSION:
        raise _env_error(
            f"triton {triton.__version__} imported, expected {C.TRITON_VERSION}"
        )
    if not callable(getattr(triton, "compile", None)):
        raise _env_error("triton.compile is missing")
    location = getattr(triton, "__file__", None)
    paths = sysconfig.get_paths()
    sites = {Path(paths[key]).resolve() for key in ("purelib", "platlib")}
    if location is None or not any(
        Path(location).resolve().is_relative_to(site) for site in sites
    ):
        raise _env_error(
            f"triton imported from {location!r}, not from site-packages "
            f"{sorted(map(str, sites))} "
            "(a directory named 'triton' is shadowing the wheel?)"
        )
    _triton = triton
    return triton


def load_kernel():
    """The ``@triton.jit`` RMSNorm kernel, loaded from its source file."""
    import_triton()
    return load_jit_kernel(KERNEL_SOURCE, C.KERNEL_SYMBOL, "tritongen_rmsnorm_src")


def load_jit_kernel(path: Path, attribute: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


# --- Erratum guard ---------------------------------------------------------------

GUARD_FEATURE = "-packed-fp32-ops"
GUARD_ARCH = "gfx950"
_GUARDED_CALLS = ("translate_to_mir", "dump_sched_dag", "translate_to_asm")
_MFMA_IR = re.compile(r"@llvm\.amdgcn\.(?:mfma|smfmac|wmma)\.")


def guard_applies(arch: str, ir: str) -> bool:
    return arch == GUARD_ARCH and _MFMA_IR.search(ir) is not None


def guarded_features(arch: str, features: str, ir: str) -> tuple[str, bool]:
    """``(features, applied)``: the features string with the guard rule applied."""
    if not guard_applies(arch, ir):
        return features, False
    return ",".join(x for x in (features, GUARD_FEATURE) if x), True


class _GuardProxy:
    """Stands in for ``triton.backends.amd.compiler.llvm``; forwards everything.

    ``fired`` is set when the rule changed a call's features during the current
    compile; ``enabled=False`` forces the guard off (negative fixtures only).
    """

    def __init__(self, module) -> None:
        self._module = module
        self.enabled = True
        self.fired = False

    def __getattr__(self, name):
        target = getattr(self._module, name)
        if name not in _GUARDED_CALLS:
            return target

        def wrapped(src, triple, arch, features, flags, *args, **kwargs):
            if self.enabled:
                features, applied = guarded_features(arch, features, src)
                self.fired = self.fired or applied
            return target(src, triple, arch, features, flags, *args, **kwargs)

        return wrapped


def install_guard() -> _GuardProxy:
    """Wrap the wheel's ``llvm`` module once; returns the (shared) proxy."""
    import_triton()
    import triton.backends.amd.compiler as amd_compiler  # noqa: PLC0415

    if not isinstance(amd_compiler.llvm, _GuardProxy):
        amd_compiler.llvm = _GuardProxy(amd_compiler.llvm)
    return amd_compiler.llvm


# --- Compile ---------------------------------------------------------------------


@dataclass(frozen=True)
class Compiled:
    hsaco: bytes
    name: str
    shared: int
    num_warps: int
    warp_size: int
    global_scratch_size: int
    profile_scratch_size: int
    guard_applied: bool


def assert_options_match_wheel_defaults() -> None:
    """The fixed HIPOptions pins must equal the wheel's own defaults."""
    import_triton()
    from triton.backends.amd.compiler import HIPOptions  # noqa: PLC0415

    defaults = {f.name: f.default for f in dataclasses.fields(HIPOptions)}
    for key, value in C.HIP_OPTIONS_FIXED.items():
        if key not in defaults or defaults[key] != value:
            raise _env_error(
                f"HIPOptions.{key} default is {defaults.get(key)!r}, pinned {value!r}"
            )


def compile_instance(
    fn,
    signature: dict,
    constexprs: dict,
    attrs: dict,
    arch: str,
    num_warps: int,
    *,
    guard: bool = True,
) -> Compiled:
    """One ``triton.compile`` with explicit ``HIPOptions``; returns the object + metadata.

    ``guard=False`` forces the erratum guard off (test-only: the negative fixture).
    """
    triton = import_triton()
    assert_options_match_wheel_defaults()
    proxy = install_guard()
    from triton.backends.compiler import GPUTarget  # noqa: PLC0415
    from triton.compiler import ASTSource  # noqa: PLC0415

    new_cache_dir()
    proxy.enabled = guard
    proxy.fired = False
    try:
        source = ASTSource(fn, signature=signature, constexprs=constexprs, attrs=attrs)
        compiled = triton.compile(
            source,
            target=GPUTarget(C.GPU_TARGET_BACKEND, arch, C.WARP_SIZE),
            options=C.hip_options(num_warps),
        )
    except Exception as exc:
        raise GeneratorError(f"{arch}: triton.compile failed: {exc}") from exc
    finally:
        proxy.enabled = True
    fired = proxy.fired

    metadata = compiled.metadata
    fields = (
        metadata._asdict() if hasattr(metadata, "_asdict") else dict(vars(metadata))
    )
    return Compiled(
        hsaco=compiled.asm["hsaco"],
        name=fields["name"],
        shared=fields["shared"],
        num_warps=fields["num_warps"],
        warp_size=fields.get("warp_size", C.WARP_SIZE),
        global_scratch_size=fields["global_scratch_size"],
        profile_scratch_size=fields["profile_scratch_size"],
        guard_applied=fired,
    )
