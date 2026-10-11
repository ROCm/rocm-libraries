# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prepare the environment, import the pinned ``triton``, and compile one instance.

``triton`` is imported lazily, so the helpers that do not compile work without it.
"""

from __future__ import annotations

import atexit
import dataclasses
import importlib.util
import os
import re
import shutil
import sys
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

    A fresh directory per compile means every object comes from a full compile,
    never from a Triton cache hit.
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
    try:
        import triton  # noqa: PLC0415
    except ImportError as exc:
        raise _env_error(
            f"triton is not importable by {sys.executable}: {exc} "
            "(run with the generator venv)"
        ) from exc

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
    spec = importlib.util.spec_from_file_location(
        "tritongen_rmsnorm_src", KERNEL_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, C.KERNEL_SYMBOL)


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
) -> Compiled:
    """One ``triton.compile`` with explicit ``HIPOptions``; returns the object + metadata."""
    triton = import_triton()
    from triton.backends.compiler import GPUTarget  # noqa: PLC0415
    from triton.compiler import ASTSource  # noqa: PLC0415

    new_cache_dir()
    try:
        source = ASTSource(fn, signature=signature, constexprs=constexprs, attrs=attrs)
        compiled = triton.compile(
            source,
            target=GPUTarget(C.GPU_TARGET_BACKEND, arch, C.WARP_SIZE),
            options=C.hip_options(num_warps),
        )
    except Exception as exc:
        raise GeneratorError(
            f"{C.ERR_COMPILE}: {arch}: triton.compile failed: {exc}"
        ) from exc

    fields = compiled.metadata._asdict()
    return Compiled(
        hsaco=compiled.asm["hsaco"],
        name=fields["name"],
        shared=fields["shared"],
        num_warps=fields["num_warps"],
        warp_size=fields["warp_size"],
        global_scratch_size=fields["global_scratch_size"],
        profile_scratch_size=fields["profile_scratch_size"],
    )
