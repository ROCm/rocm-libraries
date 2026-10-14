# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Prepare the environment, import the pinned ``triton``, and compile one instance.

``triton`` is imported lazily, so the helpers that do not compile work without it.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import importlib.util
import os
import re
import shutil
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import _family as F
from ._family import GeneratorError

# --- Environment -------------------------------------------------------------------

# Refused outright: either one makes triton.compile return something other than a
# compile of the kernel source.
REFUSED_ENV = ("TRITON_INTERPRET", "TRITON_KERNEL_OVERRIDE")
# Every other variable is removed before triton is imported; only these survive.
ALLOWED_ENV = ("HOME", "PATH", "ROCM_PATH")
# Set by the generator itself (TRITON_CACHE_DIR is a fresh directory per compile).
FIXED_ENV = {"TRITON_DISABLE_LINE_INFO": "1"}

_prepared = False
_cache_dirs: list[str] = []


def _env_error(message: str) -> GeneratorError:
    return GeneratorError(f"{F.ERR_ENV}: {message}")


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
    """Reduce this process's environment to the allowlist plus the fixed values.
    Idempotent; must precede ``import triton``."""
    global _prepared
    if _prepared:
        return
    for name in REFUSED_ENV:
        if os.environ.get(name):
            raise _env_error(f"{name} is set; refusing to generate under it")
    for name in [n for n in os.environ if n not in ALLOWED_ENV]:
        del os.environ[name]
    os.environ.update(FIXED_ENV)
    _prepared = True


def env_record() -> dict:
    """The compile environment for the manifest: the allowlisted names and the fixed
    values, never the host's values."""
    return {
        "allowed": list(ALLOWED_ENV),
        **FIXED_ENV,
        "TRITON_CACHE_DIR": "<fresh per compile>",
    }


def wheel_sha256() -> str:
    """The wheel hash from the ``triton==`` line of ``requirements.txt``."""
    for line in F.REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        if line.startswith("triton=="):
            match = re.search(r"--hash=sha256:([0-9a-f]{64})", line)
            if match is None:
                raise _env_error(
                    f"{F.REQUIREMENTS}: triton line carries no sha256 hash"
                )
            sha = match.group(1)
            if sha != F.TRITON_WHEEL_SHA256:
                raise _env_error(
                    f"{F.REQUIREMENTS} hash {sha} differs from the pinned "
                    f"{F.TRITON_WHEEL_SHA256}"
                )
            return sha
    raise _env_error(f"{F.REQUIREMENTS}: no triton== line")


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

    if triton.__version__ != F.TRITON_VERSION:
        raise _env_error(
            f"triton {triton.__version__} imported, expected {F.TRITON_VERSION}"
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


def libtriton_sha256() -> str:
    """SHA256 of the imported wheel's ``triton/_C/libtriton.so`` (measured)."""
    triton = import_triton()
    path = Path(triton.__file__).resolve().parent / "_C" / "libtriton.so"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise _env_error(f"cannot read {path}: {exc}") from exc


def load_kernel(op):
    """The ``@triton.jit`` kernel of op contract ``op``, loaded from its source file."""
    import_triton()
    spec = importlib.util.spec_from_file_location(
        f"tritongen_{op.OP}_src", F.TRITON_DIR / op.KERNEL_SOURCE_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, op.KERNEL_SYMBOL)


# --- Compile ---------------------------------------------------------------------


@dataclass(frozen=True)
class Compiled:
    hsaco: bytes
    name: str
    dynamic_lds_bytes: int
    num_warps: int
    warp_size: int
    global_scratch_size: int
    profile_scratch_size: int


def assert_options_match_wheel_defaults() -> None:
    """The fixed HIPOptions pins must equal the wheel's own defaults."""
    import_triton()
    from triton.backends.amd.compiler import HIPOptions  # noqa: PLC0415

    defaults = {f.name: f.default for f in dataclasses.fields(HIPOptions)}
    for key, value in F.HIP_OPTIONS_FIXED.items():
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
            target=GPUTarget(F.GPU_TARGET_BACKEND, arch, F.WARP_SIZE),
            options=F.hip_options(num_warps),
        )
    except Exception as exc:
        raise GeneratorError(
            f"{F.ERR_COMPILE}: {arch}: triton.compile failed: {exc}"
        ) from exc

    fields = compiled.metadata._asdict()
    return Compiled(
        hsaco=compiled.asm["hsaco"],
        name=fields["name"],
        dynamic_lds_bytes=fields["shared"],
        num_warps=fields["num_warps"],
        warp_size=fields["warp_size"],
        global_scratch_size=fields["global_scratch_size"],
        profile_scratch_size=fields["profile_scratch_size"],
    )
