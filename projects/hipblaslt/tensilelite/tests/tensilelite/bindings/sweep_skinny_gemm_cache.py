# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The kernel cache directory of the skinny GEMM tools.

The kernel names encode the compile-time knobs but not the generator or its
kernarg ABI, so a code object cached by an older emitter would be reused
silently. The cache directory is therefore <work-dir>/hsaco/<fingerprint>, where
the fingerprint hashes the content of the extension modules that emit and
assemble the kernels plus the target, the code object version and the preload,
so a rebuilt generator gets a fresh directory. The Python-side mapping
(skinnyConfig) is not hashed: a kernarg ABI change it causes is caught by
checkKernargs of skinny_gemm_support, the defense in depth. Nothing is ever
deleted here: older directories are only counted.
"""

import functools
import hashlib
import importlib.util
import os

import skinny_gemm_support as support
from sweep_skinny_gemm_config import SweepConfigError

GENERATOR_MODULES = ("tensilelite_codegen", "tensilelite_jit")
# The code object version and kernarg preload the tools emit their kernels with.
SWEEP_CODE_OBJECT_VERSION = support.DEFAULT_CODE_OBJECT_VERSION
SWEEP_PRELOAD_DWORDS = 0
FINGERPRINT_CHARS = 16


def module_path(name: str) -> str:
    """The file of a module, found without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SweepConfigError(f"cannot find the {name} module on PYTHONPATH")
    return spec.origin


def fingerprint_of(paths: list[str], abi: str) -> str:
    """Hashes the content of the files (not their paths, so the same build in
    another directory hashes the same) and the abi string."""
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as stream:
            digest.update(hashlib.sha256(stream.read()).digest())
    digest.update(abi.encode())
    return digest.hexdigest()[:FINGERPRINT_CHARS]


def abi_key() -> str:
    return (
        f"{support.TARGET} cov{SWEEP_CODE_OBJECT_VERSION} "
        f"preload{SWEEP_PRELOAD_DWORDS}"
    )


@functools.cache
def generator_fingerprint() -> str:
    """Computed once per process; the workers get the directory instead."""
    return fingerprint_of([module_path(name) for name in GENERATOR_MODULES], abi_key())


def kernel_cache_dir(work_dir: str, fingerprint: str) -> str:
    return os.path.join(work_dir, "hsaco", fingerprint)


def describe_stale(root: str, fingerprint: str) -> str:
    """The older fingerprint directories and loose .hsaco files of the cache
    root, as a sentence tail; empty when there are none."""
    dirs = files = 0
    for entry in os.scandir(root):
        if entry.is_dir():
            if entry.name != fingerprint:
                dirs += 1
        elif entry.name.endswith(".hsaco"):
            files += 1
    parts = []
    if dirs:
        parts.append(f"{dirs} older fingerprint director{'y' if dirs == 1 else 'ies'}")
    if files:
        parts.append(f"{files} unfingerprinted .hsaco file(s)")
    if not parts:
        return ""
    return (
        f"; {' and '.join(parts)} in {root} are out of date and kept "
        "(remove them manually)"
    )


def prepare_kernel_cache(work_dir: str, fingerprint: str) -> tuple[str, str]:
    """Creates the cache directory; returns it and the message to announce."""
    path = kernel_cache_dir(work_dir, fingerprint)
    new = not os.path.isdir(path)
    os.makedirs(path, exist_ok=True)
    if new:
        message = (
            f"kernel cache {path}: new generator fingerprint {fingerprint}, "
            "every kernel is built again"
        )
    else:
        message = f"kernel cache {path} (generator fingerprint {fingerprint})"
    return path, message + describe_stale(os.path.dirname(path), fingerprint)
