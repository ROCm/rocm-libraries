# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-disk ``KernelArtifact`` cache for the norm2d/batchnorm2d AOT sweeps.

``norm_bench_common``'s docstring explains why this family has no persistent
HSACO cache dir like conv's ``kernel_cache.KernelCache``: its candidate space
is normally a few dozen entries, cheap enough to just compile on demand. That
assumption breaks for batchnorm2d's ``n_groups > 1`` masked tail on large-C
shapes, where the candidate count runs into the hundreds -- compiling all of
them inline, inside a GPU-device-pinned shard worker (``run_norm_shards.py``),
serializes CPU-bound compile time behind 8-way GPU parallelism and starves it
of the other ~248 idle cores on an 8-GPU box. The fix is the same shape conv
already uses (``benchmark_implicit_gemm_conv.py``'s ``--compile-all`` /
``--run-from-cache``): compile once, across as many CPU workers as the box
has (``norm_compile_all.py``), then have the GPU-pinned run phase load
already-built artifacts instead of building them.

``KernelArtifact`` (``rocke.helpers.compile.KernelArtifact``) is a plain
``@dataclass`` of picklable data (a frozen ``KernelDef`` IR tree, IR/LLVM
text, hsaco bytes, timings) built before any GPU/HIP handle exists, so the
cache is just ``pickle.dump``/``pickle.load`` of the whole object -- no need
to reconstruct a ``KernelLauncher`` from raw hsaco+signature separately; every
existing caller already does ``KernelLauncher.from_artifact(artifact)``, which
keeps working unchanged on an unpickled artifact.

The key includes the full case shape (not just ``candidate.name``) for a
shape-specialized candidate: the compiled kernel is shape-specialized (its
``kernel_name`` embeds N/C/H/W, e.g.
``rocke_batchnorm2d_fwd_f16_N1C64H28W28_b1024_v8``), so two cases with the
same candidate but different dims must not collide.

A **dynamic** (AOT-generic) candidate -- ``spec.dynamic is True``, see
``instances/common/batchnorm2d.py``/``layernorm2d.py``'s ``dynamic`` spec
field -- is the one case where this does NOT hold: its compiled kernel
genuinely serves any runtime N/H/W (or ``cols``), so ``dims`` is dropped
from the key entirely and ``candidate.name`` alone (already unique per
tuning config -- block_size/vec/channels_per_block/n_groups/fast_reduce,
see ``dispatch/families/batchnorm.py``/``norm.py``'s dynamic-candidate
``spec_id``) is the cache key. This is what lets ``norm_compile_all.py``
compile a dynamic candidate exactly once and have every MIOpen-driver case
that resolves to it hit the SAME cache entry, mirroring how conv's
``KernelCache`` keys on ``KernelIdentity`` (a shape-free tuning-config
tuple) rather than on a specific problem.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

from benchmarks.common.norm_bench_common import compile_candidate


def _code_digest() -> str:
    """Digest of the kernel-source modules a cached artifact is built from.

    Part of every key, so editing a kernel builder invalidates the cache instead
    of serving the old binary under an unchanged candidate name.
    """
    global _CODE_DIGEST
    if _CODE_DIGEST is None:
        import rocke.instances.common.batchnorm2d as _bn
        import rocke.instances.common.batchnorm2d_ro as _ro
        import rocke.instances.common.layernorm2d as _ln

        h = hashlib.sha1()
        for mod in (_bn, _ro, _ln):
            h.update(Path(mod.__file__).read_bytes())
        _CODE_DIGEST = h.hexdigest()[:12]
    return _CODE_DIGEST


_CODE_DIGEST: "str | None" = None


def _load_or_compile(path: Path, name: str, build) -> Any:
    """Return the pickled artifact at *path*; on a miss run ``build()`` (honouring
    ``ROCKE_NORM_CACHE_ONLY=1``, which turns a miss into an error) and store it."""
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)
    if os.environ.get("ROCKE_NORM_CACHE_ONLY") == "1":
        raise RuntimeError(
            f"AOT cache miss (ROCKE_NORM_CACHE_ONLY=1): {name} -> {path}"
        )
    artifact = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".pkl.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # atomic: a reader never observes a partial file
    return artifact


def _hashed_path(cache_dir: str, prefix: str, raw: str) -> Path:
    digest = hashlib.sha1(f"{raw}|{_code_digest()}".encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"{prefix}{digest}.pkl"


def cache_key(
    *,
    family: str,
    dims: Mapping[str, int],
    dtype: str,
    arch: str,
    candidate_name: str,
    dynamic: bool = False,
    kernel_name: str = "",
) -> str:
    if dynamic:
        # Shape-free: this candidate's compiled kernel admits any runtime
        # dims, so including them would scatter one tuning config's
        # artifact across as many cache entries as shapes tried against it.
        # The candidate name alone is not unique: one candidate serves both
        # training and inference, so the spec's kernel name (which carries
        # those flags) is part of the key.
        raw = f"{family}|dynamic|{dtype}|{arch}|{candidate_name}|{kernel_name}"
    else:
        dims_part = ",".join(f"{k}={v}" for k, v in sorted(dims.items()))
        raw = f"{family}|{dims_part}|{dtype}|{arch}|{candidate_name}"
    digest = hashlib.sha1(f"{raw}|{_code_digest()}".encode("utf-8")).hexdigest()[:24]
    return f"{family}_{digest}"


def compile_candidate_cached(
    spec: Any,
    candidate: Any,
    *,
    arch: str,
    cache_dir: str,
    family: str,
    dims: Mapping[str, int],
    dtype: str,
):
    """Load *candidate*'s ``KernelArtifact`` from *cache_dir*, compiling + caching on a miss.

    ``dims`` is ignored for the cache key when ``spec.dynamic`` is true
    (see :func:`cache_key`); the build itself is unaffected -- a dynamic
    spec builds the same way regardless of which case asked for it.
    """
    dynamic = bool(getattr(spec, "dynamic", False))
    path = Path(cache_dir) / (
        cache_key(
            family=family,
            dims=dims,
            dtype=dtype,
            arch=arch,
            candidate_name=candidate.name,
            dynamic=dynamic,
            kernel_name=spec.kernel_name() if hasattr(spec, "kernel_name") else "",
        )
        + ".pkl"
    )
    return _load_or_compile(
        path, candidate.name, lambda: compile_candidate(spec, candidate, arch=arch)
    )


def compile_split_cached(
    cand: Any, kind: str, *, dtype: str, arch: str, cache_dir: str
):
    """Load / build one kernel (``kind`` = ``"reduce"`` or ``"apply"``) of a
    dynamic split candidate. The key is the kernel name alone (shape- and
    split-plan-free): the split geometry is a runtime kernarg."""
    from rocke.helpers.compile import compile_kernel

    spec = cand.spec(dtype, arch)
    kname = spec.kernel_name(kind)
    raw = f"batchnorm2d_split|dynamic|{dtype}|{arch}|{cand.name}|{kname}"

    def build():
        kdef = (
            cand.build_reduce(spec, arch)
            if kind == "reduce"
            else cand.build_apply(spec, arch)
        )
        return compile_kernel(kdef, arch=arch)

    return _load_or_compile(
        _hashed_path(cache_dir, "batchnorm2d_split_", raw), kname, build
    )


def compile_nhwc_cached(cand: Any, kind: str, *, dtype: str, arch: str, cache_dir: str):
    """Load / build one kernel of an NHWC candidate. The key is the candidate and
    kernel name alone: shape and launch plan are runtime kernel arguments."""
    from rocke.helpers.compile import compile_kernel

    spec = cand.spec(dtype, arch)
    kname = spec.kernel_name(kind)
    raw = f"batchnorm2d_nhwc|dynamic|{dtype}|{arch}|{cand.name}|{kname}"
    return _load_or_compile(
        _hashed_path(cache_dir, "batchnorm2d_nhwc_", raw),
        kname,
        lambda: compile_kernel(cand.build(kind, spec, arch), arch=arch),
    )


def compile_mcw_cached(cand: Any, *, dtype: str, arch: str, cache_dir: str):
    """Load / build the kernel of a multi-channel-per-wave candidate. The key is the
    candidate and kernel name alone: the shape is a runtime kernel argument."""
    from rocke.helpers.compile import compile_kernel

    spec = cand.spec(dtype, arch)
    kname = spec.kernel_name()
    raw = f"batchnorm2d_mcw|dynamic|{dtype}|{arch}|{cand.name}|{kname}"
    return _load_or_compile(
        _hashed_path(cache_dir, "batchnorm2d_mcw_", raw),
        kname,
        lambda: compile_kernel(cand.build("fused", spec, arch), arch=arch),
    )


def compile_ro_cached(cand: Any, *, dtype: str, arch: str, cache_dir: str):
    """Load / build the kernel of a read-once candidate. The key is the candidate
    and kernel name alone: shape and launch plan are runtime kernel arguments."""
    from rocke.helpers.compile import compile_kernel

    spec = cand.spec(dtype, arch)
    kname = spec.kernel_name("bwd" if cand.backward else "train")
    raw = f"batchnorm2d_ro|dynamic|{dtype}|{arch}|{cand.name}|{kname}"
    return _load_or_compile(
        _hashed_path(cache_dir, "batchnorm2d_ro_", raw),
        kname,
        lambda: compile_kernel(cand.build(spec), arch=arch),
    )
