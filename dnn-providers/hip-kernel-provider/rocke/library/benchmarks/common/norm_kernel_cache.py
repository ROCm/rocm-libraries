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
import pickle
from pathlib import Path
from typing import Any, Mapping

from benchmarks.common.norm_bench_common import compile_candidate


def cache_key(
    *,
    family: str,
    dims: Mapping[str, int],
    dtype: str,
    arch: str,
    candidate_name: str,
    dynamic: bool = False,
) -> str:
    if dynamic:
        # Shape-free: this candidate's compiled kernel admits any runtime
        # dims, so including them would scatter one tuning config's
        # artifact across as many cache entries as shapes tried against it.
        raw = f"{family}|dynamic|{dtype}|{arch}|{candidate_name}"
    else:
        dims_part = ",".join(f"{k}={v}" for k, v in sorted(dims.items()))
        raw = f"{family}|{dims_part}|{dtype}|{arch}|{candidate_name}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
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
        )
        + ".pkl"
    )
    if path.is_file():
        with path.open("rb") as fh:
            return pickle.load(fh)

    artifact = compile_candidate(spec, candidate, arch=arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # atomic: a reader never observes a partial file
    return artifact
