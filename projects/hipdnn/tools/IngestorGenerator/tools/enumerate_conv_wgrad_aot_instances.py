#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocKE gfx950 backward-weight conv AOT instance lists the hipkernel pack ships.

The wgrad sibling of ``enumerate_conv_dgrad_aot_instances.py``. The catalog is rocKE's
own registered instance list
(``dispatch/grouped_convolution_instances/gfx950_wgrad.json``, read with
:func:`read_rocke_catalog`) -- rocKE is the only source of "what instances exist"; this
tool mirrors nothing by hand. :data:`EXCLUDES` (optional; empty when absent) names
instances rocKE registers that the pack must not ship (for example a rocKE kernel known
to compute wrong results), each with a reason -- the one legitimate thing hand curation
used to do beyond mirroring rocKE.

Implicit-GEMM entries must be identities of the wgrad AOT sweep
(``benchmarks/common/kernel_sweep.py``) and direct entries wgrad identities of the
direct sweep (``benchmarks/common/direct_kernel_sweep.py``), for gfx950, fp16 or bf16.
Every entry must also convert (``make_conv_wgrad_aot_config.py``); nothing is dropped
silently.

The five partial-slot direct families (``kernels.conv.direct_variants.DIRECT_WGRAD_FOLDED``)
launch a fold helper (``direct_wgrad_fold``) the labels do not name. rocKE's own
``direct_wgrad_fold_knobs()`` decides its knobs, in launch order, from the folded
identity's filter and channels-per-group; this tool asks it rather than restating the
rule, so a pipeline whose fold changes comes out right too. Likewise a two-stage
implicit-GEMM identity launches the Stage 2 fold of its f32 scratch
(``direct_wgrad_ws_reduce``), which the dispatcher names
(``dispatch.grouped_convolution._instance_helpers``). Every direct identity, on a graph
whose dW is 16-bit, also launches the cast of its fp32 dW into it: that same Stage 2 at
one slab (``convolution_backward_filter.CAST_WS_REPLICAS``). Helpers are never
selectable on their own.

The output is one JSON list per dtype (:data:`INSTANCE_LIST`), one compact identity
per line, ordered by family (implicit GEMM, direct, helper), then ``stable_hash``::

    python tools/enumerate_conv_wgrad_aot_instances.py --rocke-root "$ROCKE" \\
        --output-dir configs
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import enumerate_conv_aot_instances as _fwd  # noqa: E402

ARCH = _fwd.ARCH
DTYPES = _fwd.DTYPES

#: File name of one dtype's instance list.
INSTANCE_LIST = "gfx950_conv_wgrad_aot_{dtype}.instances.json"
DEFAULT_OUTPUT_DIR = _fwd.DEFAULT_OUTPUT_DIR
#: Known-bug exclusions: instances rocKE registers that this pack must not ship,
#: each with a reason. JSON list of ``{"label": ..., "reason": ...}``; absent
#: file means no exclusions.
EXCLUDES = DEFAULT_OUTPUT_DIR / "gfx950_conv_wgrad_aot.excludes.json"
#: The family order of the written lists.
FAMILIES = ("implicit", "direct", "helper")

_bind_rocke = _fwd._bind_rocke
read_rocke_catalog = _fwd.read_rocke_catalog
read_excludes = _fwd.read_excludes
stable_hash = _fwd.stable_hash
compact_identity = _fwd.compact_identity
write_instances = _fwd.write_instances
identity_label = _fwd.identity_label


def _converter():
    """``make_conv_wgrad_aot_config``, the tool that consumes the list."""
    import make_conv_wgrad_aot_config

    return make_conv_wgrad_aot_config


def family(identity: dict) -> str:
    """``implicit``, ``direct`` or ``helper`` (the wgrad partial-slot fold)."""
    if identity["algorithm"] == "implicit_gemm":
        return "implicit"
    if identity["direction"] == "direct_wgrad_helper":
        return "helper"
    return "direct"


def sweep_hashes(dtype: str, jobs: int) -> set:
    """``stable_hash`` of every valid wgrad identity of rocKE's implicit-GEMM AOT
    grid and of its direct grid (the fold helper included), and of the Stage 2
    each two-stage implicit-GEMM one launches, for ``dtype``."""
    from benchmarks.common import kernel_sweep
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(ARCH)
    implicit = kernel_sweep.enumerate_jobs(
        arch=ARCH, dtype=dtype, target=target, directions=("wgrad",), jobs=jobs
    )
    direct = enumerate_direct_jobs(
        arch=ARCH, target=target, directions=("wgrad",), dtype=dtype, jobs=jobs
    )
    from benchmarks.common.kernel_cache import KernelIdentity
    from dispatch.grouped_convolution import _instance_helpers

    identities = [job.identity for job in [*implicit, *direct]]
    # The two-stage Stage 2 is no sweep job of its own: it is the helper the
    # dispatcher names for a two-stage implicit-GEMM identity; nor is the direct
    # pipelines' cast, which is that Stage 2 at one slab.
    stage2 = [h for i in identities if i.needs_stage2 for h in _instance_helpers(i)]
    casts = [
        KernelIdentity.from_dict(c)
        for i in identities
        for c in cast_identities(i.to_dict())
    ]
    return {i.stable_hash() for i in [*identities, *stage2, *casts]}


def helper_identities(identity: dict) -> list:
    """The fold-helper identity a folded wgrad direct identity's pipeline launches, as
    the direct sweep pulls it in (``direct_kernel_sweep._caps_jobs``); ``[]`` for
    anything else.

    The fold and its knobs come from rocKE's own ``direct_wgrad_fold_knobs()``, which
    is what the sweep and the launcher both read, so a pipeline that changes its fold
    needs no change here."""
    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.kernel_cache import KernelIdentity
    from kernels.conv.direct_variants import (
        DIRECT_WGRAD_FOLD,
        DIRECT_WGRAD_FOLDED,
        direct_wgrad_fold_knobs,
    )

    variant = identity["algorithm"]
    if variant not in DIRECT_WGRAD_FOLDED:
        return []
    ident = KernelIdentity.from_dict(identity)
    caps = dks._caps_of(ident)
    fold_knobs = direct_wgrad_fold_knobs(variant, caps.KH, caps.KW, caps.cpg)
    helper = dks._helper_identity(
        ident.arch, ident.wave_size, DIRECT_WGRAD_FOLD, caps, fold_knobs, ident.dtype_a
    )
    from dataclasses import replace

    return [replace(helper, llvm_flavor=ident.llvm_flavor).to_dict()]


def stage2_identities(identity: dict) -> list:
    """The Stage 2 a two-stage implicit-GEMM identity launches, as the dispatcher
    names it (``_instance_helpers``); ``[]`` for anything else."""
    from benchmarks.common.kernel_cache import KernelIdentity
    from dispatch.grouped_convolution import _instance_helpers

    ident = KernelIdentity.from_dict(identity)
    if ident.direction != "wgrad" or not ident.needs_stage2:
        return []
    return [helper.to_dict() for helper in _instance_helpers(ident)]


def cast_identities(identity: dict) -> list:
    """The cast a wgrad direct identity launches on a 16-bit dW: the Stage 2 of its
    dtype at ``CAST_WS_REPLICAS`` slabs; ``[]`` for anything else."""
    from dataclasses import replace

    from benchmarks.common.kernel_cache import KernelIdentity
    from builders.common.convolution_backward_filter import CAST_WS_REPLICAS
    from kernels.conv.direct_aot_jobs import _ws_reduce_identity

    if identity["direction"] != "direct_wgrad":
        return []
    ident = KernelIdentity.from_dict(identity)
    cast = _ws_reduce_identity(ident.arch, ident.wave_size, CAST_WS_REPLICAS, ident.dtype_a)
    return [replace(cast, llvm_flavor=ident.llvm_flavor).to_dict()]


def with_helpers(identities: list) -> list:
    """``identities`` plus the helper each entry needs (the fold of a folded direct
    entry and every direct entry's cast, the Stage 2 of a two-stage implicit-GEMM one),
    each once, in order."""
    out, seen = [], set()
    for identity in identities:
        for entry in [
            identity,
            *helper_identities(identity),
            *cast_identities(identity),
            *stage2_identities(identity),
        ]:
            key = stable_hash(entry)
            if key not in seen:
                seen.add(key)
                out.append(entry)
    return out


def validate(identities: list, jobs: int = 1) -> list:
    """``(stable_hash, reason)`` for each identity the converter rejects."""
    converted = _converter().instance_specs(identities, jobs)
    return [
        (stable_hash(identity), reason)
        for identity, (spec, reason) in zip(identities, converted)
        if spec is None
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root",
        type=Path,
        required=True,
        help="rocKE tree (platform/, library/).",
    )
    parser.add_argument(
        "--excludes",
        type=Path,
        default=EXCLUDES,
        help="Known-bug exclusions from rocKE's catalog, JSON "
        '[{"label":..., "reason":...}, ...] (default: %(default)s; absent '
        "file means none).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for the per-dtype lists ({INSTANCE_LIST}; default: %(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    _bind_rocke(args.rocke_root)

    rocke_catalog = read_rocke_catalog("wgrad")
    excludes = read_excludes(args.excludes)
    rocke_labels = {identity_label(i) for i in rocke_catalog}
    stale = set(excludes) - rocke_labels
    if stale:
        print(
            f"{args.excludes}: exclude(s) not in rocKE's gfx950 wgrad catalog: "
            f"{sorted(stale)}",
            file=sys.stderr,
        )
        return 1
    listed = [i for i in rocke_catalog if identity_label(i) not in excludes]
    others = sorted({i["dtype_a"] for i in listed} - set(DTYPES))
    if others:
        print(
            f"rocKE's gfx950 wgrad catalog: dtype(s) {others} are not packaged",
            file=sys.stderr,
        )
        return 1
    helpers_listed = [i for i in listed if family(i) == "helper"]
    if helpers_listed:
        print(
            f"rocKE's gfx950 wgrad catalog: helper kernels are derived, not "
            f"registered: {identity_label(helpers_listed[0])}",
            file=sys.stderr,
        )
        return 1

    lists, failed = {}, False
    print(f"{'dtype':6s} {'implicit':>8s} {'direct':>6s} {'helper':>6s}")
    for dtype in sorted(DTYPES):
        identities = with_helpers([i for i in listed if i["dtype_a"] == dtype])
        swept = sweep_hashes(dtype, args.jobs)
        for identity in identities:
            if stable_hash(identity) not in swept:
                print(
                    f"not a {ARCH} wgrad sweep identity: {identity_label(identity)}",
                    file=sys.stderr,
                )
                failed = True
        for instance_id, reason in validate(identities, args.jobs):
            print(f"unsupported instance {instance_id}: {reason}", file=sys.stderr)
            failed = True
        families = collections.defaultdict(list)
        for identity in identities:
            families[family(identity)].append((stable_hash(identity), identity))
        print(
            f"{dtype:6s} "
            + " ".join(f"{len(families[f]):{w}d}" for f, w in zip(FAMILIES, (8, 6, 6)))
        )
        lists[dtype] = [i for f in FAMILIES for _, i in sorted(families[f])]
    if failed:
        print("unsupported catalog entries; nothing written", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dtype, identities in lists.items():
        path = args.output_dir / INSTANCE_LIST.format(dtype=dtype)
        write_instances(path, identities)
        print(f"{len(identities)} instance(s) written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
