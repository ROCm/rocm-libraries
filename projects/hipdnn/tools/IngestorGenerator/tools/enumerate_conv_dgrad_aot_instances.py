#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocKE gfx950 backward-data conv AOT instance lists the hipkernel pack ships.

The dgrad sibling of ``enumerate_conv_aot_instances.py``. The catalog is a curated list
of rocKE kernels (:data:`CATALOG_LABELS`), one ``KernelIdentity.label()`` per line.
Implicit-GEMM entries must be identities of the dgrad AOT sweep
(``benchmarks/common/kernel_sweep.py``) and direct entries dgrad identities of the
direct sweep (``benchmarks/common/direct_kernel_sweep.py``), for gfx950, fp16 or bf16.
Every entry must also convert (``make_conv_dgrad_aot_config.py``); nothing is dropped
silently. Kernels left out of the catalog on purpose (for example a rocKE kernel known
to compute wrong results) are removed from the labels file, with the reason in its
``#`` header.

The MFMA direct pipeline (``direct_grouped_dgrad_mfma``) launches weight-transform
helper kernels the labels do not name: the transpose always, the reorganize (keyed on
``fold_k32``) when the entry splits K over waves or walks it at run time. This tool
adds exactly the helpers the listed MFMA entries need, as the direct sweep pulls them
in, so the pack can load them; they are never selectable on their own.

The output is one JSON list per dtype (:data:`INSTANCE_LIST`), one compact identity
per line, ordered by family (implicit GEMM, direct, helpers), then ``stable_hash``::

    python tools/enumerate_conv_dgrad_aot_instances.py --rocke-root "$ROCKE" \\
        --output-dir configs
"""

from __future__ import annotations

import argparse
import collections
import json
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
INSTANCE_LIST = "gfx950_conv_dgrad_aot_{dtype}.instances.json"
DEFAULT_OUTPUT_DIR = _fwd.DEFAULT_OUTPUT_DIR
#: The curated catalog: one ``KernelIdentity.label()`` per line; blank lines and
#: lines starting with ``#`` are ignored.
CATALOG_LABELS = DEFAULT_OUTPUT_DIR / "gfx950_conv_dgrad_aot.labels.txt"
#: The family order of the written lists.
FAMILIES = ("implicit", "direct", "helper")

_bind_rocke = _fwd._bind_rocke
read_labels = _fwd.read_labels
stable_hash = _fwd.stable_hash
compact_identity = _fwd.compact_identity
write_instances = _fwd.write_instances
identity_label = _fwd.identity_label


def _converter():
    """``make_conv_dgrad_aot_config``, the tool that consumes the list."""
    import make_conv_dgrad_aot_config

    return make_conv_dgrad_aot_config


def family(identity: dict) -> str:
    """``implicit``, ``direct`` or ``helper`` (an MFMA weight transform)."""
    if identity["algorithm"] == "implicit_gemm":
        return "implicit"
    if identity["direction"] == "direct_dgrad_helper":
        return "helper"
    return "direct"


def sweep_hashes(dtype: str, jobs: int) -> set:
    """``stable_hash`` of every valid dgrad identity of rocKE's implicit-GEMM AOT grid
    and of its direct grid (helpers included), for ``dtype``."""
    from benchmarks.common import kernel_sweep
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(ARCH)
    implicit = kernel_sweep.enumerate_jobs(
        arch=ARCH, dtype=dtype, target=target, directions=("dgrad",), jobs=jobs
    )
    direct = enumerate_direct_jobs(
        arch=ARCH, target=target, directions=("dgrad",), dtype=dtype, jobs=jobs
    )
    return {job.identity.stable_hash() for job in [*implicit, *direct]}


def helper_identities(identity: dict) -> list:
    """The helper identities an MFMA dgrad identity's pipeline launches, as the direct
    sweep pulls them in (``direct_kernel_sweep._caps_jobs``); [] for anything else."""
    from dataclasses import replace

    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.kernel_cache import KernelIdentity

    if identity["algorithm"] != "direct_grouped_dgrad_mfma":
        return []
    ident = KernelIdentity.from_dict(identity)
    caps = dks._caps_of(ident)
    knobs = json.loads(ident.knobs) if ident.knobs else {}
    helpers = [(dks._TRANSPOSE, {})]
    if dks._uses_coalesced_weights(knobs):
        helpers.append((dks._REORGANIZE, {"fold_k32": knobs["fold_k32"]}))
    out = []
    for variant, helper_knobs in helpers:
        helper = dks._helper_identity(
            ident.arch, ident.wave_size, variant, caps, helper_knobs, ident.dtype_a
        )
        out.append(replace(helper, llvm_flavor=ident.llvm_flavor).to_dict())
    return out


def with_helpers(identities: list) -> list:
    """``identities`` plus every helper their MFMA entries need, each once, in order."""
    out, seen = [], set()
    for identity in identities:
        for entry in [identity, *helper_identities(identity)]:
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
        "--labels",
        type=Path,
        default=CATALOG_LABELS,
        help="The curated catalog, one KernelIdentity label per line "
        "(default: %(default)s).",
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

    try:
        listed = read_labels(args.labels)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    others = sorted({i["dtype_a"] for i in listed} - set(DTYPES))
    if others:
        print(f"{args.labels}: dtype(s) {others} are not packaged", file=sys.stderr)
        return 1
    helpers_listed = [i for i in listed if family(i) == "helper"]
    if helpers_listed:
        print(
            f"{args.labels}: helper kernels are derived, not listed: "
            f"{identity_label(helpers_listed[0])}",
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
                    f"not a {ARCH} dgrad sweep identity: {identity_label(identity)}",
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
