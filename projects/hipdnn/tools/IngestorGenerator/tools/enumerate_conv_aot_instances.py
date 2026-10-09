#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Enumerate the rocKE gfx950 forward-conv AOT instances the hipkernel pack ships.

The catalog is rocKE's own sweep grids, per dtype, in two families, with no
selection of ours: the only limits are the sweeps' own validity checks.

Implicit GEMM is every valid gfx950 forward implicit-GEMM ``KernelIdentity`` of the
AOT sweep grid (``benchmarks/common/kernel_sweep.py``): every tile and warp
geometry, pipeline, epilogue, load and store width, and K loop (plain, unrolled,
async DMA with its chunk widths) the sweep enumerates. Every one must convert: an
identity ``make_conv_aot_config.py`` refuses fails the run.

Direct is every forward identity of the direct sweep grid
(``benchmarks/common/direct_kernel_sweep.py``) that ``make_conv_aot_config.py``
accepts, which is the set of direct kernels the packaged launch runs. The others
are counted per algorithm with the converter's reason and left out, so the
catalog grows by itself as the converter learns to package more of them.

Which instance serves a given problem is decided at runtime, not here.

The output is one JSON list per dtype (:data:`INSTANCE_LIST`), one identity per
line, ordered by family (implicit GEMM first), then ``stable_hash``. Each entry
is compact (:func:`compact_identity`): the ``KernelIdentity`` fields without a
plain default, plus every field that differs from its default;
``KernelIdentity.from_dict`` restores the rest, so the entry keeps its
``stable_hash()``. The lists are the input of ``make_conv_aot_config.py
--instances``, and each entry is validated with that tool's ``instance_spec``
before anything is written::

    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --output-dir configs
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
from pathlib import Path

ARCH = "gfx950"
DTYPES = ("fp16", "bf16")

#: File name of one dtype's instance list. One file per dtype keeps each below the
#: 50 MB a source host accepts.
INSTANCE_LIST = "gfx950_conv_fwd_aot_{dtype}.instances.json"

_HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = _HERE.parent / "configs"


def _bind_rocke(rocke_root: Path) -> None:
    for sub in ("platform/python", "library"):
        path = str((rocke_root / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _converter():
    """``make_conv_aot_config``, the tool that consumes the list."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import make_conv_aot_config

    return make_conv_aot_config


def grid_identities(dtype: str, jobs: int) -> list:
    """Every valid identity of rocKE's forward implicit-GEMM AOT grid, as dicts."""
    from benchmarks.common import kernel_sweep
    from rocke.core.arch import ArchTarget

    found = kernel_sweep.enumerate_jobs(
        arch=ARCH,
        dtype=dtype,
        target=ArchTarget.from_gfx(ARCH),
        directions=("fwd",),
        jobs=jobs,
    )
    return [job.identity.to_dict() for job in found]


def direct_identities(dtype: str, jobs: int) -> tuple:
    """``(packaged, refused)``: the forward identities of rocKE's direct sweep grid
    that ``make_conv_aot_config.py`` accepts, as dicts, and a ``Counter`` of the
    others by ``(algorithm, reason)``."""
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from rocke.core.arch import ArchTarget

    converter = _converter()
    found = enumerate_direct_jobs(
        arch=ARCH,
        target=ArchTarget.from_gfx(ARCH),
        directions=("fwd",),
        dtype=dtype,
        jobs=jobs,
    )
    packaged, refused = [], collections.Counter()
    for job in found:
        identity = job.identity.to_dict()
        spec, reason = converter.instance_spec(identity)
        if spec is None:
            refused[(identity["algorithm"], reason)] += 1
        else:
            packaged.append(identity)
    return packaged, refused


def stable_hash(identity: dict) -> str:
    from benchmarks.common.kernel_cache import KernelIdentity

    return KernelIdentity.from_dict(identity).stable_hash()


def validate(identities: list, jobs: int = 1) -> list:
    """``(stable_hash, reason)`` for each identity the converter rejects."""
    converted = _converter().instance_specs(identities, jobs)
    return [
        (stable_hash(identity), reason)
        for identity, (spec, reason) in zip(identities, converted)
        if spec is None
    ]


def compact_identity(identity: dict) -> dict:
    """``identity`` without the fields that hold their ``KernelIdentity`` default.

    A field with no plain default is always kept: the required ones, and
    ``llvm_flavor``, whose default is this process's flavor and which
    ``KernelIdentity.from_dict`` fills with ``""`` when it is missing. A value
    equal to its default but of another type (``0`` for ``False``) is kept, so
    ``from_dict`` rebuilds exactly the identity, and its ``stable_hash()``.
    """
    from benchmarks.common.kernel_cache import KernelIdentity

    compact = {}
    for field in dataclasses.fields(KernelIdentity):
        value = identity[field.name]
        default = field.default
        if (
            default is dataclasses.MISSING
            or type(value) is not type(default)
            or value != default
        ):
            compact[field.name] = value
    return compact


def write_instances(path: Path, identities: list) -> None:
    """A JSON list, one :func:`compact_identity` per line."""
    body = ",\n".join(
        json.dumps(compact_identity(i), separators=(",", ":")) for i in identities
    )
    path.write_text(f"[\n{body}\n]\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root",
        type=Path,
        required=True,
        help="rocKE tree (platform/, library/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for the per-dtype lists ({INSTANCE_LIST}; default: "
        "%(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    _bind_rocke(args.rocke_root)

    lists, failed = {}, False
    print(f"{'dtype':6s} {'implicit':>8s} {'direct grid':>11s} {'direct':>6s}")
    for dtype in sorted(DTYPES):
        implicit = sorted(
            (stable_hash(i), i) for i in grid_identities(dtype, args.jobs)
        )
        if len({h for h, _ in implicit}) != len(implicit):
            print(f"{dtype}: the grid repeats an identity", file=sys.stderr)
            return 1
        rejected = validate([i for _, i in implicit], args.jobs)
        for instance_id, reason in rejected:
            print(f"unsupported instance {instance_id}: {reason}", file=sys.stderr)
        failed |= bool(rejected)
        direct, refused = direct_identities(dtype, args.jobs)
        direct = sorted(((stable_hash(i), i) for i in direct), key=lambda p: p[0])
        print(
            f"{dtype:6s} {len(implicit):8d} "
            f"{len(direct) + sum(refused.values()):11d} {len(direct):6d}"
        )
        for (algorithm, reason), count in sorted(refused.items()):
            print(f"  direct {algorithm}: {count} not packaged ({reason})")
        lists[dtype] = [i for _, i in implicit] + [i for _, i in direct]
    if failed:
        print("unsupported implicit-GEMM instance(s); nothing written", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dtype, identities in lists.items():
        path = args.output_dir / INSTANCE_LIST.format(dtype=dtype)
        write_instances(path, identities)
        print(f"{len(identities)} instance(s) written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
