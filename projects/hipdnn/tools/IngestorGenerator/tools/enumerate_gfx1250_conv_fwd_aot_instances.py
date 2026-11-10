#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocKE gfx1250 forward-conv AOT instance lists the hipkernel pack ships.

The gfx1250 sibling of ``enumerate_conv_aot_instances.py``, reusing its helpers. The
catalog is rocKE's own registered list (``dispatch/grouped_convolution_instances/
gfx1250_fwd.json``); this tool mirrors nothing by hand. :data:`EXCLUDES` (optional;
empty when absent) names instances rocKE registers that the pack must not ship, each
with a reason. Every entry must be an identity of rocKE's forward sweep grids for
gfx1250 (``benchmarks/common/kernel_sweep.py``, ``benchmarks/common/
direct_kernel_sweep.py``) and must convert: an identity
``make_gfx1250_conv_fwd_aot_config.py`` refuses fails the run. Nothing is dropped
silently.

The sweep it checks identities against lowers with this process's LLVM flavor
(:func:`check_flavor`), same as the committed lists embed; rerun under the catalog's
own flavor (``ROCKE_LLVM_FLAVOR``) rather than skipping the check.

The output is one JSON list per dtype (:data:`INSTANCE_LIST`), one compact identity
per line (``enumerate_conv_aot_instances.compact_identity``), **in the order rocKE
registers them**: the forward list is ordered, and the config tool turns that order
into each kernel's catalog priority. The lists are the input of
``make_gfx1250_conv_fwd_aot_config.py --instances``::

    ROCKE_LLVM_FLAVOR=llvm24 python tools/enumerate_gfx1250_conv_fwd_aot_instances.py \\
        --rocke-root "$ROCKE" --output-dir configs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import enumerate_conv_aot_instances as _fwd  # noqa: E402
import enumerate_conv_dgrad_aot_instances as _dgrad  # noqa: E402

ARCH = "gfx1250"
DIRECTION = "fwd"
DTYPES = ("fp16", "bf16")
#: File name of one dtype's instance list.
INSTANCE_LIST = "gfx1250_conv_fwd_aot_{dtype}.instances.json"
DEFAULT_OUTPUT_DIR = _fwd.DEFAULT_OUTPUT_DIR
#: Known-bug exclusions, as ``enumerate_conv_aot_instances.EXCLUDES``.
EXCLUDES = DEFAULT_OUTPUT_DIR / "gfx1250_conv_fwd_aot.excludes.json"

_bind_rocke = _fwd._bind_rocke
read_excludes = _fwd.read_excludes
compact_identity = _fwd.compact_identity
write_instances = _fwd.write_instances
identity_label = _fwd.identity_label
stable_hash = _fwd.stable_hash
check_flavor = _dgrad.check_flavor


def _converter():
    import make_gfx1250_conv_fwd_aot_config

    return make_gfx1250_conv_fwd_aot_config


def read_rocke_catalog() -> list:
    """rocKE's registered gfx1250 forward instances, as dicts, in registration order.
    rocKE must already be bound."""
    from dispatch.grouped_convolution import (
        _INSTANCE_DIR,
        _INSTANCE_LISTS,
        load_instance_list,
    )

    path = _INSTANCE_DIR / _INSTANCE_LISTS[(ARCH, DIRECTION)]
    return [i.to_dict() for i in load_instance_list(path).identities]


def sweep_hashes(dtype: str, jobs: int) -> set:
    """``stable_hash`` of every valid forward identity of rocKE's implicit-GEMM AOT
    grid and of its direct grid for gfx1250, for ``dtype``, under this process's LLVM
    flavor.

    The direct grid's pool workers name their identities for the flavor *they*
    resolve, which a reused worker pool (``forkserver``) may have fixed under another
    ``ROCKE_LLVM_FLAVOR``; each identity is restamped with this process's flavor so
    the answer does not depend on the pool."""
    from dataclasses import replace

    from benchmarks.common import kernel_sweep
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from kernels.conv.kernel_identity import current_llvm_flavor
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(ARCH)
    implicit = kernel_sweep.enumerate_jobs(
        arch=ARCH, dtype=dtype, target=target, directions=("fwd",), jobs=jobs
    )
    direct = enumerate_direct_jobs(
        arch=ARCH, target=target, directions=("fwd",), dtype=dtype, jobs=jobs
    )
    flavor = current_llvm_flavor()
    return {
        replace(job.identity, llvm_flavor=flavor).stable_hash()
        for job in [*implicit, *direct]
    }


def validate(identities: list) -> list:
    """``(stable_hash, reason)`` for each identity the config tool refuses."""
    return [
        (stable_hash(identity), reason)
        for identity, (spec, reason) in zip(
            identities, _converter().instance_specs(identities)
        )
        if spec is None
    ]


def catalog_lists(excludes: dict, jobs: int) -> tuple:
    """``(lists, problems)``: per dtype the identities to ship, in registration order,
    and every reason the catalog cannot be written."""
    catalog = read_rocke_catalog()
    labels = {identity_label(i) for i in catalog}
    problems = [
        f"exclude not in rocKE's {ARCH} {DIRECTION} catalog: {label}"
        for label in sorted(set(excludes) - labels)
    ]
    listed = [i for i in catalog if identity_label(i) not in excludes]
    problems += [
        f"dtype {dtype} is not packaged"
        for dtype in sorted({i["dtype_a"] for i in listed} - set(DTYPES))
    ]
    lists = {}
    for dtype in DTYPES:
        identities = [i for i in listed if i["dtype_a"] == dtype]
        swept = sweep_hashes(dtype, jobs)
        problems += [
            f"not a {ARCH} forward sweep identity: {identity_label(i)}"
            for i in identities
            if stable_hash(i) not in swept
        ]
        problems += [
            f"unsupported instance {instance_id}: {reason}"
            for instance_id, reason in validate(identities)
        ]
        lists[dtype] = identities
    return lists, problems


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
        help="Known-bug exclusions from rocKE's catalog (default: %(default)s; absent "
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
        help="Worker processes for the sweep grids (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    _bind_rocke(args.rocke_root)

    rc = check_flavor(read_rocke_catalog())
    if rc:
        return rc
    lists, problems = catalog_lists(read_excludes(args.excludes), args.jobs)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
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
