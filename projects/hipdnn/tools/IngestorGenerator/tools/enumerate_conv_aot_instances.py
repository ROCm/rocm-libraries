#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocKE gfx950 forward-conv AOT instance lists the hipkernel pack ships.

The catalog is exactly what rocKE's own forward dispatcher lists
(:data:`DISPATCHER_INSTANCE_LIST`, ``dispatch/grouped_convolution_instances/
gfx950_fwd.json``): the dispatcher is the single source of truth for which gfx950
forward instances exist, and this engine ships every one of them as AOT. Pass
``--labels`` (one ``KernelIdentity.label()`` per line) to build from a curated
subset instead -- a one-off override, not the normal path. Implicit-GEMM entries
must be identities of the AOT sweep grid (``kernels/conv/aot_jobs.py``, enumerated
by ``benchmarks/common/kernel_sweep.py``) and direct entries identities of the
direct sweep grid (``kernels/conv/direct_aot_jobs.py``, enumerated by
``benchmarks/common/direct_kernel_sweep.py``), for gfx950, forward, fp16 or bf16.
Every entry must also convert: an identity ``make_conv_aot_config.py`` refuses,
because the packaged launch does not run it, fails the run. Nothing is dropped
silently.

Which instance serves a given problem, and which serving instances run, is decided
at runtime, not here.

The output is one JSON list per dtype (:data:`INSTANCE_LIST`), one identity per
line, ordered by family (implicit GEMM first), then ``stable_hash``. Each entry
is compact (:func:`compact_identity`): the ``KernelIdentity`` fields without a
plain default, plus every field that differs from its default;
``KernelIdentity.from_dict`` restores the rest, so the entry keeps its
``stable_hash()``. The lists are the input of ``make_conv_aot_config.py
--instances``::

    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --output-dir configs
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

ARCH = "gfx950"
DTYPES = ("fp16", "bf16")

#: File name of one dtype's instance list. One file per dtype keeps each below the
#: 50 MB a source host accepts.
INSTANCE_LIST = "gfx950_conv_fwd_aot_{dtype}.instances.json"

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import conv_aot_common as _common  # noqa: E402

DEFAULT_OUTPUT_DIR = _HERE.parent / "configs"

#: rocKE's own forward dispatcher instance list, relative to --rocke-root: the
#: default source of the catalog (every instance it lists, none it doesn't).
DISPATCHER_INSTANCE_LIST = Path(
    "library/dispatch/grouped_convolution_instances/gfx950_fwd.json"
)

#: Known-bug exclusions: instances rocKE registers that this pack must not ship,
#: each with a reason. JSON list of ``{"label": ..., "reason": ...}``; absent
#: file means no exclusions.
EXCLUDES = DEFAULT_OUTPUT_DIR / "gfx950_conv_fwd_aot.excludes.json"


_bind_rocke = _common.bind_rocke


def _converter():
    """``make_conv_aot_config``, the tool that consumes the list."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import make_conv_aot_config

    return make_conv_aot_config


def read_labels(path: Path) -> list:
    """The identities ``path`` lists, as dicts, in file order.

    Raises ``ValueError`` on a malformed label, on one that does not spell its
    identity exactly (``label()`` of the parsed identity must be the line), and on a
    repeated identity.
    """
    from kernels.conv.kernel_identity import KernelIdentity

    identities, seen = [], set()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        label = line.strip()
        if not label or label.startswith("#"):
            continue
        identity = KernelIdentity.from_label(label)
        if identity.label() != label:
            raise ValueError(
                f"{path}:{lineno}: {label!r} reads back as {identity.label()!r}"
            )
        if identity.stable_hash() in seen:
            raise ValueError(f"{path}:{lineno}: {label!r} is listed twice")
        seen.add(identity.stable_hash())
        identities.append(identity.to_dict())
    return identities


def read_dispatcher_instances(rocke_root: Path) -> list:
    """Every identity rocKE's own forward dispatcher lists (:data:`DISPATCHER_INSTANCE_LIST`),
    as dicts, in the dispatcher's canonical (label-sorted) order."""
    from dispatch.grouped_convolution import load_instance_list

    lst = load_instance_list(rocke_root / DISPATCHER_INSTANCE_LIST)
    return [identity.to_dict() for identity in lst.identities]


def read_rocke_catalog(direction: str) -> list:
    """rocKE's own registered instances of ``(ARCH, direction)``, as dicts, in
    rocKE's on-disk order -- rocKE is a set, not a priority order
    (``dispatch.grouped_convolution``'s module docstring), so this makes no
    claim about that order mattering; it is simply whatever rocKE's loader
    returns. rocKE must already be bound (:func:`_bind_rocke`)."""
    return _common.read_rocke_catalog(direction, arch=ARCH)


#: ``label -> reason`` of :data:`EXCLUDES`-style instances; ``{}`` when ``path``
#: does not exist. Shared with the dgrad tool (``conv_aot_common.py``).
read_excludes = _common.read_excludes


def sweep_hashes(dtype: str, jobs: int) -> set:
    """``stable_hash`` of every valid forward identity of rocKE's implicit-GEMM AOT
    grid and of its direct grid, for ``dtype``."""
    from benchmarks.common import kernel_sweep
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(ARCH)
    implicit = kernel_sweep.enumerate_jobs(
        arch=ARCH, dtype=dtype, target=target, directions=("fwd",), jobs=jobs
    )
    direct = enumerate_direct_jobs(
        arch=ARCH, target=target, directions=("fwd",), dtype=dtype, jobs=jobs
    )
    return {job.identity.stable_hash() for job in [*implicit, *direct]}


stable_hash = _common.stable_hash


def validate(identities: list, jobs: int = 1) -> list:
    """``(stable_hash, reason)`` for each identity the converter rejects."""
    converted = _converter().instance_specs(identities, jobs)
    return [
        (stable_hash(identity), reason)
        for identity, (spec, reason) in zip(identities, converted)
        if spec is None
    ]


compact_identity = _common.compact_identity
write_instances = _common.write_instances


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
        default=None,
        help="A curated subset override, one KernelIdentity label per line "
        "(default: every instance rocKE's forward dispatcher, "
        f"{DISPATCHER_INSTANCE_LIST}, lists).",
    )
    parser.add_argument(
        "--excludes",
        type=Path,
        default=EXCLUDES,
        help=f"Known-bug exclusions ({EXCLUDES.name}-style: JSON list of "
        '{"label": ..., "reason": ...}; default: %(default)s). Applied to '
        "the dispatcher source only, after --labels already picked its own "
        "subset.",
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

    try:
        if args.labels is not None:
            listed = read_labels(args.labels)
        else:
            listed = read_dispatcher_instances(args.rocke_root)
            excludes = read_excludes(args.excludes)
            if excludes:
                listed = _common.apply_excludes(
                    listed, excludes, args.excludes, "rocKE's gfx950 fwd dispatcher"
                )
                if listed is None:
                    return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    others = sorted({i["dtype_a"] for i in listed} - set(DTYPES))
    if others:
        source = args.labels if args.labels is not None else DISPATCHER_INSTANCE_LIST
        print(f"{source}: dtype(s) {others} are not packaged", file=sys.stderr)
        return 1

    lists, failed = {}, False
    print(f"{'dtype':6s} {'implicit':>8s} {'direct':>6s}")
    for dtype in sorted(DTYPES):
        identities = [i for i in listed if i["dtype_a"] == dtype]
        swept = sweep_hashes(dtype, args.jobs)
        for identity in identities:
            if stable_hash(identity) not in swept:
                print(
                    f"not a {ARCH} forward sweep identity: {identity_label(identity)}",
                    file=sys.stderr,
                )
                failed = True
        for instance_id, reason in validate(identities, args.jobs):
            print(f"unsupported instance {instance_id}: {reason}", file=sys.stderr)
            failed = True
        families = collections.defaultdict(list)
        for identity in identities:
            implicit = identity["algorithm"] == "implicit_gemm"
            families[implicit].append((stable_hash(identity), identity))
        implicit, direct = sorted(families[True]), sorted(families[False])
        print(f"{dtype:6s} {len(implicit):8d} {len(direct):6d}")
        lists[dtype] = [i for _, i in implicit] + [i for _, i in direct]
    if failed:
        print("unsupported catalog entries; nothing written", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dtype, identities in lists.items():
        path = args.output_dir / INSTANCE_LIST.format(dtype=dtype)
        write_instances(path, identities)
        print(f"{len(identities)} instance(s) written to {path}")
    return 0


identity_label = _common.identity_label


if __name__ == "__main__":
    raise SystemExit(main())
