#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocKE forward-conv AOT instance lists a hipkernel pack ships.

One tool for both forward engines, chosen by ``--arch`` (gfx950 by default, or
gfx1250, ``make_conv_aot_config.ARCHES``). The rest of this docstring describes
gfx950; gfx1250 differs where it says so.

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
line, ordered by family (implicit GEMM first), then ``stable_hash``; a gfx1250 list
keeps the dispatcher list's order instead, which is that catalog's priority order
(``make_conv_aot_config.priorities``). Each entry
is compact (:func:`compact_identity`): the ``KernelIdentity`` fields without a
plain default, plus every field that differs from its default;
``KernelIdentity.from_dict`` restores the rest, so the entry keeps its
``stable_hash()``. The lists are the input of ``make_conv_aot_config.py
--instances``::

    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --output-dir configs
    ROCKE_LLVM_FLAVOR=llvm24 python tools/enumerate_conv_aot_instances.py \\
        --arch gfx1250 --rocke-root "$ROCKE" --output-dir configs
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ARCH = "gfx950"
DTYPES = ("fp16", "bf16")

#: File name of one dtype's instance list (gfx950's; :func:`instance_list` for
#: another arch). One file per dtype keeps each below the 50 MB a source host
#: accepts.
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


def instance_list(arch: str = ARCH) -> str:
    """The file name of one of ``arch``'s per-dtype lists (``{dtype}`` left in)."""
    return f"{arch}_conv_fwd_aot_{{dtype}}.instances.json"


def dispatcher_instance_list(arch: str = ARCH) -> Path:
    """rocKE's forward dispatcher list of ``arch``, relative to --rocke-root."""
    return DISPATCHER_INSTANCE_LIST.with_name(f"{arch}_fwd.json")


def excludes_path(arch: str = ARCH) -> Path:
    """``arch``'s known-bug exclusion list (absent: none)."""
    return DEFAULT_OUTPUT_DIR / f"{arch}_conv_fwd_aot.excludes.json"


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


def read_dispatcher_instances(rocke_root: Path, arch: str = ARCH) -> list:
    """Every identity rocKE's own forward dispatcher lists for ``arch``
    (:func:`dispatcher_instance_list`), as dicts, in the list's on-disk order."""
    from dispatch.grouped_convolution import load_instance_list

    lst = load_instance_list(rocke_root / dispatcher_instance_list(arch))
    return [identity.to_dict() for identity in lst.identities]


def read_rocke_catalog(direction: str, arch: str = ARCH) -> list:
    """rocKE's own registered instances of ``(arch, direction)``, as dicts, in
    rocKE's on-disk order (for fwd, the list's priority order; see
    ``dispatch.grouped_convolution``'s module docstring). rocKE must already be
    bound (:func:`_bind_rocke`)."""
    return _common.read_rocke_catalog(direction, arch=arch)


#: ``label -> reason`` of :data:`EXCLUDES`-style instances; ``{}`` when ``path``
#: does not exist. Shared with the dgrad tool (``conv_aot_common.py``).
read_excludes = _common.read_excludes


def sweep_hashes(dtype: str, jobs: int, arch: str = ARCH) -> set:
    """``stable_hash`` of every valid forward identity of rocKE's implicit-GEMM AOT
    grid and of its direct grid for ``arch``, for ``dtype``, under this process's
    LLVM flavor.

    The direct grid's pool workers name their identities for the flavor *they*
    resolve, which a reused worker pool (``forkserver``) may have fixed under another
    ``ROCKE_LLVM_FLAVOR``; each identity is restamped with this process's flavor so
    the answer does not depend on the pool."""
    from dataclasses import replace

    from benchmarks.common import kernel_sweep
    from benchmarks.common.direct_kernel_sweep import enumerate_direct_jobs
    from kernels.conv.kernel_identity import current_llvm_flavor
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    implicit = kernel_sweep.enumerate_jobs(
        arch=arch, dtype=dtype, target=target, directions=("fwd",), jobs=jobs
    )
    direct = enumerate_direct_jobs(
        arch=arch, target=target, directions=("fwd",), dtype=dtype, jobs=jobs
    )
    flavor = current_llvm_flavor()
    return {
        replace(job.identity, llvm_flavor=flavor).stable_hash()
        for job in [*implicit, *direct]
    }


stable_hash = _common.stable_hash


def validate(identities: list, jobs: int = 1, arch: str = ARCH) -> list:
    """``(stable_hash, reason)`` for each identity the converter rejects."""
    converted = _converter().instance_specs(identities, jobs, arch)
    return [
        (stable_hash(identity), reason)
        for identity, (spec, reason) in zip(identities, converted)
        if spec is None
    ]


compact_identity = _common.compact_identity
write_instances = _common.write_instances


def check_flavor(identities: list) -> int:
    """0 when this process lowers the LLVM flavor the catalog's identities are named
    for; otherwise an error naming ``ROCKE_LLVM_FLAVOR`` (the dgrad tool's check)."""
    import enumerate_conv_dgrad_aot_instances

    return enumerate_conv_dgrad_aot_instances.check_flavor(identities)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rocke-root",
        type=Path,
        required=True,
        help="rocKE tree (platform/, library/).",
    )
    parser.add_argument(
        "--arch",
        default=ARCH,
        choices=("gfx950", "gfx1250"),
        help="The engine's architecture (default: %(default)s).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="A curated subset override, one KernelIdentity label per line "
        "(default: every instance rocKE's forward dispatcher list of --arch "
        f"lists, {DISPATCHER_INSTANCE_LIST} for gfx950).",
    )
    parser.add_argument(
        "--excludes",
        type=Path,
        default=None,
        help=f"Known-bug exclusions ({EXCLUDES.name}-style: JSON list of "
        '{"label": ..., "reason": ...}; default: <arch>_conv_fwd_aot.excludes.json '
        "beside the lists). Applied to the dispatcher source only, after --labels "
        "already picked its own subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for the per-dtype lists ({instance_list('<arch>')}; default: "
        "%(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_common.default_jobs(),
        help="Worker processes (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    arch = args.arch
    excludes_file = args.excludes if args.excludes is not None else excludes_path(arch)
    _bind_rocke(args.rocke_root)
    ordered = _converter().ARCHES[arch].ordered

    try:
        if args.labels is not None:
            listed = read_labels(args.labels)
        else:
            listed = read_dispatcher_instances(args.rocke_root, arch)
            excludes = read_excludes(excludes_file)
            if excludes:
                listed = _common.apply_excludes(
                    listed, excludes, excludes_file, f"rocKE's {arch} fwd dispatcher"
                )
                if listed is None:
                    return 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if arch != ARCH:
        rc = check_flavor(listed)
        if rc:
            return rc
    others = sorted({i["dtype_a"] for i in listed} - set(DTYPES))
    if others:
        source = (
            args.labels if args.labels is not None else dispatcher_instance_list(arch)
        )
        print(f"{source}: dtype(s) {others} are not packaged", file=sys.stderr)
        return 1

    lists, failed = {}, False
    print(f"{'dtype':6s} {'implicit':>8s} {'direct':>6s}")
    for dtype in sorted(DTYPES):
        identities = [i for i in listed if i["dtype_a"] == dtype]
        swept = sweep_hashes(dtype, args.jobs, arch)
        for identity in identities:
            if stable_hash(identity) not in swept:
                print(
                    f"not a {arch} forward sweep identity: {identity_label(identity)}",
                    file=sys.stderr,
                )
                failed = True
        for instance_id, reason in validate(identities, args.jobs, arch):
            print(f"unsupported instance {instance_id}: {reason}", file=sys.stderr)
            failed = True
        families = collections.defaultdict(list)
        for identity in identities:
            implicit = identity["algorithm"] == "implicit_gemm"
            families[implicit].append((stable_hash(identity), identity))
        implicit, direct = sorted(families[True]), sorted(families[False])
        print(f"{dtype:6s} {len(implicit):8d} {len(direct):6d}")
        if ordered:
            # The dispatcher list's own order: the catalog's priority order.
            lists[dtype] = identities
        else:
            lists[dtype] = [i for _, i in implicit] + [i for _, i in direct]
    if failed:
        print("unsupported catalog entries; nothing written", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dtype, identities in lists.items():
        path = args.output_dir / instance_list(arch).format(dtype=dtype)
        write_instances(path, identities)
        print(f"{len(identities)} instance(s) written to {path}")
    return 0


identity_label = _common.identity_label


if __name__ == "__main__":
    raise SystemExit(main())
