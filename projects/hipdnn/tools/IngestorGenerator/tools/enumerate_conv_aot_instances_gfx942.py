#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the rocKE gfx942 forward-conv AOT instance lists the hipkernel pack ships.

The catalog is a curated list of rocKE kernels (:data:`CATALOG_LABELS`): one
``KernelIdentity.label()`` per line, which :meth:`KernelIdentity.from_label` turns
back into the identity.

Unlike gfx950's ``enumerate_conv_aot_instances.py``, this tool does not also
require every entry to be reproducible from the current checkout's default sweep
grid (``kernels/conv/aot_jobs.py`` / ``kernels/conv/direct_aot_jobs.py``,
enumerated by ``benchmarks/common/kernel_sweep.py`` /
``benchmarks/common/direct_kernel_sweep.py``): this gfx942 catalog
(``instances_new.txt``) was produced by a different run and every one of its
labels fails that provenance check even though rocKE's own dispatcher (its
registered instance list, ``dispatch/grouped_convolution_instances/gfx942_fwd.json``,
validated through ``load_instance_list``) and the packaged-launch adapter
(:func:`validate`, below) both accept it -- see ``mi300plan.md`` Faza 2/3. The
dispatcher's registered list is this catalog's source of truth.

Every entry must still convert: an identity ``make_conv_aot_config_gfx942.py``
refuses, because the packaged launch does not run it, fails the run. Nothing is
dropped silently.

Which instance serves a given problem, and which serving instances run, is decided
at runtime, not here.

The output is one JSON list per dtype (:data:`INSTANCE_LIST`), one identity per
line, ordered by family (implicit GEMM first), then ``stable_hash``. Each entry
is compact (:func:`compact_identity`): the ``KernelIdentity`` fields without a
plain default, plus every field that differs from its default;
``KernelIdentity.from_dict`` restores the rest, so the entry keeps its
``stable_hash()``. The lists are the input of ``make_conv_aot_config_gfx942.py
--instances``::

    python tools/enumerate_conv_aot_instances_gfx942.py --rocke-root "$ROCKE" \\
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

ARCH = "gfx942"
DTYPES = ("fp16", "bf16")

#: File name of one dtype's instance list. One file per dtype keeps each below the
#: 50 MB a source host accepts.
INSTANCE_LIST = "gfx942_conv_fwd_aot_{dtype}.instances.json"

_HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = _HERE.parent / "configs"

#: The curated catalog: one ``KernelIdentity.label()`` per line; blank lines and
#: lines starting with ``#`` are ignored.
CATALOG_LABELS = DEFAULT_OUTPUT_DIR / "gfx942_conv_fwd_aot.labels.txt"


def _bind_rocke(rocke_root: Path) -> None:
    for sub in ("platform/python", "library"):
        path = str((rocke_root / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _converter():
    """``make_conv_aot_config_gfx942``, the tool that consumes the list."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import make_conv_aot_config_gfx942

    return make_conv_aot_config_gfx942


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


def stable_hash(identity: dict) -> str:
    from kernels.conv.kernel_identity import KernelIdentity

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
    from kernels.conv.kernel_identity import KernelIdentity

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
        listed = read_labels(args.labels)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    others = sorted({i["dtype_a"] for i in listed} - set(DTYPES))
    if others:
        print(f"{args.labels}: dtype(s) {others} are not packaged", file=sys.stderr)
        return 1

    lists, failed = {}, False
    print(f"{'dtype':6s} {'implicit':>8s} {'direct':>6s}")
    for dtype in sorted(DTYPES):
        identities = [i for i in listed if i["dtype_a"] == dtype]
        # No sweep-hash provenance check here (gfx950's enumerate_conv_aot_instances.py
        # has one, via sweep_hashes()/kernel_sweep.enumerate_jobs): this gfx942 catalog
        # (instances_new.txt) was not generated by the current checkout's default
        # sweep grid, so every one of its 681 identities reads back as "not a gfx942
        # forward sweep identity" even though rocKE's dispatcher (grouped_convolution_
        # instances/gfx942_fwd.json, validated through load_instance_list) and this
        # file's own gfx942_conv_fwd_aot_spec_for_identity both accept all 681 -- see
        # mi300plan.md Faza 2/3. The dispatcher's registered instance list, not the
        # sweep grid, is this catalog's source of truth; validate() below (the
        # packaged-launch adapter check) is the real buildability gate.
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


if __name__ == "__main__":
    raise SystemExit(main())
