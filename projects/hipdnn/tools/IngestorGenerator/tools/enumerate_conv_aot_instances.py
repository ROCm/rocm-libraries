#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Enumerate the rocKE gfx950 forward-conv AOT instances the hipkernel pack ships.

The catalog is a subset of rocKE's own sweep grids, per dtype, in two families,
chosen by a committed selection (:data:`SELECTION`, read by :class:`Selection`).
The selection keeps the configurations that measured as contenders on some problem
in an offline study and keeps every problem served; it holds identities only, never
the measurements behind them.

Implicit GEMM is every valid gfx950 forward implicit-GEMM ``KernelIdentity`` of the
AOT sweep grid (``benchmarks/common/kernel_sweep.py``) that the selection keeps
(:func:`excluded_reason`): its config, the tile and warp geometry together with a
pipeline and a K loop, is a selected config, and the identity is not one of the
selection's excluded instances (code objects that spill registers). The async DMA K
loop is never kept. A kept config keeps every width pair and epilogue the sweep
enumerates for it; the runtime selection picks the widest a problem admits. Every
kept identity must convert: an identity ``make_conv_aot_config.py`` refuses fails
the run.

Direct is every forward identity of the direct sweep grid
(``benchmarks/common/direct_kernel_sweep.py``) that ``make_conv_aot_config.py``
accepts and the selection lists. Identities the converter refuses are counted per
algorithm with its reason and left out.

The selection names instances by the ``stable_hash`` of their fp16 identity and
applies to every dtype alike. A selected config or instance the sweep no longer
enumerates fails the run, so the selection cannot silently go stale.

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

#: File name of the committed selection, beside the lists.
SELECTION = "gfx950_conv_fwd_aot.selection.json"

#: The dtype whose ``stable_hash`` names an instance in the selection.
SELECTION_DTYPE = "fp16"

_HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = _HERE.parent / "configs"
DEFAULT_SELECTION = DEFAULT_OUTPUT_DIR / SELECTION


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


def k_loop(identity: dict) -> str:
    """The K loop an implicit-GEMM identity runs: ``plain``, ``unroll`` or ``async``."""
    if identity["async_dma"]:
        return "async"
    return "unroll" if identity["unroll_k"] else "plain"


def selection_hash(identity: dict) -> str:
    """The ``stable_hash`` of ``identity`` in :data:`SELECTION_DTYPE`: the name the
    selection gives the instance in every dtype."""
    neutral = dict(identity)
    neutral["dtype_a"] = neutral["dtype_b"] = neutral["dtype_d"] = SELECTION_DTYPE
    return stable_hash(neutral)


@dataclasses.dataclass(frozen=True)
class Selection:
    """The committed choice of instances (:data:`SELECTION`).

    ``configs`` holds ``(geometry, pipeline, k_loop)`` triples, the geometry being
    ``geometry_fields`` in order. ``excluded`` names implicit-GEMM instances of a
    selected config that are left out, and ``direct`` the direct instances kept, both
    by :func:`selection_hash`.
    """

    geometry_fields: tuple
    configs: frozenset
    excluded: frozenset
    direct: frozenset

    @classmethod
    def from_dict(cls, document: dict) -> "Selection":
        implicit = document["implicit_gemm"]
        return cls(
            geometry_fields=tuple(implicit["geometry_fields"]),
            configs=frozenset(
                (tuple(config["geometry"]), pipeline, loop)
                for config in implicit["configs"]
                for pipeline, loop in config["pipeline_k_loop"]
            ),
            excluded=frozenset(implicit["excluded_instances"]),
            direct=frozenset(document["direct"]["instances"]),
        )

    @classmethod
    def load(cls, path: Path) -> "Selection":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def config_of(self, identity: dict) -> tuple:
        """``identity``'s ``(geometry, pipeline, k_loop)``."""
        geometry = tuple(identity[field] for field in self.geometry_fields)
        return geometry, identity["pipeline"], k_loop(identity)


def excluded_reason(identity: dict, selection: Selection):
    """Why the catalog leaves out an implicit-GEMM identity, or None to ship it.

    - The async DMA K loop, which no selected config uses.
    - A config the selection does not keep.
    - An instance the selection excludes from a kept config.
    """
    if identity["async_dma"]:
        return "async DMA K loop"
    if selection.config_of(identity) not in selection.configs:
        return "config not selected"
    if selection_hash(identity) in selection.excluded:
        return "instance excluded"
    return None


def sweep_identities(dtype: str, jobs: int) -> list:
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


def grid_identities(dtype: str, jobs: int, selection: Selection) -> tuple:
    """``(kept, swept)``: the identities of :func:`sweep_identities` the selection
    keeps, and all of them."""
    swept = sweep_identities(dtype, jobs)
    return [i for i in swept if excluded_reason(i, selection) is None], swept


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


def stale_selection(selection: Selection, swept: list, direct: list) -> list:
    """What the selection names that the sweep does not enumerate: one line per
    config, excluded instance or direct instance."""
    configs = {selection.config_of(i) for i in swept if not i["async_dma"]}
    hashes = {selection_hash(i) for i in swept}
    direct_hashes = {selection_hash(i) for i in direct}
    stale = [f"config {c}" for c in sorted(selection.configs - configs)]
    stale += [f"excluded instance {h}" for h in sorted(selection.excluded - hashes)]
    stale += [f"direct instance {h}" for h in sorted(selection.direct - direct_hashes)]
    return stale


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
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="The committed selection (default: %(default)s).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    _bind_rocke(args.rocke_root)
    selection = Selection.load(args.selection)

    lists, failed = {}, False
    print(f"{'dtype':6s} {'implicit':>8s} {'direct grid':>11s} {'direct':>6s}")
    for dtype in sorted(DTYPES):
        kept, swept = grid_identities(dtype, args.jobs, selection)
        implicit = sorted((stable_hash(i), i) for i in kept)
        if len({h for h, _ in implicit}) != len(implicit):
            print(f"{dtype}: the grid repeats an identity", file=sys.stderr)
            return 1
        rejected = validate([i for _, i in implicit], args.jobs)
        for instance_id, reason in rejected:
            print(f"unsupported instance {instance_id}: {reason}", file=sys.stderr)
        failed |= bool(rejected)
        packaged, refused = direct_identities(dtype, args.jobs)
        stale = stale_selection(selection, swept, packaged)
        for line in stale:
            print(
                f"{dtype}: the selection names a {line} the sweep lacks",
                file=sys.stderr,
            )
        failed |= bool(stale)
        direct = sorted(
            (
                (stable_hash(i), i)
                for i in packaged
                if selection_hash(i) in selection.direct
            ),
            key=lambda pair: pair[0],
        )
        print(
            f"{dtype:6s} {len(implicit):8d} "
            f"{len(packaged) + sum(refused.values()):11d} {len(direct):6d}"
        )
        for (algorithm, reason), count in sorted(refused.items()):
            print(f"  direct {algorithm}: {count} not packaged ({reason})")
        lists[dtype] = [i for _, i in implicit] + [i for _, i in direct]
    if failed:
        print(
            "unsupported implicit-GEMM instance(s) or a stale selection; nothing written",
            file=sys.stderr,
        )
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dtype, identities in lists.items():
        path = args.output_dir / INSTANCE_LIST.format(dtype=dtype)
        write_instances(path, identities)
        print(f"{len(identities)} instance(s) written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
