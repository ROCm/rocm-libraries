#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Enumerate the rocKE gfx950 forward-conv AOT instances the hipkernel pack ships.

The instances are a filtered slice of rocKE's own AOT sweep grid
(``benchmarks/common/kernel_sweep.py``): every valid gfx950 forward implicit-GEMM
``KernelIdentity`` for each dtype, then

1. the knob rule (:data:`KNOB_RULE`): only the listed values of each compile-time
   knob are kept; every other knob value is dominated by a kept one;
2. rule R1: drop every instance whose compiled kernel uses scratch (VGPR spills).
   A spilling kernel is never the fastest one, and each has a spill-free sibling
   of the same tile geometry.

R1 needs a compile of every candidate, so its verdict is committed as an exclusion
list of ``stable_hash()`` values (``configs/gfx950_conv_fwd_aot.spills.json``). The
default run reads that list; ``--recompute-spills`` rebuilds it by compiling every
candidate on the CPU (no GPU) in a process pool. COMGR's own cache
(``~/.cache/comgr``) makes a repeat run much cheaper than a cold one. The list
records a digest of the candidates it was computed over, so a changed grid or knob
rule is refused instead of silently filtered with a stale list.

The output is a JSON list with one identity per line for every dtype, sorted by
``(dtype, stable_hash)``. Each entry is compact (:func:`compact_identity`): the
``KernelIdentity`` fields without a plain default, plus every field that differs
from its default; ``KernelIdentity.from_dict`` restores the rest, so the entry
keeps its ``stable_hash()``. The list is the input of
``make_conv_aot_config.py --instances``, and each entry is validated with that
tool's ``instance_spec`` before anything is written::

    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --output configs/gfx950_conv_fwd_aot.instances.json
    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --recompute-spills --output configs/gfx950_conv_fwd_aot.instances.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

ARCH = "gfx950"
DTYPES = ("fp16", "bf16")

#: The knob rule: the values each compile-time knob may take. An identity whose
#: knob holds any other value is dropped. Every value the grid has for a knob that
#: is not listed is dominated by a listed one.
KNOB_RULE = {
    # compv4 emits compv3's code plus one s_setprio, and is never faster than it;
    # mem alone loses on strided filters, so both stay.
    "pipeline": ("mem", "compv3"),
    # The default epilogue is never faster than cshuffle, and cshuffle with C width
    # 1 serves every K/groups, so no problem loses coverage.
    "epilogue": ("cshuffle",),
    # A and B load widths (always equal in the grid). Width 2 is never needed:
    # 8 or 4 serve wherever it does. Width 4 is needed for channel counts 8 does
    # not divide, width 1 is the universal fallback.
    "vector_size_a": (8, 4, 1),
    "vector_size_b": (8, 4, 1),
    # C store width: as for A/B, width 2 is never needed.
    "vector_size_c": (8, 4, 1),
    # K loop: plain and unrolled both win; the async DMA loop never does.
    "unroll_k": (False, True),
    "async_dma": (False,),
}

#: Rule R1, as applied to the code-object metadata of the compiled kernel.
R1 = (
    "drop instances whose kernel uses scratch: "
    ".vgpr_spill_count > 0 or .private_segment_fixed_size > 0"
)

_HERE = Path(__file__).resolve().parent
DEFAULT_SPILLS = _HERE.parent / "configs" / "gfx950_conv_fwd_aot.spills.json"

#: Code-object metadata fields R1 reads (``llvm-readelf --notes``).
_NOTE_FIELDS = {
    "vgpr_spill": re.compile(r"\.vgpr_spill_count:\s+(\d+)"),
    "scratch_bytes": re.compile(r"\.private_segment_fixed_size:\s+(\d+)"),
}


def _bind_rocke(rocke_root: Path) -> None:
    for sub in ("platform/python", "library"):
        path = str((rocke_root / sub).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def knob_rule_rejection(identity: dict) -> str:
    """The first knob of ``identity`` outside :data:`KNOB_RULE`, as a reason; ``""``
    when the identity passes."""
    for knob, kept in KNOB_RULE.items():
        if identity[knob] not in kept:
            return f"{knob}={identity[knob]!r} not in {kept!r}"
    return ""


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


def stable_hash(identity: dict) -> str:
    from benchmarks.common.kernel_cache import KernelIdentity

    return KernelIdentity.from_dict(identity).stable_hash()


def candidates_digest(hashes) -> str:
    """SHA-256 over the sorted candidate hashes R1 was computed for."""
    return hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest()


def _readelf() -> str:
    return str(Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/llvm-readelf")


def scratch_usage(hsaco: bytes) -> dict:
    """``{"vgpr_spill": n, "scratch_bytes": n}`` from a code object's metadata."""
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
        handle.write(hsaco)
        handle.flush()
        notes = subprocess.run(
            [_readelf(), "--notes", handle.name],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    usage = {}
    for name, pattern in _NOTE_FIELDS.items():
        match = pattern.search(notes)
        if match is None:
            raise RuntimeError(f"code-object metadata has no {pattern.pattern!r}")
        usage[name] = int(match.group(1))
    return usage


def spills(usage: dict) -> bool:
    return usage["vgpr_spill"] > 0 or usage["scratch_bytes"] > 0


def _init_worker(rocke_root: str) -> None:
    _bind_rocke(Path(rocke_root))


def _compile_one(identity: dict):
    """Compile one candidate on the CPU; ``(identity, usage, error)``."""
    from benchmarks.common import kernel_sweep
    from benchmarks.common.kernel_cache import KernelIdentity
    from rocke.helpers.compile import lower_kernel_for_comgr
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    try:
        job = kernel_sweep.job_for_identity(KernelIdentity.from_dict(identity))
        kernel, _ = kernel_sweep.build_kernel(job, ARCH, identity["dtype_a"])
        lowered = lower_kernel_for_comgr(kernel, arch=ARCH, backend="python")
        hsaco, _ = build_hsaco_from_llvm_ir(
            lowered.llvm_text, isa=lowered.isa, options=list(lowered.options)
        )
        return identity, scratch_usage(hsaco), ""
    except Exception as exc:  # noqa: BLE001 - reported per identity by the caller
        return identity, None, f"{type(exc).__name__}: {exc}"[:400]


def spill_entry(identity: dict, usage: dict) -> dict:
    """One exclusion-list entry: the hash, plus enough to recognise the kernel."""
    return {
        "instance_id": stable_hash(identity),
        "dtype": identity["dtype_a"],
        "tile": [identity["tile_m"], identity["tile_n"], identity["tile_k"]],
        "warps": [identity["warp_m"], identity["warp_n"]],
        "warp_tile": [
            identity["warp_tile_m"],
            identity["warp_tile_n"],
            identity["warp_tile_k"],
        ],
        "pipeline": identity["pipeline"],
        "vector_size_ab": identity["vector_size_a"],
        "vector_size_c": identity["vector_size_c"],
        "unroll_k": identity["unroll_k"],
        "vgpr_spill": usage["vgpr_spill"],
        "scratch_bytes": usage["scratch_bytes"],
    }


def compute_spills(rocke_root: Path, candidates: list, jobs: int, log=print) -> list:
    """Compile every candidate; the exclusion-list entries of those that spill."""
    os.environ.setdefault("ROCM_PATH", "/opt/rocm")
    os.environ["ROCKE_BACKEND"] = "python"
    entries, failures = [], []
    started = time.perf_counter()
    with get_context("spawn").Pool(
        jobs,
        initializer=_init_worker,
        initargs=(str(rocke_root),),
        maxtasksperchild=50,
    ) as pool:
        for done, (identity, usage, error) in enumerate(
            pool.imap_unordered(_compile_one, candidates, chunksize=1), 1
        ):
            if error:
                failures.append((stable_hash(identity), error))
            elif spills(usage):
                entries.append(spill_entry(identity, usage))
            if done % 2000 == 0 or done == len(candidates):
                log(
                    f"  compiled {done}/{len(candidates)}, {len(entries)} spill "
                    f"({time.perf_counter() - started:.0f}s)"
                )
    for instance_id, error in failures:
        print(f"compile failed {instance_id}: {error}", file=sys.stderr)
    if failures:
        raise SystemExit(f"{len(failures)} candidate(s) failed to compile")
    return sorted(entries, key=lambda e: (e["dtype"], e["instance_id"]))


def write_spill_list(path: Path, entries: list, candidates: dict) -> None:
    """Write the exclusion list. ``candidates``: dtype -> candidate hashes."""
    header = {
        "generated_by": "tools/enumerate_conv_aot_instances.py --recompute-spills",
        "arch": ARCH,
        "rule": R1,
        "knob_rule": {knob: list(kept) for knob, kept in KNOB_RULE.items()},
        "candidates": {
            dtype: {"count": len(hashes), "sha256": candidates_digest(hashes)}
            for dtype, hashes in sorted(candidates.items())
        },
    }
    lines = ["{"]
    for key, value in header.items():
        if isinstance(value, dict):
            inner = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in value.items()]
            lines.append(f" {json.dumps(key)}: {{\n" + ",\n".join(inner) + "\n },")
        else:
            lines.append(f" {json.dumps(key)}: {json.dumps(value)},")
    lines.append(' "spills": [')
    lines.append(",\n".join(f"  {json.dumps(e)}" for e in entries))
    lines.append(" ]\n}\n")
    path.write_text("\n".join(lines))


def read_spill_list(path: Path, candidates: dict) -> set:
    """The excluded hashes, after checking the list was computed for ``candidates``."""
    document = json.loads(path.read_text())
    for dtype, hashes in candidates.items():
        recorded = document["candidates"].get(dtype)
        if recorded != {"count": len(hashes), "sha256": candidates_digest(hashes)}:
            raise SystemExit(
                f"{path} was computed for another {dtype} candidate set (the rocKE "
                "grid or the knob rule changed); rerun with --recompute-spills"
            )
    return {entry["instance_id"] for entry in document["spills"]}


def validate(identities: list) -> list:
    """``(stable_hash, reason)`` for each identity the converter rejects."""
    sys.path.insert(0, str(_HERE))
    from make_conv_aot_config import instance_spec

    rejected = []
    for identity in identities:
        spec, reason = instance_spec(identity)
        if spec is None:
            rejected.append((stable_hash(identity), reason))
    return rejected


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
        "--output", type=Path, required=True, help="Instance list (JSON) to write."
    )
    parser.add_argument(
        "--spills",
        type=Path,
        default=DEFAULT_SPILLS,
        help="R1 exclusion list (default: %(default)s).",
    )
    parser.add_argument(
        "--recompute-spills",
        action="store_true",
        help="Compile every candidate (CPU only) and rewrite --spills.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    _bind_rocke(args.rocke_root)

    grid, candidates = {}, {}
    for dtype in DTYPES:
        grid[dtype] = grid_identities(dtype, args.jobs)
        candidates[dtype] = [i for i in grid[dtype] if not knob_rule_rejection(i)]
    hashes = {d: [stable_hash(i) for i in ids] for d, ids in candidates.items()}

    if args.recompute_spills:
        started = time.perf_counter()
        everything = [i for dtype in DTYPES for i in candidates[dtype]]
        print(f"R1: compiling {len(everything)} candidate(s) with {args.jobs} jobs")
        entries = compute_spills(args.rocke_root, everything, args.jobs)
        write_spill_list(args.spills, entries, hashes)
        print(
            f"R1: {len(entries)} spill(s) written to {args.spills} "
            f"({time.perf_counter() - started:.0f}s)"
        )
    excluded = read_spill_list(args.spills, hashes)

    kept = []
    print(f"{'dtype':6s} {'grid':>7s} {'knob rule':>9s} {'R1':>7s} {'distinct':>8s}")
    for dtype in sorted(DTYPES):
        survivors = [
            (h, i)
            for h, i in zip(hashes[dtype], candidates[dtype])
            if h not in excluded
        ]
        distinct = len({h for h, _ in survivors})
        print(
            f"{dtype:6s} {len(grid[dtype]):7d} {len(candidates[dtype]):9d} "
            f"{len(survivors):7d} {distinct:8d}"
        )
        kept.extend(sorted(survivors, key=lambda pair: pair[0]))
    identities = [identity for _, identity in kept]

    rejected = validate(identities)
    for instance_id, reason in rejected:
        print(f"unsupported instance {instance_id}: {reason}", file=sys.stderr)
    if rejected:
        print(
            f"{len(rejected)} unsupported instance(s); nothing written", file=sys.stderr
        )
        return 1
    write_instances(args.output, identities)
    print(f"{len(identities)} instance(s) written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
