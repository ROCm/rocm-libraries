#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Enumerate the rocKE gfx950 forward-conv AOT instances the hipkernel pack ships.

The instances come from rocKE's own sweep grids, per dtype, in two families.

Implicit GEMM is a filtered slice of the AOT sweep grid
(``benchmarks/common/kernel_sweep.py``): every valid gfx950 forward implicit-GEMM
``KernelIdentity``, then

1. the knob rule (:data:`KNOB_RULE`, :data:`VECTOR_SIZES`,
   :data:`EPILOGUE_BY_VECTOR_SIZE_C`): only the listed knob values, load/store
   width pairs and epilogues are kept. The rule is a policy choice of which knob
   combinations to ship; narrowing it further is left to later steps;
2. rule R1: drop every instance whose compiled kernel uses scratch (VGPR spills).
   ``--keep-spilling`` skips R1 and keeps every candidate of the knob rule.

R1 needs a compile of every candidate, so its verdict is committed as an exclusion
list of ``stable_hash()`` values (``configs/gfx950_conv_fwd_aot.spills.json``). The
default run reads that list; ``--recompute-spills`` rebuilds it by compiling every
candidate on the CPU (no GPU) in a process pool. COMGR's own cache
(``~/.cache/comgr``) makes a repeat run much cheaper than a cold one. The list
records a digest of the candidates it was computed over, so a changed grid or knob
rule is refused instead of silently filtered with a stale list.

Direct depthwise is every forward identity of the direct sweep grid
(``benchmarks/common/direct_kernel_sweep.py``) whose algorithm the packaged direct
launch runs (``direct_depthwise`` and ``direct_depthwise_spatial``) and that
``make_conv_aot_config.py`` accepts. The direct grid validates each kernel itself,
so neither rule above applies to it.

The output is a JSON list with one identity per line, ordered by dtype, then family
(implicit GEMM first), then ``stable_hash``. Each entry is compact
(:func:`compact_identity`): the
``KernelIdentity`` fields without a plain default, plus every field that differs
from its default; ``KernelIdentity.from_dict`` restores the rest, so the entry
keeps its ``stable_hash()``. The list is the input of
``make_conv_aot_config.py --instances``, and each entry is validated with that
tool's ``instance_spec`` before anything is written::

    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --output configs/gfx950_conv_fwd_aot.instances.json
    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --recompute-spills --output configs/gfx950_conv_fwd_aot.instances.json
    python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" \\
        --keep-spilling --output "$WORK"/with_spilling.instances.json
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

#: The knob rule, part 1: the values each listed compile-time knob may take. An
#: identity whose knob holds any other value is dropped.
KNOB_RULE = {
    # All three implicit-GEMM pipelines of the grid ship.
    "pipeline": ("mem", "compv3", "compv4"),
    # K loop: plain or unrolled. The async DMA loop is not shipped.
    "unroll_k": (False, True),
    "async_dma": (False,),
}

#: The knob rule, part 2: the ``(load width, store width)`` pairs that ship, where
#: the load width is ``vector_size_a``, always equal to ``vector_size_b``, and the
#: store width is ``vector_size_c``. Equal widths 1, 2, 4 and 8, plus the two mixed
#: pairs of widths 1 and 4, so that channel counts that only one side's width
#: divides keep a wide load or a wide store.
VECTOR_SIZES = ((1, 1), (2, 2), (4, 4), (8, 8), (1, 4), (4, 1))

#: The knob rule, part 3: the epilogue is tied to the store width. A store width
#: of 1 ships the default epilogue, a wider one cshuffle. In the grid the default
#: epilogue only exists with a store width of 1.
EPILOGUE_BY_VECTOR_SIZE_C = {1: "default", 2: "cshuffle", 4: "cshuffle", 8: "cshuffle"}

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


def _converter():
    """``make_conv_aot_config``, the tool that consumes the list."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    import make_conv_aot_config

    return make_conv_aot_config


def knob_rule_rejection(identity: dict) -> str:
    """Why ``identity`` falls outside the knob rule, as a reason naming the first
    offending knob; ``""`` when the identity passes."""
    for knob, kept in KNOB_RULE.items():
        if identity[knob] not in kept:
            return f"{knob}={identity[knob]!r} not in {kept!r}"
    load, store = identity["vector_size_a"], identity["vector_size_c"]
    if identity["vector_size_b"] != load:
        return f"vector_size_b={identity['vector_size_b']!r} != vector_size_a={load!r}"
    if (load, store) not in VECTOR_SIZES:
        return f"vector_size_a/c=({load!r}, {store!r}) not in {VECTOR_SIZES!r}"
    epilogue = EPILOGUE_BY_VECTOR_SIZE_C[store]
    if identity["epilogue"] != epilogue:
        return (
            f"epilogue={identity['epilogue']!r} with vector_size_c={store!r} "
            f"is not {epilogue!r}"
        )
    return ""


def knob_rule_document() -> dict:
    """The knob rule as JSON data, recorded in the spill list."""
    return {
        **{knob: list(kept) for knob, kept in KNOB_RULE.items()},
        "vector_sizes": [list(pair) for pair in VECTOR_SIZES],
        "epilogue_by_vector_size_c": {
            str(width): epilogue
            for width, epilogue in EPILOGUE_BY_VECTOR_SIZE_C.items()
        },
    }


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
    """``(packaged, grid_count)``: the forward identities of rocKE's direct sweep grid
    that the packaged direct launch runs, as dicts, and how many forward identities the
    grid has in all. An identity is packaged when its algorithm is one of
    ``make_conv_aot_config.DIRECT_ALGORITHMS`` and the converter accepts it."""
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
    forward = [job.identity.to_dict() for job in found]
    packaged = [
        identity
        for identity in forward
        if identity["algorithm"] in converter.DIRECT_ALGORITHMS
        and converter.instance_spec(identity)[0] is not None
    ]
    return packaged, len(forward)


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
        "epilogue": identity["epilogue"],
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
        "knob_rule": knob_rule_document(),
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
    rejected = []
    for identity in identities:
        spec, reason = _converter().instance_spec(identity)
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
        "--keep-spilling",
        action="store_true",
        help="Skip R1: keep the candidates that spill as well.",
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
    excluded = set() if args.keep_spilling else read_spill_list(args.spills, hashes)

    kept = []
    print(
        f"{'dtype':6s} {'grid':>7s} {'knob rule':>9s} {'R1':>7s} {'distinct':>8s} "
        f"{'direct grid':>11s} {'direct':>6s}"
    )
    for dtype in sorted(DTYPES):
        survivors = [
            (h, i)
            for h, i in zip(hashes[dtype], candidates[dtype])
            if h not in excluded
        ]
        distinct = len({h for h, _ in survivors})
        direct, direct_grid = direct_identities(dtype, args.jobs)
        print(
            f"{dtype:6s} {len(grid[dtype]):7d} {len(candidates[dtype]):9d} "
            f"{len(survivors):7d} {distinct:8d} {direct_grid:11d} {len(direct):6d}"
        )
        kept.extend(sorted(survivors, key=lambda pair: pair[0]))
        kept.extend(sorted(((stable_hash(i), i) for i in direct), key=lambda p: p[0]))
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
