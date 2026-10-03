# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Step 0 exhaustive lever sweep for UniversalGemm fp16 RCR on gfx1250.

Implements step 0 of ``dsl_docs/optimization/optimization_runbook.md``: before
changing the kernel body or concluding a gap is structural, prove the *current*
implementation cannot already hit the target under a different configuration.

The runbook's core warning is that a curated preset menu does not satisfy step 0
-- it silently skips exactly the default-off levers most likely to be mis-picked.
So this driver enumerates the lever space from the spec dataclasses themselves
and classifies every combination against the real validator
(``gemm_universal.is_valid_spec``) rather than a duplicated copy of its rules.

Every variant lands in exactly one bucket, and the reason is recorded:

``legal``        built successfully; resources parsed from the HSACO ELF notes.
``spec_fail``    ``__post_init__`` rejected the field combination.
``gated``        ``is_valid_spec`` rejected it, with the validator's own reason.
``build_fail``   lowering or comgr failed.

Coverage that is deliberately bounded is reported in ``bounded_coverage``, and
levers the hardware supports but the builder cannot reach are reported in
``unreachable``. The runbook is explicit that silent truncation reads as
"covered everything" when it did not, so both are part of the output contract.

Output is a JSON lever manifest. It carries lever names, build outcomes, and
static resource counts only -- never a timing. Measured performance is subject
to ``platform/AGENTS.md`` compliance and belongs in the protected tracker, not
in this tree.

Usage::

    PYTHONPATH=platform/python python -m rocke.examples.gfx1250.gemm.step0_lever_sweep \\
        --stage geometry --out /path/to/manifest.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from rocke.assets import dsl_docs_dir
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
)

ARCH = "gfx1250"
ISA = f"amdgcn-amd-amdhsa--{ARCH}"

# The sole fp16 WMMA atom on gfx1250 (arch_specs.json: wmma_gfx1250_f32_16x16x32_f16).
# gfx1201's 16x16x16 does not exist here, so the atom is not a lever.
WARP_TILE = (16, 16, 32)
WAVE_SIZE = 32


# --------------------------------------------------------------------------
# Lever domains
# --------------------------------------------------------------------------
# Geometry is swept first and alone: the trait levers below all sit *on top of*
# a tile/warp geometry, so a trait sweep against an unvetted geometry measures
# the wrong thing. This is the runbook's "sweep one family at a time".

GEOMETRY_DOMAINS: Dict[str, Tuple[Any, ...]] = {
    "tile_m": (32, 64, 128, 256),
    "tile_n": (32, 64, 128, 256),
    "tile_k": (32, 64, 128),  # multiples of warp_tile_k=32
    "warp_m": (1, 2, 4),
    "warp_n": (1, 2, 4),
    # warp_k is pinned to 1: see DEFECTS["warp_k_dead"]. Sweeping it only
    # generates guaranteed rejections (432/432 in the first geometry pass).
    "warp_k": (1,),
}

TRAIT_DOMAINS: Dict[str, Tuple[Any, ...]] = {
    "pipeline": ("mem", "wmma_v1"),
    "scheduler": ("intrawave", "interwave"),
    "waves_per_eu": (None, 2, 3),
    "lds_k_pad": (0, 8),
    "lds_swizzle": (False, True),
    # split_k is pinned to 1: the validator rejects split_k > 1 as CDNA-only
    # ("got family 'wmma' on gfx1250"), so it is recorded in GATED_OFF, not swept.
    "split_k": (1,),
    "persistent": (False, True),
    # pad_* is swept as one coupled flag: 4096^3 is tile-aligned so the tail
    # path is unused, and the runbook asks us to prove a lever neutral rather
    # than assume it. Sweeping all three independently would 8x the space to
    # re-prove the same thing.
    "pad_all": (False, True),
}

# Hard-gated off for the WMMA path by gemm_universal.py:506-533. Sweeping these
# would only generate guaranteed SPEC-FAILs, so they are recorded, not swept.
GATED_OFF = {
    "epilogue": "WMMA path supports only the 'default' epilogue",
    "preshuffle_b": "WMMA path does not support preshuffle_b",
    "direct_to_lds": "WMMA path does not support direct_to_lds",
    "dtl_prefetch": "WMMA path does not support dtl_prefetch",
    "active_tile_skip": "WMMA path does not support active_tile_skip",
    "chiplet_swizzle": "WMMA path does not support chiplet_swizzle (no XCDs)",
    "warp_tile_*": "only the 16x16x32 atom exists for fp16 on gfx1250",
    "split_k": "split_k > 1 is CDNA-only; rejected for family 'wmma' on gfx1250",
    "warp_k": "see DEFECTS['warp_k_dead'] -- unusable on every arch, not just this one",
}

# Levers the hardware has but the builder cannot express. These are the leading
# structural candidates if the swept ceiling misses the target -- and they are
# precisely the class of lever the runbook warns goes missing from a preset menu.
UNREACHABLE = {
    "pipeline=wavelet": (
        "helpers/schedule.py:453 defines a gfx1250 'wavelet' pipeline "
        "(load/math wave specialization over gfx1250's separate VMEM and WMMA "
        "issue slots) but it is absent from the Pipeline literal at "
        "gemm_universal.py:122, so no spec can select it."
    ),
    "async_global_to_lds": (
        "arch_specs.json records has_async_global_lds=true for gfx1250 "
        "(global_load_async_to_lds_b128 + s_wait_asynccnt, GPU-validated via "
        "the DTLA path), but the WMMA gate forces direct_to_lds=False, so the "
        "async DRAM->LDS path is unreachable from UniversalGemmSpec."
    ),
    "ds_load_tr16_b128": (
        "arch_specs.json notes a GPU-validated hardware transpose-LDS read on "
        "gfx1250; has_ds_read_tr=false denotes only the absence of the gfx9 "
        "ABI, not the absence of the capability. Not wired into this builder."
    ),
}


# Builder defects found while enumerating. Recorded rather than worked around
# silently, because each one removes a lever a reader would assume was swept.
DEFECTS = {
    "warp_k_dead": (
        "warp_k > 1 is unconditionally rejected for UniversalGemm on every arch "
        "(verified gfx950 / gfx942 / gfx1250). helpers/spec.py:284 "
        "derive_block_size -- documented as the single source of truth -- uses "
        "warp_m*warp_n*warp_k*wave_size, but gemm_universal.py:544 validates "
        "against warp_m*warp_n*wave_size, omitting warp_k. So the auto-derived "
        "block_size can never satisfy the validator. Either the derivation or "
        "the check is wrong; pinned to warp_k=1 here and reported separately."
    ),
}

# gfx1250 per-wave architectural VGPR ceiling (arch_specs.json limits.vgprs).
# A reported count above this means the allocator failed, not that the kernel
# legitimately uses that many.
VGPR_CEILING = 256


def is_resource_clean(res: Dict[str, int]) -> Tuple[bool, str]:
    """Whether a built variant is free of spilling.

    ``vgpr_spill_count`` alone is not sufficient: a variant can report zero
    VGPR spills while still carrying a nonzero ``scratch_size`` (spilled to
    memory) or a ``vgpr_count`` above the architectural ceiling, which means
    the allocator gave up. The runbook treats any spill as invalidating a perf
    comparison, so all three are gates.
    """
    if res.get("vgpr_spill_count", 0) > 0:
        return False, f"vgpr_spill_count={res['vgpr_spill_count']}"
    if res.get("sgpr_spill_count", 0) > 0:
        return False, f"sgpr_spill_count={res['sgpr_spill_count']}"
    if res.get("scratch_size", 0) > 0:
        return False, f"scratch_size={res['scratch_size']}"
    if res.get("vgpr_count", 0) > VGPR_CEILING:
        return False, f"vgpr_count={res['vgpr_count']} > ceiling {VGPR_CEILING}"
    return True, ""


@dataclass
class Variant:
    """One enumerated configuration and what became of it."""

    name: str
    levers: Dict[str, Any]
    bucket: str = "legal"
    reason: str = ""
    resources: Dict[str, int] = field(default_factory=dict)
    build_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "levers": self.levers,
            "bucket": self.bucket,
        }
        if self.reason:
            out["reason"] = self.reason
        if self.resources:
            out["resources"] = self.resources
        if self.build_ms:
            out["build_ms"] = round(self.build_ms, 1)
        return out


def _make_spec(levers: Dict[str, Any]) -> UniversalGemmSpec:
    """Build a spec from a flat lever dict. May raise from ``__post_init__``."""
    pad = bool(levers.get("pad_all", False))
    tile = TileSpec(
        tile_m=levers["tile_m"],
        tile_n=levers["tile_n"],
        tile_k=levers["tile_k"],
        warp_m=levers["warp_m"],
        warp_n=levers["warp_n"],
        warp_k=levers["warp_k"],
        warp_tile_m=WARP_TILE[0],
        warp_tile_n=WARP_TILE[1],
        warp_tile_k=WARP_TILE[2],
    )
    trait = TraitSpec(
        pipeline=levers.get("pipeline", "mem"),
        scheduler=levers.get("scheduler", "intrawave"),
        epilogue="default",  # gated: the only legal value on the WMMA path
        pad_m=pad,
        pad_n=pad,
        pad_k=pad,
        persistent=levers.get("persistent", False),
        waves_per_eu=levers.get("waves_per_eu"),
        lds_k_pad=levers.get("lds_k_pad", 0),
        lds_swizzle=levers.get("lds_swizzle", False),
        split_k=levers.get("split_k", 1),
    )
    return UniversalGemmSpec(
        name=_variant_name(levers),
        tile=tile,
        trait=trait,
        data=DataSpec(),  # fp16 / fp16 / fp16 / fp32, RCR
        wave_size=WAVE_SIZE,
    )


def _variant_name(levers: Dict[str, Any]) -> str:
    """A name that encodes the hypothesis, per runbook 12.4."""
    parts = [f"t{levers['tile_m']}x{levers['tile_n']}x{levers['tile_k']}"]
    parts.append(f"w{levers['warp_m']}x{levers['warp_n']}x{levers['warp_k']}")
    for key in ("pipeline", "scheduler"):
        if key in levers:
            parts.append(str(levers[key]))
    for key, tag in (
        ("waves_per_eu", "wpe"),
        ("lds_k_pad", "pad"),
        ("split_k", "sk"),
    ):
        if levers.get(key):
            parts.append(f"{tag}{levers[key]}")
    for key, tag in (("lds_swizzle", "swz"), ("persistent", "pers"), ("pad_all", "padt")):
        if levers.get(key):
            parts.append(tag)
    return "gemm_fp16_rcr_" + "_".join(parts)


def enumerate_variants(domains: Dict[str, Tuple[Any, ...]]) -> Iterator[Dict[str, Any]]:
    keys = list(domains)
    for combo in itertools.product(*(domains[k] for k in keys)):
        yield dict(zip(keys, combo))


def _classify(levers: Dict[str, Any]) -> Variant:
    """Bucket a lever combination without compiling it."""
    try:
        spec = _make_spec(levers)
    except (ValueError, TypeError) as exc:
        return Variant(
            name=_variant_name(levers), levers=levers, bucket="spec_fail", reason=str(exc)
        )
    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        return Variant(name=spec.name, levers=levers, bucket="gated", reason=why)
    return Variant(name=spec.name, levers=levers, bucket="legal")


def _build_one(levers: Dict[str, Any]) -> Dict[str, Any]:
    """Compile one legal variant. Runs in a worker process."""
    # Imported here so the fork workers do not pay for them when classification
    # already rejected the variant.
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    sys.path.insert(0, str(_PROBE_DIR))
    from probe_occupancy import parse_hsaco_notes

    var = _classify(levers)
    if var.bucket != "legal":
        return var.as_dict()

    t0 = time.time()
    try:
        spec = _make_spec(levers)
        ir = lower_kernel_to_llvm(build_universal_gemm(spec, ARCH), arch=ARCH)
        hsaco, _ = build_hsaco_from_llvm_ir(ir, isa=ISA)
        var.resources = parse_hsaco_notes(hsaco)
        var.resources["hsaco_bytes"] = len(hsaco)
    except Exception as exc:  # noqa: BLE001 - a build failure is data, not a crash
        var.bucket = "build_fail"
        var.reason = f"{type(exc).__name__}: {exc}"
    var.build_ms = (time.time() - t0) * 1000.0
    return var.as_dict()


# Non-package assets resolve through rocke.assets, never per-file parents[N]
# math (CLAUDE.md, "Hard rules").
_PROBE_DIR = dsl_docs_dir() / "optimization" / "utilities" / "tools" / "dsl_probes"


def run_sweep(
    domains: Dict[str, Tuple[Any, ...]],
    *,
    parallel: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    combos = list(enumerate_variants(domains))
    truncated = 0
    if limit is not None and len(combos) > limit:
        truncated = len(combos) - limit
        combos = combos[:limit]
    if truncated:
        print(f"[warn] --limit dropped {truncated} combinations from the sweep")

    n = parallel or max(1, (os.cpu_count() or 4) - 2)
    if n == 1:
        return [_build_one(c) for c in combos]
    ctx = mp.get_context("fork")  # avoid re-importing rocke under spawn
    with ctx.Pool(n) as pool:
        return pool.map(_build_one, combos)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, int] = {}
    for r in results:
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1

    legal = [r for r in results if r["bucket"] == "legal"]
    spills = [r for r in legal if not is_resource_clean(r.get("resources", {}))[0]]

    # Distinct validator reasons, so a dominant gate is visible at a glance.
    reasons: Dict[str, int] = {}
    for r in results:
        if r["bucket"] in ("gated", "spec_fail"):
            key = r.get("reason", "")[:90]
            reasons[key] = reasons.get(key, 0) + 1

    unclean_reasons: Dict[str, int] = {}
    for r in spills:
        why = is_resource_clean(r.get("resources", {}))[1].split("=")[0]
        unclean_reasons[why] = unclean_reasons.get(why, 0) + 1

    return {
        "buckets": buckets,
        "built_ok": len(legal),
        "with_spills": len(spills),
        "spill_free": len(legal) - len(spills),
        "unclean_by_reason": unclean_reasons,
        "top_rejection_reasons": dict(
            sorted(reasons.items(), key=lambda kv: -kv[1])[:8]
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--stage",
        choices=("geometry", "traits", "all"),
        default="geometry",
        help="which lever family to sweep (runbook 12.2: one family at a time)",
    )
    p.add_argument("--out", type=Path, default=None, help="lever manifest JSON path")
    p.add_argument("--parallel", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap variants (logged)")
    p.add_argument(
        "--base-geometry",
        default="",
        help="for --stage traits: 'tm,tn,tk,wm,wn,wk' to sweep traits on top of",
    )
    args = p.parse_args(argv)

    domains: Dict[str, Tuple[Any, ...]] = {}
    if args.stage in ("geometry", "all"):
        domains.update(GEOMETRY_DOMAINS)
    if args.stage in ("traits", "all"):
        if args.stage == "traits":
            if not args.base_geometry:
                p.error("--stage traits requires --base-geometry tm,tn,tk,wm,wn,wk")
            vals = [int(v) for v in args.base_geometry.split(",")]
            if len(vals) != 6:
                p.error("--base-geometry needs exactly 6 comma-separated ints")
            domains.update(
                {
                    k: (v,)
                    for k, v in zip(
                        ("tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_k"),
                        vals,
                    )
                }
            )
        domains.update(TRAIT_DOMAINS)

    total = 1
    for v in domains.values():
        total *= len(v)
    print(f"[step0] arch={ARCH} stage={args.stage} enumerated={total} combinations")

    t0 = time.time()
    results = run_sweep(domains, parallel=args.parallel, limit=args.limit)
    elapsed = time.time() - t0

    summary = summarize(results)
    manifest = {
        "arch": ARCH,
        "instance": "UniversalGemmSpec",
        "dtypes": {"a": "fp16", "b": "fp16", "c": "fp16", "acc": "fp32"},
        "layout": "RCR",
        "wave_size": WAVE_SIZE,
        "atom": f"{WARP_TILE[0]}x{WARP_TILE[1]}x{WARP_TILE[2]} (wmma_gfx1250_f32_16x16x32_f16)",
        "stage": args.stage,
        "swept_levers": {k: list(v) for k, v in domains.items()},
        "gated_off": GATED_OFF,
        "unreachable": UNREACHABLE,
        "defects": DEFECTS,
        "bounded_coverage": _bounded_coverage(args),
        "summary": summary,
        "variants": results,
        "sweep_seconds": round(elapsed, 1),
        "note": (
            "Static build/resource data only. No timings: measured performance is "
            "subject to platform/AGENTS.md compliance and does not belong in-tree."
        ),
    }

    out = args.out or (Path(tempfile.mkdtemp(prefix="rocke_step0_")) / "lever_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))

    print(f"[step0] {elapsed:.1f}s  buckets={summary['buckets']}")
    print(f"[step0] spill-free legal variants: {summary['spill_free']}")
    print(f"[step0] manifest: {out}")
    return 0


def _bounded_coverage(args: argparse.Namespace) -> Dict[str, str]:
    """Every place coverage was deliberately capped, and why."""
    out: Dict[str, str] = {
        "pad_m/pad_n/pad_k": (
            "swept as one coupled 'pad_all' flag rather than 3 independent bools; "
            "4096^3 is tile-aligned so the tail path is unused"
        ),
        "staging": (
            "geometry and traits are swept in separate stages (runbook 12.2), not "
            "as one cartesian product; the trait stage is conditioned on the "
            "geometry winner"
        ),
    }
    if args.limit is not None:
        out["--limit"] = f"variant count capped at {args.limit} by the caller"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
