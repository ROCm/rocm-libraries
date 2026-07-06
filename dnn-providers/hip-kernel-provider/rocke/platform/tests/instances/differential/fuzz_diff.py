#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# fuzz_diff.py -- random-spec differential fuzzer for the dual lowering engine.
#
# The fixed sampled-config parity harnesses (run_diff.py / ir_artifact_diff.py)
# pin a hand-chosen list of specs per family. That leaves the *edges* of the
# valid spec space -- odd tile shapes, off-default knob combinations, the less
# common atoms/pipelines -- unexercised. This fuzzer randomly walks the valid
# spec space of each target family and byte-compares the two lowering paths on
# every buildable spec it samples, so the differential coverage is no longer
# limited to the curated list.
#
# COMPARISON ORACLE (the IR-artifact identity, same invariant ir_artifact_diff
# proves on the fixed configs):
#
#     ir_text = serialize( build_<family>(spec) )          # Python authors IR
#     ll_ref  = lower_kernel_to_llvm(kernel, arch)         # Python lowers it
#     ll_art  = ir_lower_cli <arch> < ir_text              # C parses + C lowers
#
#     IDENTICAL  ll_art == ll_ref byte-for-byte            (seam collapses)
#     MISMATCH   both non-empty, bytes differ              (real divergence)
#     SKIPPED    spec failed its validity gate (not built) (expected, counted)
#     ERROR      the artifact path could not be exercised  (build/parse/lower gap)
#
# This is the single-tree oracle: both sides are *this* tree's Python authoring
# plus the C lowerer reading that tree's serialized IR. It is deliberately NOT a
# C-build-vs-Python-build comparison, which on the current branch would surface
# the pre-merge scheduler-hint divergence (the C engine matches the merge-target
# Python, not the current-branch Python, on those families) as false drift. The
# IR-artifact identity has no such asymmetry: a mismatch here is a genuine parser
# round-trip gap or a real lowering divergence, not a tree-skew artifact.
#
# Every sampled spec is run through its family validity gate FIRST; invalid
# combinations are counted and skipped, so only buildable specs are diffed. The
# RNG is seeded (CLI --seed) so a run -- and any mismatch it finds -- is exactly
# reproducible. A real mismatch is printed with the offending spec (enough to
# rebuild it) and the first differing line, and the process exits nonzero.

import argparse
import hashlib
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[2]  # rocKE root (differential -> instances -> tests -> rocKE)
PYROOT = ROCKE / "Python"  # holds rocke
if str(PYROOT) not in sys.path:
    sys.path.insert(0, str(PYROOT))

TIMEOUT = 120  # seconds per CLI lowering call

DEFAULT_CLI = str(Path(tempfile.gettempdir()) / "rocke_irart" / "ir_lower_cli")
DEFAULT_SEED = 1234


# ---------------------------------------------------------------------------
# Python-side imports (lazy so a missing module fails with a clear message).
# ---------------------------------------------------------------------------
def _ck_imports():
    # Use the NATIVE Python lowerer (not the backend-dispatching public
    # lower_kernel_to_llvm): this fuzzer is the Python-reference side of the
    # python-vs-C-CLI differential, so it must emit the Python engine's .ll
    # regardless of the package-default backend.
    from rocke.core.ir_serialize import serialize

    try:
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as native
    except ImportError:  # pragma: no cover - older reference tree
        from rocke import lower_kernel_to_llvm as native

    return native, serialize


# ---------------------------------------------------------------------------
# Per-family samplers.
#
# Each sampler is a callable rng -> (spec_obj, repr_str, arch). The family entry
# bundles it with the family's validity gate and IR builder. The gate is run
# before building so invalid combinations are skipped (not diffed). repr_str is a
# compact, re-buildable description printed on mismatch so a finding is minimal
# and reproducible by hand.
# ---------------------------------------------------------------------------

# Universal GEMM is the richest space: tile/warp geometry, the four CDNA MFMA
# atoms, three+ pipelines, two epilogues, two dtypes, plus the optional knobs.
_GEMM_ATOMS_F16 = [(16, 16, 16), (16, 16, 32), (32, 32, 8), (32, 32, 16)]
_GEMM_ATOMS_BF16 = [(16, 16, 16), (16, 16, 32)]
_GEMM_PIPELINES = ["mem", "compv3", "compv4"]
_GEMM_EPILOGUES = ["default", "cshuffle"]
_GEMM_WARP_GRID = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (2, 4),
    (4, 2),
    (4, 4),
]
# k-atom counts (tile_k = warp_tile_k * count); mfma counts per warp in m/n.
_GEMM_KCOUNT = [1, 2, 4, 8]
_GEMM_MFMA_MN = [1, 2, 3]


def _sample_gemm(rng):
    """Sample a UniversalGemmSpec across the full valid surface, including a few
    optional knobs. Geometry is constructed from the atom + warp grid + mfma
    counts so divisibility holds by construction; the family gate still vets the
    LDS budget, atom availability, and block-size identity."""
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    dtype = rng.choice(["fp16", "bf16"])
    atom = rng.choice(_GEMM_ATOMS_F16 if dtype == "fp16" else _GEMM_ATOMS_BF16)
    wtm, wtn, wtk = atom
    warp_m, warp_n = rng.choice(_GEMM_WARP_GRID)
    mfma_m = rng.choice(_GEMM_MFMA_MN)
    mfma_n = rng.choice(_GEMM_MFMA_MN)
    kcount = rng.choice(_GEMM_KCOUNT)
    tile_m = warp_m * wtm * mfma_m
    tile_n = warp_n * wtn * mfma_n
    tile_k = wtk * kcount

    pipeline = rng.choice(_GEMM_PIPELINES)
    epilogue = rng.choice(_GEMM_EPILOGUES)

    # Optional knobs, each toggled with low probability so the bulk of samples
    # stay near the common path and the knobs still get exercised over a run.
    knobs = {}
    if rng.random() < 0.15:
        knobs["chiplet_swizzle"] = True
    if rng.random() < 0.15:
        knobs["lds_swizzle"] = True
    if rng.random() < 0.15:
        knobs["lds_k_pad"] = rng.choice([8, 16])
    if rng.random() < 0.10:
        knobs["persistent"] = True
    if rng.random() < 0.10:
        knobs["pad_m"] = knobs["pad_n"] = knobs["pad_k"] = True
    if rng.random() < 0.10:
        knobs["waves_per_eu"] = 2
    if rng.random() < 0.10:
        knobs["preshuffle_b"] = True
    if rng.random() < 0.10:
        knobs["direct_to_lds"] = True
        if rng.random() < 0.5:
            knobs["dtl_prefetch"] = True
    batched = rng.random() < 0.20
    if batched and rng.random() < 0.4:
        knobs["active_tile_skip"] = True

    trait = TraitSpec(pipeline=pipeline, epilogue=epilogue, **knobs)
    spec = UniversalGemmSpec(
        name="fuzz_gemm",
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
        ),
        trait=trait,
        data=DataSpec(dtype_a=dtype, dtype_b=dtype, dtype_c=dtype),
        wave_size=64,
        block_size=warp_m * warp_n * 64,
        batched=batched,
    )
    rep = (
        f"gemm dtype={dtype} tile={tile_m}x{tile_n}x{tile_k} "
        f"warp={warp_m}x{warp_n}x1 wt={wtm}x{wtn}x{wtk} "
        f"pipeline={pipeline} epilogue={epilogue} batched={batched} knobs={knobs}"
    )
    return spec, rep, "gfx950"


def _gemm_valid(spec):
    from rocke.instances.common.gemm_universal import is_valid_spec

    return is_valid_spec(spec, arch="gfx950")


def _gemm_build(spec):
    from rocke.instances.common.gemm_universal import build_universal_gemm

    return build_universal_gemm(spec, arch="gfx950")


# Batched GEMM: the same universal geometry surface routed through the batched
# wrapper (extra batch index + strides), validated by the shared GEMM gate.
def _sample_batched_gemm(rng):
    from rocke.instances.common.batched_gemm import BatchedGemmSpec
    from rocke.instances.common.gemm_universal import TileSpec, TraitSpec

    dtype = rng.choice(["fp16", "bf16"])
    atom = rng.choice(_GEMM_ATOMS_F16 if dtype == "fp16" else _GEMM_ATOMS_BF16)
    wtm, wtn, wtk = atom
    warp_m, warp_n = rng.choice(_GEMM_WARP_GRID)
    mfma_m = rng.choice(_GEMM_MFMA_MN)
    mfma_n = rng.choice(_GEMM_MFMA_MN)
    kcount = rng.choice(_GEMM_KCOUNT)
    tile_m = warp_m * wtm * mfma_m
    tile_n = warp_n * wtn * mfma_n
    tile_k = wtk * kcount
    pipeline = rng.choice(_GEMM_PIPELINES)
    epilogue = rng.choice(_GEMM_EPILOGUES)

    spec = BatchedGemmSpec(
        name="fuzz_bgemm",
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=wtm,
            warp_tile_n=wtn,
            warp_tile_k=wtk,
        ),
        trait=TraitSpec(pipeline=pipeline, epilogue=epilogue),
        wave_size=64,
        block_size=warp_m * warp_n * 64,
        dtype=dtype,
    )
    rep = (
        f"batched_gemm dtype={dtype} tile={tile_m}x{tile_n}x{tile_k} "
        f"warp={warp_m}x{warp_n}x1 wt={wtm}x{wtn}x{wtk} "
        f"pipeline={pipeline} epilogue={epilogue}"
    )
    return spec, rep, "gfx950"


def _batched_gemm_valid(spec):
    from rocke.instances.common.batched_gemm import is_valid_spec

    return is_valid_spec(spec, arch="gfx950")


def _batched_gemm_build(spec):
    from rocke.instances.common.batched_gemm import build_batched_gemm

    return build_batched_gemm(spec, arch="gfx950")


# Elementwise: op x dtype x block_size x vec, including a couple of deliberately
# out-of-range draws so the validity gate gets exercised too.
_EW_UNARY = [
    "copy",
    "neg",
    "abs",
    "relu",
    "gelu_tanh",
    "quick_gelu",
    "silu",
    "swish",
    "tanh",
    "sigmoid",
    "exp2",
]
_EW_BINARY = ["add", "sub", "mul", "max", "min", "swiglu", "geglu"]
_EW_BLOCK = [64, 128, 256, 512, 1024, 96]  # 96 is invalid -> gate skip
_EW_VEC = [2, 4, 8, 3]  # 3 is invalid -> gate skip


def _sample_elementwise(rng):
    from rocke.instances.common.elementwise import ElementwiseSpec

    op = rng.choice(_EW_UNARY + _EW_BINARY)
    dtype = rng.choice(["f16", "bf16"])
    block_size = rng.choice(_EW_BLOCK)
    vec = rng.choice(_EW_VEC)
    spec = ElementwiseSpec(op=op, dtype=dtype, block_size=block_size, vec=vec)
    rep = f"elementwise op={op} dtype={dtype} block_size={block_size} vec={vec}"
    return spec, rep, "gfx950"


def _elementwise_valid(spec):
    from rocke.instances.common.elementwise import is_valid_spec

    return is_valid_spec(spec)


def _elementwise_build(spec):
    from rocke.instances.common.elementwise import build_elementwise

    return build_elementwise(spec)


# Reduce (2D row reduction): op x dtype x block_size x vec x n_per_block. The
# gate requires n_per_block divisible by block_size*vec, so most random
# n_per_block draws are intentionally pruned (exercises the gate heavily).
_RED_OPS = ["sum", "max", "min", "mean", "prod"]


def _sample_reduce(rng):
    from rocke.instances.common.reduce import Reduce2DSpec

    op = rng.choice(_RED_OPS)
    dtype = rng.choice(["f16", "bf16"])
    block_size = rng.choice([64, 128, 256, 512, 1024])
    vec = rng.choice([2, 4, 8])
    # Bias toward n_per_block that are multiples of block_size*vec so a useful
    # fraction passes the gate, while still sampling some that do not.
    chunk = block_size * vec
    mult = rng.choice([1, 2, 3, 4])
    n_per_block = chunk * mult if rng.random() < 0.8 else rng.choice([1000, 4096])
    spec = Reduce2DSpec(
        n_per_block=n_per_block,
        op=op,
        block_size=block_size,
        vec=vec,
        dtype=dtype,
    )
    rep = (
        f"reduce op={op} dtype={dtype} block_size={block_size} "
        f"vec={vec} n_per_block={n_per_block}"
    )
    return spec, rep, "gfx950"


def _reduce_valid(spec):
    from rocke.instances.common.reduce import is_valid_spec

    return is_valid_spec(spec)


def _reduce_build(spec):
    from rocke.instances.common.reduce import build_reduce2d

    return build_reduce2d(spec)


# Family registry. name -> (sampler, validity_gate, builder).
FAMILIES = {
    "gemm": (_sample_gemm, _gemm_valid, _gemm_build),
    "batched_gemm": (_sample_batched_gemm, _batched_gemm_valid, _batched_gemm_build),
    "elementwise": (_sample_elementwise, _elementwise_valid, _elementwise_build),
    "reduce": (_sample_reduce, _reduce_valid, _reduce_build),
}


# ---------------------------------------------------------------------------
# The oracle: serialize -> Python-lower (ref) vs C-CLI-lower (artifact).
# ---------------------------------------------------------------------------
def run_cli(cli, arch, ir_text):
    try:
        p = subprocess.run(
            [str(cli), arch],
            input=ir_text.encode("utf-8"),
            capture_output=True,
            timeout=TIMEOUT,
        )
        return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, b"", "TIMEOUT"


def first_diff(ref, art):
    """Short description of where two .ll texts first diverge."""
    ref_lines = ref.decode("utf-8", "replace").splitlines()
    art_lines = art.decode("utf-8", "replace").splitlines()
    n = min(len(ref_lines), len(art_lines))
    for i in range(n):
        if ref_lines[i] != art_lines[i]:
            return "line %d:\n      ref %r\n      art %r" % (
                i + 1,
                ref_lines[i][:90],
                art_lines[i][:90],
            )
    if len(ref_lines) != len(art_lines):
        return "len differs: ref %d lines, art %d lines (%d vs %d bytes)" % (
            len(ref_lines),
            len(art_lines),
            len(ref),
            len(art),
        )
    return "byte-only diff (%d vs %d bytes)" % (len(ref), len(art))


def diff_one(cli, build_fn, spec, arch, lower, serialize):
    """Build IR for one valid spec, serialize, lower both ways, byte-compare.

    Returns ("IDENTICAL"|"MISMATCH"|"ERROR", note). The Python build/serialize/
    lower happens in-process; only the C lowering shells out to the CLI."""
    try:
        kernel = build_fn(spec)
    except Exception as e:  # noqa: BLE001
        return "ERROR", f"python build raised: {type(e).__name__}: {e}"
    try:
        ir_text = serialize(kernel)
    except Exception as e:  # noqa: BLE001
        return "ERROR", f"python serialize raised: {type(e).__name__}: {e}"
    try:
        ll_ref = lower(kernel, arch=arch).encode("utf-8")
    except Exception as e:  # noqa: BLE001
        return "ERROR", f"python lower raised: {type(e).__name__}: {e}"

    rc, ll_art, err = run_cli(cli, arch, ir_text)
    if not ll_art:
        last = (err.strip().splitlines() or ["(no stderr)"])[-1]
        return "ERROR", f"C parse/lower produced no output (rc={rc}): {last}"
    if ll_art == ll_ref:
        return "IDENTICAL", ""
    return "MISMATCH", first_diff(ll_ref, ll_art)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def fuzz_family(cli, name, count, base_seed, lower, serialize, verbose):
    """Generate `count` random specs for one family at a deterministic per-family
    seed (base_seed + a stable family offset) so each family is independently
    reproducible. Returns a stats dict and the list of mismatches found."""
    sampler, gate, build_fn = FAMILIES[name]
    # Stable, distinct stream per family so --only does not change the specs a
    # given family sees, and the same seed reproduces byte-for-byte across runs.
    # The per-family offset is a STABLE digest of the family name (Python's
    # built-in hash() is salted per process and would break reproducibility).
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") & 0xFFFF
    rng = random.Random(base_seed + offset)

    stats = {"generated": 0, "invalid": 0, "identical": 0, "mismatch": 0, "error": 0}
    mismatches = []
    errors = []
    for _ in range(count):
        spec, rep, arch = sampler(rng)
        stats["generated"] += 1
        try:
            ok, _reason = gate(spec)
        except Exception as e:  # noqa: BLE001
            # A gate that raises on a spec it should have cleanly rejected is
            # itself a finding; surface it as an error rather than a skip.
            stats["error"] += 1
            errors.append((rep, f"validity gate raised: {type(e).__name__}: {e}"))
            continue
        if not ok:
            stats["invalid"] += 1
            continue
        verdict, note = diff_one(cli, build_fn, spec, arch, lower, serialize)
        if verdict == "IDENTICAL":
            stats["identical"] += 1
            if verbose:
                print(f"    IDENTICAL  {rep}")
        elif verdict == "MISMATCH":
            stats["mismatch"] += 1
            mismatches.append((rep, note))
            print(f"    MISMATCH   {rep}\n      {note}")
        else:
            stats["error"] += 1
            errors.append((rep, note))
            print(f"    ERROR      {rep}\n      {note}")
    return stats, mismatches, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (reproducible; default {DEFAULT_SEED})",
    )
    ap.add_argument(
        "--count",
        type=int,
        default=200,
        help="random specs to generate PER family (default 200)",
    )
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated family-name substrings to include "
        f"(families: {', '.join(FAMILIES)})",
    )
    ap.add_argument(
        "--cli",
        default=DEFAULT_CLI,
        help="path to the compiled ir_lower_cli tool",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="print every IDENTICAL spec too (default prints only mismatch/error)",
    )
    args = ap.parse_args()

    cli = Path(args.cli)
    if not cli.exists():
        sys.exit(
            f"ir_lower_cli not found: {cli}\n"
            "build it: cmake -S <rocKE> -B /tmp/rocke_irart -DCMAKE_BUILD_TYPE="
            "Release && cmake --build /tmp/rocke_irart --target rocke_core -j && "
            "c++ -std=c++20 -I <rocKE>/Cpp/include <rocKE>/tests/core/"
            "ir_lower_cli.cpp /tmp/rocke_irart/librocke_core.a -lm "
            "-o /tmp/rocke_irart/ir_lower_cli\n"
            "or pass --cli <path>"
        )

    lower, serialize = _ck_imports()

    fams = list(FAMILIES)
    if args.only:
        subs = [s for s in args.only.split(",") if s]
        fams = [f for f in fams if any(s in f for s in subs)]
        if not fams:
            sys.exit(f"--only matched no families (have: {', '.join(FAMILIES)})")

    print(f"random-spec differential fuzz  seed={args.seed}  count/family={args.count}")
    print("oracle: Python build IR -> serialize -> C-lower (CLI) vs Python-lower")
    print(f"cli={cli}  families={fams}")
    print("-" * 78)

    total = {"generated": 0, "invalid": 0, "identical": 0, "mismatch": 0, "error": 0}
    all_mismatch = []
    all_error = []
    for name in fams:
        print(f"[{name}]")
        stats, mism, errs = fuzz_family(
            cli, name, args.count, args.seed, lower, serialize, args.verbose
        )
        for k in total:
            total[k] += stats[k]
        all_mismatch += [(name, *m) for m in mism]
        all_error += [(name, *e) for e in errs]
        print(
            f"  generated={stats['generated']} invalid-skipped={stats['invalid']} "
            f"identical={stats['identical']} mismatch={stats['mismatch']} "
            f"error={stats['error']}"
        )

    print("\n" + "=" * 78)
    print("SUMMARY -- random-spec differential fuzz")
    print("=" * 78)
    print(f"  total generated       : {total['generated']}")
    print(f"  invalid-skipped       : {total['invalid']}")
    print(f"  buildable & diffed    : {total['identical'] + total['mismatch']}")
    print(f"  IDENTICAL             : {total['identical']}")
    print(f"  MISMATCH              : {total['mismatch']}")
    print(f"  ERROR                 : {total['error']}")
    if all_mismatch:
        print("\n  MISMATCHES (real divergence -- reproducible with the seed above):")
        for fam, rep, note in all_mismatch:
            print(f"    [{fam}] {rep}\n      {note}")
    if all_error:
        print("\n  ERRORS (artifact path could not be exercised):")
        for fam, rep, note in all_error:
            print(f"    [{fam}] {rep}\n      {note}")

    # Real mismatches fail the run. ERRORs (e.g. a build/parse path gap) are also
    # surfaced as a nonzero exit -- they mean the oracle could not be applied,
    # which is itself a regression worth catching.
    return 1 if (all_mismatch or all_error) else 0


if __name__ == "__main__":
    raise SystemExit(main())
