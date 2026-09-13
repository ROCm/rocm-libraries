#!/usr/bin/env python3
"""Build a RANKABLE flyDSL attention catalog: every functional tuple, every scheduling knob.

The shipped pack has one kernel per functional tuple (dtype, head_size, num_query_heads,
num_kv_heads, causal) -- which is exactly what ``kernel_match`` pins (FlydslAttentionNative
.cpp:564), so for any graph exactly one kernel is applicable and a catalog ranker has
nothing to rank. L2 collection says so in as many words: "held-out corpus has no evaluable
candidate ranking".

Anything ``kernel_match`` does NOT pin is a free knob: same functional tuple, several
kernels, all applicable, enumeration hands them to L2 as candidates. ``build_flash_attn_real
.py`` already names the three that qualify -- "Tuning knobs that DON'T change the kernarg
ABI (pure codegen/occupancy/scheduling): each distinct value yields a functionally-identical
HSACO with different perf, which is how we generate OVERLAPPING instances of one functional
tuple for the autotune experiment". ``num_kv_splits`` stays out for the reason given there:
it adds a reduction workspace and changes the launch protocol.

Writes the HSACOs, a regenerated ``.kdp.json`` whose metadata declares the knobs as INTEGER
fields (the collection UED exposes every int KMD field, so that is what makes them
addressable), and the matching ``.kmd.json``.

    python build_flydsl_variants.py --out <dir> [--waves 1 2 4] [--stagger 1 0]
                                    [--lazy 1 0] [--setprio 1]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRATCH = _HERE.parent
sys.path.insert(0, str(_SCRATCH))

from flydsl_build.extract_hsaco import extract_bin_attr  # noqa: E402

#: The pack whose functional tuples are reproduced. Its kernels are the baseline of the
#: grid: knobs at their shipped values must reproduce the shipped binaries.
BASE_KDP = _SCRATCH / "flydsl_descriptors_attention" / "flydsl_attention" / "flydsl_attention.kdp.json"

#: Stable per-variant identity: a UUID5 over the variant's own name, so rebuilding the same
#: grid yields the same descriptor ids and a corpus collected against one build still joins
#: against the next. A minted UUID would make every rebuild a different catalog.
_NAMESPACE = uuid.UUID("f1d5b1a7-0000-4a00-9000-000000000000")

DTYPE_ARG = {"BF16": "bf16", "FP16": "f16"}


def variant_name(meta: dict) -> str:
    """`flash_attn_real_h<H>[kv<N>]_d<D>_<mask>_<dtype>[_w<N>][_nostag][_norescale]_gfx950.hsaco`

    The shipped naming, extended once per knob that is off its default. Keeping the default
    spelling identical means the 21 shipped names are a subset of this grid's names.
    """
    heads, kv = int(meta["num_query_heads"]), int(meta["num_kv_heads"])
    code = f"h{heads}" if heads == kv else f"h{heads}kv{kv}"
    mask = "causal" if int(meta["causal"]) else "noncausal"
    knob = ""
    if int(meta["waves_per_eu"]) != 2:
        knob += f"_w{int(meta['waves_per_eu'])}"
    if not int(meta["stagger"]):
        knob += "_nostag"
    if not int(meta["lazy_rescale"]):
        knob += "_norescale"
    if not int(meta["setprio"]):
        knob += "_noprio"
    return (f"flash_attn_real_{code}_d{int(meta['head_size'])}_{mask}_"
            f"{DTYPE_ARG[meta['dtype']]}{knob}_gfx950.hsaco")


def functional_tuples(kdp: dict) -> list[dict]:
    """One entry per (dtype, head_size, num_query_heads, num_kv_heads, causal) in the pack."""
    seen: dict[tuple, dict] = {}
    for kernel in kdp["kernelDescriptors"]:
        meta = kernel["metadata"]
        key = (meta["dtype"], int(meta["head_size"]), int(meta["num_query_heads"]),
               int(meta["num_kv_heads"]), int(meta["causal"]))
        seen.setdefault(key, {"dtype": meta["dtype"], "head_size": int(meta["head_size"]),
                              "num_query_heads": int(meta["num_query_heads"]),
                              "num_kv_heads": int(meta["num_kv_heads"]),
                              "causal": int(meta["causal"]),
                              "symbol": meta["symbol"],
                              "block_threads": int(meta["block_threads"]),
                              "block_m": int(meta["block_m"])})
    return list(seen.values())


def build_one(base: dict, waves: int, stagger: int, lazy: int, setprio: int, out_dir: Path) -> dict:
    """Compile one variant and return its kernel-descriptor metadata."""
    import torch
    from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module

    dump_dir = Path(tempfile.mkdtemp(prefix="flydsl-variant-"))
    os.environ["FLYDSL_DUMP_IR"] = "1"
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)
    os.environ["COMPILE_ONLY"] = "1"
    os.environ.setdefault("ARCH", "gfx950")

    dtype_str = DTYPE_ARG[base["dtype"]]
    launch = build_flash_attn_dualwave_swp_module(
        num_heads=base["num_query_heads"],
        head_dim=base["head_size"],
        causal=bool(base["causal"]),
        dtype_str=dtype_str,
        num_kv_heads=base["num_kv_heads"],
        waves_per_eu=waves,
        dualwave_swp_enable_stagger=bool(stagger),
        dualwave_swp_lazy_rescale=bool(lazy),
        dualwave_swp_setprio=bool(setprio),
    )
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype_str]
    batch, seqlen = 1, 256  # runtime scalars; any legal prefill shape triggers the compile
    q = torch.zeros((batch, seqlen, base["num_query_heads"], base["head_size"]), dtype=torch_dtype)
    k = torch.zeros((batch, seqlen, base["num_kv_heads"], base["head_size"]), dtype=torch_dtype)
    v = torch.zeros((batch, seqlen, base["num_kv_heads"], base["head_size"]), dtype=torch_dtype)
    o = torch.empty((batch, seqlen, base["num_query_heads"], base["head_size"]), dtype=torch_dtype)
    launch(q, k, v, o, batch, seqlen)

    stage19 = next(dump_dir.glob("*/19_gpu_module_to_binary.mlir"))
    blob = extract_bin_attr(stage19.read_text(encoding="utf-8", errors="surrogateescape"))
    if blob[:4] != b"\x7fELF":
        raise SystemExit(f"not ELF: {blob[:4]!r}")

    meta = dict(base)
    meta.update({"waves_per_eu": waves, "stagger": stagger,
                 "lazy_rescale": lazy, "setprio": setprio})
    meta["hsaco"] = variant_name(meta)
    (out_dir / meta["hsaco"]).write_bytes(blob)
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path, help="directory for HSACOs + descriptors")
    parser.add_argument("--waves", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--stagger", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--lazy", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--setprio", type=int, nargs="+", default=[1])
    parser.add_argument("--limit-tuples", type=int, default=0,
                        help="build only the first N functional tuples (smoke runs)")
    args = parser.parse_args(argv)

    base_kdp = json.loads(BASE_KDP.read_text(encoding="utf-8"))
    tuples = functional_tuples(base_kdp)
    if args.limit_tuples:
        tuples = tuples[: args.limit_tuples]
    grid = list(itertools.product(args.waves, args.stagger, args.lazy, args.setprio))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tuples)} functional tuples x {len(grid)} knob settings = "
          f"{len(tuples) * len(grid)} kernels")

    kernels, failures = [], []
    for base in tuples:
        for waves, stagger, lazy, setprio in grid:
            try:
                meta = build_one(base, waves, stagger, lazy, setprio, args.out)
            except Exception as error:  # noqa: BLE001 - one variant must not end the grid
                failures.append((base, (waves, stagger, lazy, setprio), str(error)[:200]))
                print(f"  FAILED {base['dtype']} h{base['num_query_heads']}kv{base['num_kv_heads']} "
                      f"w{waves} s{stagger} l{lazy} p{setprio}: {str(error)[:120]}")
                continue
            kernels.append({
                "version": "1.0",
                "id": str(uuid.uuid5(_NAMESPACE, meta["hsaco"])),
                "name": meta["hsaco"].removesuffix("_gfx950.hsaco"),
                "kernel_source": {"kind": "embedded_source", "source_file": "FlydslAttentionNative.cpp",
                                  "entry_point": meta["symbol"]},
                "metadata": meta,
                "priority": 0,
            })
            print(f"  {meta['hsaco']}")

    if not kernels:
        print("FATAL: no kernel built")
        return 1

    pack = dict(base_kdp)
    pack["kernelDescriptors"] = kernels
    (args.out / "flydsl_attention.kdp.json").write_text(json.dumps(pack, indent=2) + "\n",
                                                        encoding="utf-8")
    # The knobs must be INT fields: `uhd_gen generate` exposes exactly the int KMD fields as
    # the collection UED's knobs, which is what makes a candidate addressable at all.
    kmd_path = BASE_KDP.parent / "flydsl_attention.kmd.json"
    kmd = json.loads(kmd_path.read_text(encoding="utf-8"))
    declared = {field["name"] for field in kmd["fields"]}
    for knob in ("waves_per_eu", "stagger", "lazy_rescale", "setprio"):
        if knob not in declared:
            kmd["fields"].append({"name": knob, "type": "int"})
    (args.out / "flydsl_attention.kmd.json").write_text(json.dumps(kmd, indent=2) + "\n",
                                                        encoding="utf-8")
    print(f"\nwrote {len(kernels)} kernels, {len(failures)} failed -> {args.out}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
