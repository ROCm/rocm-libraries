#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Turn a rocKE attention KDP's kernel geometries into integration-test bundles.

A UHD is only worth training when several kernels compete for the same problem.
rocKE kernels bake their geometry in, so the kernel matcher pins dtype, head
counts, sequence lengths, head size, batch and causality before ranking starts --
what is left to rank is the free knobs, and only kernels sharing one geometry ever
compete.

That makes the KDP itself the authority on which problems are worth measuring: a
geometry with one kernel has nothing to rank, and a geometry the pack does not
carry cannot be served at all. This reads the KDP, keeps the geometries with
enough competitors to be informative, and writes one bundle per geometry.

This exists because `corpus_gen` emits `commands.txt` for a `hipdnn_bench` that
this repository does not build. When that lands, corpus_gen's problem set should
replace this script -- it derives problems from an operation's declared space
rather than from whatever kernels happen to be packed, which is the more general
question. Until then a corpus of packed geometries is the one that can actually
be measured.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

GEOMETRY = ("dtype", "head_size", "num_query_heads", "num_kv_heads",
            "seqlen_q", "seqlen_kv", "batch", "causal")

# The KMD spells dtypes the rocKE way; a bundle spells them the data SDK's way.
# The graph JSON spellings, which are the FlatBuffer DataType enum's own names
# (flatbuffers_sdk/utilities/json/Common.hpp), not the KMD's and not the encoder's.
# The asymmetry is real: BFLOAT16 is written out, HALF is not "float16". Spelling it
# `float16` did not fail conversion -- it produced a graph the backend then refused at
# from_binary, so every fp16 case failed after the bundle looked fine on disk.
DTYPE_TO_BUNDLE = {"BF16": "bfloat16", "FP16": "half"}
DTYPE_TO_DIR = {"BF16": "bf16", "FP16": "fp16"}
BYTES_PER_ELEMENT = {"BF16": 2, "FP16": 2}


def bshd_strides(batch: int, heads: int, seq: int, head_size: int) -> list[int]:
    """Strides for a BSHD tensor whose dims are declared [B, H, S, D].

    Read off a shipped bundle rather than derived from a layout enum: dims are
    B,H,S,D but memory is seq-major, so the head stride is D and the seq stride is
    H*D. Getting these two the wrong way round produces a graph the matcher still
    accepts and the kernel reads transposed.
    """
    return [seq * heads * head_size, head_size, heads * head_size, 1]


def tensor(uid: int, name: str, batch: int, heads: int, seq: int,
           head_size: int, dtype: str) -> dict:
    return {
        "uid": uid,
        "name": name,
        "dims": [batch, heads, seq, head_size],
        "strides": bshd_strides(batch, heads, seq, head_size),
        "data_type": DTYPE_TO_BUNDLE[dtype],
        "virtual": False,
    }


def bundle_for(geom: dict) -> dict:
    """One SdpaFwd bundle, shaped like the shipped bshd ones."""
    dt = geom["dtype"]
    b, hq, hkv = geom["batch"], geom["num_query_heads"], geom["num_kv_heads"]
    sq, skv, d = geom["seqlen_q"], geom["seqlen_kv"], geom["head_size"]

    attributes = {
        "generate_stats": None,
        # Present-but-null, as the shipped bundles carry them: the reader distinguishes
        # an absent key from an explicit null, and a missing one reads as unset rather
        # than as declined.
        "dropout_probability": None,
        "max_seq_len_kv": None,
        # The attention window. Present-but-null for the same reason as the two above,
        # and it is not cosmetic here: omitting them made `json::to<Graph>` reject the
        # bundle, so every nomask case was logged as INVALID_GRAPH_SCHEMA and silently
        # not registered -- no test, no failure, just 165 problems missing from a sweep
        # that looked like it had run everything.
        "left_bound": None,
        "right_bound": None,
        "alibi_mask": False,
        "padding_mask": False,
        "causal_mask": False,
        "causal_mask_bottom_right": False,
        "attn_scale_value": 1.0 / math.sqrt(d),
        "diagonal_alignment": "TOP_LEFT",
        "mma_core_mode": "float",
        "implementation": "AUTO",
    }
    if geom["causal"]:
        # Causality is expressed as a window, not the causal_mask flag: the shipped
        # causal bundles carry left_bound/right_bound and leave causal_mask False.
        attributes["left_bound"] = -1
        attributes["right_bound"] = 0

    inputs = {k: None for k in (
        "attn_mask_tensor_uid", "scale_tensor_uid", "seq_len_q_tensor_uid",
        "seq_len_kv_tensor_uid", "seed_tensor_uid", "offset_tensor_uid",
        "dropout_mask_tensor_uid", "dropout_scale_tensor_uid",
        "page_table_k_tensor_uid", "page_table_v_tensor_uid",
        "block_mask_tensor_uid", "sink_token_tensor_uid", "descale_q_tensor_uid",
        "descale_k_tensor_uid", "descale_v_tensor_uid", "descale_s_tensor_uid",
        "scale_s_tensor_uid", "scale_o_tensor_uid")}
    inputs.update({"q_tensor_uid": 0, "k_tensor_uid": 1, "v_tensor_uid": 2})
    outputs = {k: None for k in (
        "stats_tensor_uid", "max_tensor_uid", "sum_exp_tensor_uid",
        "rng_dump_tensor_uid", "amax_s_tensor_uid", "amax_o_tensor_uid")}
    outputs["o_tensor_uid"] = 3

    return {
        "nodes": [{"type": "SdpaAttributes", "compute_data_type": "float",
                   "name": "", "inputs": inputs, "outputs": outputs,
                   "attributes": attributes}],
        "tensors": [
            tensor(0, "Q", b, hq, sq, d, dt),
            tensor(1, "K", b, hkv, skv, d, dt),
            tensor(2, "V", b, hkv, skv, d, dt),
            tensor(3, "O", b, hq, sq, d, dt),
        ],
        "io_data_type": DTYPE_TO_BUNDLE[dt],
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "name": "",
    }


def footprint_bytes(geom: dict) -> int:
    w = BYTES_PER_ELEMENT[geom["dtype"]]
    b, hq, hkv = geom["batch"], geom["num_query_heads"], geom["num_kv_heads"]
    sq, skv, d = geom["seqlen_q"], geom["seqlen_kv"], geom["head_size"]
    return w * (2 * b * hq * sq * d + 2 * b * hkv * skv * d)


def stratified(geoms: list[dict], limit: int) -> list[dict]:
    """Spread the pick across dtype/head_size/causal/GQA rather than taking the
    largest, because a corpus concentrated on one corner trains a model that is
    excellent there and useless elsewhere -- the shape RFC 0019.13 §11.2 wants the
    per-regime table to expose."""
    buckets: dict = collections.defaultdict(list)
    for g in geoms:
        gqa = "mha" if g["num_query_heads"] == g["num_kv_heads"] else "gqa"
        buckets[(g["dtype"], g["head_size"], g["causal"], gqa)].append(g)
    for v in buckets.values():
        v.sort(key=footprint_bytes)
    picked, keys = [], sorted(buckets)
    i = 0
    while len(picked) < limit and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            picked.append(buckets[k].pop(0))
        i += 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kdp", required=True, help="gfx942_attention_dense.kdp.json")
    ap.add_argument("--out", required=True, help="integration-test-bundles root")
    ap.add_argument("--tier", default="generated", help="bundle tier directory")
    ap.add_argument("--min-candidates", type=int, default=3,
                    help="skip geometries with fewer competing kernels (default 3)")
    ap.add_argument("--max-bytes", type=int, default=2 * 1024**3,
                    help="skip geometries whose tensors exceed this (default 2 GiB)")
    ap.add_argument("--limit", type=int, default=120, help="bundles to emit")
    args = ap.parse_args()

    kernels = json.load(open(args.kdp))["kernelDescriptors"]
    counts: collections.Counter = collections.Counter()
    for k in kernels:
        m = k.get("metadata", {})
        counts[tuple(m.get(g) for g in GEOMETRY)] += 1

    eligible, skipped_small, skipped_big = [], 0, 0
    for tup, n in counts.items():
        geom = dict(zip(GEOMETRY, tup))
        if any(v is None for v in tup) or geom["dtype"] not in DTYPE_TO_BUNDLE:
            continue
        if n < args.min_candidates:
            skipped_small += 1
            continue
        if footprint_bytes(geom) > args.max_bytes:
            skipped_big += 1
            continue
        geom["_candidates"] = n
        eligible.append(geom)

    picked = stratified(eligible, args.limit)
    root = Path(args.out) / args.tier / "SdpaFwd" / "bshd"
    for g in picked:
        name = (f"hd{g['head_size']}_b{g['batch']}_hq{g['num_query_heads']}"
                f"_hkv{g['num_kv_heads']}_sq{g['seqlen_q']}_skv{g['seqlen_kv']}"
                f"_{'causal' if g['causal'] else 'nomask'}")
        d = root / DTYPE_TO_DIR[g["dtype"]] / name / "Gen"
        d.mkdir(parents=True, exist_ok=True)
        (d / "Gen.json").write_text(json.dumps(bundle_for(g), indent=4) + "\n")

    total_candidates = sum(g["_candidates"] for g in picked)
    print(f"kernels                 {len(kernels)}")
    print(f"distinct geometries     {len(counts)}")
    print(f"  < {args.min_candidates} candidates       {skipped_small} skipped (nothing to rank)")
    print(f"  over byte budget      {skipped_big} skipped")
    print(f"eligible                {len(eligible)}")
    print(f"bundles written         {len(picked)}  -> {root}")
    print(f"expected sweep records  {total_candidates} across {len(picked)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
