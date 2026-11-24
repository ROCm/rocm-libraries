#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adds a `layout` parameter to the SDPA corpus declarations, so a corpus stratifies over it.

The shipped declarations bake BSHD strides (token-major) into every Q/K/V/O tensor, which keeps
rocKE's dense kernels (BSHD only) in a live-engine corpus. An evaluation of engine selection
needs the other layouts callers hand hipDNN too, because they change which engines compete
(rocKE declines all but BSHD) and how fast the stride-taking engines run:

  bshd        [B, S, H, D] memory: FlashAttention, AITER, vLLM          (the shipped default)
  bhsd        [B, H, S, D] memory: PyTorch eager / Hugging Face after .contiguous()
  sbhd        [S, B, H, D] memory: Megatron
  qkv_packed  Q, K and V are views into one fused [B, S, Hq + 2 Hkv, D] projection (self-attention
              only, so seqlen_q == seqlen_k); O and the gradients of O stay BSHD

Dims stay (B, H, S, D) in every layout; only the strides change. `ragged` keeps BSHD: hipDNN's
ragged addressing needs token-major packing. The layout is a facet column of its own
(form_label axis `layout`), so `--form-floor` counts every layout.

usage: add_sdpa_layouts.py <operations dir> <out dir>   (writes sdpa_fwd/sdpa_bwd.opmeta.json)
"""
import json
import sys
from pathlib import Path

LAYOUTS = ["bshd", "bhsd", "sbhd", "qkv_packed"]
PACKED_OPERANDS = {"q", "k", "v", "dq", "dk", "dv"}
# operand -> (heads, seqlen)
OPERANDS = {
    "q": ("$q.heads", "$q.seqlen_q"),
    "o": ("$q.heads", "$q.seqlen_q"),
    "dO": ("$q.heads", "$q.seqlen_q"),
    "dq": ("$q.heads", "$q.seqlen_q"),
    "k": ("$q.heads_kv", "$q.seqlen_k"),
    "v": ("$q.heads_kv", "$q.seqlen_k"),
    "dk": ("$q.heads_kv", "$q.seqlen_k"),
    "dv": ("$q.heads_kv", "$q.seqlen_k"),
}
D = "$q.head_dim"


def mul(*terms):
    out = terms[0]
    for term in terms[1:]:
        out = {"*": [out, term]}
    return out


def strides(operand):
    """[batch, head, seq, dim] element strides of @p operand for each layout, as one expression."""
    heads, seq = OPERANDS[operand]
    by_layout = {
        "bshd": [mul(seq, heads, D), D, mul(heads, D), 1],
        "bhsd": [mul(heads, seq, D), mul(seq, D), D, 1],
        "sbhd": [mul(heads, D), D, mul("$q.batch", heads, D), 1],
    }
    if operand in PACKED_OPERANDS:
        row = mul({"+": ["$q.heads", mul(2, "$q.heads_kv")]}, D)
        by_layout["qkv_packed"] = [mul(seq, row), D, row, 1]
    else:
        by_layout["qkv_packed"] = by_layout["bshd"]
    # One nested `if` per axis, falling through to bshd.
    axes = []
    for axis in range(4):
        expr = by_layout["bshd"][axis]
        for layout in ("qkv_packed", "sbhd", "bhsd"):
            value = by_layout[layout][axis]
            if value != expr:
                expr = {"if": [{"==": ["$q.layout", layout]}, value, expr]}
        axes.append(expr)
    return axes


def transform(document):
    document["parameters"]["layout"] = {
        "type": "enum",
        "values": LAYOUTS,
        "description": (
            "Memory layout of Q/K/V/O (and their gradients) behind the logical [B, H, S, D] dims: "
            "bshd token-major (FlashAttention, AITER, vLLM), bhsd head-major (PyTorch eager), sbhd "
            "sequence-major (Megatron), qkv_packed Q/K/V views of one fused QKV projection "
            "(self-attention; O stays bshd). Engines that take no stride arguments (rocKE dense) "
            "serve bshd only."
        ),
    }
    document["constraints"] += [
        {"or": [{"!=": ["$q.form", "ragged"]}, {"==": ["$q.layout", "bshd"]}]},
        {"or": [{"!=": ["$q.layout", "qkv_packed"]}, {"==": ["$q.seqlen_q", "$q.seqlen_k"]}]},
    ]
    document["form_label"].append(
        {
            "name": "layout",
            "labels": [
                {"label": layout, "when": {"==": ["$q.layout", layout]}}
                for layout in LAYOUTS[1:]
            ],
            "otherwise": "bshd",
        }
    )
    builder = document["graph_builder"]
    builder["// layout"] = (
        "Strides follow $q.layout (see the parameter); dims stay (B, H, S, D). bshd is the shipped "
        "token-major layout; ragged graphs are bshd only."
    )
    for argument in builder["arguments"]:
        name = argument.get("name", "")
        if name.endswith("Strides") and name[: -len("Strides")] in OPERANDS:
            assert argument.get("kind") == "expr", name
            argument["value"] = strides(name[: -len("Strides")])
    return document


def main(source, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for operation in ("sdpa_fwd", "sdpa_bwd"):
        document = json.loads((Path(source) / f"{operation}.opmeta.json").read_text())
        (target / f"{operation}.opmeta.json").write_text(
            json.dumps(transform(document), indent=2) + "\n"
        )
        print(f"{operation}: layout axis added -> {target / f'{operation}.opmeta.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
