#!/usr/bin/env python3
"""Emit family.json for the gfx950 activation and layernorm2d families.

Reproduces rocKE's ElementwiseSpec / LayerNorm2D{,Dynamic}Spec kernel_name(), which
are arch-independent. Plain stdlib so it runs before the rocKE pyenv exists.
"""

import json
import os
import sys

ARCH = "gfx950"

# ---------------- activation (pointwise) ----------------
ACT_OPS = ["silu", "gelu_tanh"]
ACT_DTYPES = ["f16", "bf16"]
ACT_TUNING = [(256, 8), (512, 8), (256, 4)]
ACT_ARGS = [
    {"name": "A", "type": "ptr"},
    {"name": "C", "type": "ptr"},
    {"name": "N", "type": "i32"},
]

# ---------------- layernorm2d ----------------
LN_CONFIGS = [
    (2048, 256, 4), (2048, 512, 4), (2048, 128, 8), (2048, 64, 8),
    (1024, 256, 4), (1024, 128, 8), (4096, 512, 4), (4096, 256, 8),
]
LN_DYNAMIC = [(256, 4), (128, 8)]
LN_ARGS = [
    {"name": "X", "type": "ptr"},
    {"name": "Gamma", "type": "ptr"},
    {"name": "Beta", "type": "ptr"},
    {"name": "Y", "type": "ptr"},
    {"name": "M", "type": "i32"},
    {"name": "N", "type": "i32"},
    {"name": "eps", "type": "f32"},
]


def activation_doc():
    kernels = []
    for op in ACT_OPS:
        for dtype in ACT_DTYPES:
            for bs, vec in ACT_TUNING:
                sym = f"rocke_elementwise_{op}_{dtype}_b{bs}_v{vec}"
                kernels.append({
                    "symbol": sym,
                    "co_file": sym + ".co",
                    "constraints": {
                        "dtype": {"equals": dtype},
                        "activation": {"equals": op},
                        # Flat 1-D problem; the grid covers a ragged tail, so the only
                        # requirement is a non-degenerate extent.
                        "numel": {"min": 1},
                    },
                    "grid": {"x": {"ceil_div": ["numel", bs * vec]}, "y": 1, "z": 1},
                    "block": [bs, 1, 1],
                    "args_signature": ACT_ARGS,
                    "workspace_bytes": 0,
                })
    return {
        "family": f"activation_{ARCH}",
        "op_kind": "pointwise",
        "arch": ARCH,
        "dtype": ["f16", "bf16"],
        "_comment": (
            "gfx950 port of the gfx1151 activation family. ElementwiseSpec has no "
            "wave_size field (per-element f32 math with a scalar tail), so this is the "
            "arch string and nothing else. silu maps to PointwiseMode SWISH_FWD "
            "(beta==1) and gelu_tanh to GELU_APPROX_TANH_FWD; exact-erf gelu has no "
            "builder and is left to fall back to native."
        ),
        "kernels": kernels,
    }


def layernorm_doc():
    kernels = []
    for dtype in ("f16", "bf16"):
        for n, bs, vec in LN_CONFIGS:
            sym = f"rocke_layernorm2d_fwd_{dtype}_N{n}_b{bs}_v{vec}"
            kernels.append({
                "symbol": sym,
                "co_file": sym + ".co",
                "constraints": {"dtype": {"equals": dtype}, "N": {"equals": n}},
                "grid": {"x": "M", "y": 1, "z": 1},
                "block": [bs, 1, 1],
                "args_signature": LN_ARGS,
                "workspace_bytes": 0,
            })
        for bs, vec in LN_DYNAMIC:
            sym = f"rocke_layernorm2d_fwd_dyn_{dtype}_b{bs}_v{vec}"
            kernels.append({
                "symbol": sym,
                "co_file": sym + ".co",
                "constraints": {
                    "dtype": {"equals": dtype},
                    # "min": 1 is mandatory beside multiple_of: N=0 satisfies
                    # multiple_of alone and would launch a degenerate grid.
                    "N": {"min": 1, "multiple_of": vec},
                },
                "grid": {"x": "M", "y": 1, "z": 1},
                "block": [bs, 1, 1],
                "args_signature": LN_ARGS,
                "workspace_bytes": 0,
            })
    return {
        "family": f"layernorm2d_{ARCH}",
        "op_kind": "layernorm",
        "arch": ARCH,
        "dtype": ["f16", "bf16"],
        "_comment": (
            "gfx950 (CDNA4, wave64) port of the gfx1151 layernorm2d family, built at "
            "wave_size=64. Static entries bake N and match exactly; the _dyn entries "
            "take N as a runtime argument and serve any vec-aligned N."
        ),
        "kernels": kernels,
    }


def write(doc, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")
    print(f"wrote {path}: {len(doc['kernels'])} kernels")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    write(activation_doc(), os.path.join(base, "activation", "family.json"))
    write(layernorm_doc(), os.path.join(base, "layernorm2d", "family.json"))
