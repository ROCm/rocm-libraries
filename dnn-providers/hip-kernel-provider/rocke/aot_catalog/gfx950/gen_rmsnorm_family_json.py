#!/usr/bin/env python3
"""Emit the gfx950 rmsnorm2d family.json.

Kept next to the producer so the two config tables cannot drift: the symbol names
here reproduce rocKE's RMSNorm2D{,Dynamic}Spec.kernel_name(), which is arch-
independent (no gfx token in the symbol). `verify_family_json.py` cross-checks the
emitted .co filenames against this file after a build, so a naming change in rocKE
fails loudly instead of silently producing an empty catalog.

Plain stdlib: runnable before the rocKE pyenv exists.
"""

import json
import os
import sys

ARCH = "gfx950"
FAMILY = "rmsnorm2d"

# Must match produce_rmsnorm2d_co.py exactly.
CONFIGS = [
    (2048, 256, 4),
    (2048, 512, 4),
    (2048, 128, 8),
    (2048, 64, 8),
    (1024, 256, 4),
    (1024, 128, 8),
    (4096, 512, 4),
    (4096, 256, 8),
]
CONFIGS_BF16 = list(CONFIGS)
DYNAMIC_CONFIGS = [(256, 4), (128, 8)]

# (X, Gamma, Y, M, N, eps) -- the RmsNormAdapter's emitted vocabulary.
ARGS = [
    {"name": "X", "type": "ptr"},
    {"name": "Gamma", "type": "ptr"},
    {"name": "Y", "type": "ptr"},
    {"name": "M", "type": "i32"},
    {"name": "N", "type": "i32"},
    {"name": "eps", "type": "f32"},
]


def static_symbol(dtype, n, bs, vec):
    return f"rocke_rmsnorm2d_fwd_{dtype}_N{n}_b{bs}_v{vec}"


def dynamic_symbol(dtype, bs, vec):
    return f"rocke_rmsnorm2d_fwd_dyn_{dtype}_b{bs}_v{vec}"


def entry(symbol, constraints, block_size):
    return {
        "symbol": symbol,
        "co_file": symbol + ".co",
        "constraints": constraints,
        "grid": {"x": "M", "y": 1, "z": 1},
        "block": [block_size, 1, 1],
        "args_signature": ARGS,
        "workspace_bytes": 0,
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "family.json"
    kernels = []

    for dtype, table in (("f16", CONFIGS), ("bf16", CONFIGS_BF16)):
        for n, bs, vec in table:
            kernels.append(
                entry(
                    static_symbol(dtype, n, bs, vec),
                    # Static kernels bake N, so exact match only.
                    {"dtype": {"equals": dtype}, "N": {"equals": n}},
                    bs,
                )
            )
        for bs, vec in DYNAMIC_CONFIGS:
            kernels.append(
                entry(
                    dynamic_symbol(dtype, bs, vec),
                    # Runtime-N: serves any N that is vec-aligned. "min": 1 is
                    # mandatory -- multiple_of alone admits N=0, a degenerate
                    # zero-extent problem that would launch a garbage grid.
                    {
                        "dtype": {"equals": dtype},
                        "N": {"min": 1, "multiple_of": vec},
                    },
                    bs,
                )
            )

    doc = {
        "family": f"{FAMILY}_{ARCH}",
        "op_kind": "rmsnorm",
        "arch": ARCH,
        "dtype": ["f16", "bf16"],
        "_comment": (
            "gfx950 (CDNA4, wave64) port of the gfx1151 rmsnorm2d family. Same "
            "algorithm and knob-space; the kernels are built at wave_size=64 "
            "instead of 32. Static entries bake N and match exactly; the _dyn "
            "entries take N as a runtime argument and serve any vec-aligned N."
        ),
        "kernels": kernels,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")
    print(f"wrote {out}: {len(kernels)} kernels")


if __name__ == "__main__":
    sys.exit(main())
