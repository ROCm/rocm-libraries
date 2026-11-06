# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compile ML4 layer kernels from JSON specs to code objects.

    python -m kernels.gfx950.ml4_mono.layer.build SPEC.json [SPEC.json ...] --out DIR

Writes ``DIR/<kernel>.hsaco`` and ``DIR/<kernel>.ll`` per spec and prints one
JSON line per kernel. Run it in a process that has loaded no other LLVM: a
host that embeds its own compiler (a JIT DSL) loads the code objects instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from rocke.helpers.compile import compile_kernel

from .kernel import build_ml4_layer
from .spec import Ml4LayerSpec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("specs", nargs="+", help="Ml4LayerSpec JSON files")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--arch", default="gfx950")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in args.specs:
        with open(path) as f:
            spec = Ml4LayerSpec.from_json(f.read())
        t0 = time.perf_counter()
        kernel = build_ml4_layer(spec, arch=args.arch)
        art = compile_kernel(kernel, arch=args.arch, backend="python")
        base = os.path.join(args.out, spec.kernel_name)
        with open(base + ".hsaco", "wb") as f:
            f.write(art.hsaco)
        with open(base + ".ll", "w") as f:
            f.write(art.llvm_text)
        try:
            from rocke.benchmark.perf.occupancy import resources

            res = {k: v for k, v in resources(art.hsaco, args.arch).items()
                   if k in ("vgpr", "agpr", "sgpr", "lds_bytes", "vgpr_spill", "sgpr_spill")}  # fmt: skip
        except Exception as exc:  # the metadata reader needs llvm-readelf
            res = {"error": str(exc)[:80]}
        print(
            json.dumps(
                {
                    "spec": path,
                    "kernel": spec.kernel_name,
                    "hsaco": base + ".hsaco",
                    "ms": round((time.perf_counter() - t0) * 1e3, 1),
                    "resources": res,
                    # the MoE sum order the build uses (re-associated: x_out
                    # matches the reference to rounding)
                    "moe_em": spec.moe_em,
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
