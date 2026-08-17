#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Prove the rocke_engine pybind binding is byte-identical to the Python engine.

For each of the 7 GEMM configs (reused from tests/parity/gemm_emit.py), call
rocke_engine.gemm_lower_llvm / gemm_serialize_ir (the C++ engine via pybind) and
compare the result to the Python engine's lower_kernel_to_llvm(build_universal_gemm)
/ ir_serialize.serialize for the same spec. They must match (sha equal)."""

import hashlib
import os
import sys
from pathlib import Path

# Directory holding the built ``rocke_engine`` pybind module. Override via
# ROCKE_PYBIND_BUILD_DIR; defaults to a sibling ``build`` dir next to this file.
sys.path.insert(
    0,
    os.environ.get(
        "ROCKE_PYBIND_BUILD_DIR", str(Path(__file__).resolve().parent / "build")
    ),
)
# composablekernel python root (so ``rocke`` is importable). Override via
# ROCKE_ROOT; defaults to the python/ root inferred relative to this file
# (bindings/ -> rocke/ -> python/).
sys.path.insert(
    0,
    os.environ.get("ROCKE_ROOT", str(Path(__file__).resolve().parents[2])),
)
import rocke_engine

from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
    build_universal_gemm,
)
from rocke import lower_kernel_to_llvm
from rocke.core.ir_serialize import serialize

ARCH = "gfx950"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# (dict for the C++ binding, UniversalGemmSpec for Python) per config.
CONFIGS = [
    (
        dict(
            name="test1",
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline="compv3",
            epilogue="default",
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_c="fp16",
            dtype_acc="fp32",
            wave_size=64,
            block_size=256,
            batched=False,
        ),
        UniversalGemmSpec(
            name="test1",
            tile=TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 16),
            trait=TraitSpec(pipeline="compv3", epilogue="default"),
            data=DataSpec(
                dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
            ),
            wave_size=64,
            block_size=256,
            batched=False,
        ),
    ),
    (
        dict(
            name="test2",
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=4,
            warp_n=4,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="compv4",
            epilogue="cshuffle",
            dtype_a="fp16",
            wave_size=64,
            block_size=1024,
            batched=False,
        ),
        UniversalGemmSpec(
            name="test2",
            tile=TileSpec(256, 256, 64, 4, 4, 1, 32, 32, 16),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=1024,
            batched=False,
        ),
    ),
    (
        dict(
            name="test3",
            tile_m=256,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=4,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
            pipeline="compv4",
            epilogue="default",
            dtype_a="bf16",
            dtype_b="bf16",
            dtype_c="bf16",
            wave_size=64,
            block_size=512,
            batched=False,
        ),
        UniversalGemmSpec(
            name="test3",
            tile=TileSpec(256, 128, 32, 2, 4, 1, 32, 32, 8),
            trait=TraitSpec(pipeline="compv4", epilogue="default"),
            data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
            wave_size=64,
            block_size=512,
            batched=False,
        ),
    ),
    (
        dict(
            name="test4",
            tile_m=128,
            tile_n=256,
            tile_k=64,
            warp_m=4,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
            pipeline="mem",
            epilogue="default",
            dtype_a="fp16",
            wave_size=64,
            block_size=256,
            batched=False,
        ),
        UniversalGemmSpec(
            name="test4",
            tile=TileSpec(128, 256, 64, 4, 1, 1, 16, 16, 32),
            trait=TraitSpec(pipeline="mem", epilogue="default"),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=256,
            batched=False,
        ),
    ),
    (
        dict(
            name="test5",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline="compv3",
            epilogue="cshuffle",
            chiplet_swizzle=True,
            dtype_a="fp16",
            wave_size=64,
            block_size=64,
            batched=False,
        ),
        UniversalGemmSpec(
            name="test5",
            tile=TileSpec(64, 64, 64, 1, 1, 1, 16, 16, 16),
            trait=TraitSpec(
                pipeline="compv3", epilogue="cshuffle", chiplet_swizzle=True
            ),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=64,
            batched=False,
        ),
    ),
    (
        dict(
            name="test6",
            tile_m=256,
            tile_n=256,
            tile_k=128,
            warp_m=4,
            warp_n=4,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="compv4",
            epilogue="default",
            direct_to_lds=True,
            dtype_a="fp16",
            wave_size=64,
            block_size=1024,
            batched=True,
        ),
        UniversalGemmSpec(
            name="test6",
            tile=TileSpec(256, 256, 128, 4, 4, 1, 32, 32, 16),
            trait=TraitSpec(pipeline="compv4", epilogue="default", direct_to_lds=True),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=1024,
            batched=True,
        ),
    ),
    (
        dict(
            name="test7",
            tile_m=192,
            tile_n=192,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=32,
            pipeline="compv4",
            epilogue="cshuffle",
            lds_swizzle=True,
            dtype_a="fp16",
            wave_size=64,
            block_size=256,
            batched=False,
        ),
        UniversalGemmSpec(
            name="test7",
            tile=TileSpec(192, 192, 32, 2, 2, 1, 32, 32, 32),
            trait=TraitSpec(pipeline="compv4", epilogue="cshuffle", lds_swizzle=True),
            data=DataSpec(dtype_a="fp16"),
            wave_size=64,
            block_size=256,
            batched=False,
        ),
    ),
]


def call(fn):
    """Run fn(); return ('ok', result) or ('reject', message)."""
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("reject", str(e))


def compare(cpp_kind, cpp_val, py_kind, py_val):
    """Both produced text -> sha must match. Both rejected -> parity (the C++
    binding wraps the message, so we don't require identical reject text, only
    that BOTH engines agree on accept/reject). One accepts + one rejects ->
    divergence."""
    if cpp_kind == "ok" and py_kind == "ok":
        return (cpp_val == py_val, sha(cpp_val), sha(py_val))
    if cpp_kind == "reject" and py_kind == "reject":
        return (True, "REJECT", "REJECT")
    return (False, cpp_kind.upper(), py_kind.upper())


def main():
    print(
        f"{'cfg':<6}{'LL cpp':<18}{'LL py':<18}{'LL':<6}"
        f"{'IR cpp':<18}{'IR py':<18}{'IR':<6}"
    )
    all_ok = True
    for cdict, pyspec in CONFIGS:
        name = cdict["name"]
        # --- .ll ---
        ck, cv = call(lambda: rocke_engine.gemm_lower_llvm(cdict, arch=ARCH))
        pk, pv = call(
            lambda: lower_kernel_to_llvm(
                build_universal_gemm(pyspec, arch=ARCH), arch=ARCH
            )
        )
        ll_match, ll_c, ll_p = compare(ck, cv, pk, pv)
        # --- ir ---
        cik, civ = call(lambda: rocke_engine.gemm_serialize_ir(cdict, arch=ARCH))
        pik, piv = call(lambda: serialize(build_universal_gemm(pyspec, arch=ARCH)))
        ir_match, ir_c, ir_p = compare(cik, civ, pik, piv)

        all_ok = all_ok and ll_match and ir_match
        print(
            f"{name:<6}{ll_c:<18}{ll_p:<18}{'OK' if ll_match else 'DIFF':<6}"
            f"{ir_c:<18}{ir_p:<18}{'OK' if ir_match else 'DIFF':<6}"
        )
        if ck == "reject" and pk == "reject":
            print(f"       both reject: {pv[:90]}")
    print()
    print(
        "ALL CONSISTENT (byte-identical where built, agree on reject otherwise)"
        if all_ok
        else "MISMATCH DETECTED"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
