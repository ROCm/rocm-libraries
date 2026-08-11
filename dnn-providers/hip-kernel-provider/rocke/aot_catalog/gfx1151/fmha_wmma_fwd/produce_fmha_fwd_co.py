# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Co-located, build-time producer for the fmha_wmma_fwd family: emit the gfx1151
# WMMA flash-attention forward .co (HSACO), for f16 and NATIVE bf16, into the
# directory given as argv[1]. Runtime never touches rocke -- the per-family
# CMakeLists runs this at build time to drop the .co next to the checked-in
# family.json (family.json is the source of truth; this script emits .co ONLY).
#
# Normally run by this family's CMakeLists via the rocke build interpreter
# (${ROCKE_PYENV_PYTHON}); the editable pyenv puts `rocke` on the path with no
# PYTHONPATH surgery. To run standalone, use that interpreter, e.g.:
#   <build>/rocke-pyenv/bin/python produce_fmha_fwd_co.py <out_dir>
# The kernel symbols are auto-derived and MUST match the co_file names in
# family.json (rocke_wmma_fmha_fwd_wmma16x16x16_H64_HQ32_HK32_{f16,bf16}_none_vgather).
#
# WHAT THIS BUILDS: rocKE's real gfx1151 WMMA attention forward
# `rocke.instances.gfx1151.wmma_fmha_fwd.build_wmma_fmha_fwd`, a thin adapter
# over the unified `mfma_attention_fwd_inner_body`. We use rocke purely as a
# *library*. The bf16 build is NATIVE (no fp16 cast): bf16 shares the f16 WMMA
# 16x16x16 fragment layout on gfx1151, so the same inner body lowers to the
# `wmma.f32.16x16x16.bf16` intrinsic when the dtype is bf16.
#
# ONE KERNEL PER DTYPE serves both LTX attention flavors: head_size (D=64) and
# num_heads (H=32) are compile-time, but seqlen_q/seqlen_k are RUNTIME kernel
# args, so the D64/H32 mask=none kernel covers self-attn (Sq=Sk=4096) AND
# cross-attn (Sq=4096, Sk=128) alike. The launch contract (grid/block/ABI,
# scale_log2 gotcha, alignment gates) is documented in family.json.

import os
import sys

from rocke.helpers.compile import compile_kernel
from rocke.instances.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    build_wmma_fmha_fwd,
    is_valid_spec,
)

ARCH = "gfx1151"

# The spec dtype spelling ("fp16"/"bf16"); the catalog dtype constraint tokens
# ("f16"/"bf16") live in family.json, not here.
_DTYPES = ["fp16", "bf16"]

# LTX-Video (and most diffusion transformers) run MHA with head_size 64 and
# 32 heads, non-causal, no mask. One kernel per dtype covers every LTX SDPA call.
_HEAD_SIZE = 64
_NUM_HEADS = 32


def _build_spec(dtype):
    return WmmaFmhaFwdSpec(
        head_size=_HEAD_SIZE,
        num_query_heads=_NUM_HEADS,
        num_kv_heads=_NUM_HEADS,  # MHA (H_kv == H)
        dtype=dtype,
        mask_mode="none",
        v_lds_stage=False,  # measured winner on gfx1151 (see the spec docstring)
    )


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    skipped = []
    for dtype in _DTYPES:
        spec = _build_spec(dtype)
        ok, why = is_valid_spec(spec, arch=ARCH)
        if not ok:
            # This producer only runs (via the per-family CMakeLists) when rocKE is
            # available and the arch matches, so every dtype is expected to compile.
            # A skip means the sweep and the checked-in family.json have diverged;
            # emitting a partial family would let the missing .co drop BOTH kernels
            # at catalog load (parseKernel throws -> parseFamily aborts the file).
            # Record it and fail below rather than exit 0 with an incomplete family.
            print(f"SKIP {dtype}: {why}")
            skipped.append(dtype)
            continue

        artifact = compile_kernel(build_wmma_fmha_fwd(spec, arch=ARCH), arch=ARCH)
        if not artifact.hsaco:
            print(f"ERROR {dtype}: compiled .co is empty", file=sys.stderr)
            return 1
        symbol = artifact.kernel_name
        co_file = symbol + ".co"
        with open(os.path.join(out_dir, co_file), "wb") as f:
            f.write(artifact.hsaco)
        print(
            f"dtype={dtype} symbol={symbol} "
            f"D={_HEAD_SIZE} H={_NUM_HEADS} mask=none "
            f"bytes={len(artifact.hsaco)} path={os.path.join(out_dir, co_file)}"
        )

    if skipped:
        print(
            f"ERROR: {len(skipped)} kernel(s) skipped ({', '.join(skipped)}); the "
            "family.json would be incomplete -- failing the build (see issue 3 / "
            "the engine README).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
