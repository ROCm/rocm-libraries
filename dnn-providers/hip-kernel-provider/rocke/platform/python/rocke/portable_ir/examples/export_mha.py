#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# export_mha.py -- build the unified-attention 2D scalar MHA/SDPA kernel (the
# same kernel benchmarked in dsl_docs/architecture/attention_2d_experiment_summary.md)
# and either export it to portable rocKE IR JSON or print its Python-lowered
# AMDGPU LLVM IR.
#
# Shape-polymorphism: head_size and dtype select the kernel *family* (head_size
# is structural -- the body unrolls over it); sequence length / batch / num_seqs
# are runtime kernel arguments and do NOT change the kernel body. So one exported
# IR per (dtype, head_size) serves S2048 / S4096 / S8192 alike. The --seqlen knob
# exists only to prove that invariance (the exported IR is identical across S).
#
#   export_mha.py --dtype fp16 --head-size 128 [--seqlen 2048] [--ll] [--arch gfx950]
import argparse
import sys

from rocke.core.ir_export import export_kernel_ir_json
from rocke.core.lower_llvm import lower_kernel_to_llvm


def build(
    dtype: str, head_size: int, seqlen: int, batch: int, num_heads: int, gqa: int
):
    # attention_unified lives in the rocke LIBRARY tree (kernels/), not the
    # platform package. Imported lazily so the platform -> library dependency
    # stays confined to this call, mirroring tests/instances/rocke_ir_parity_harness.py.
    from kernels.common.attention_unified import (
        UnifiedAttention2DSpec,
        UnifiedAttentionProblem,
        build_unified_attention_2d,
    )

    # MHA when gqa==1 (num_kv_heads == num_query_heads); GQA otherwise.
    num_kv_heads = max(1, num_heads // gqa)
    problem = UnifiedAttentionProblem(
        total_q=batch * seqlen,  # B*S_q; runtime grid, not baked into the body
        num_seqs=batch,  # runtime
        num_query_heads=num_heads,  # runtime grid
        num_kv_heads=num_kv_heads,  # runtime grid
        head_size=head_size,  # STRUCTURAL (body unrolls over it)
        block_size=16,  # paged-KV block modulus (structural)
        max_seqlen_q=seqlen,  # runtime
        max_seqlen_k=seqlen,  # runtime
        dtype=dtype,  # STRUCTURAL (fp16/bf16)
    )
    spec = UnifiedAttention2DSpec(problem=problem)
    return build_unified_attention_2d(spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument(
        "--gqa", type=int, default=1, help="query heads per kv head (1 = MHA)"
    )
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument(
        "--ll", action="store_true", help="print Python-lowered .ll instead of IR JSON"
    )
    args = ap.parse_args()

    kernel = build(
        args.dtype, args.head_size, args.seqlen, args.batch, args.num_heads, args.gqa
    )
    if args.ll:
        sys.stdout.write(lower_kernel_to_llvm(kernel, arch=args.arch))
    else:
        sys.stdout.write(export_kernel_ir_json(kernel, target_hint=args.arch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
