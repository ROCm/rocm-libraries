# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone single-shape combo-kernel launcher for rocprofv3 profiling.

Usage: _profile_one.py <sw:0|128> <num_seqs> <iters>
Launches the production-dispatched 2D kernel for a d64/b32/GQA8/sinks shape.
"""

import sys

sys.path.insert(0, "Python")
sys.path.insert(0, "dsl_docs/optimization/utilities/tools/stage1_benchmark")

import torch  # noqa: E402
from _ua_shape_utils import UAShape, make_inputs  # noqa: E402
from rocke.instances import (
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)  # noqa: E402

sw = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ns = int(sys.argv[2]) if len(sys.argv) > 2 else 149
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 300

rec = {
    "kind": "2d",
    "ALL_DECODE": False,
    "q_shape": [8192, 64, 64],
    "k_shape": [8192, 32, 8, 64],
    "v_shape": [8192, 32, 8, 64],
    "block_table_shape": [ns, 64],
    "max_seqlen_q": 1000,
    "max_seqlen_k": 1020,
    "num_seqs": ns,
    "head_size": 64,
    "block_size": 32,
    "num_query_heads": 64,
    "num_kv_heads": 8,
    "softmax_scale": 0.125,
    "softcap": 0.0,
    "window_size": [sw - 1, 0] if sw else [-1, -1],
    "has_sinks": True,
    "has_alibi": False,
    "has_output_scale": False,
    "q_dtype": "torch.bfloat16",
    "k_dtype": "torch.bfloat16",
    "v_dtype": "torch.bfloat16",
    "out_dtype": "torch.bfloat16",
}
shape = UAShape.from_record(rec, source_file="prof", call_idx=0)
data = make_inputs(shape, seed=0, cap_blocks=6000)
problem = UnifiedAttentionProblem(
    total_q=shape.total_q,
    num_seqs=shape.num_seqs,
    num_query_heads=64,
    num_kv_heads=8,
    head_size=64,
    block_size=32,
    max_seqlen_q=shape.max_seqlen_q,
    max_seqlen_k=shape.max_seqlen_k,
    dtype="bf16",
    sliding_window=sw,
    softcap=0.0,
    use_sinks=True,
    use_alibi=False,
    use_qq_bias=False,
    use_fp8=False,
    num_sms=256,
)
out = torch.empty_like(data["query"])
st = torch.cuda.current_stream().cuda_stream
kw = dict(
    problem=problem,
    q=data["query"],
    k=data["key_cache"],
    v=data["value_cache"],
    out=out,
    cu_seqlens_q=data["cu_seqlens_q"],
    seqused_k=data["kv_lens"],
    softmax_scale=data["scale"],
    block_table=data["block_tables"],
    softcap=0.0,
    sinks=data["sinks"],
    backend="auto",
    stream=st,
)
# warmup (compiles + caches)
for _ in range(5):
    run_unified_attention_torch(**kw)
torch.cuda.synchronize()
print(f"profiling sw={sw} ns={ns} iters={iters}", flush=True)
for _ in range(iters):
    run_unified_attention_torch(**kw)
torch.cuda.synchronize()
print("done", flush=True)
