# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure ordinary saved-spec decode calls; write absolute data privately.

Synthetic BF16 one-token full attention on gfx942/gfx950. Internal and outer
HIP graphs are disabled. See CASE_STUDY_DECODE_HOST_OVERHEAD.md for comparison
controls. Torch is imported only when running the GPU harness.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

DEFAULT_CASES = [
    {
        "label": "b1-k2048",
        "batch": 1,
        "kv_length": 2048,
        "query_heads": 32,
        "kv_heads": 4,
        "head_size": 128,
        "block_size": 16,
    },
    {
        "label": "b4-k4096",
        "batch": 4,
        "kv_length": 4096,
        "query_heads": 32,
        "kv_heads": 4,
        "head_size": 128,
        "block_size": 16,
    },
]


def reference(q, k, v, lengths):
    """Independent FP64 reference over quantized, contiguous logical KV."""
    output = np.empty(q.shape, dtype=np.float64)
    ratio = q.shape[1] // k.shape[2]
    for batch, length in enumerate(lengths):
        for head in range(q.shape[1]):
            kv_head = head // ratio
            keys = k[batch, :length, kv_head].astype(np.float64)
            values = v[batch, :length, kv_head].astype(np.float64)
            scores = keys @ q[batch, head].astype(np.float64) / math.sqrt(q.shape[-1])
            weights = np.exp(scores - scores.max())
            output[batch, head] = (weights / weights.sum()) @ values
    return output


def check(actual, expected):
    if not np.isfinite(actual).all() or not np.allclose(
        actual, expected, atol=0.01, rtol=0.02
    ):
        raise AssertionError(
            "BF16 decode failed independent FP64 reference (atol=0.01, rtol=0.02)"
        )
    return {
        "passed": True,
        "max_abs_error": float(np.max(np.abs(actual - expected))),
        "atol": 0.01,
        "rtol": 0.02,
        "reference": "numpy_fp64_quantized_inputs",
    }


def measure(call, *, stream, warmup, iters, samples):
    from rocke.runtime import no_fence, time_launches, wait_stream_and_release

    with no_fence():
        for _ in range(warmup):
            call()
    wait_stream_and_release(stream)
    host = []
    for _ in range(samples):
        try:
            with no_fence():
                start = time.perf_counter_ns()
                for _ in range(iters):
                    call()
                elapsed = time.perf_counter_ns() - start
        finally:
            wait_stream_and_release(stream)
        host.append(elapsed / iters)
    events = [
        time_launches(call, warmup=warmup, iters=iters, stream=stream)
        for _ in range(samples)
    ]
    # First sample is retained in raw data and excluded from the medians.
    return {
        "host_enqueue_ns": host,
        "hip_event_ms": events,
        "host_median_ns": statistics.median(host[1:]),
        "event_median_ms": statistics.median(events[1:]),
    }


def kernel_llvm_hashes(spec, *, arch):
    from rocke import lower_kernel_to_llvm

    return [
        hashlib.sha256(lower_kernel_to_llvm(kernel, arch=arch).encode()).hexdigest()
        for kernel in spec.build(arch)
    ]


def run_case(case, *, arch, mode, timing):
    import torch
    from dispatch.attention import (
        AttentionRequest,
        attention_execution_candidates,
        attention_tuning_spec,
    )
    from kernels import UnifiedAttentionProblem
    from kernels.common import attention_unified as au
    from rocke.runtime import wait_stream_and_release

    b, sk = int(case["batch"]), int(case["kv_length"])
    hq, hk, dim, block = (
        int(case[name])
        for name in ("query_heads", "kv_heads", "head_size", "block_size")
    )
    if min(b, sk, hq, hk, dim, block) <= 0 or hq % hk:
        raise ValueError("positive dimensions and integral GQA ratio required")
    maxima = [sk] if mode == "stable" else [sk, sk + 1, sk + 2]
    capacity = math.ceil(max(maxima) / block) * block
    generator = torch.Generator(device="cpu").manual_seed(13700)
    q_host = (torch.randn((b, hq, dim), generator=generator) * 0.3).to(torch.bfloat16)
    k_host = (torch.randn((b, capacity, hk, dim), generator=generator) * 0.3).to(
        torch.bfloat16
    )
    v_host = (torch.randn((b, capacity, hk, dim), generator=generator) * 0.3).to(
        torch.bfloat16
    )
    expected = reference(
        q_host.float().numpy(), k_host.float().numpy(), v_host.float().numpy(), [sk] * b
    )
    blocks = b * capacity // block
    tensors = {
        "q": q_host.to("cuda"),
        "out": torch.empty((b, hq, dim), device="cuda", dtype=torch.bfloat16),
        "k": k_host.reshape(blocks, block, hk, dim).to("cuda"),
        "v": v_host.reshape(blocks, block, hk, dim).to("cuda"),
        "cu_seqlens_q": torch.arange(b + 1, dtype=torch.int32).to("cuda"),
        "seqused_k": torch.full((b,), sk, dtype=torch.int32).to("cuda"),
        "block_table": torch.arange(blocks, dtype=torch.int32)
        .reshape(b, capacity // block)
        .to("cuda"),
    }
    stream = int(torch.cuda.current_stream().cuda_stream)
    candidates = {c.spec_id: c for c in attention_execution_candidates()}
    spec_id = case.get("spec_id", f"{arch}_u3d_splitkv_seg128_t1xb")
    if spec_id not in candidates or not spec_id.startswith(f"{arch}_u3d_"):
        raise ValueError(f"case requires an executable {arch} 3D spec: {spec_id}")
    calls, specs, problems = [], [], []
    for maximum in maxima:
        req = AttentionRequest(
            batch=b,
            nhead_q=hq,
            nhead_k=hk,
            seqlen_q=1,
            seqlen_k=maximum,
            hdim_q=dim,
            hdim_v=dim,
            arch=arch,
            dtype="bf16",
            kv_block_size=block,
            num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
        )
        spec = attention_tuning_spec(req, spec_id, knobs=case.get("knobs"))
        problem = UnifiedAttentionProblem(
            total_q=b,
            num_seqs=b,
            num_query_heads=hq,
            num_kv_heads=hk,
            head_size=dim,
            block_size=block,
            max_seqlen_q=1,
            max_seqlen_k=maximum,
            dtype="bf16",
            num_cus=req.num_cus,
            num_kv_blocks=blocks,
        )
        bound = candidates[spec_id].bind_torch(
            req, spec, dict(tensors, problem=problem), stream=stream
        )
        calls.append(bound.launch)
        specs.append(spec.with_num_kv_blocks(blocks))
        problems.append(problem)
    keys = [spec.cache_key() for spec in specs]
    if any(key != keys[0] for key in keys):
        raise ValueError(
            "changing maxima cross a saved-spec bucket; choose compatible maxima"
        )
    index = 0

    def call():
        nonlocal index
        calls[index % len(calls)]()
        index += 1

    for launch in calls:
        launch()
    wait_stream_and_release(stream)
    before = check(tensors["out"].cpu().float().numpy(), expected)
    prepared = au._3D_PIPELINES[au._shape_key(keys[0], problems[0])]
    compiled = au._ATTN_3D_TILED_CACHE[keys[0]]
    geometry = {
        "segment_grid": list(prepared.seg_config.grid),
        "segment_block": list(prepared.seg_config.block),
        "reduce_grid": list(prepared.red_config.grid),
        "reduce_block": list(prepared.red_config.block),
    }
    hashes = {
        "hsaco": [hashlib.sha256(compiled[i]).hexdigest() for i in (0, 2)],
        "llvm": kernel_llvm_hashes(specs[0], arch=arch),
    }
    result = measure(call, stream=stream, **timing)
    # Poison output before a fresh launch so the post-check proves GPU writes.
    tensors["out"].copy_(torch.full((b, hq, dim), float("nan"), dtype=torch.bfloat16))
    call()
    wait_stream_and_release(stream)
    after = check(tensors["out"].cpu().float().numpy(), expected)
    if au._3D_GRAPHS:
        raise AssertionError("internal graph cache populated during eager measurement")
    clear = getattr(au, "clear_attention_3d_eager_cache", None)
    if clear is not None:
        clear()
    return dict(
        case=case,
        mode=mode,
        maxima=maxima,
        spec_id=spec_id,
        spec=asdict(specs[0].kernel_spec),
        geometry=geometry,
        hashes=hashes,
        before=before,
        after=after,
        **result,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, choices=("gfx942", "gfx950"))
    parser.add_argument(
        "--cases",
        type=Path,
        help="JSON list of synthetic dimensions, optional executable spec_id and knobs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="private absolute-data JSON destination",
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument(
        "--adapter",
        help="optional environment adapter module with install() before Torch import",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args(argv)
    if args.samples < 2 or args.iters < 1 or args.warmup < 0:
        parser.error("samples >= 2, iters >= 1 and warmup >= 0 required")
    for arch in ("gfx942", "gfx950"):
        os.environ[f"HIPDNN_{arch.upper()}_3D_GRAPH"] = "0"
    if args.adapter:
        importlib.import_module(args.adapter).install()
    import torch
    from kernels.common.attention_unified import _resolve_attention_arch

    torch.set_num_threads(1)
    if _resolve_attention_arch() != args.arch:
        raise RuntimeError("assigned GPU architecture disagrees with requested lane")
    cases = json.loads(args.cases.read_text()) if args.cases else DEFAULT_CASES
    props = torch.cuda.get_device_properties(0)
    provenance = {
        "source_revision": args.source_revision,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trial": args.trial,
        "arch": args.arch,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "numpy": np.__version__,
        "gpu": str(props),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "hip_library": os.environ.get("ROCKE_HIP_LIB"),
        "comgr_library": os.environ.get("ROCKE_COMGR_LIB"),
        "llvm_flavor": os.environ.get("ROCKE_LLVM_FLAVOR"),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "internal_graphs": False,
        "outer_graphs": False,
        "dtype": "bf16",
        "input_generation": "seeded_cpu_torch_then_h2d",
        "timing": {
            "warmup": args.warmup,
            "iters": args.iters,
            "samples": args.samples,
            "discard_first": True,
        },
    }
    results = []
    for case in cases:
        for mode in ("stable", "max_series"):
            result = run_case(
                case,
                arch=args.arch,
                mode=mode,
                timing={
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "samples": args.samples,
                },
            )
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {"provenance": provenance, "results": results},
                    indent=2,
                    default=str,
                )
                + "\n"
            )
            print(
                json.dumps({"case": case["label"], "mode": mode, "numeric": "passed"}),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
