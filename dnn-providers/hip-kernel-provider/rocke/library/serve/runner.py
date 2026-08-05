# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The measured lanes: does the planned kernel compute the right answer, and is
it faster than what the workload runs today.

This is the only optional stage. It needs torch and a device, so every entry
point here degrades to a stated reason rather than an exception -- a caller that
planned successfully on a CPU box must get its plan back, not a traceback.

Correctness reference
---------------------
The fast path is checked against rocKE's own scalar attention kernel, which
exists for exactly this purpose. That bounds what the check can prove: it
catches tiling, geometry, and codegen faults in the fast path -- the errors that
actually occur -- but not a misreading of paged-KV semantics shared by both
kernels, since they share none of the fast path but do share the semantics. The
stronger check against an independent torch reference lives in the parity
harnesses under ``builders/*/attention/``; it is not duplicated here.

Baseline
--------
Speedup is reported against AITER's Triton ``unified_attention`` when it is
importable, because that is the kernel the traced workload is actually running
and therefore the only honest thing to beat. With no baseline present the lane
reports ``None`` rather than falling back to something easier to beat: a
speedup against a reference no one deploys would be a number that reads like
evidence without being any.
"""

from __future__ import annotations

import math
import sys
from typing import Any

#: Cap on reference work (query rows x KV length). The scalar kernel is O(n^2)
#: and deliberately unoptimized, so a production prefill shape can take minutes.
#: Past this the verify lane declines and says so.
_VERIFY_WORK_LIMIT = 1 << 22

#: The attention parity gate, matching ``tests/differential/numeric_attention``
#: (``rtol=0.0, atol=2e-2``) and the example harnesses. Softmax attention adds an
#: exp and a length-normalization on top of two matmuls, so the fast path
#: accumulates in a different order than the scalar reference and a tighter
#: bound would reject ordering noise as a fault. Purely absolute is deliberate:
#: attention outputs cross zero, where a relative bound is meaningless.
ATTENTION_ATOL = 2e-2
ATTENTION_RTOL = 0.0

_DTYPE_TO_TORCH_NAME = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}
_PATH_TO_BACKEND = {"2d": "tiled", "3d": "3d"}


def torch_gpu_available() -> tuple[bool, str]:
    """Return whether a torch-visible GPU is usable, and why not when it isn't."""
    try:
        import torch
    except ImportError as exc:
        return False, f"torch unavailable: {exc}"
    try:
        if not torch.cuda.is_available():
            return False, "no torch-visible GPU"
    except (RuntimeError, AssertionError) as exc:
        return False, f"torch GPU probe failed: {exc}"
    return True, ""


def _load_shape_utils():
    """Import the loose ``_ua_shape_utils`` helper via the sanctioned locator.

    It lives outside any package under the platform source root, so it needs the
    explicit accessor rather than an ordinary import. This mirrors what the
    attention benchmarks already do.
    """
    from rocke.assets import shape_utils_dir

    path = str(shape_utils_dir())
    if path not in sys.path:
        sys.path.insert(0, path)
    import _ua_shape_utils

    return _ua_shape_utils


def shape_from_problem(
    problem: dict[str, Any], *, softmax_scale: float, causal: bool = True
):
    """Build the benchmark harness's ``UAShape`` from a runtime problem.

    Two fields the wire format does not carry have to be derived, because they
    describe the paged cache rather than the attention math: ``max_blocks_per_seq``
    follows from the KV length and the block size, and ``num_blocks`` is sized to
    hold every sequence's blocks so the block table indices stay in range.
    """
    utils = _load_shape_utils()

    block_size = int(problem["block_size"])
    num_seqs = int(problem["num_seqs"])
    max_seqlen_k = int(problem["max_seqlen_k"])
    max_blocks_per_seq = max(1, math.ceil(max_seqlen_k / block_size))

    dtype = str(problem["dtype"]).lower()
    q_name = _DTYPE_TO_TORCH_NAME.get(dtype)
    if q_name is None:
        raise ValueError(f"unsupported dtype {dtype!r} for the measured lanes")
    # An fp8 KV cache stores K/V narrower than Q; everything else is uniform.
    kv_name = "torch.float8_e4m3fnuz" if problem.get("use_fp8") else q_name

    sliding_window = int(problem.get("sliding_window") or 0)
    window = (sliding_window - 1, 0) if sliding_window > 0 else (-1, -1)

    scale = float(softmax_scale) or 1.0 / math.sqrt(float(problem["head_size"]))

    return utils.UAShape(
        source_file="rocke-serve",
        line_idx=0,
        call_idx=0,
        kind="unified_attention",
        all_decode=int(problem.get("max_seqlen_q") or 1) == 1,
        num_seqs=num_seqs,
        total_q=int(problem["total_q"]),
        num_query_heads=int(problem["num_query_heads"]),
        num_kv_heads=int(problem["num_kv_heads"]),
        head_size=int(problem["head_size"]),
        block_size=block_size,
        num_blocks=max(1, num_seqs * max_blocks_per_seq),
        max_blocks_per_seq=max_blocks_per_seq,
        max_seqlen_q=int(problem["max_seqlen_q"]),
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        softcap=float(problem.get("softcap") or 0.0),
        window_size=window,
        has_sinks=bool(problem.get("use_sinks")),
        has_alibi=bool(problem.get("use_alibi")),
        has_output_scale=False,
        q_dtype=q_name,
        k_dtype=kv_name,
        v_dtype=kv_name,
        out_dtype=q_name,
    )


def _stream_handle() -> int:
    import torch

    return int(torch.cuda.current_stream().cuda_stream)


def _time(call_once, *, warmup: int, iters: int, stream: int) -> float:
    from rocke.runtime import synchronize_and_release, time_launches

    ms = time_launches(call_once, warmup=warmup, iters=iters, stream=stream)
    synchronize_and_release(stream)
    return ms


def _rocke_call(problem_obj, data, shape, *, out, backend: str, stream: int):
    from kernels.common.attention_unified import run_unified_attention_torch

    def call_once():
        run_unified_attention_torch(
            problem=problem_obj,
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_seqlens_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(shape.softcap),
            sinks=data["sinks"],
            alibi_slopes=data["alibi_slopes"],
            backend=backend,
            stream=stream,
        )

    return call_once


def verify(problem_obj, data, shape, *, backend: str, stream: int) -> dict[str, Any]:
    """Compare the planned fast path against the scalar reference kernel."""
    import torch

    work = int(shape.total_q) * int(shape.max_seqlen_k)
    if work > _VERIFY_WORK_LIMIT:
        return {
            "ran": False,
            "reason": f"reference work {work} exceeds limit {_VERIFY_WORK_LIMIT}",
        }

    fast = torch.empty_like(data["output"])
    reference = torch.empty_like(data["output"])
    try:
        _rocke_call(
            problem_obj, data, shape, out=fast, backend=backend, stream=stream
        )()
        _rocke_call(
            problem_obj, data, shape, out=reference, backend="scalar", stream=stream
        )()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - a failed lane must not abort the run
        return {"ran": False, "reason": f"reference launch failed: {exc!r}"}

    a = fast.float()
    b = reference.float()
    diff = (a - b).abs()
    allowed = ATTENTION_ATOL + ATTENTION_RTOL * b.abs()
    margin = float((diff - allowed).max())
    # Reported for diagnosis only, never as the gate. Attention outputs pass
    # through zero, so a relative error taken against a near-zero reference is
    # large whenever the absolute error is utterly negligible.
    denom = b.abs().clamp_min(1e-12)
    return {
        "ran": True,
        "passed": margin <= 0.0,
        "margin": margin,
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float((diff / denom).max()),
        "atol": ATTENTION_ATOL,
        "rtol": ATTENTION_RTOL,
        "reference": "rocke_scalar",
    }


def _bench_baseline(
    data, shape, *, warmup: int, iters: int, stream: int
) -> dict[str, Any]:
    """Time AITER's Triton ``unified_attention`` on the same inputs."""
    try:
        from aiter.ops.triton.attention.unified_attention import unified_attention
    except Exception as exc:  # noqa: BLE001 - optional baseline
        return {"ran": False, "reason": f"aiter Triton baseline unavailable: {exc!r}"}

    out = data["output"]

    def call_once():
        unified_attention(
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_seqlens_q"],
            max_seqlen_q=data["max_query_len"],
            seqused_k=data["kv_lens"],
            max_seqlen_k=data["max_kv_len"],
            softmax_scale=data["scale"],
            causal=True,
            window_size=tuple(shape.window_size),
            block_table=data["block_tables"],
            softcap=shape.softcap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=data["alibi_slopes"],
            sinks=data["sinks"],
        )

    try:
        ms = _time(call_once, warmup=warmup, iters=iters, stream=stream)
    except Exception as exc:  # noqa: BLE001
        return {"ran": False, "reason": f"baseline launch failed: {exc!r}"}
    return {
        "ran": True,
        "latency_ms": ms,
        "framework": "triton_aiter_unified_attention",
    }


def measure_plan(
    plan: dict[str, Any],
    *,
    iterations: int = 20,
    warmup: int = 5,
    seed: int = 0,
    do_verify: bool = True,
    do_baseline: bool = True,
) -> dict[str, Any]:
    """Verify and time one planned shape. Never raises."""
    from kernels.common.attention_unified import UnifiedAttentionProblem

    result: dict[str, Any] = {"signature": plan.get("signature", "")}
    problem = dict(plan.get("problem") or {})
    backend = _PATH_TO_BACKEND.get(str(plan.get("path") or ""), "auto")
    result["backend"] = backend

    try:
        utils = _load_shape_utils()
        shape = shape_from_problem(
            problem, softmax_scale=float(plan.get("softmax_scale") or 0.0)
        )
        data = utils.make_inputs(shape, seed=seed)
        problem_obj = UnifiedAttentionProblem(**problem)
        stream = _stream_handle()
    except Exception as exc:  # noqa: BLE001 - setup failure is a lane failure
        return {**result, "ran": False, "reason": f"input setup failed: {exc!r}"}

    if do_verify:
        result["verify"] = verify(
            problem_obj, data, shape, backend=backend, stream=stream
        )
    try:
        ms = _time(
            _rocke_call(
                problem_obj,
                data,
                shape,
                out=data["output"],
                backend=backend,
                stream=stream,
            ),
            warmup=warmup,
            iters=iterations,
            stream=stream,
        )
        result["rocke"] = {"ran": True, "latency_ms": ms}
    except Exception as exc:  # noqa: BLE001
        return {**result, "ran": False, "reason": f"rocke launch failed: {exc!r}"}

    if do_baseline:
        result["baseline"] = _bench_baseline(
            data, shape, warmup=warmup, iters=iterations, stream=stream
        )
        base = result["baseline"]
        if base.get("ran") and ms > 0:
            result["speedup"] = float(base["latency_ms"]) / ms

    result["ran"] = True
    return result
