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


# ==========================================================================
# Fused MoE
# ==========================================================================
#
# The MoE lane is driven as two subprocesses rather than in-process, and that
# is a deliberate inversion of how attention is measured here.
#
# The reason is that the two halves cannot share an interpreter. The MoE
# harness is torch-free by construction -- it drives HIP through ctypes and
# builds its inputs, its fp8 block-scale quantization, its routing and its
# numpy oracle without importing torch, and asserts as much at startup. The
# incumbent it must be timed against is vLLM's Triton ``fused_experts``, which
# is nothing but torch. The existing benchmarks already keep that boundary (the
# baseline lives in its own file for exactly this reason), so the runner honors
# it instead of re-litigating it.
#
# The alternative -- reimplementing input generation here -- would duplicate
# the fp8 quantization, the top-k routing, the block-sorted layout and the
# reference oracle. Those are the parts most likely to be subtly wrong, and a
# second copy would let the measured lane and the benchmark disagree about what
# the kernel even computes.

import json
import os
import subprocess
from pathlib import Path

#: Where the two harnesses live. They are loose scripts, not an importable
#: package, so they are located by path like ``_ua_shape_utils`` is.
_MOE_BENCH_DIRS = (
    Path(__file__).resolve().parents[2]
    / "platform"
    / "python"
    / "rocke"
    / "examples"
    / "gfx950"
    / "fused_mega_moe",
)

#: Wire dims -> the harness's named shape. The harness generates ~600 MB of
#: expert weights per shape and caches them by name, so it takes a name rather
#: than loose dimensions.
_MOE_SHAPE_BY_DIMS = {
    (32, 128, 8, 2048, 768): "qwen3",
}

#: The seed ``Weights`` defaults to. The cache directory name embeds it, and
#: the routing inside that directory is what makes the two lanes comparable.
_MOE_WEIGHT_SEED = 11939


def _moe_routing_dir(shape_name: str, experts: int) -> Path:
    """Where the numpy harness cached this shape's routing.

    Sharing it with the Triton baseline is not a nicety. The layer's latency is
    set by how many experts the routing activated -- that is what fixes the
    weight bytes streamed -- so two independently seeded routings produce two
    different amounts of work and a speedup between them measures nothing. The
    baseline harness takes ``--routing-from`` for exactly this reason.
    """
    root = str(os.environ.get("ROCKE_MOE_BENCH_CACHE") or "").strip()
    base = Path(root) if root else _moe_bench_dir() / ".cache"
    return base / f"{shape_name}_e{experts}_seed{_MOE_WEIGHT_SEED}"


def _moe_bench_dir() -> Path:
    explicit = str(os.environ.get("ROCKE_MOE_BENCH_DIR") or "").strip()
    if explicit and (Path(explicit) / "bench_moe_mega_fp8.py").is_file():
        return Path(explicit)
    for candidate in _MOE_BENCH_DIRS:
        if (candidate / "bench_moe_mega_fp8.py").is_file():
            return candidate
    raise FileNotFoundError(
        "MoE benchmark harness not found; set ROCKE_MOE_BENCH_DIR to the "
        "directory holding bench_moe_mega_fp8.py"
    )


def moe_shape_name(problem: dict[str, Any]) -> str:
    """Map a wire problem onto one of the harness's named shapes.

    The harness generates ~600 MB of expert weights per shape and caches them
    by name, so it takes a shape name rather than loose dimensions. Only the
    cohort the dispatcher claims is mappable, which is the same restriction the
    dispatcher already enforced -- this is a lookup, not a second gate.
    """
    key = (
        int(problem["tokens"]),
        int(problem["experts"]),
        int(problem["topk"]),
        int(problem["hidden"]),
        int(problem["intermediate"]),
    )
    name = _MOE_SHAPE_BY_DIMS.get(key)
    if name is None:
        raise ValueError(f"no benchmark shape for MoE dims {key}")
    return name


def _spec_json_for(plan: dict[str, Any]) -> dict[str, Any]:
    """The benchmark ``Config`` fields for the kernel dispatch selected.

    Read out of the plan's own spec rather than restated, so measuring the
    dispatched kernel cannot silently become measuring a different one. The
    tuning record supplies only the fields the benchmark ``Config`` needs and
    the spec does not carry; everything the spec does carry wins, because the
    spec is what was dispatched.
    """
    from rocke.instances.common.moe_fused_mega_fp8_tuned import BASE_KNOBS

    spec = dict(plan.get("spec") or {})
    fields = dict(BASE_KNOBS)
    fields["label"] = "dispatched"
    fields.update({k: v for k, v in spec.items() if k in fields or k == "tile_m"})
    return fields


def _tuned_config_dir() -> str:
    """A ``VLLM_TUNED_CONFIG_FOLDER`` for the baseline, when one is available.

    This is load-bearing for honesty, not a performance nicety. vLLM ships no
    Triton config for ``E=128,N=768,fp8_w8a8,block_shape=[128,128]``, so it
    falls back to a default and says so:

        Using default MoE config. Performance might be sub-optimal!

    Timing against that fallback measures a missing JSON file, not a kernel.
    Tuning it is worth 1.048x on this shape -- enough to erase the mega-kernel's
    entire reported margin -- so the comparison has to be made against the
    tuned config or it is not a comparison at all.
    """
    explicit = str(os.environ.get("VLLM_TUNED_CONFIG_FOLDER") or "").strip()
    if explicit and Path(explicit).is_dir():
        return explicit
    bundled = str(os.environ.get("ROCKE_MOE_TUNED_CONFIGS") or "").strip()
    if bundled and Path(bundled).is_dir():
        return bundled
    return ""


def _run_bench(
    script: str,
    argv: list[str],
    *,
    work: Path,
    timeout_s: int,
    python_bin: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Run one harness and read back its JSON. Never raises."""
    bench_dir = _moe_bench_dir()
    out_path = work / f"{Path(script).stem}.json"
    cmd = [python_bin, "-u", str(bench_dir / script), *argv, "--json", str(out_path)]
    env = {**os.environ, **(extra_env or {})}
    # The harnesses import ``rocke`` from the platform source root.
    platform_python = bench_dir.parents[3]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(platform_python)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, env=env
        )
    except subprocess.TimeoutExpired:
        return False, f"{script} exceeded {timeout_s}s", {}
    except OSError as exc:
        return False, f"{script} failed to start: {exc}", {}
    if not out_path.is_file():
        tail = (proc.stderr or proc.stdout or "")[-400:]
        return False, f"{script} wrote no result (rc={proc.returncode}): {tail}", {}
    try:
        return True, "", json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"{script} wrote malformed JSON: {exc}", {}


def measure_moe_plan(
    plan: dict[str, Any],
    *,
    iterations: int = 50,
    warmup: int = 25,
    do_verify: bool = True,
    do_baseline: bool = True,
    timeout_s: int = 1800,
    work_dir: str = "",
) -> dict[str, Any]:
    """Verify and time one planned MoE layer. Never raises.

    Correctness here is stronger than the attention lane's: the harness checks
    the kernel against a numpy oracle that recomputes the whole layer -- routing,
    fp8 block-scale dequant, SiLU, requantization and both GEMMs -- in float64,
    rather than against another rocKE kernel. So a shared misreading of the
    semantics is not invisible the way it is for attention.
    """
    result: dict[str, Any] = {"signature": plan.get("signature", "")}
    problem = dict(plan.get("problem") or {})
    work = Path(work_dir or ".") / "moe_measure"
    work.mkdir(parents=True, exist_ok=True)

    try:
        shape_name = moe_shape_name(problem)
    except (KeyError, ValueError) as exc:
        return {**result, "ran": False, "reason": str(exc)}

    python_bin = str(os.environ.get("ROCKE_MOE_PYTHON") or sys.executable)
    spec_path = work / "dispatched_spec.json"
    spec_path.write_text(
        json.dumps(_spec_json_for(plan), indent=2, sort_keys=True), encoding="utf-8"
    )

    ok, reason, payload = _run_bench(
        "bench_moe_mega_fp8.py",
        [
            "--shape", shape_name,
            "--spec-json", str(spec_path),
            "--iters", str(iterations),
            "--warmup", str(warmup),
            "--phase", "full",
        ],
        work=work,
        timeout_s=timeout_s,
        python_bin=python_bin,
    )
    if not ok:
        return {**result, "ran": False, "reason": reason}

    rows = [r for r in (payload.get("rows") or []) if r.get("us") is not None]
    if not rows:
        note = ((payload.get("rows") or [{}])[0]).get("note") or "no row produced"
        return {**result, "ran": False, "reason": f"rocke lane did not run: {note}"}
    row = rows[0]
    ms = float(row["us"]) / 1000.0
    result["rocke"] = {
        "ran": True,
        "latency_ms": ms,
        "latency_us": float(row["us"]),
        "achieved_gbs": row.get("gbs"),
        "grid": row.get("grid"),
    }

    if do_verify:
        rel = row.get("rel")
        tol = float(payload.get("tol") or 1.5e-2)
        result["verify"] = (
            {
                "ran": True,
                "passed": float(rel) <= tol,
                "max_rel_diff": float(rel),
                "rtol": tol,
                "reference": "numpy_float64_oracle",
            }
            if rel is not None
            else {"ran": False, "reason": "harness reported no parity figure"}
        )

    if do_baseline:
        baseline_argv = [
            "--shape", shape_name,
            "--iters", str(iterations),
            "--warmup", str(warmup),
        ]
        routing = _moe_routing_dir(shape_name, int(problem["experts"]))
        shared_routing = (routing / "topk_ids.npy").is_file()
        if shared_routing:
            baseline_argv += ["--routing-from", str(routing)]
        tuned_dir = _tuned_config_dir()
        ok, reason, base = _run_bench(
            "bench_triton_baseline.py",
            baseline_argv,
            work=work,
            timeout_s=timeout_s,
            python_bin=python_bin,
            extra_env={"VLLM_TUNED_CONFIG_FOLDER": tuned_dir} if tuned_dir else None,
        )
        result["baseline"] = (
            {
                "ran": True,
                "latency_ms": float(base["latency_ms"]),
                "framework": base.get("framework", "vllm_triton_fused_experts"),
                "active_experts": base.get("active_experts"),
                "read_peak_gbs": base.get("read_peak_gbs"),
                "shared_routing": shared_routing,
                "tuned_config": bool(tuned_dir),
                "tuned_config_dir": tuned_dir,
            }
            if ok
            else {"ran": False, "reason": reason}
        )
        # Both conditions can hold at once, and each on its own is enough to
        # make the ratio unreportable, so they accumulate rather than overwrite.
        warnings: list[str] = []
        if ok and not tuned_dir:
            # Without this the lane reports a speedup over vLLM's fallback,
            # which on this shape is worth 1.048x on its own -- more than the
            # margin being claimed.
            warnings.append(
                "baseline ran vLLM's default MoE config; no tuned config folder "
                "was found, so this speedup includes the cost of a missing "
                "vendor JSON rather than measuring the kernel. Set "
                "VLLM_TUNED_CONFIG_FOLDER (or ROCKE_MOE_TUNED_CONFIGS)"
            )
        if ok and not shared_routing:
            # The two lanes activated different expert counts, so they did
            # different amounts of work and the number is not like-for-like.
            warnings.append(
                f"routing not shared (no cache at {routing}); the baseline "
                "seeded its own routing, so the expert count -- and therefore "
                "the weight traffic -- may differ from rocKE's"
            )
        if warnings:
            result["baseline"]["warnings"] = warnings
        if ok and ms > 0:
            result["speedup"] = float(base["latency_ms"]) / ms
            # The layer streams a fixed number of weight bytes, so the share of
            # achievable read bandwidth says whether a speedup has any room
            # left in it. Reported next to the ratio because on this workload
            # the roofline, not the incumbent, is the binding constraint.
            peak = float(base.get("read_peak_gbs") or 0.0)
            achieved = float(row.get("gbs") or 0.0)
            if peak > 0 and achieved > 0:
                result["roofline_fraction"] = achieved / peak

    result["ran"] = True
    return result
