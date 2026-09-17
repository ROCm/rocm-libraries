#!/usr/bin/env python3
"""Isolated ``rocke_dense`` attention-comparison lane.

The module is intentionally stdlib-only at import time so its support and
feature-detection contract can be unit-tested without torch, a GPU, or a rocKE
checkout. Runtime dependencies are imported lazily by :func:`main`.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, replace
from inspect import signature
from pathlib import Path
from typing import Any, Callable


LANE = "rocke_dense"
SUPPORTED_DTYPES = {"bf16", "fp16"}
SUPPORTED_HEAD_SIZES = {64, 128}
CORRECTNESS_WARNING_THRESHOLD = 5e-2


def firstline(error: BaseException) -> str:
    text = str(error)
    return text.splitlines()[0] if text else type(error).__name__


def classify_case(case: dict) -> tuple[bool, str]:
    """Conservative v1 cohort for dense self-attention prefill."""
    try:
        batch = int(case["B"])
        sq = int(case["Sq"])
        skv = int(case["Skv"])
        hq = int(case["Hq"])
        hkv = int(case["Hkv"])
        head_size = int(case["D"])
    except (KeyError, TypeError, ValueError) as error:
        return False, f"invalid dense-prefill shape: {firstline(error)}"

    dtype = str(case.get("dtype", "")).lower()
    mask = str(case.get("mask", "causal")).lower()
    try:
        window = int(case.get("window", 0) or 0)
        softcap = float(case.get("softcap", 0.0) or 0.0)
    except (TypeError, ValueError) as error:
        return False, f"invalid dense-prefill option: {firstline(error)}"
    if batch <= 0 or sq <= 0 or skv <= 0 or hq <= 0 or hkv <= 0:
        return False, "dense-prefill dimensions must be positive"
    if dtype not in SUPPORTED_DTYPES:
        return False, f"dtype={dtype!r} is not bf16/fp16"
    if head_size not in SUPPORTED_HEAD_SIZES:
        return False, f"head_size={head_size} is not 64/128"
    if hq % hkv:
        return False, f"Hq={hq} is not divisible by Hkv={hkv}"
    if sq != skv:
        return False, "cross-length/decode attention is outside dense-prefill v1"
    if mask not in {"causal", "full"}:
        return False, f"mask={mask!r} is outside dense-prefill v1"
    if window:
        return False, "sliding-window attention is outside dense-prefill v1"
    if bool(case.get("varlen")) or bool(case.get("doc_mask")):
        return False, "varlen/doc-mask attention is outside dense-prefill v1"
    if bool(case.get("has_sinks")):
        return False, "attention sinks are unsupported"
    if bool(case.get("has_alibi")):
        return False, "ALiBi is unsupported"
    if bool(case.get("has_output_scale")):
        return False, "output scaling is unsupported"
    if softcap:
        return False, "softcap is unsupported"
    return True, "ok"


def classify_paged_case(case: dict, arch: str) -> tuple[bool, str]:
    """Narrow cohort for gfx950 paged-KV sliding-window prefill."""
    try:
        batch = int(case["B"])
        sq = int(case["Sq"])
        skv = int(case["Skv"])
        hq = int(case["Hq"])
        hkv = int(case["Hkv"])
        head_size = int(case["D"])
        block_size = int(case.get("block", 0))
        window = int(case.get("window", 0) or 0)
        softcap = float(case.get("softcap", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError) as error:
        return False, f"invalid paged dense-prefill shape: {firstline(error)}"

    dtype = str(case.get("dtype", "")).lower()
    mask = str(case.get("mask", "causal")).lower()
    if arch != "gfx950":
        return False, f"paged dense-prefill requires gfx950, got {arch or 'unknown'}"
    if batch != 1:
        return False, f"paged dense-prefill requires B=1, got B={batch}"
    if min(sq, skv, hq, hkv) <= 0:
        return False, "paged dense-prefill dimensions must be positive"
    if dtype not in SUPPORTED_DTYPES:
        return False, f"dtype={dtype!r} is not bf16/fp16"
    if head_size != 128:
        return False, f"paged dense-prefill requires head_size=128, got {head_size}"
    if hq % hkv:
        return False, f"Hq={hq} is not divisible by Hkv={hkv}"
    if sq != skv:
        return False, "paged dense-prefill requires self-attention with Sq=Skv"
    if mask != "swin" or window <= 0:
        return False, "paged dense-prefill requires a positive sliding window"
    if block_size not in {16, 32, 64} or skv % block_size:
        return False, (
            f"paged dense-prefill requires block 16/32/64 dividing Skv, "
            f"got block={block_size} Skv={skv}"
        )
    if bool(case.get("varlen")) or bool(case.get("doc_mask")):
        return False, "varlen/doc-mask attention is unsupported"
    if bool(case.get("has_sinks")):
        return False, "attention sinks are unsupported"
    if bool(case.get("has_alibi")):
        return False, "ALiBi is unsupported"
    if bool(case.get("has_output_scale")):
        return False, "output scaling is unsupported"
    if softcap:
        return False, "softcap is unsupported"
    return True, "ok"


@dataclass(frozen=True)
class DenseAPI:
    state: str  # available | unavailable | error
    reason: str
    AttentionRequest: Any = None
    dispatch_attention: Callable[..., Any] | None = None
    dense_spec_for_request: Callable[..., Any] | None = None
    run_attention_dense_torch: Callable[..., Any] | None = None
    paged_available: bool = False
    paged_reason: str = "paged dense API unavailable"


def resolve_dense_api(
    arch: str = "",
    *,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> DenseAPI:
    """Feature-detect the staged source; never compare commit hashes."""
    try:
        dispatch_spec = find_spec("dispatch.attention")
    except ModuleNotFoundError:
        dispatch_spec = None
    if dispatch_spec is None:
        return DenseAPI("unavailable", "source has no dispatch.attention module")

    try:
        dispatch_module = import_module("dispatch.attention")
    except Exception as error:  # implementation exists but is broken
        return DenseAPI(
            "error", f"dispatch.attention import failed: {firstline(error)}"
        )

    required = ("AttentionRequest", "dispatch_attention")
    missing = [name for name in required if not hasattr(dispatch_module, name)]
    if missing:
        return DenseAPI("unavailable", "source lacks dense API: " + ", ".join(missing))

    try:
        kernels_module = import_module("kernels")
    except ModuleNotFoundError as error:
        if error.name == "kernels":
            return DenseAPI("unavailable", "source has no top-level kernels module")
        return DenseAPI("error", f"kernels import failed: {firstline(error)}")
    except Exception as error:
        return DenseAPI("error", f"kernels import failed: {firstline(error)}")
    if arch == "gfx942":
        dispatch_arch = getattr(dispatch_module, "gfx942", None)
        kernels_arch = getattr(kernels_module, "attention_dense_gfx942", None)
        spec_builder = getattr(dispatch_arch, "dense_spec_for_request", None)
        runner = getattr(kernels_arch, "run_attention_dense_torch", None)
        missing = []
        if spec_builder is None:
            missing.append("dispatch.attention.gfx942.dense_spec_for_request")
        if runner is None:
            missing.append("kernels.attention_dense_gfx942.run_attention_dense_torch")
        if missing:
            return DenseAPI(
                "unavailable", "source lacks gfx942 dense API: " + ", ".join(missing)
            )
        spec_type = getattr(kernels_arch, "AttentionDenseSpec", None)
    else:
        spec_builder = getattr(dispatch_module, "dense_spec_for_request", None)
        runner = getattr(kernels_module, "run_attention_dense_torch", None)
        missing = []
        if spec_builder is None:
            missing.append("dispatch.attention.dense_spec_for_request")
        if runner is None:
            missing.append("kernels.run_attention_dense_torch")
        if missing:
            return DenseAPI(
                "unavailable", "source lacks dense API: " + ", ".join(missing)
            )
        spec_type = getattr(kernels_module, "AttentionDenseSpec", None)

    paged_available = False
    paged_reason = "paged dense-prefill requires gfx950"
    if arch == "gfx950":
        paged_fields = {
            "paged",
            "persistent",
            "sliding_window",
            "block_size",
            "num_kv_blocks",
        }
        fields = set(getattr(spec_type, "__dataclass_fields__", {}))
        missing_fields = sorted(paged_fields - fields)
        try:
            runner_parameters = set(signature(runner).parameters)
        except (TypeError, ValueError):
            runner_parameters = set()
        missing_parameters = sorted(
            {"block_tables", "kv_lens", "validate_paged"} - runner_parameters
        )
        if missing_fields:
            paged_reason = "source lacks paged dense spec fields: " + ", ".join(
                missing_fields
            )
        elif missing_parameters:
            paged_reason = "source lacks paged dense runner arguments: " + ", ".join(
                missing_parameters
            )
        else:
            paged_available = True
            paged_reason = "ok"

    return DenseAPI(
        "available",
        "ok",
        AttentionRequest=dispatch_module.AttentionRequest,
        dispatch_attention=dispatch_module.dispatch_attention,
        dense_spec_for_request=spec_builder,
        run_attention_dense_torch=runner,
        paged_available=paged_available,
        paged_reason=paged_reason,
    )


def load_resolved_sha(artifacts_path: str) -> str:
    try:
        manifest = json.loads((Path(artifacts_path) / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown"
    value = manifest.get("resolved_sha")
    return str(value) if value else "unknown"


def attention_flops(case: dict) -> float:
    batch, sq, skv = int(case["B"]), int(case["Sq"]), int(case["Skv"])
    hq, head_size = int(case["Hq"]), int(case["D"])
    window = int(case.get("window", 0) or 0)
    if str(case.get("mask", "causal")).lower() == "swin" and window > 0:
        offset = skv - sq
        pairs = sum(min(index + offset + 1, window) for index in range(sq))
        return 4.0 * batch * hq * head_size * pairs
    flops = 4.0 * batch * hq * sq * skv * head_size
    return flops * 0.5 if case.get("mask", "causal") == "causal" else flops


def timing_stats(timing: dict, case: dict, warmup: int, timed: int) -> dict:
    flops = attention_flops(case)
    us = float(timing["median_ms"]) * 1e3
    return {
        "timing": {
            "benchmark_iteration_count": len(timing["benchmark_iterations_ms"]),
            "warmup_executions_per_iteration": warmup,
            "timed_executions_per_iteration": timed,
            "excluded_initial_iterations": timing["excluded_initial_iterations"],
            "benchmark_iterations": [
                {
                    "amortized_us": value * 1e3,
                    "tflops": flops / ((value * 1e3) / 1e6) / 1e12,
                }
                for value in timing["benchmark_iterations_ms"]
            ],
        },
        "us": us,
        "spread_pct": 100.0
        * (timing["max_ms"] - timing["min_ms"])
        / max(timing["median_ms"], 1e-9),
        "tflops": flops / (us / 1e6) / 1e12,
    }


def make_attention_request(api: DenseAPI, case: dict, arch: str, num_sms: int):
    """Build the upstream request without imposing a local architecture gate."""
    fields = dict(
        batch=int(case["B"]),
        nhead_q=int(case["Hq"]),
        nhead_k=int(case["Hkv"]),
        seqlen_q=int(case["Sq"]),
        seqlen_k=int(case["Skv"]),
        hdim_q=int(case["D"]),
        hdim_v=int(case["D"]),
        arch=arch,
        mask_type=1 if case.get("mask", "causal") in {"causal", "swin"} else 0,
        kv_block_size=int(case.get("block", 64)),
        dtype=str(case["dtype"]).lower(),
        algorithm="attention_dense",
    )
    sliding_window = int(case.get("window") or 0)
    if sliding_window:
        fields["sliding_window"] = sliding_window
    # rocKE renamed the device-CU knob num_sms -> num_cus (a no-alias terminology
    # cutover). Try the current name, fall back to the old, so this lane works
    # across the rename boundary when benchmarking arbitrary rocKE SHAs.
    try:
        return api.AttentionRequest(num_cus=num_sms, **fields)
    except TypeError:
        return api.AttentionRequest(num_sms=num_sms, **fields)


def select_dense_spec(api: DenseAPI, request):
    """Obtain the launch-ready spec after upstream accepted the request."""
    return api.dense_spec_for_request(request)


def select_paged_dense_spec(api: DenseAPI, request, case: dict):
    """Apply the kernel-only paged fields to rocKE's tuned gfx950 base spec."""
    base = select_dense_spec(api, request)
    block_size = int(case["block"])
    return replace(
        base,
        paged=True,
        persistent=False,
        sliding_window=int(case["window"]),
        block_size=block_size,
        num_kv_blocks=int(case["Skv"]) // block_size,
    )


def paged_kv_inputs(torch, k, v, case: dict):
    """Return paged cache views plus the deterministic logical-to-physical map."""
    block_size = int(case["block"])
    num_blocks = int(case["Skv"]) // block_size
    hkv, head_size = int(case["Hkv"]), int(case["D"])
    k_cache = k.reshape(num_blocks, block_size, hkv, head_size).contiguous()
    v_cache = v.reshape(num_blocks, block_size, hkv, head_size).contiguous()
    block_tables = (
        torch.arange(num_blocks, device="cuda", dtype=torch.int32)
        .reshape(1, num_blocks)
        .contiguous()
    )
    kv_lens = torch.tensor([int(case["Skv"])], device="cuda", dtype=torch.int32)
    return k_cache, v_cache, block_tables, kv_lens


def launch_dense(
    api: DenseAPI,
    *,
    spec,
    q,
    k,
    v,
    out,
    scale,
    stream,
    arch,
    block_tables=None,
    kv_lens=None,
):
    """Forward the detected architecture to the upstream public runner."""
    arguments = dict(
        spec=spec,
        q=q,
        k=k,
        v=v,
        out=out,
        scale=scale,
        stream=stream,
        arch=arch,
    )
    if block_tables is not None or kv_lens is not None:
        arguments.update(
            block_tables=block_tables,
            kv_lens=kv_lens,
            validate_paged=False,
        )
    return api.run_attention_dense_torch(**arguments)


def selected_kernel_name(dispatch_result, dense_spec) -> str:
    dispatched_spec = getattr(dispatch_result, "spec", None)
    override = getattr(dispatched_spec, "kernel_name_override", None)
    return str(override) if override else str(dense_spec.kernel_name())


def main() -> int:
    shard = os.environ["ATTN_SHARD"]
    output = os.environ["ATTN_OUT_C"]
    arch = os.environ.get("ATTN_ARCH", "")
    artifacts_path = os.environ.get(
        "ARTIFACTS_PATH", "/artifacts/rocke_attention_comparison"
    )
    benchmark_iterations = int(os.environ["ATTN_BENCHMARK_ITERATIONS"])
    timed_executions = int(os.environ["ATTN_TIMED_EXECUTIONS"])
    warmup_executions = int(os.environ["ATTN_WARMUP_EXECUTIONS"])
    num_sms = int(os.environ["ATTN_NUM_SMS"])
    source_sha = load_resolved_sha(artifacts_path)
    if os.environ.get("ROCKE_CUSTOM_COMPILER") == "1":
        from rocke.runtime import comgr as _rocke_comgr

        _rocke_comgr._resolve_lib()
    try:
        progress_fd = int(os.environ.get("ATTN_PROGRESS_FD", "-1"))
        if progress_fd < 0:
            raise OSError("progress fd disabled")
        progress_stream = os.fdopen(os.dup(progress_fd), "w", buffering=1)
    except (OSError, ValueError):
        progress_stream = sys.stdout

    def progress(message: str) -> None:
        progress_stream.write(message + "\n")
        progress_stream.flush()

    cases = []
    with open(shard) as source:
        for line in source:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and stripped.startswith("attention ")
            ):
                cases.append(json.loads(stripped[len("attention ") :]))

    api = resolve_dense_api(arch)
    print(f"rocke_dense API state={api.state} reason={api.reason} source={source_sha}")
    if api.state != "available":
        key = "na" if api.state == "unavailable" else "error"
        with open(output, "w") as destination:
            for case in cases:
                destination.write(
                    json.dumps(
                        {
                            "idx": case["idx"],
                            LANE: {
                                key: f"attention_dense {api.reason} in source {source_sha}"
                            },
                        }
                    )
                    + "\n"
                )
                progress(
                    f"[C] idx={case['idx']} status={'na' if key == 'na' else 'error'}"
                )
        return 0

    import torch

    sys.path.insert(0, "/scripts")
    try:
        from attn_reference import reference

        reference_import_note = None
    except Exception as error:
        reference = None
        reference_import_note = f"reference import unavailable: {firstline(error)}"
    try:
        from numerics_utils import sdpa_err_stats

        err_stats_import_note = None
    except Exception as error:
        sdpa_err_stats = None
        err_stats_import_note = (
            f"error statistics import unavailable: {firstline(error)}"
        )
    from rocke.runtime import synchronize_and_release, time_launches

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}

    def measure(call, stream):
        call()
        torch.cuda.synchronize()
        values = []
        for _ in range(benchmark_iterations):
            values.append(
                time_launches(
                    call,
                    warmup=warmup_executions,
                    iters=timed_executions,
                    stream=stream,
                )
            )
        used = values[1:] if len(values) > 1 else values
        ordered = sorted(used)
        return {
            "benchmark_iterations_ms": values,
            "excluded_initial_iterations": 1 if len(values) > 1 else 0,
            "median_ms": ordered[len(ordered) // 2],
            "min_ms": min(used),
            "max_ms": max(used),
        }

    with open(output, "w") as destination:
        for position, case in enumerate(cases, 1):
            q = k = v = launch_k = launch_v = out = ref = None
            block_tables = kv_lens = None
            result = None
            paged = str(case.get("mask", "causal")).lower() == "swin"
            eligible, reason = (
                classify_paged_case(case, arch) if paged else classify_case(case)
            )
            if not eligible:
                result = {"na": reason}
            elif paged and not api.paged_available:
                result = {"na": api.paged_reason}
            else:
                try:
                    request = make_attention_request(api, case, arch, num_sms)
                    dispatch_result = None
                    if paged:
                        try:
                            dense_spec = select_paged_dense_spec(api, request, case)
                        except (TypeError, ValueError, NotImplementedError) as error:
                            result = {"na": firstline(error)}
                    else:
                        try:
                            dispatch_result = api.dispatch_attention(request)
                        except (ValueError, NotImplementedError) as error:
                            result = {"na": firstline(error)}
                        else:
                            dense_spec = select_dense_spec(api, request)
                    if result is not None:
                        destination.write(
                            json.dumps({"idx": case["idx"], LANE: result}) + "\n"
                        )
                        destination.flush()
                        progress(
                            f"[C] ({position}/{len(cases)}) idx={case['idx']} "
                            f"{case.get('model')}/{case.get('variant')} rocKE-dense=-"
                        )
                        continue

                    dtype = dtype_map[str(case["dtype"]).lower()]
                    batch = int(case["B"])
                    sq, skv = int(case["Sq"]), int(case["Skv"])
                    hq, hkv = int(case["Hq"]), int(case["Hkv"])
                    head_size = int(case["D"])
                    generator = torch.Generator(device="cuda").manual_seed(0)
                    q = (
                        torch.randn(
                            batch,
                            sq,
                            hq,
                            head_size,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )
                        * 0.3
                    )
                    k = (
                        torch.randn(
                            batch,
                            skv,
                            hkv,
                            head_size,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )
                        * 0.3
                    )
                    v = (
                        torch.randn(
                            batch,
                            skv,
                            hkv,
                            head_size,
                            device="cuda",
                            dtype=dtype,
                            generator=generator,
                        )
                        * 0.3
                    )
                    launch_k, launch_v = k, v
                    if paged:
                        launch_k, launch_v, block_tables, kv_lens = paged_kv_inputs(
                            torch, k, v, case
                        )
                    out = torch.empty_like(q)
                    scale = 1.0 / math.sqrt(head_size)
                    stream = int(torch.cuda.current_stream().cuda_stream)

                    def call():
                        launch_dense(
                            api,
                            spec=dense_spec,
                            q=q,
                            k=launch_k,
                            v=launch_v,
                            out=out,
                            scale=scale,
                            stream=stream,
                            arch=arch,
                            block_tables=block_tables,
                            kv_lens=kv_lens,
                        )

                    call()
                    torch.cuda.synchronize()
                    ref_note = reference_import_note
                    try:
                        ref = (
                            reference(case, q, k, v) if reference is not None else None
                        )
                    except Exception as error:
                        ref = None
                        ref_note = f"ref unavailable: {firstline(error)}"
                    max_abs = (
                        float((out.float() - ref.float()).abs().max())
                        if ref is not None
                        else None
                    )
                    errors = {"mean_err": None, "rms_err": None}
                    if ref is not None and sdpa_err_stats is not None:
                        try:
                            stats = sdpa_err_stats(ref, out, dtype)
                            errors = {
                                "mean_err": stats["mean"],
                                "rms_err": stats["rms"],
                            }
                        except Exception:
                            pass
                    timing = measure(call, stream)
                    try:
                        synchronize_and_release(stream)
                    except Exception:
                        torch.cuda.synchronize()
                    result = {
                        "path": "attention_dense",
                        "kernel_name": selected_kernel_name(
                            dispatch_result, dense_spec
                        ),
                        "source_sha": source_sha,
                        "max_abs": max_abs,
                        **errors,
                        **timing_stats(
                            timing, case, warmup_executions, timed_executions
                        ),
                    }
                    if ref_note:
                        result["ref_note"] = ref_note
                    if err_stats_import_note:
                        result["err_stats_note"] = err_stats_import_note
                    if max_abs is not None and max_abs > CORRECTNESS_WARNING_THRESHOLD:
                        result["warn"] = (
                            f"max_abs {max_abs:.2e}>{CORRECTNESS_WARNING_THRESHOLD:.0e}"
                        )
                except Exception as error:
                    traceback.print_exc()
                    result = {"error": f"{type(error).__name__}: {firstline(error)}"}
                finally:
                    del q, k, v, launch_k, launch_v, out, ref, block_tables, kv_lens
                    torch.cuda.empty_cache()

            destination.write(json.dumps({"idx": case["idx"], LANE: result}) + "\n")
            destination.flush()
            tflops = (
                f"{result['tflops']:.1f}"
                if isinstance(result, dict)
                and isinstance(result.get("tflops"), (int, float))
                else "-"
            )
            progress(
                f"[C] ({position}/{len(cases)}) idx={case['idx']} "
                f"{case.get('model')}/{case.get('variant')} rocKE-dense={tflops}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
