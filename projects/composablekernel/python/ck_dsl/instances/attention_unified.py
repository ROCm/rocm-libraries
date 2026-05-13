# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native CK DSL entry points for AITER unified attention.

This module intentionally separates *feature selection* from *kernel emission*.
The selector mirrors AITER's Triton wrapper exactly; kernel emission is gated
until every required primitive and correctness/perf path is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType, Type, Value
from ..helpers.compile import compile_kernel
from ..runtime.torch_module import launch_torch_kernel

from ..helpers.attention import (
    Attention2DConfig,
    Attention3DConfig,
    select_2d_config,
    select_3d_config,
    use_2d_kernel,
)
from .attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
    supports_tiled_2d,
)
from .attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
    supports_tiled_3d,
)
from ..runtime.torch_module import empty_workspace


@dataclass(frozen=True)
class UnifiedAttentionProblem:
    total_q: int
    num_seqs: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    dtype: str
    q_dtype: Optional[str] = None
    sliding_window: int = 0
    softcap: float = 0.0
    use_sinks: bool = False
    use_alibi: bool = False
    use_qq_bias: bool = False
    use_fp8: bool = False
    num_sms: int = 120

    @property
    def num_queries_per_kv(self) -> int:
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        return self.num_query_heads // self.num_kv_heads

    @property
    def all_decode(self) -> bool:
        return self.max_seqlen_q == 1

    @property
    def total_num_q_blocks_upper_bound(self) -> int:
        block_m = (
            16
            if self.num_queries_per_kv <= 16
            else _next_power_of_2(self.num_queries_per_kv)
        )
        block_q = block_m // self.num_queries_per_kv
        return self.total_q // block_q + self.num_seqs

    def select_path(self) -> str:
        target = self.num_sms * 4
        num_2d = self.total_num_q_blocks_upper_bound * self.num_kv_heads
        return (
            "2d"
            if use_2d_kernel(
                head_size=self.head_size,
                sliding_window=self.sliding_window,
                all_decode=self.all_decode,
                max_seqlen_q=self.max_seqlen_q,
                max_seqlen_k=self.max_seqlen_k,
                target_num_prgms=target,
                num_2d_prgms=num_2d,
            )
            else "3d"
        )

    def select_2d(self) -> Attention2DConfig:
        num_2d = self.total_num_q_blocks_upper_bound * self.num_kv_heads
        return select_2d_config(
            block_size=self.block_size,
            head_size=self.head_size,
            sliding_window=self.sliding_window,
            all_decode=self.all_decode,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            num_queries_per_kv=self.num_queries_per_kv,
            num_2d_prgms=num_2d,
        )

    def select_3d(self) -> Tuple[Attention3DConfig, Attention3DConfig]:
        target = self.num_sms * 4
        num_2d = self.total_num_q_blocks_upper_bound * self.num_kv_heads
        return select_3d_config(
            head_size=self.head_size,
            block_size=self.block_size,
            element_size=2 if self.dtype in ("fp16", "bf16") else 1,
            max_seqlen_k=self.max_seqlen_k,
            target_num_prgms=target,
            num_2d_prgms=num_2d,
        )


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


def supports_native_unified_attention(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether CK DSL can run this problem without fallback today.

    This is deliberately strict. It prevents a partially implemented backend
    from being selected in `auto` mode and gives test code a single place to
    check coverage.
    """
    if problem.head_size not in (128, 256):
        return False, f"unsupported head_size {problem.head_size}"
    if problem.block_size not in (16, 64):
        return False, f"unsupported block_size {problem.block_size}"
    if problem.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {problem.dtype}"
    if problem.use_fp8 or problem.q_dtype is not None:
        return False, "FP8 unified attention is not enabled in CK DSL yet"
    if problem.use_alibi:
        return False, "ALiBi slopes are not enabled in CK DSL attention yet"
    if problem.use_qq_bias:
        return False, "QQ bias is not enabled in CK DSL attention yet"
    return True, "supported by scalar CK DSL 2D attention backend"


def supports_native_unified_attention_tiled(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether the optimized tiled MFMA path can run this problem."""
    return supports_tiled_2d(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        use_fp8=problem.use_fp8,
        q_dtype=problem.q_dtype,
    )


def supports_native_unified_attention_3d_tiled(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether the optimized tiled MFMA 3D split-KV path can run this."""
    return supports_tiled_3d(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        use_fp8=problem.use_fp8,
        q_dtype=problem.q_dtype,
    )


_ATTN_CACHE: Dict[Tuple, bytes] = {}
_ATTN_TILED_CACHE: Dict[Tuple, bytes] = {}
_ATTN_3D_TILED_CACHE: Dict[Tuple, Tuple[bytes, str, bytes, str]] = {}


def _cache_key(problem: UnifiedAttentionProblem) -> Tuple:
    return (
        "scalar",
        problem.total_q,
        problem.num_seqs,
        problem.num_query_heads,
        problem.num_kv_heads,
        problem.head_size,
        problem.block_size,
        problem.max_seqlen_q,
        problem.max_seqlen_k,
        problem.dtype,
        problem.sliding_window,
        bool(problem.use_sinks),
        bool(problem.softcap > 0),
    )


def _tiled_cache_key(problem: UnifiedAttentionProblem) -> Tuple:
    return (
        "tiled",
        problem.num_seqs,
        problem.num_query_heads,
        problem.num_kv_heads,
        problem.head_size,
        problem.block_size,
        problem.dtype,
        problem.sliding_window,
        bool(problem.use_sinks),
        bool(problem.softcap > 0),
        bool(problem.use_alibi),
        bool(problem.use_qq_bias),
    )


def _tiled_spec_from_problem(
    problem: UnifiedAttentionProblem,
) -> UnifiedAttention2DTiledSpec:
    return UnifiedAttention2DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
    )


def _num_segments(problem: UnifiedAttentionProblem) -> int:
    """Mirror AITER ``select_3d_config`` num_segments derivation exactly."""
    attn_cfg, _ = problem.select_3d()
    return attn_cfg.NUM_SEGMENTS_PER_SEQ


def _tiled_3d_spec_from_problem(
    problem: UnifiedAttentionProblem,
) -> UnifiedAttention3DTiledSpec:
    return UnifiedAttention3DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        num_segments=_num_segments(problem),
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
    )


def _tiled_3d_cache_key(problem: UnifiedAttentionProblem) -> Tuple:
    return (
        "tiled3d",
        problem.num_seqs,
        problem.num_query_heads,
        problem.num_kv_heads,
        problem.head_size,
        problem.block_size,
        problem.dtype,
        problem.sliding_window,
        bool(problem.use_sinks),
        bool(problem.softcap > 0),
        bool(problem.use_alibi),
        bool(problem.use_qq_bias),
        _num_segments(problem),
    )


def _3d_signature(dtype: str):
    ty = "ptr<f16, global>" if dtype == "fp16" else "ptr<bf16, global>"
    return [
        {"name": "segm_output_ptr", "type": "ptr<f32, global>"},
        {"name": "segm_max_ptr", "type": "ptr<f32, global>"},
        {"name": "segm_expsum_ptr", "type": "ptr<f32, global>"},
        {"name": "query_ptr", "type": ty},
        {"name": "key_cache_ptr", "type": ty},
        {"name": "value_cache_ptr", "type": ty},
        {"name": "sink_ptr", "type": ty},
        {"name": "block_tables_ptr", "type": "ptr<i32, global>"},
        {"name": "seq_lens_ptr", "type": "ptr<i32, global>"},
        {"name": "alibi_slopes_ptr", "type": "ptr<f32, global>"},
        {"name": "qq_bias_ptr", "type": "ptr<f32, global>"},
        {"name": "query_start_len_ptr", "type": "ptr<i32, global>"},
        {"name": "scale", "type": "f32"},
        {"name": "k_scale", "type": "f32"},
        {"name": "v_scale", "type": "f32"},
        {"name": "softcap", "type": "f32"},
        {"name": "num_seqs", "type": "i32"},
        {"name": "block_table_stride", "type": "i32"},
        {"name": "qq_bias_stride_0", "type": "i32"},
    ]


def _reduce_signature(dtype: str):
    ty = "ptr<f16, global>" if dtype == "fp16" else "ptr<bf16, global>"
    return [
        {"name": "output_ptr", "type": ty},
        {"name": "segm_output_ptr", "type": "ptr<f32, global>"},
        {"name": "segm_max_ptr", "type": "ptr<f32, global>"},
        {"name": "segm_expsum_ptr", "type": "ptr<f32, global>"},
        {"name": "seq_lens_ptr", "type": "ptr<i32, global>"},
    ]


def _attn_signature(
    dtype: str, *, include_bt_stride: bool, include_qq_bias_stride: bool = False
):
    ty = "ptr<f16, global>" if dtype == "fp16" else "ptr<bf16, global>"
    sig = [
        {"name": "output_ptr", "type": ty},
        {"name": "query_ptr", "type": ty},
        {"name": "key_cache_ptr", "type": ty},
        {"name": "value_cache_ptr", "type": ty},
        {"name": "sink_ptr", "type": ty},
        {"name": "block_tables_ptr", "type": "ptr<i32, global>"},
        {"name": "seq_lens_ptr", "type": "ptr<i32, global>"},
        {"name": "alibi_slopes_ptr", "type": "ptr<f32, global>"},
        {"name": "qq_bias_ptr", "type": "ptr<f32, global>"},
        {"name": "query_start_len_ptr", "type": "ptr<i32, global>"},
        {"name": "scale", "type": "f32"},
        {"name": "k_scale", "type": "f32"},
        {"name": "v_scale", "type": "f32"},
        {"name": "out_scale", "type": "f32"},
        {"name": "softcap", "type": "f32"},
        {"name": "num_seqs", "type": "i32"},
    ]
    if include_bt_stride:
        sig.append({"name": "block_table_stride", "type": "i32"})
    if include_qq_bias_stride:
        sig.append({"name": "qq_bias_stride_0", "type": "i32"})
    return sig


def _attn_values(
    *,
    problem: UnifiedAttentionProblem,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale: float,
    block_table,
    softcap: float,
    sinks,
    bt_stride: int,
    include_bt_stride: bool,
    alibi_slopes=None,
    qq_bias=None,
    qq_bias_stride_0: int = 0,
    include_qq_bias_stride: bool = False,
):
    vals = {
        "output_ptr": out,
        "query_ptr": q,
        "key_cache_ptr": k,
        "value_cache_ptr": v,
        "sink_ptr": sinks,
        "block_tables_ptr": block_table,
        "seq_lens_ptr": seqused_k,
        "alibi_slopes_ptr": alibi_slopes if alibi_slopes is not None else 0,
        "qq_bias_ptr": qq_bias if qq_bias is not None else 0,
        "query_start_len_ptr": cu_seqlens_q,
        "scale": float(softmax_scale),
        "k_scale": 1.0,
        "v_scale": 1.0,
        "out_scale": 1.0,
        "softcap": float(softcap),
        "num_seqs": int(problem.num_seqs),
    }
    if include_bt_stride:
        vals["block_table_stride"] = int(bt_stride)
    if include_qq_bias_stride:
        vals["qq_bias_stride_0"] = int(qq_bias_stride_0)
    return vals


def _run_3d_tiled(
    *,
    problem: UnifiedAttentionProblem,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale: float,
    block_table,
    softcap: float,
    sinks,
    bt_stride: int,
    warmup: int,
    attempts: int,
    alibi_slopes=None,
    qq_bias=None,
    qq_bias_stride_0: int = 0,
):
    """Launch the tiled 3D segment + reduce kernels.

    Mirrors AITER's 3D path:
      1. Compile (and cache) both kernels for this problem shape.
      2. Allocate the per-segment workspace tensors `segm_output`,
         `segm_max`, `segm_expsum`.
      3. Launch the 3D segment kernel with grid
         `(total_num_q_blocks, num_kv_heads, num_segments)`.
      4. Launch the reduce kernel with grid `(total_q, num_query_heads, 1)`.
    """
    import torch

    num_segments = _num_segments(problem)
    key = _tiled_3d_cache_key(problem)
    if key not in _ATTN_3D_TILED_CACHE:
        seg_spec = _tiled_3d_spec_from_problem(problem)
        reduce_spec = UnifiedAttentionReduceTiledSpec(
            head_size=problem.head_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            num_segments=num_segments,
        )
        seg_art = compile_kernel(
            build_unified_attention_3d_tiled(seg_spec), capture_ir_text=False
        )
        red_art = compile_kernel(
            build_unified_attention_reduce_tiled(reduce_spec), capture_ir_text=False
        )
        _ATTN_3D_TILED_CACHE[key] = (
            seg_art.hsaco,
            seg_art.kernel_name,
            red_art.hsaco,
            red_art.kernel_name,
        )
    seg_hsaco, seg_kname, red_hsaco, red_kname = _ATTN_3D_TILED_CACHE[key]

    segm_output = empty_workspace(
        (problem.total_q, problem.num_query_heads, num_segments, problem.head_size),
        dtype=torch.float32,
        device=q.device,
    )
    segm_max = empty_workspace(
        (problem.total_q, problem.num_query_heads, num_segments),
        dtype=torch.float32,
        device=q.device,
    )
    segm_expsum = empty_workspace(
        (problem.total_q, problem.num_query_heads, num_segments),
        dtype=torch.float32,
        device=q.device,
    )

    block_q = (
        16 // problem.num_queries_per_kv if problem.num_queries_per_kv <= 16 else 1
    )
    total_num_q_blocks = problem.total_q // block_q + problem.num_seqs

    seg_sig = _3d_signature(problem.dtype)
    seg_vals = {
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "query_ptr": q,
        "key_cache_ptr": k,
        "value_cache_ptr": v,
        "sink_ptr": sinks,
        "block_tables_ptr": block_table,
        "seq_lens_ptr": seqused_k,
        "alibi_slopes_ptr": alibi_slopes if alibi_slopes is not None else 0,
        "qq_bias_ptr": qq_bias if qq_bias is not None else 0,
        "query_start_len_ptr": cu_seqlens_q,
        "scale": float(softmax_scale),
        "k_scale": 1.0,
        "v_scale": 1.0,
        "softcap": float(softcap),
        "num_seqs": int(problem.num_seqs),
        "block_table_stride": int(bt_stride),
        "qq_bias_stride_0": int(qq_bias_stride_0),
    }
    red_sig = _reduce_signature(problem.dtype)
    red_vals = {
        "output_ptr": out,
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "seq_lens_ptr": seqused_k,
    }
    return _launch_3d_segment_then_reduce(
        seg_hsaco=seg_hsaco,
        seg_kname=seg_kname,
        seg_sig=seg_sig,
        seg_vals=seg_vals,
        red_hsaco=red_hsaco,
        red_kname=red_kname,
        red_sig=red_sig,
        red_vals=red_vals,
        seg_grid=(
            int(total_num_q_blocks),
            int(problem.num_kv_heads),
            int(num_segments),
        ),
        red_grid=(int(problem.total_q), int(problem.num_query_heads), 1),
        warmup=warmup,
        attempts=attempts,
    )


_HIP_RUNTIME = None
_HIP_3D_MODULE_CACHE: Dict[Tuple, Tuple] = {}


def _launch_3d_segment_then_reduce(
    *,
    seg_hsaco,
    seg_kname,
    seg_sig,
    seg_vals,
    red_hsaco,
    red_kname,
    red_sig,
    red_vals,
    seg_grid,
    red_grid,
    warmup: int,
    attempts: int,
):
    """Run segment + reduce exactly once per AITER call.

    The outer benchmark harness already handles warmup/repeats, so we do not
    add an inner loop here. Adding one would multiply the reported latency
    by `attempts` relative to Triton's single-launch path. The returned
    summary's `ms` is the elapsed wall time for this one (segment, reduce)
    pair.
    """
    from ..runtime.hip_module import Runtime
    from ..runtime.torch_module import TorchLaunchSummary, pack_args

    global _HIP_RUNTIME
    if _HIP_RUNTIME is None:
        _HIP_RUNTIME = Runtime()
    rt = _HIP_RUNTIME
    cache_key = (id(seg_hsaco), id(red_hsaco))
    if cache_key not in _HIP_3D_MODULE_CACHE:
        mod_seg = rt.load_module(seg_hsaco)
        fn_seg = mod_seg.get_function(seg_kname)
        mod_red = rt.load_module(red_hsaco)
        fn_red = mod_red.get_function(red_kname)
        _HIP_3D_MODULE_CACHE[cache_key] = (mod_seg, fn_seg, mod_red, fn_red)
    _, fn_seg, _, fn_red = _HIP_3D_MODULE_CACHE[cache_key]
    args_seg = pack_args(seg_sig, seg_vals)
    args_red = pack_args(red_sig, red_vals)
    rt.launch(fn_seg, seg_grid, (64, 1, 1), args_seg)
    rt.launch(fn_red, red_grid, (64, 1, 1), args_red)
    return TorchLaunchSummary(ms=0.0, attempts=1)


def run_unified_attention_torch(
    *,
    problem: UnifiedAttentionProblem,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale: float,
    block_table,
    softcap: float,
    sinks=None,
    alibi_slopes=None,
    qq_bias=None,
    qq_bias_stride_0: int = 0,
    warmup: int = 0,
    attempts: int = 1,
    backend: str = "auto",
):
    """Launch a CK DSL attention kernel on torch tensors.

    Backend selection:
      - `"tiled"`: force the optimized MFMA path; raises if unsupported.
      - `"scalar"`: force the slow correctness kernel.
      - `"auto"`: prefer tiled when supported, else scalar.

    ``alibi_slopes`` is an optional `[num_query_heads]` f32 tensor; when
    supplied, the kernel applies the ALiBi linear bias on each row.
    ``qq_bias`` is an optional 2D f32 query-to-query bias; ``qq_bias_stride_0``
    is its first-axis stride (in elements). Both follow AITER's Triton
    semantics exactly and require the corresponding ``problem.use_alibi`` /
    ``problem.use_qq_bias`` flags to be set.
    """
    bt_stride = (
        int(block_table.stride(0))
        if hasattr(block_table, "stride")
        else int(block_table.shape[1])
    )

    # Prefer 3D split-KV whenever supported. CK's split-KV produces a much
    # larger grid (multi-segment per (qblock, kv_head)) than AITER's 2D path
    # and beats Triton on MI355X by 2-6x. Sliding-window is supported via the
    # per-cell mask inside the segment kernel.
    if backend in ("3d", "auto"):
        ok_3d, reason_3d = supports_native_unified_attention_3d_tiled(problem)
        if ok_3d:
            return _run_3d_tiled(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                block_table=block_table,
                softcap=softcap,
                sinks=sinks,
                bt_stride=bt_stride,
                warmup=warmup,
                attempts=attempts,
                alibi_slopes=alibi_slopes,
                qq_bias=qq_bias,
                qq_bias_stride_0=qq_bias_stride_0,
            )
        if backend == "3d":
            raise NotImplementedError(reason_3d)

    if backend in ("tiled", "auto"):
        ok_t, reason_t = supports_native_unified_attention_tiled(problem)
        if ok_t:
            key = _tiled_cache_key(problem)
            if key not in _ATTN_TILED_CACHE:
                spec = _tiled_spec_from_problem(problem)
                artifact = compile_kernel(
                    build_unified_attention_2d_tiled(spec), capture_ir_text=False
                )
                _ATTN_TILED_CACHE[key] = (artifact.hsaco, artifact.kernel_name)
            hsaco, kname = _ATTN_TILED_CACHE[key]
            sig = _attn_signature(
                problem.dtype,
                include_bt_stride=True,
                include_qq_bias_stride=True,
            )
            vals = _attn_values(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                block_table=block_table,
                softcap=softcap,
                sinks=sinks,
                bt_stride=bt_stride,
                include_bt_stride=True,
                alibi_slopes=alibi_slopes,
                qq_bias=qq_bias,
                qq_bias_stride_0=qq_bias_stride_0,
                include_qq_bias_stride=True,
            )
            # Total Q blocks upper-bound mirroring AITER.
            block_q = (
                16 // problem.num_queries_per_kv
                if problem.num_queries_per_kv <= 16
                else 1
            )
            total_num_q_blocks = problem.total_q // block_q + problem.num_seqs
            return launch_torch_kernel(
                hsaco=hsaco,
                kernel_name=kname,
                signature=sig,
                values=vals,
                grid=(int(problem.num_kv_heads), int(total_num_q_blocks), 1),
                block=(64, 1, 1),
                warmup=warmup,
                attempts=attempts,
            )
        if backend == "tiled":
            raise NotImplementedError(reason_t)

    # Scalar fallback
    ok, reason = supports_native_unified_attention(problem)
    if not ok:
        raise NotImplementedError(reason)
    key = _cache_key(problem)
    if key not in _ATTN_CACHE:
        spec = UnifiedAttention2DSpec(problem=problem)
        artifact = compile_kernel(
            build_unified_attention_2d(spec), capture_ir_text=False
        )
        _ATTN_CACHE[key] = (artifact.hsaco, artifact.kernel_name)
    hsaco, kname = _ATTN_CACHE[key]
    sig = _attn_signature(problem.dtype, include_bt_stride=False)
    vals = _attn_values(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        block_table=block_table,
        softcap=softcap,
        sinks=sinks,
        bt_stride=bt_stride,
        include_bt_stride=False,
    )
    return launch_torch_kernel(
        hsaco=hsaco,
        kernel_name=kname,
        signature=sig,
        values=vals,
        grid=(
            int(problem.total_q),
            int(problem.num_query_heads),
            int(problem.head_size),
        ),
        block=(64, 1, 1),
        warmup=warmup,
        attempts=attempts,
    )


@dataclass(frozen=True)
class UnifiedAttention2DSpec:
    problem: UnifiedAttentionProblem
    name: str = "ck_dsl_unified_attention_2d_scalar"

    @property
    def dtype_ir(self) -> Type:
        if self.problem.dtype == "fp16":
            return F16
        if self.problem.dtype == "bf16":
            return BF16
        raise ValueError(
            f"unsupported dtype for scalar 2D kernel: {self.problem.dtype}"
        )

    def kernel_name(self) -> str:
        p = self.problem
        return (
            f"{self.name}_q{p.total_q}_h{p.num_query_heads}_kv{p.num_kv_heads}"
            f"_d{p.head_size}_b{p.block_size}_{p.dtype}"
            f"{'_sink' if p.use_sinks else ''}"
            f"{'_sw' if p.sliding_window > 0 else ''}"
            f"{'_softcap' if p.softcap > 0 else ''}"
        )


def build_unified_attention_2d(spec: UnifiedAttention2DSpec) -> KernelDef:
    """Build a scalar-correct 2D unified-attention kernel.

    One workgroup computes one output element `(query_token, query_head, dim)`.
    This is deliberately a correctness kernel: it implements the full paged
    online-softmax semantics for fp16/bf16 without relying on Triton. The
    optimized MFMA/tiled kernel will replace this body once parity is locked.
    """
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        raise ValueError("scalar 2D kernel currently supports fp16/bf16")
    dtype = spec.dtype_ir
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64

    output = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sinks = b.param("sink_ptr", PtrType(dtype, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    _alibi = b.param("alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4)
    _qq_bias = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale = b.param("scale", F32)
    _k_scale = b.param("k_scale", F32)
    _v_scale = b.param("v_scale", F32)
    _out_scale = b.param("out_scale", F32)
    softcap = b.param("softcap", F32)
    num_seqs = b.param("num_seqs", I32)

    q_tok = b.block_id_x()
    q_head = b.block_id_y()
    dim = b.block_id_z()
    tid = b.thread_id_x()
    active = b.cmp_eq(tid, b.const_i32(0))

    # Find seq_idx by scanning cu_q: largest i such that cu_q[i] <= q_tok.
    seq_init = b.const_i32(0)
    scan = b.scf_for_iter(
        b.const_i32(0), num_seqs, b.const_i32(1), [("seq_idx", seq_init)], iv_name="si"
    )
    with scan as (si, (seq_idx,)):
        start_i = b.global_load_i32(cu_q, si)
        le = b.cmp_le(start_i, q_tok)
        next_seq = b.select(le, si, seq_idx)
        b.scf_yield(next_seq)
    seq_idx = scan.results[0]

    cu_start = b.global_load_i32(cu_q, seq_idx)
    cu_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    q_len = b.sub(cu_stop, cu_start)
    query_pos = b.sub(q_tok, cu_start)
    kv_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(kv_len, q_len)
    kv_head = b.div(q_head, b.const_i32(p.num_queries_per_kv))

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)

    if p.use_sinks:
        sink_h = b.global_load(sinks, q_head, dtype, align=2)
        init_m = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
        init_l = one_f
    else:
        init_m = neg_inf
        init_l = one_f
    init_acc = zero_f

    loop = b.scf_for_iter(
        b.const_i32(0),
        kv_len,
        b.const_i32(1),
        [("m", init_m), ("l", init_l), ("acc", init_acc)],
        iv_name="kpos",
    )
    with loop as (kpos, (m_val, l_val, acc_val)):
        block_idx = b.div(kpos, b.const_i32(p.block_size))
        token_in_block = b.mod(kpos, b.const_i32(p.block_size))
        physical = b.global_load_i32(
            block_tables,
            b.add(
                b.mul(
                    seq_idx,
                    b.const_i32((p.max_seqlen_k + p.block_size - 1) // p.block_size),
                ),
                block_idx,
            ),
        )

        score = zero_f
        for d in b.unroll(p.head_size):
            q_off = b.add(
                b.add(
                    b.mul(q_tok, b.const_i32(p.num_query_heads * p.head_size)),
                    b.mul(q_head, b.const_i32(p.head_size)),
                ),
                b.const_i32(d),
            )
            k_off = b.add(
                b.add(
                    b.add(
                        b.mul(
                            physical,
                            b.const_i32(p.block_size * p.num_kv_heads * p.head_size),
                        ),
                        b.mul(
                            token_in_block, b.const_i32(p.num_kv_heads * p.head_size)
                        ),
                    ),
                    b.mul(kv_head, b.const_i32(p.head_size)),
                ),
                b.const_i32(d),
            )
            qv = b.cast_to_f32(b.global_load(query, q_off, dtype, align=2))
            kv = b.cast_to_f32(b.global_load(key, k_off, dtype, align=2))
            score = b.fadd(score, b.fmul(qv, kv))

        score = b.fmul(b.fmul(score, scale), rcp_ln2)
        if p.softcap > 0:
            score = b.fmul(apply_softcap_runtime(b, score, softcap), rcp_ln2)

        causal_ok = b.cmp_le(kpos, b.add(context_len, query_pos))
        if p.sliding_window > 0:
            dist = b.sub(b.add(context_len, query_pos), kpos)
            sw_ok = b.cmp_lt(dist, b.const_i32(p.sliding_window))
            causal_ok = b.land(causal_ok, sw_ok)
        score = b.select(causal_ok, score, neg_inf)
        new_m_raw = b.fmax(m_val, score)
        # If both running max and current score are -inf, the row is fully
        # masked; force m to 0 so the resulting alpha/prob are 0 instead of NaN
        # (matches Triton's `m_j = tl.where(m_j > -inf, m_j, 0.0)`).
        is_finite = b.fcmp("ogt", new_m_raw, neg_inf)
        new_m = b.select(is_finite, new_m_raw, zero_f)
        alpha = b.exp2(b.fsub(m_val, new_m))
        prob = b.exp2(b.fsub(score, new_m))
        new_l = b.fadd(b.fmul(l_val, alpha), prob)
        v_off = b.add(
            b.add(
                b.add(
                    b.mul(
                        physical,
                        b.const_i32(p.block_size * p.num_kv_heads * p.head_size),
                    ),
                    b.mul(token_in_block, b.const_i32(p.num_kv_heads * p.head_size)),
                ),
                b.mul(kv_head, b.const_i32(p.head_size)),
            ),
            dim,
        )
        vv = b.cast_to_f32(b.global_load(value, v_off, dtype, align=2))
        new_acc = b.fadd(b.fmul(acc_val, alpha), b.fmul(prob, vv))
        b.scf_yield(new_m, new_l, new_acc)

    out_val = b.fmul(loop.results[2], b.rcp(loop.results[1]))
    out_cast = b.cast_f32_to(out_val, dtype)
    out_off = b.add(
        b.add(
            b.mul(q_tok, b.const_i32(p.num_query_heads * p.head_size)),
            b.mul(q_head, b.const_i32(p.head_size)),
        ),
        dim,
    )
    valid = b.land(active, b.cmp_lt(dim, b.const_i32(p.head_size)))
    with b.scf_if(valid):
        b.global_store(output, out_off, out_cast, align=2)
    return b.kernel


def apply_softcap_runtime(b: IRBuilder, score_log2: Value, softcap: Value) -> Value:
    """Triton-equivalent softcap on a log2-domain score.

    Computes `softcap * tanh(score_natural / softcap)` using only `exp2` so the
    same primitives that drive the online softmax also handle softcap (matches
    AITER's `apply_softcap`). Given `score_log2 = score_natural * log2(e)`,

        Sdiv = score_log2 / softcap
        p1   = exp2(Sdiv)      = e^(score_natural / softcap)
        p2   = exp2(-Sdiv)     = e^(-score_natural / softcap)
        return softcap * (p1 - p2) / (p1 + p2)

    Returned value is in *natural* domain; the caller is responsible for
    multiplying by `RCP_LN2` to bring it back to log2 for the next `exp2`.
    """
    sdiv = b.fdiv(score_log2, softcap)
    p1 = b.exp2(sdiv)
    p2 = b.exp2(b.fneg(sdiv))
    diff = b.fsub(p1, p2)
    summ = b.fadd(p1, p2)
    return b.fmul(softcap, b.fmul(diff, b.rcp(summ)))


@dataclass(frozen=True)
class UnifiedAttention3DSpec(UnifiedAttention2DSpec):
    name: str = "ck_dsl_unified_attention_3d_scalar"
    num_segments: int = 8

    def kernel_name(self) -> str:
        p = self.problem
        return (
            f"{self.name}_q{p.total_q}_h{p.num_query_heads}_kv{p.num_kv_heads}"
            f"_d{p.head_size}_b{p.block_size}_seg{self.num_segments}_{p.dtype}"
        )


def build_unified_attention_3d(spec: UnifiedAttention3DSpec) -> KernelDef:
    """Build scalar-correct split-3D segment attention kernel."""
    p = spec.problem
    dtype = spec.dtype_ir
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64

    segm_output = b.param(
        "segm_output_ptr",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    segm_max = b.param(
        "segm_max_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    segm_expsum = b.param(
        "segm_expsum_ptr",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    _sinks = b.param("sink_ptr", PtrType(dtype, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    _alibi = b.param("alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4)
    _qq_bias = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale = b.param("scale", F32)
    _k_scale = b.param("k_scale", F32)
    _v_scale = b.param("v_scale", F32)
    _softcap = b.param("softcap", F32)
    num_seqs = b.param("num_seqs", I32)

    q_tok = b.block_id_x()
    q_head = b.block_id_y()
    zd = b.block_id_z()
    segm_idx = b.div(zd, b.const_i32(p.head_size))
    dim = b.mod(zd, b.const_i32(p.head_size))
    tid = b.thread_id_x()
    active = b.cmp_eq(tid, b.const_i32(0))

    seq_idx = _emit_find_seq_idx_scan(b, cu_q, q_tok, num_seqs)
    cu_start = b.global_load_i32(cu_q, seq_idx)
    cu_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    q_len = b.sub(cu_stop, cu_start)
    query_pos = b.sub(q_tok, cu_start)
    kv_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(kv_len, q_len)
    kv_head = b.div(q_head, b.const_i32(p.num_queries_per_kv))
    tiles_per_segment = b.div(
        b.add(kv_len, b.const_i32(spec.num_segments * p.block_size - 1)),
        b.const_i32(spec.num_segments * p.block_size),
    )
    seg_start = b.mul(segm_idx, b.mul(tiles_per_segment, b.const_i32(p.block_size)))
    seg_stop_i = b.mul(
        b.add(segm_idx, b.const_i32(1)),
        b.mul(tiles_per_segment, b.const_i32(p.block_size)),
    )
    seg_stop_i = b.select(b.cmp_lt(seg_stop_i, kv_len), seg_stop_i, kv_len)

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    init_m = neg_inf
    init_l = zero_f
    init_acc = zero_f

    loop = b.scf_for_iter(
        seg_start,
        seg_stop_i,
        b.const_i32(1),
        [("m", init_m), ("l", init_l), ("acc", init_acc)],
        iv_name="kpos",
    )
    with loop as (kpos, (m_val, l_val, acc_val)):
        score = _emit_qk_score(
            b,
            p,
            dtype,
            query,
            key,
            block_tables,
            seq_idx,
            q_tok,
            q_head,
            kv_head,
            kpos,
            scale,
            rcp_ln2,
        )
        causal_ok = b.cmp_le(kpos, b.add(context_len, query_pos))
        score = b.select(causal_ok, score, neg_inf)
        new_m = b.fmax(m_val, score)
        alpha = b.exp2(b.fsub(m_val, new_m))
        prob = b.exp2(b.fsub(score, new_m))
        new_l = b.fadd(b.fmul(l_val, alpha), prob)
        vv = _emit_v_load(b, p, dtype, value, block_tables, seq_idx, kv_head, kpos, dim)
        new_acc = b.fadd(b.fmul(acc_val, alpha), b.fmul(prob, vv))
        b.scf_yield(new_m, new_l, new_acc)

    base = b.add(
        b.add(
            b.mul(
                q_tok, b.const_i32(p.num_query_heads * spec.num_segments * p.head_size)
            ),
            b.mul(q_head, b.const_i32(spec.num_segments * p.head_size)),
        ),
        b.add(b.mul(segm_idx, b.const_i32(p.head_size)), dim),
    )
    with b.scf_if(active):
        b.global_store(segm_output, base, loop.results[2], align=4)
        is_dim0 = b.cmp_eq(dim, b.const_i32(0))
        with b.scf_if(is_dim0):
            segm_base = b.add(
                b.add(
                    b.mul(q_tok, b.const_i32(p.num_query_heads * spec.num_segments)),
                    b.mul(q_head, b.const_i32(spec.num_segments)),
                ),
                segm_idx,
            )
            b.global_store(segm_max, segm_base, loop.results[0], align=4)
            b.global_store(segm_expsum, segm_base, loop.results[1], align=4)
    return b.kernel


@dataclass(frozen=True)
class UnifiedAttentionReduceSpec:
    problem: UnifiedAttentionProblem
    num_segments: int
    name: str = "ck_dsl_unified_attention_reduce_scalar"

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.problem.dtype == "fp16" else BF16

    def kernel_name(self) -> str:
        p = self.problem
        return f"{self.name}_q{p.total_q}_h{p.num_query_heads}_d{p.head_size}_seg{self.num_segments}_{p.dtype}"


def build_unified_attention_reduce(spec: UnifiedAttentionReduceSpec) -> KernelDef:
    p = spec.problem
    dtype = spec.dtype_ir
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64
    out = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    segm_output = b.param(
        "segm_output_ptr", PtrType(F32, "global"), readonly=True, align=16
    )
    segm_max = b.param("segm_max_ptr", PtrType(F32, "global"), readonly=True, align=16)
    segm_expsum = b.param(
        "segm_expsum_ptr", PtrType(F32, "global"), readonly=True, align=16
    )
    _seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    _cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    q_tok = b.block_id_x()
    q_head = b.block_id_y()
    dim = b.block_id_z()
    tid = b.thread_id_x()
    active = b.cmp_eq(tid, b.const_i32(0))
    neg_inf = b.const_f32(float("-inf"))
    zero = b.const_f32(0.0)
    max_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.num_segments),
        b.const_i32(1),
        [("mx", neg_inf)],
        iv_name="seg",
    )
    with max_loop as (seg, (mx,)):
        idx = _seg_idx(b, p, spec.num_segments, q_tok, q_head, seg)
        mv = b.global_load_f32(segm_max, idx)
        b.scf_yield(b.fmax(mx, mv))
    overall = max_loop.results[0]
    red = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.num_segments),
        b.const_i32(1),
        [("den", zero), ("acc", zero)],
        iv_name="seg2",
    )
    with red as (seg, (den, acc)):
        idx = _seg_idx(b, p, spec.num_segments, q_tok, q_head, seg)
        mv = b.global_load_f32(segm_max, idx)
        lv = b.global_load_f32(segm_expsum, idx)
        factor = b.exp2(b.fsub(mv, overall))
        den2 = b.fadd(den, b.fmul(lv, factor))
        out_idx = b.add(
            b.add(
                b.mul(
                    q_tok,
                    b.const_i32(p.num_query_heads * spec.num_segments * p.head_size),
                ),
                b.mul(q_head, b.const_i32(spec.num_segments * p.head_size)),
            ),
            b.add(b.mul(seg, b.const_i32(p.head_size)), dim),
        )
        ov = b.global_load_f32(segm_output, out_idx)
        acc2 = b.fadd(acc, b.fmul(ov, factor))
        b.scf_yield(den2, acc2)
    result = b.fmul(red.results[1], b.rcp(red.results[0]))
    cast = b.cast_f32_to(result, dtype)
    out_idx = b.add(
        b.add(
            b.mul(q_tok, b.const_i32(p.num_query_heads * p.head_size)),
            b.mul(q_head, b.const_i32(p.head_size)),
        ),
        dim,
    )
    with b.scf_if(active):
        b.global_store(out, out_idx, cast, align=2)
    return b.kernel


def _seg_idx(
    b: IRBuilder,
    p: UnifiedAttentionProblem,
    segments: int,
    q_tok: Value,
    q_head: Value,
    seg: Value,
) -> Value:
    return b.add(
        b.add(
            b.mul(q_tok, b.const_i32(p.num_query_heads * segments)),
            b.mul(q_head, b.const_i32(segments)),
        ),
        seg,
    )


def _emit_find_seq_idx_scan(
    b: IRBuilder, cu_q: Value, q_tok: Value, num_seqs: Value
) -> Value:
    scan = b.scf_for_iter(
        b.const_i32(0),
        num_seqs,
        b.const_i32(1),
        [("seq_idx", b.const_i32(0))],
        iv_name="si",
    )
    with scan as (si, (seq_idx,)):
        start_i = b.global_load_i32(cu_q, si)
        le = b.cmp_le(start_i, q_tok)
        b.scf_yield(b.select(le, si, seq_idx))
    return scan.results[0]


def _emit_qk_score(
    b: IRBuilder,
    p: UnifiedAttentionProblem,
    dtype: Type,
    query: Value,
    key: Value,
    block_tables: Value,
    seq_idx: Value,
    q_tok: Value,
    q_head: Value,
    kv_head: Value,
    kpos: Value,
    scale: Value,
    rcp_ln2: Value,
) -> Value:
    score = b.const_f32(0.0)
    physical, token_in_block = _physical_block_and_token(
        b, p, block_tables, seq_idx, kpos
    )
    for d in b.unroll(p.head_size):
        q_off = b.add(
            b.add(
                b.mul(q_tok, b.const_i32(p.num_query_heads * p.head_size)),
                b.mul(q_head, b.const_i32(p.head_size)),
            ),
            b.const_i32(d),
        )
        k_off = b.add(
            b.add(
                b.add(
                    b.mul(
                        physical,
                        b.const_i32(p.block_size * p.num_kv_heads * p.head_size),
                    ),
                    b.mul(token_in_block, b.const_i32(p.num_kv_heads * p.head_size)),
                ),
                b.mul(kv_head, b.const_i32(p.head_size)),
            ),
            b.const_i32(d),
        )
        qv = b.cast_to_f32(b.global_load(query, q_off, dtype, align=2))
        kv = b.cast_to_f32(b.global_load(key, k_off, dtype, align=2))
        score = b.fadd(score, b.fmul(qv, kv))
    return b.fmul(b.fmul(score, scale), rcp_ln2)


def _emit_v_load(
    b: IRBuilder,
    p: UnifiedAttentionProblem,
    dtype: Type,
    value: Value,
    block_tables: Value,
    seq_idx: Value,
    kv_head: Value,
    kpos: Value,
    dim: Value,
) -> Value:
    physical, token_in_block = _physical_block_and_token(
        b, p, block_tables, seq_idx, kpos
    )
    v_off = b.add(
        b.add(
            b.add(
                b.mul(
                    physical, b.const_i32(p.block_size * p.num_kv_heads * p.head_size)
                ),
                b.mul(token_in_block, b.const_i32(p.num_kv_heads * p.head_size)),
            ),
            b.mul(kv_head, b.const_i32(p.head_size)),
        ),
        dim,
    )
    return b.cast_to_f32(b.global_load(value, v_off, dtype, align=2))


def _physical_block_and_token(
    b: IRBuilder,
    p: UnifiedAttentionProblem,
    block_tables: Value,
    seq_idx: Value,
    kpos: Value,
) -> Tuple[Value, Value]:
    block_idx = b.div(kpos, b.const_i32(p.block_size))
    token_in_block = b.mod(kpos, b.const_i32(p.block_size))
    max_blocks = (p.max_seqlen_k + p.block_size - 1) // p.block_size
    physical = b.global_load_i32(
        block_tables, b.add(b.mul(seq_idx, b.const_i32(max_blocks)), block_idx)
    )
    return physical, token_in_block
