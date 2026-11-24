# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free paged unified-attention harness for the gfx950 numeric tests.

Scenarios, seeded inputs, a float64 NumPy reference and the launch lanes
(``2d`` direct, ``3d``, ``auto``, plus any backend name), with no torch.
Values are rounded to the kernel dtype before the reference reads them.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
from dataclasses import dataclass

import numpy as np
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray

ARCH = get_device_arch()

_BACKEND = {"2d": "tiled", "3d": "3d", "auto": "auto"}


@dataclass(frozen=True)
class Scenario:
    name: str
    seq_lens: list[tuple[int, int]]
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    dtype: str
    sliding_window: int | None = None
    softcap: float = 0.0
    use_sinks: bool = False
    num_blocks: int = 32768
    use_alibi: bool = False
    use_qq_bias: bool = False
    qq_bias_stride_0: int = 0
    # Band: diagonal alignment and right bound (0 causal, -1 unbounded, R > 0
    # lookahead); ``sliding_window`` is the left bound.
    causal_top_left: bool = False
    right_bound: int = 0


_MIXED = [(512, 512), (700, 200), (200, 700), (64, 64)]


def topleft_scenarios() -> list[Scenario]:
    def tl(name, seq_lens, **kw):
        base = dict(
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype="fp16",
        )
        base.update(kw)
        return Scenario(
            name=f"tl_{name}", seq_lens=seq_lens, causal_top_left=True, **base
        )

    lt = [(256, 1024), (100, 300), (1, 700)]
    gt = [(1024, 256), (300, 100), (700, 1)]
    return [
        tl("q_lt_kv", lt),
        tl("q_gt_kv", gt),
        tl("mixed", _MIXED),
        tl("mixed_bf16_b64", _MIXED, dtype="bf16", block_size=64),
        tl(
            "gqa64x8_prefill",
            [(2048, 4096), (1500, 700)],
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
        ),
        tl("d64", _MIXED, head_size=64, num_query_heads=8, num_kv_heads=1),
        tl(
            "d256_bf16",
            [(1024, 2048), (900, 300)],
            head_size=256,
            dtype="bf16",
            num_query_heads=64,
            num_kv_heads=8,
        ),
        tl("sinks", _MIXED, use_sinks=True),
        tl("softcap", _MIXED, softcap=30.0),
        tl("sliding_window", [(300, 1024), (500, 200), (256, 256)], sliding_window=64),
        tl("alibi", _MIXED, use_alibi=True),
        tl("qq_bias", _MIXED, use_qq_bias=True, qq_bias_stride_0=1024),
        tl("decode_q1", [(1, 1024), (1, 4096), (1, 33)]),
        tl(
            "d256_decode_q1",
            [(1, 2048), (1, 4096), (1, 512)],
            head_size=256,
            dtype="bf16",
        ),
    ]


def nomask_scenarios() -> list[Scenario]:
    def nm(name, seq_lens, **kw):
        base = dict(
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype="fp16",
        )
        base.update(kw)
        return Scenario(name=f"nm_{name}", seq_lens=seq_lens, right_bound=-1, **base)

    return [
        nm("equal", [(512, 512), (257, 257), (64, 64)]),
        nm("q_lt_kv", [(256, 1024), (100, 300), (1, 700)]),
        nm("q_gt_kv", [(1024, 256), (300, 100), (700, 1)]),
        nm("mixed", _MIXED),
        nm("mixed_bf16_b64", _MIXED, dtype="bf16", block_size=64),
        nm(
            "gqa64x8_prefill",
            [(2048, 4096), (1500, 700)],
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
        ),
        nm("d64", _MIXED, head_size=64, num_query_heads=8, num_kv_heads=1),
        nm(
            "d256_bf16",
            [(1024, 2048), (900, 300)],
            head_size=256,
            dtype="bf16",
            num_query_heads=64,
            num_kv_heads=8,
        ),
        nm("sinks", _MIXED, use_sinks=True),
        nm("softcap", _MIXED, softcap=30.0),
        nm("decode_q1", [(1, 1024), (1, 4096), (1, 33)]),
        nm(
            "d256_decode_q1",
            [(1, 2048), (1, 4096), (1, 512)],
            head_size=256,
            dtype="bf16",
        ),
    ]


def band_scenarios() -> list[Scenario]:
    def bd(name, seq_lens, *, tl, window=0, right=-1, **kw):
        base = dict(
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype="fp16",
        )
        base.update(kw)
        align = "tl" if tl else "br"
        return Scenario(
            name=f"bd_{align}_{name}",
            seq_lens=seq_lens,
            causal_top_left=tl,
            right_bound=right,
            sliding_window=window or None,
            **base,
        )

    out: list[Scenario] = []
    for tl in (False, True):
        out += [
            bd("left_only_w64", _MIXED, tl=tl, window=64),
            bd("two_sided_w64_r32", _MIXED, tl=tl, window=64, right=32),
            bd("lookahead_r16", _MIXED, tl=tl, right=16),
            bd("causal_w100", _MIXED, tl=tl, window=100, right=0),
            bd("q_lt_kv_w128", [(256, 1024), (100, 300), (1, 700)], tl=tl, window=128),
            bd(
                "q_gt_kv_w128_r8",
                [(1024, 256), (300, 100), (700, 1)],
                tl=tl,
                window=128,
                right=8,
            ),
            bd(
                "bf16_b64_w96_r48",
                _MIXED,
                tl=tl,
                window=96,
                right=48,
                dtype="bf16",
                block_size=64,
            ),
            bd(
                "gqa64x8_w256",
                [(2048, 4096), (1500, 700)],
                tl=tl,
                window=256,
                num_query_heads=64,
                num_kv_heads=8,
                dtype="bf16",
            ),
            bd(
                "gqa64x8_r64",
                [(2048, 4096), (1500, 700)],
                tl=tl,
                right=64,
                num_query_heads=64,
                num_kv_heads=8,
                dtype="bf16",
            ),
            bd(
                "d64_w64_r32",
                _MIXED,
                tl=tl,
                window=64,
                right=32,
                head_size=64,
                num_query_heads=8,
                num_kv_heads=1,
            ),
            bd(
                "d256_bf16_w128_r16",
                [(1024, 2048), (900, 300)],
                tl=tl,
                window=128,
                right=16,
                head_size=256,
                dtype="bf16",
                num_query_heads=64,
                num_kv_heads=8,
            ),
            bd("sinks_w64_r32", _MIXED, tl=tl, window=64, right=32, use_sinks=True),
            bd("softcap_w64", _MIXED, tl=tl, window=64, softcap=30.0),
            bd("decode_q1_r8", [(1, 1024), (1, 4096), (1, 33)], tl=tl, right=8),
        ]
    return out


# --- inputs -----------------------------------------------------------------


def _normal(rng, shape, dtype, *, chunk_elems=1 << 25):
    """Unit-normal values encoded as ``dtype`` host bits, drawn a chunk at a time
    so a multi-GiB cache needs no fp32 copy."""
    host = np.empty(shape, dtype=np.float16 if dtype == "fp16" else np.uint16)
    row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    rows = max(1, chunk_elems // row)
    for start in range(0, shape[0], rows):
        stop = min(shape[0], start + rows)
        host[start:stop] = an.encode(
            rng.standard_normal((stop - start, *shape[1:]), dtype=np.float32), dtype
        )
    return host


def make_inputs(s: Scenario, seed: int = 0):
    """Host inputs: ``query`` / ``key_cache`` / ``value_cache`` / ``sinks`` are
    ``s.dtype`` host bits (:func:`rocke.numeric` encoding); the rest are NumPy."""
    rng = np.random.default_rng(seed)
    query_lens = [x[0] for x in s.seq_lens]
    kv_lens_list = [x[1] for x in s.seq_lens]
    num_seqs = len(s.seq_lens)
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)
    max_blocks = (max_kv_len + s.block_size - 1) // s.block_size
    cache_shape = (s.num_blocks, s.block_size, s.num_kv_heads, s.head_size)
    data = {
        "query": _normal(
            rng, (sum(query_lens), s.num_query_heads, s.head_size), s.dtype
        ),
        "key_cache": _normal(rng, cache_shape, s.dtype),
        "value_cache": _normal(rng, cache_shape, s.dtype),
        "cu_q": np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32),
        "kv_lens": np.array(kv_lens_list, dtype=np.int32),
        "query_lens": query_lens,
        "kv_lens_list": kv_lens_list,
        "block_tables": rng.integers(
            0, s.num_blocks, (num_seqs, max_blocks), dtype=np.int32
        ),
        "scale": s.head_size**-0.5,
        "sinks": (_normal(rng, (s.num_query_heads,), s.dtype) if s.use_sinks else None),
        "alibi_slopes": (
            -np.linspace(0.05, 0.5, s.num_query_heads, dtype=np.float32)
            if s.use_alibi
            else None
        ),
        "qq_bias": None,
        "max_query_len": max_query_len,
        "max_kv_len": max_kv_len,
    }
    if s.use_qq_bias:
        n0 = max(max_query_len, s.qq_bias_stride_0)
        data["qq_bias"] = (
            rng.standard_normal((n0, s.qq_bias_stride_0), dtype=np.float32) * 0.1
        )
    return data


def copy_inputs(data, **replace):
    """``data`` with the named arrays replaced and every other array shared."""
    return {**data, **replace}


def upload(rt, data):
    """Device buffers for ``data`` (the dtype of each cache is its host dtype)."""
    dev = {}
    for name in (
        "query",
        "key_cache",
        "value_cache",
        "cu_q",
        "kv_lens",
        "block_tables",
        "sinks",
        "alibi_slopes",
        "qq_bias",
    ):
        host = data[name]
        dev[name] = None if host is None else DeviceArray(rt, host)
    return dev


# --- reference --------------------------------------------------------------


def reference(s: Scenario, data) -> np.ndarray:
    """float64 paged attention for the band ``(left = sliding_window, right =
    right_bound)`` around the top-left or bottom-right diagonal.

    A row with no visible key is 0. Soft cap, ALiBi, QQ bias and sink column follow
    the kernels' order: scale, soft cap, bias, mask, softmax (sink column dropped
    after normalising).
    """
    dt = s.dtype
    hq, hkv, d, bs = s.num_query_heads, s.num_kv_heads, s.head_size, s.block_size
    group = hq // hkv
    scale = data["scale"]
    query = an.decode(data["query"], dt).astype(np.float64)
    sinks = (
        None
        if data["sinks"] is None
        else an.decode(data["sinks"], dt).astype(np.float64)
    )
    out = np.zeros(query.shape, dtype=np.float64)
    window = s.sliding_window or 0
    start = 0
    for i, (q_len, kv_len) in enumerate(s.seq_lens):
        ctx = 0 if s.causal_top_left else kv_len - q_len
        pages = data["block_tables"][i, : -(-kv_len // bs)]
        k = an.decode(data["key_cache"][pages], dt).reshape(-1, hkv, d)[:kv_len]
        v = an.decode(data["value_cache"][pages], dt).reshape(-1, hkv, d)[:kv_len]
        k = k.astype(np.float64)
        v = v.astype(np.float64)
        q_pos = np.arange(q_len)[:, None]
        k_pos = np.arange(kv_len)[None, :]
        diag = q_pos + ctx
        keep = np.ones((q_len, kv_len), dtype=bool)
        if s.right_bound >= 0:
            keep &= k_pos <= diag + s.right_bound
        if window:
            keep &= (diag - k_pos) < window
        empty = ~keep.any(axis=1)
        bias = np.zeros((q_len, kv_len), dtype=np.float64)
        if data["qq_bias"] is not None:
            qq = data["qq_bias"]
            for qp in range(min(q_len, qq.shape[0])):
                lo = max(ctx, 0)
                hi = min(kv_len, qq.shape[1] + ctx)
                if lo < hi:
                    bias[qp, lo:hi] = qq[qp, lo - ctx : hi - ctx]
        for h in range(hq):
            sc = (query[start : start + q_len, h, :] @ k[:, h // group, :].T) * scale
            if s.softcap > 0:
                sc = s.softcap * np.tanh(sc / s.softcap)
            if data["alibi_slopes"] is not None:
                sc = sc + float(data["alibi_slopes"][h]) * (k_pos - ctx)
            sc = np.where(keep, sc + bias, -np.inf)
            if sinks is not None:
                sc = np.concatenate(
                    [sc, np.full((q_len, 1), sinks[h], dtype=np.float64)], axis=1
                )
            m = sc.max(axis=1, keepdims=True)
            m[~np.isfinite(m)] = 0.0
            p = np.exp(sc - m)
            denom = p.sum(axis=1, keepdims=True)
            p /= np.where(denom > 0, denom, 1.0)
            if sinks is not None:
                p = p[:, :-1]
            o = p @ v[:, h // group, :]
            o[empty] = 0.0
            out[start : start + q_len, h, :] = o
        start += q_len
    return out


_LAST = {"key": None, "value": None}


def cached_case(s: Scenario):
    """``(data, reference)`` for ``s`` at seed 0, kept for the scenario last asked
    for, so the lanes of one scenario share inputs and reference. Do not mutate
    ``data``; use :func:`copy_inputs`."""
    key = repr(s)
    if _LAST["key"] != key:
        _LAST["key"] = None
        _LAST["value"] = None
        data = make_inputs(s)
        _LAST["value"] = (data, reference(s, data).astype(np.float32))
        _LAST["key"] = key
    return _LAST["value"]


# --- launch -----------------------------------------------------------------


def problem_for(s: Scenario, data, **overrides):
    from kernels import UnifiedAttentionProblem

    fields = dict(
        total_q=data["query"].shape[0],
        num_seqs=len(s.seq_lens),
        num_query_heads=s.num_query_heads,
        num_kv_heads=s.num_kv_heads,
        head_size=s.head_size,
        block_size=s.block_size,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype=s.dtype,
        sliding_window=s.sliding_window or 0,
        softcap=float(s.softcap),
        use_sinks=data["sinks"] is not None,
        use_alibi=data["alibi_slopes"] is not None,
        use_qq_bias=data["qq_bias"] is not None,
        use_fp8=False,
        num_cus=120,
        num_kv_blocks=int(data["key_cache"].shape[0]),
        compile_backend=os.environ.get("ROCKE_ATTENTION_COMPILE_BACKEND") or None,
        causal_top_left=s.causal_top_left,
        right_bound=s.right_bound,
    )
    fields.update(overrides)
    return UnifiedAttentionProblem(**fields)


class StorageArray(DeviceArray):
    """A device array that also reports its storage, as the additive-bias bounds
    check reads it from a tensor."""

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return types.SimpleNamespace(nbytes=lambda: self._host.nbytes)


@contextlib.contextmanager
def workspace_allocator(rt):
    """Stand in for the few ``torch`` calls the split-KV launcher makes: scratch
    from ``rocke.runtime`` device buffers (NaN-filled) and the default stream.
    No torch is imported."""
    null = contextlib.nullcontext
    fake = types.ModuleType("torch")
    fake.float32 = "float32"

    def empty(shape, dtype=None, device=None):
        return DeviceArray(rt, np.full(tuple(shape), np.nan, dtype=np.float32))

    fake.empty = empty
    fake.cuda = types.SimpleNamespace(
        device=lambda _d=None: null(),
        stream=lambda _s=None: null(),
        ExternalStream=lambda stream, device=None: stream,
        current_device=lambda: 0,
        current_stream=lambda _d=None: types.SimpleNamespace(cuda_stream=0),
        is_current_stream_capturing=lambda: False,
    )
    saved = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        yield
    finally:
        if saved is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved


def launch(rt, s: Scenario, data, path: str, *, problem=None, dev=None):
    """Run the lane ``path`` (``"2d"``, ``"3d"``, ``"auto"`` or a backend name) on
    ``data`` and return the output as fp32. The output buffer starts as NaN."""
    from kernels import run_unified_attention_torch

    dev = upload(rt, data) if dev is None else dev
    problem = problem_for(s, data) if problem is None else problem
    out = an.empty_device(rt, data["query"].shape, s.dtype)
    qq_bias = dev["qq_bias"]
    qq_stride = int(qq_bias.stride(0)) if qq_bias is not None else 0
    if path == "2d":
        _launch_2d(rt, s, data, dev, problem, out, qq_stride)
    else:
        with workspace_allocator(rt):
            run_unified_attention_torch(
                problem=problem,
                q=dev["query"],
                k=dev["key_cache"],
                v=dev["value_cache"],
                out=out,
                cu_seqlens_q=dev["cu_q"],
                seqused_k=dev["kv_lens"],
                softmax_scale=data["scale"],
                block_table=dev["block_tables"],
                softcap=float(s.softcap),
                sinks=dev["sinks"],
                alibi_slopes=dev["alibi_slopes"],
                qq_bias=qq_bias,
                qq_bias_stride_0=qq_stride,
                backend=_BACKEND.get(path, path),
                use_graph=False,
            )
    return an.read(out, s.dtype)


def _launch_2d(rt, s, data, dev, problem, out, qq_stride):
    """Build and launch the production 2D spec directly, as the dispatcher would."""
    from kernels import build_unified_attention_2d_tiled, supports_tiled_2d
    from kernels.common import attention_unified as au
    from rocke import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    if not au._d256_gfx950_fast(problem, ARCH):
        ok, reason = supports_tiled_2d(
            head_size=s.head_size,
            block_size=s.block_size,
            dtype=s.dtype,
            num_queries_per_kv=problem.num_queries_per_kv,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            use_fp8=problem.use_fp8,
            q_dtype=problem.q_dtype,
            num_warps=4,
            kv_storage_dtype=None,
            tile_size=2 * s.block_size,
        )
        if not ok:
            raise NotImplementedError(reason)
    spec = au._tiled_spec_from_problem(problem, ARCH)
    kernel = build_unified_attention_2d_tiled(spec)
    if au._select_2d_compile_backend(problem, ARCH) == "hipcc":
        from rocke.helpers.compile import compile_kernel_via_hipcc

        artifact = compile_kernel_via_hipcc(kernel)
    else:
        artifact = compile_kernel(kernel, capture_ir_text=False)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=au._tiled_2d_signature(ARCH, s.dtype),
        cache_key=("prod_2d_direct", spec.kernel_name()),
    )
    vals = au._attn_values(
        problem=problem,
        q=dev["query"],
        k=dev["key_cache"],
        v=dev["value_cache"],
        out=out,
        cu_seqlens_q=dev["cu_q"],
        seqused_k=dev["kv_lens"],
        softmax_scale=data["scale"],
        block_table=dev["block_tables"],
        softcap=float(s.softcap),
        sinks=dev["sinks"],
        bt_stride=int(dev["block_tables"].stride(0)),
        include_bt_stride=True,
        alibi_slopes=dev["alibi_slopes"],
        qq_bias=dev["qq_bias"],
        qq_bias_stride_0=qq_stride,
        include_qq_bias_stride=True,
        kv_addr_limit=au._spec_kv_addr_limit(problem, spec),
    )
    total_num_q_blocks = data["query"].shape[0] // spec.block_q + len(s.seq_lens)
    launcher(
        vals,
        config=LaunchConfig(
            grid=(int(s.num_kv_heads), int(total_num_q_blocks), 1),
            block=(64 * spec.num_warps, 1, 1),
            stream=0,
        ),
    )
