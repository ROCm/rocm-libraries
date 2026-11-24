# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unified paged attention LSE output -- split-KV (3D) reduce and tiled 2D
epilogues on gfx942/gfx950 -- vs an independent float64 NumPy paged reference.

Inputs, buffers and the reference are NumPy and ``rocke.runtime`` device arrays; no
torch is imported. The split-KV (3D) lanes launch under ``workspace_allocator``, the
scratch-buffer stub that stands in for the few ``torch`` calls the split-KV runner
makes. The one test that needs a CUDA graph on gfx942 uses real torch and skips
without it.
"""

import contextlib
import dataclasses
import itertools
import math
import types
from unittest import mock

import numpy as np
import pytest

from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from kernels.common import attention_unified as au
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from . import _unified_paged_numpy as U
from ._attention_device_array import DeviceArray

_ARCH = get_device_arch()


def _arch():
    return _ARCH if _ARCH in ("gfx942", "gfx950") else None


requires_cdna3_or_4 = pytest.mark.skipif(
    _arch() is None, reason=f"needs a gfx942 or gfx950 GPU, found {_ARCH}"
)


@pytest.fixture(scope="module")
def rt():
    return an.runtime()


def _tolerance(dtype):
    return an.tolerance(dtype)


def _max_abs(out, ref):
    return float(np.abs(np.asarray(out, np.float64) - ref).max())


def _host(x):
    return x.read() if hasattr(x, "read") else np.asarray(x)


def _paged_reference(
    case,
    q_lens,
    kv_lens,
    scale,
    softcap,
    sliding_window=0,
    additive_bias=None,
):
    """float64 O and natural-log LSE with the unified bottom-right causal mask
    (key k visible to query p iff k <= diag, diag = p + kv_len - q_len) and, when
    ``sliding_window`` > 0, diag - k < sliding_window. Rows with no key get
    O = 0 and LSE = -inf."""
    q = case.q.astype(np.float64)
    hq, d = q.shape[1], q.shape[2]
    hkv, bs = case.k_cache.shape[2], case.k_cache.shape[1]
    group = hq // hkv
    outs, lses, start = [], [], 0
    with np.errstate(all="ignore"):
        for i, (ql, kl) in enumerate(zip(q_lens, kv_lens)):
            nblk = (kl + bs - 1) // bs
            blocks = case.block_table[i, :nblk]
            k = case.k_cache[blocks].reshape(-1, hkv, d)[:kl].astype(np.float64)
            v = case.v_cache[blocks].reshape(-1, hkv, d)[:kl].astype(np.float64)
            k = k.transpose(1, 0, 2)  # [hkv, kl, d]
            v = v.transpose(1, 0, 2)
            p = np.arange(ql).reshape(-1, 1)
            kk = np.arange(kl).reshape(1, -1)
            diag = p + (kl - ql)
            visible = kk <= diag
            if sliding_window > 0:
                visible = visible & (diag - kk < sliding_window)
            o_seq = np.zeros((hq, ql, d))
            lse_seq = np.full((hq, ql), -np.inf)
            for h0 in range(0, hq, 4):
                heads = np.arange(h0, min(h0 + 4, hq))
                kvh = heads // group
                s = (
                    np.matmul(
                        q[start : start + ql, heads].transpose(1, 0, 2),
                        k[kvh].transpose(0, 2, 1),
                    )
                    * scale
                )
                if softcap > 0:
                    s = softcap * np.tanh(s / softcap)
                if additive_bias is not None:
                    # [seq, q_head, q_row, key], added after softcap, as the kernels do.
                    s = s + additive_bias[i, heads, :ql, :kl]
                s = np.where(visible, s, -np.inf)
                if kl:
                    m = s.max(axis=-1)
                else:
                    m = np.full(s.shape[:2], -np.inf)
                mm = np.where(np.isfinite(m), m, 0.0)
                total = np.exp(s - mm[..., None]).sum(axis=-1)
                lse = mm + np.log(total)
                has = np.isfinite(lse)
                prob = np.exp(s - np.where(has, lse, 0.0)[..., None])
                prob = np.where(has[..., None], prob, 0.0)
                o_seq[heads] = np.matmul(prob, v[kvh])
                lse_seq[heads] = lse
            outs.append(o_seq.transpose(1, 0, 2))
            lses.append(lse_seq.T)
            start += ql
    return np.concatenate(outs), np.concatenate(lses)[..., None]


@dataclasses.dataclass
class _Case:
    problem: UnifiedAttentionProblem
    dtype: str
    q: np.ndarray
    k_cache: np.ndarray
    v_cache: np.ndarray
    cu: np.ndarray
    seqused: np.ndarray
    block_table: np.ndarray


def _case(
    q_lens,
    kv_lens,
    *,
    hq=32,
    hkv=8,
    d=128,
    dtype="bf16",
    bs=16,
    softcap=0.0,
    sliding_window=0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    total_q, num_seqs = sum(q_lens), len(q_lens)
    max_blocks = max(1, max((kl + bs - 1) // bs for kl in kv_lens))
    num_blocks = num_seqs * max_blocks + 3
    q = an.randn(rng, (total_q, hq, d), dtype)
    k_cache = an.randn(rng, (num_blocks, bs, hkv, d), dtype)
    v_cache = an.randn(rng, (num_blocks, bs, hkv, d), dtype)
    # Shuffled physical pages: catches any block-table indexing shortcut.
    perm = rng.permutation(num_blocks)[: num_seqs * max_blocks]
    block_table = perm.reshape(num_seqs, max_blocks).astype(np.int32)
    cu = np.array([0, *itertools.accumulate(q_lens)], dtype=np.int32)
    seqused = np.array(kv_lens, dtype=np.int32)
    problem = UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(max(kv_lens), 1),
        dtype=dtype,
        softcap=softcap,
        sliding_window=sliding_window,
        num_kv_blocks=num_blocks,
        emit_lse=True,
    )
    return _Case(problem, dtype, q, k_cache, v_cache, cu, seqused, block_table)


def _upload(rt, case):
    return types.SimpleNamespace(
        q=an.to_device(rt, case.q, case.dtype),
        k=an.to_device(rt, case.k_cache, case.dtype),
        v=an.to_device(rt, case.v_cache, case.dtype),
        cu=DeviceArray(rt, case.cu),
        su=DeviceArray(rt, case.seqused),
        bt=DeviceArray(rt, case.block_table),
    )


_CUDA = types.SimpleNamespace(type="cuda", index=0)


def _on_cuda(*arrays):
    """Tag buffers with one shared device, as the dispatcher bindings require."""
    for array in arrays:
        array.device = _CUDA
    return arrays


def _nan_lse(rt, problem, fill=np.nan):
    return an.lse_device(rt, (problem.total_q, problem.num_query_heads, 1), fill=fill)


def _run(
    rt,
    case,
    *,
    lse,
    softcap,
    backend="3d",
    tuning_spec=None,
    problem=None,
    dev=None,
    scratch=None,
    **extra,
):
    """One launch; returns O as fp32. The split-KV lanes (``backend="3d"``, or an
    ``auto`` route that reaches it when ``scratch`` is set) run under the
    scratch-buffer stub."""
    dev = _upload(rt, case) if dev is None else dev
    problem = case.problem if problem is None else problem
    out = an.empty_device(rt, case.q.shape, case.dtype)
    use_scratch = backend == "3d" if scratch is None else scratch
    if use_scratch:
        extra.setdefault("use_graph", False)  # no CUDA graphs without torch
    with U.workspace_allocator(rt) if use_scratch else contextlib.nullcontext():
        run_unified_attention_torch(
            problem=problem,
            q=dev.q,
            k=dev.k,
            v=dev.v,
            out=out,
            cu_seqlens_q=dev.cu,
            seqused_k=dev.su,
            softmax_scale=1.0 / math.sqrt(case.q.shape[-1]),
            block_table=dev.bt,
            softcap=softcap,
            backend=backend,
            tuning_spec=tuning_spec,
            lse=lse,
            **extra,
        )
    return an.read(out, case.dtype)


_SHAPES = [
    # (q_lens, kv_lens)               what it covers
    ([1, 1, 1, 1], [4096, 777, 1500, 2049]),  # decode, ragged KV, partial last page
    ([4, 2], [3000, 1029]),  # spec-decode
    ([64], [64]),  # s_q == s_kv
    ([5], [3]),  # bottom-right s_q > s_kv: rows 0,1 empty
    ([1, 1], [0, 600]),  # kv_len == 0: whole sequence empty
]


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("q_lens,kv_lens", _SHAPES)
def test_split_kv_output_and_lse(rt, dtype, d, q_lens, kv_lens):
    case = _case(q_lens, kv_lens, d=d, dtype=dtype)
    lse = _nan_lse(rt, case.problem)
    out = _run(rt, case, lse=lse, softcap=0.0)
    off = _run(
        rt,
        case,
        problem=dataclasses.replace(case.problem, emit_lse=False),
        lse=None,
        softcap=0.0,
    )
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(d), 0.0)
    tol = _tolerance(dtype)
    lse = _host(lse)
    assert np.array_equal(out, off, equal_nan=True), "LSE-on changed O"
    assert _max_abs(out, o_ref) < tol
    assert np.array_equal(np.isneginf(lse), np.isneginf(lse_ref))
    finite = np.isfinite(lse_ref)
    assert not np.isnan(lse).any()
    assert np.abs(lse[finite] - lse_ref[finite]).max() < tol


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_split_kv_lse_nan_query_row_stays_nan(rt, dtype):
    """The split-KV reduce follows the tiled 2D rule: a NaN query row gives
    NaN O and NaN LSE for that (token, head) only, never the keyless -inf.
    Its merged sum is NaN, and !(l == 0) keeps row_has_mass true."""
    q_lens, kv_lens = [1, 1, 1, 1], [4096, 777, 1500, 2049]
    nan_token, nan_head = 1, 5
    case = _case(q_lens, kv_lens, dtype=dtype)
    case.q[nan_token, nan_head] = np.nan
    problem = case.problem
    lse = _nan_lse(rt, problem, fill=np.inf)
    out = _run(rt, case, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(
        case, q_lens, kv_lens, 1.0 / math.sqrt(case.q.shape[-1]), 0.0
    )
    lse = _host(lse)
    assert not np.isposinf(lse).any(), "an LSE row was left unwritten"
    assert np.isnan(lse[nan_token, nan_head]).all(), "NaN row got a non-NaN LSE"
    assert np.isnan(out[nan_token, nan_head]).all(), "NaN row was hidden in O"
    rest = np.ones((problem.total_q, problem.num_query_heads), dtype=bool)
    rest[nan_token, nan_head] = False
    assert np.isfinite(lse[..., 0][rest]).all(), "NaN leaked into another LSE"
    assert np.isfinite(out[rest]).all(), "NaN leaked into another (row, head)"
    assert _max_abs(out[rest], o_ref[rest]) < _tolerance(dtype)
    np.testing.assert_allclose(
        lse[..., 0][rest], lse_ref[..., 0][rest], atol=_LSE_ATOL, rtol=_LSE_RTOL
    )


@requires_cdna3_or_4
@pytest.mark.gpu
def test_split_kv_lse_with_softcap(rt):
    q_lens, kv_lens = [1, 1], [2048, 900]
    case = _case(q_lens, kv_lens, softcap=30.0)
    lse = _nan_lse(rt, case.problem)
    _run(rt, case, lse=lse, softcap=30.0)
    _, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(128), 30.0)
    assert _max_abs(_host(lse), lse_ref) < _tolerance("bf16")


def _assert_matches_reference(out, lse, o_ref, lse_ref, tol):
    lse = _host(lse)
    assert np.isfinite(out).all(), "non-finite O"
    assert _max_abs(out, o_ref) < tol
    assert not np.isnan(lse).any(), "an LSE row was left unwritten"
    assert np.array_equal(np.isneginf(lse), np.isneginf(lse_ref))
    finite = np.isfinite(lse_ref)
    assert np.abs(lse[finite] - lse_ref[finite]).max() < tol


# ---------------------------------------------------------------------------
# Tiled 2D LSE (gfx942 / gfx950 build_unified_attention_2d_tiled)
# ---------------------------------------------------------------------------

_LSE_ATOL = 1e-3
_LSE_RTOL = 1e-3

# Prefill shapes the tiled 2D path serves. With 32 query / 8 KV heads every one
# routes to 2D under backend="auto" (UnifiedAttentionProblem.select_path).
_PREFILL_SHAPES = [
    # (q_lens, kv_lens)                    what it covers
    ([256], [256]),  # s_q == s_kv; the gfx942 2D-graph-eligible shape
    ([37, 300, 1], [37, 812, 900]),  # ragged, causal s_q < s_kv, partial pages
    (
        [5],
        [3],
    ),  # bottom-right s_q > s_kv: rows 0,1 see no key (m=-inf, l=0 after Part 0)
    ([3, 64], [0, 64]),  # kv_len == 0: zero-trip KV loop (m=-inf, l=1)
]


def _forbid_split_kv(*_args, **_kwargs):
    raise AssertionError("a tiled 2D LSE test reached the split-KV (3D) path")


def _case2d(q_lens, kv_lens, **kw):
    """``_case`` for a tiled 2D LSE launch; fails fast, with the gate's reason,
    if the tiled 2D gate refuses the LSE problem on this GPU."""
    case = _case(q_lens, kv_lens, **kw)
    ok, why = au.supports_native_unified_attention_tiled(case.problem, _arch())
    assert ok, f"the tiled 2D gate refuses this LSE problem: {why}"
    return case


def _run2d(
    rt,
    case,
    *,
    lse,
    softcap,
    backend,
    tuning_spec=None,
    problem=None,
    dev=None,
    **extra,
):
    """One 2D launch. The split-KV runner is patched to fail, so a pass proves
    the tiled 2D path wrote O (and LSE)."""
    with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
        return _run(
            rt,
            case,
            lse=lse,
            softcap=softcap,
            backend=backend,
            tuning_spec=tuning_spec,
            problem=problem,
            dev=dev,
            **extra,
        )


def _run2d_lse_off(rt, case, *, softcap, backend, tuning_spec=None, **extra):
    """The LSE-off launch of the kernel the LSE-on request runs. With LSE on,
    the gfx942 4-warp cohorts fall back to the default builder and the gfx942
    2D graph is bypassed; force both here so the bitwise comparison covers
    one kernel with and without its LSE store. Both are no-ops on gfx950."""
    off = dataclasses.replace(case.problem, emit_lse=False)
    with mock.patch.object(
        au, "_gfx942_4warp_eligible", lambda problem, arch: False
    ), mock.patch.object(au, "_enable_2d_graph_replay", lambda problem, arch: False):
        return _run2d(
            rt,
            case,
            problem=off,
            lse=None,
            softcap=softcap,
            backend=backend,
            tuning_spec=tuning_spec,
            **extra,
        )


def _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype):
    lse = _host(lse)
    assert not np.isnan(lse).any(), "an LSE row was left unwritten"
    assert np.array_equal(out, off, equal_nan=True), "LSE-on changed O"
    assert np.isfinite(out).all(), "non-finite O"
    assert _max_abs(out, o_ref) < _tolerance(dtype)
    keyless = np.isneginf(lse_ref[..., 0])  # [total_q, Hq]
    assert (out[keyless] == 0).all(), "a row with no visible key has O != 0"
    assert np.array_equal(
        np.isneginf(lse), np.isneginf(lse_ref)
    ), "the -inf (keyless) LSE rows differ from the reference"
    assert not np.isposinf(lse).any(), "+inf LSE"
    finite = np.isfinite(lse_ref)
    np.testing.assert_allclose(
        lse[finite], lse_ref[finite], atol=_LSE_ATOL, rtol=_LSE_RTOL
    )


# One explicit dispatcher geometry per tiled 2D epilogue branch (flags =
# dispatch/attention/axes.py _CODEPATH_KNOBS).
_BRANCH_VARIANT = {
    "gfx942": {
        "transposed32": "transposed_x8_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32x8_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
    "gfx950": {
        "transposed32": "transposed32_nw1_mw32_t2xb_llvm",
        "mfma32": "wide32_nw1_mw32_t2xb_llvm",
        "striped16": "narrow_nw1_mw16_t1xb_llvm",
    },
}
# A: transposed 32x32 (16 broadcast slots, and the 1-slot scalar state that
# auto routing ships); B: non-transposed 32x32 Acc_lds; C: striped 16x16.
_BRANCHES = ("transposed32", "transposed32_s1", "mfma32", "striped16")


def _epilogue_branch(kernel_spec) -> str:
    wide = bool(
        getattr(kernel_spec, "use_mfma_32x32", False)
        or getattr(kernel_spec, "use_mfma_32x32x8", False)
    )
    if not wide:
        return "striped16"
    return "transposed32" if kernel_spec.use_transposed_qk_32x32 else "mfma32"


def _branch_spec(problem, branch):
    """The explicit tuning spec that forces ``branch`` for ``problem``."""
    from dispatch.attention.unified_rules import canonicalize_tuning_spec

    arch = _arch()
    if arch == "gfx942":
        from dispatch.attention.gfx942_unified import (
            GFX942_TUNING_VARIANTS as variants,
        )
    else:
        from dispatch.attention.gfx950_unified import (
            GFX950_TUNING_VARIANTS as variants,
        )
    base = branch.removesuffix("_s1")
    spec_id = f"{arch}_u2d_{_BRANCH_VARIANT[arch][base]}"
    variant = next(v for v in variants if v.spec_id == spec_id)
    knobs = {"use_transposed_scalar_state": True} if branch.endswith("_s1") else {}
    spec, why = canonicalize_tuning_spec(problem, variant, knobs)
    assert spec is not None, f"{spec_id} {knobs}: {why}"
    ks = spec.kernel_spec
    assert _epilogue_branch(ks) == base, f"{spec_id} selects {_epilogue_branch(ks)}"
    assert bool(getattr(ks, "use_transposed_scalar_state", False)) == (
        branch.endswith("_s1")
    )
    assert ks.emit_lse is problem.emit_lse
    return spec


def _branch_pair(rt, case, branch, *, softcap=0.0):
    """LSE-on and LSE-off tiled launches forced to ``branch``: ``(out, lse, off)``."""
    off_problem = dataclasses.replace(case.problem, emit_lse=False)
    lse = _nan_lse(rt, case.problem)
    out = _run2d(
        rt,
        case,
        lse=lse,
        softcap=softcap,
        backend="tiled",
        tuning_spec=_branch_spec(case.problem, branch),
    )
    off = _run2d(
        rt,
        case,
        problem=off_problem,
        lse=None,
        softcap=softcap,
        backend="tiled",
        tuning_spec=_branch_spec(off_problem, branch),
    )
    return out, lse, off


# Decode with ragged KV and spec-decode: the shapes the split-KV path serves.
_DECODE_SHAPES = [([1, 1, 1, 1], [4096, 777, 1500, 2049]), ([4, 2], [3000, 1029])]


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _DECODE_SHAPES)
def test_split_kv_lse_with_one_query_head_per_kv_head(rt, q_lens, kv_lens):
    """qpk = 1 (Hq == Hkv): a different query-head-to-row layout than GQA."""
    case = _case(q_lens, kv_lens, hq=8, hkv=8)
    lse = _nan_lse(rt, case.problem)
    out = _run(rt, case, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(128), 0.0)
    _assert_matches_reference(out, lse, o_ref, lse_ref, _tolerance("bf16"))


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("window", [128, 1000])
@pytest.mark.parametrize("q_lens,kv_lens", _DECODE_SHAPES)
def test_split_kv_lse_with_sliding_window(rt, q_lens, kv_lens, window):
    """Sliding window empties the segments left of the window for every row."""
    case = _case(q_lens, kv_lens, sliding_window=window)
    lse = _nan_lse(rt, case.problem)
    out = _run(rt, case, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(
        case, q_lens, kv_lens, 1.0 / math.sqrt(128), 0.0, window
    )
    _assert_matches_reference(out, lse, o_ref, lse_ref, _tolerance("bf16"))


_PINNED_SPLIT_KV_SPEC = {
    "gfx942": "gfx942_u3d_splitkv_seg16_t1xb",
    "gfx950": "gfx950_u3d_splitkv_seg16_t1xb",
}


@requires_cdna3_or_4
@pytest.mark.gpu
def test_pinned_split_kv_spec_end_to_end(rt):
    """The dispatcher route: a request pinned to a split-KV tuning spec with an
    explicit segment count, bound to real tensors and launched."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    q_lens, kv_lens = [1, 1, 1, 1], [4096, 777, 1500, 2049]
    case = _case(q_lens, kv_lens)
    dev = _upload(rt, case)
    request = AttentionRequest(
        batch=4,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1,
        seqlen_k=4096,
        hdim_q=128,
        hdim_v=128,
        arch=_arch(),
        mask_type=2,  # bottom-right causal, as in _paged_reference
        dtype="bf16",
        emit_lse=True,
        algorithm="unified_tuning",
        spec_id=_PINNED_SPLIT_KV_SPEC[_arch()],
    )
    result = dispatch_attention(request)
    assert result.spec.reduce_spec.emit_lse is True
    assert result.spec.reduce_spec.num_segments == 16
    out, lse = an.empty_device(rt, case.q.shape, "bf16"), _nan_lse(rt, case.problem)
    tensors = dict(
        q=dev.q,
        k=dev.k,
        v=dev.v,
        out=out,
        cu_seqlens_q=dev.cu,
        seqused_k=dev.su,
        block_table=dev.bt,
        lse=lse,
    )
    _on_cuda(*tensors.values())
    # Scratch-buffer stub, and no 3D graph replay (that needs torch.cuda.CUDAGraph).
    with U.workspace_allocator(rt), mock.patch.object(
        au, "_enable_3d_graph_replay", lambda problem, arch: False
    ):
        result.bind_torch(tensors).launch()
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(128), 0.0)
    _assert_matches_reference(
        an.read(out, "bf16"), lse, o_ref, lse_ref, _tolerance("bf16")
    )


# (q_len, kv_len): the last KV tile holds only keys row 0 cannot see (bottom-right
# causal: row 0 sees kv_len - q_len + 1 keys), for any tile of 16, 32 or 64; 5/3
# also leaves rows 0 and 1 with no visible key at all.
_MASKED_TILE_SHAPES = [(4, 17), (4, 33), (4, 65), (5, 3)]


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("emit_lse", [True, False])
@pytest.mark.parametrize("q_len,kv_len", _MASKED_TILE_SHAPES)
def test_masked_segment_does_not_underflow_the_merge(rt, q_len, kv_len, emit_lse):
    """A segment whose tiles are all masked for a row stores max=0, sum=0. With
    every visible logit at -200 (Q=2, K=-12.5, D=64, scale 1/8), letting that
    max set the merge max underflows every real segment: O became 0 and LSE
    -inf for rows that do see keys. Keyless rows must stay O=0, LSE=-inf."""
    case = _case([q_len], [kv_len], d=64)
    case.q[...] = 2.0
    case.k_cache[...] = -12.5
    problem = dataclasses.replace(case.problem, emit_lse=emit_lse)
    lse = _nan_lse(rt, problem) if emit_lse else None
    out = _run(rt, case, problem=problem, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(case, [q_len], [kv_len], 0.125, 0.0)
    assert np.isfinite(out).all(), "the merge produced a non-finite O"
    assert _max_abs(out, o_ref) < _tolerance("bf16")
    if emit_lse:
        lse = _host(lse)
        assert np.array_equal(np.isneginf(lse), np.isneginf(lse_ref))
        finite = np.isfinite(lse_ref)
        assert finite.any(), "the case has no row with visible keys"
        assert np.abs(lse[finite] - lse_ref[finite]).max() < _tolerance("bf16")


@requires_cdna3_or_4
@pytest.mark.gpu
def test_auto_backend_writes_lse_on_a_2d_routed_shape(rt):
    """A shape the runner used to refuse: backend='auto' routes it to tiled
    2D, which now writes LSE itself (the split-KV runner is patched to fail)."""
    case = _case2d([256], [256])
    assert case.problem.select_path() == "2d"
    lse = _nan_lse(rt, case.problem)
    out = _run2d(rt, case, lse=lse, softcap=0.0, backend="auto")
    off = _run2d_lse_off(rt, case, softcap=0.0, backend="auto")
    o_ref, lse_ref = _paged_reference(case, [256], [256], 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(out, off, lse, o_ref, lse_ref, "bf16")


def _forbid_tiled_2d(*_args, **_kwargs):
    raise AssertionError("an LSE request the tiled 2D gate declines reached it")


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [([2048], [2048]), ([37, 300, 1], [37, 812, 900])],
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_auto_hands_lse_to_split_kv_when_tiled_2d_declines(rt, dtype, q_lens, kv_lens):
    """gfx942 D256 prefill at block size 32 routes to 2D, but the tiled 2D
    gate declines it with LSE on (LDS budget). backend='auto' runs split-KV,
    which writes LSE (the tiled 2D launcher is patched to fail)."""
    if _arch() != "gfx942":
        pytest.skip("the tiled 2D LDS gate declines this shape on gfx942 only")
    case = _case(q_lens, kv_lens, d=256, dtype=dtype, bs=32)
    problem = case.problem
    assert problem.select_path() == "2d"
    ok, why = au.supports_native_unified_attention_tiled(problem, "gfx942")
    assert not ok and "LDS budget" in why
    lse = _nan_lse(rt, problem)
    with mock.patch.object(au, "_get_2d_launcher", _forbid_tiled_2d):
        out = _run(rt, case, lse=lse, softcap=0.0, backend="auto", scratch=True)
    off = _run(
        rt,
        case,
        problem=dataclasses.replace(problem, emit_lse=False),
        lse=None,
        softcap=0.0,
    )
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(256), 0.0)
    assert np.array_equal(out, off, equal_nan=True), "LSE-on changed the split-KV O"
    _assert_matches_reference(out, lse, o_ref, lse_ref, _tolerance(dtype))


class _TorchDev:
    """Real-torch buffers, for the one test whose gfx942 half captures a CUDA graph."""

    def __init__(self, torch):
        self.torch = torch

    def float_tensor(self, values, dtype):
        t = self.torch
        tdt = t.float16 if dtype == "fp16" else t.bfloat16
        return t.from_numpy(np.ascontiguousarray(values, np.float32)).to(tdt).cuda()

    def i32(self, values):
        return self.torch.from_numpy(np.ascontiguousarray(values, np.int32)).cuda()

    def empty_out(self, shape, dtype):
        t = self.torch
        tdt = t.float16 if dtype == "fp16" else t.bfloat16
        return t.full(shape, float("nan"), dtype=tdt, device="cuda")

    def nan_lse(self, shape):
        t = self.torch
        return t.full(shape, float("nan"), dtype=t.float32, device="cuda")

    def poison(self, out, dtype):
        out.fill_(float("nan"))

    def read(self, t, dtype):
        return t.float().cpu().numpy()

    def sync(self):
        self.torch.cuda.synchronize()


class _NumpyDev:
    """``rocke.runtime`` buffers: the torch-free twin of :class:`_TorchDev`."""

    def __init__(self, rt):
        self.rt = rt

    def float_tensor(self, values, dtype):
        return an.to_device(self.rt, values, dtype)

    def i32(self, values):
        return DeviceArray(self.rt, np.ascontiguousarray(values, np.int32))

    def empty_out(self, shape, dtype):
        return an.empty_device(self.rt, shape, dtype)

    def nan_lse(self, shape):
        return an.lse_device(self.rt, shape)

    def poison(self, out, dtype):
        out.write(an.encode(np.full(out.shape, np.nan, np.float32), dtype))

    def read(self, t, dtype):
        return an.decode(t.read(), dtype) if dtype in an.DTYPES else t.read()

    def sync(self):
        pass


@requires_cdna3_or_4
@pytest.mark.gpu
def test_cached_lse_off_2d_graph_never_serves_an_lse_request(rt):
    """Regression: the gfx942 2D graph fast path replays a cache hit before any
    support gate, keyed on _cheap_2d_sig (no emit_lse) + buffer ids. An LSE-off
    graph captured on these exact buffers must not be replayed for an LSE-on
    request: the replay would write O and leave lse unwritten.

    On gfx942 the test first proves the hazard is live (the LSE-off launch
    captures a graph whose key the LSE-on request would hit). Then the LSE-on
    call must write O and LSE, match the reference, and add no graph. gfx950 has
    no 2D graph path, so there only the served-and-correct half runs. The gfx942
    graph capture is a torch.cuda.CUDAGraph, so that half runs on real torch
    tensors and skips without torch."""
    if _arch() == "gfx942":
        torch = pytest.importorskip(
            "torch", reason="the gfx942 2D graph capture needs torch.cuda.CUDAGraph"
        )
        dev = _TorchDev(torch)
    else:
        dev = _NumpyDev(rt)
    case = _case2d([256], [256])
    problem = case.problem
    off_problem = dataclasses.replace(problem, emit_lse=False)
    shape = case.q.shape
    out = dev.empty_out(shape, "bf16")
    args = dict(
        q=dev.float_tensor(case.q, "bf16"),
        k=dev.float_tensor(case.k_cache, "bf16"),
        v=dev.float_tensor(case.v_cache, "bf16"),
        out=out,
        cu_seqlens_q=dev.i32(case.cu),
        seqused_k=dev.i32(case.seqused),
        softmax_scale=1.0 / math.sqrt(128),
        block_table=dev.i32(case.block_table),
        softcap=0.0,
        backend="auto",
    )
    before = set(au._2D_GRAPHS)
    run_unified_attention_torch(problem=off_problem, **args)  # warm the 2D graph
    captured = set(au._2D_GRAPHS) - before
    run_unified_attention_torch(problem=off_problem, **args)  # cache hit: replay
    dev.sync()
    if _arch() == "gfx942":
        assert len(captured) == 1, "LSE-off launch did not capture a 2D graph"
        (key,) = captured
        sig = au._cheap_2d_sig(problem)  # the LSE-on request's signature
        assert key[: len(sig)] == sig and id(out) in key[len(sig) :]
        assert set(au._2D_GRAPHS) - before == captured, "second launch was not a replay"
    else:
        assert not captured, "2D graph replay is expected only on gfx942"
    off_out = dev.read(out, "bf16").copy()
    graphs = set(au._2D_GRAPHS)
    # A replay writes ``out``, never ``lse``: poison both in place (same
    # tensors, so the captured graph key still matches).
    dev.poison(out, "bf16")
    lse = dev.nan_lse((problem.total_q, problem.num_query_heads, 1))
    with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
        run_unified_attention_torch(problem=problem, lse=lse, **args)
    dev.sync()
    assert set(au._2D_GRAPHS) == graphs, "an LSE request captured a 2D graph"
    o_ref, lse_ref = _paged_reference(case, [256], [256], 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(
        dev.read(out, "bf16"), off_out, dev.read(lse, "float32"), o_ref, lse_ref, "bf16"
    )


class _View:
    """A tensor-shaped view of ``length`` elements of a shared device ``pool`` at
    element ``offset``, so two buffers can overlap by exactly one 4-byte element."""

    device = None
    is_cuda = True

    def __init__(self, pool, offset, shape, dtype):
        self._pool = pool
        self._offset = offset
        self.shape = tuple(shape)
        self.dtype = dtype

    def data_ptr(self):
        return self._pool.data_ptr() + 4 * self._offset

    def stride(self, dim=None):
        strides = [int(np.prod(self.shape[i + 1 :])) for i in range(len(self.shape))]
        return tuple(strides) if dim is None else strides[dim]

    def is_contiguous(self):
        return True

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return 4

    def dim(self):
        return len(self.shape)

    def storage_offset(self):
        return self._offset

    def untyped_storage(self):
        return types.SimpleNamespace(nbytes=lambda: self._pool._host.nbytes)


def _aliases_last_lse_element(rt, lse_shape, like_shape, like_dtype):
    """An LSE view and a ``like``-shaped 4-byte tensor whose first element shares
    memory with the last element of the LSE view: the smallest possible
    overlap, and otherwise a valid input."""
    n = int(np.prod(lse_shape))
    m = int(np.prod(like_shape))
    pool = DeviceArray(rt, np.zeros(n - 1 + m, dtype=np.float32))
    return (
        _View(pool, 0, lse_shape, "torch.float32"),
        _View(pool, n - 1, like_shape, like_dtype),
    )


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["3d", "tiled"])
@pytest.mark.parametrize(
    "case_name",
    [
        "rows",
        "heads",
        "dtype",
        "strided",
        "host",
        "cu_seqlens_q",
        "seqused_k",
        "block_table",
        "additive_bias",
    ],
)
def test_malformed_lse_buffer_is_refused_before_launch(rt, case_name, backend):
    case = _case([4, 2], [3000, 1029])
    problem = case.problem
    dev = _upload(rt, case)
    cu, su, bt = dev.cu, dev.su, dev.bt
    tq, hq = problem.total_q, problem.num_query_heads
    lse = an.lse_device(rt, (tq, hq, 1))
    if case_name == "rows":
        lse = an.lse_device(rt, (tq - 1, hq, 1))
    elif case_name == "heads":
        lse = an.lse_device(rt, (tq, hq - 1, 1))
    elif case_name == "dtype":
        lse = DeviceArray(rt, np.full((tq, hq, 1), np.nan, dtype=np.float16))
    elif case_name == "strided":
        wide = np.full((tq, hq, 2), np.nan, dtype=np.float32)
        lse = DeviceArray(rt, wide, shape=(tq, hq, 1))
    elif case_name == "host":
        lse.is_cuda = False
    elif case_name == "cu_seqlens_q":
        lse, cu = _aliases_last_lse_element(rt, lse.shape, cu.shape, "torch.int32")
    elif case_name == "seqused_k":
        lse, su = _aliases_last_lse_element(rt, lse.shape, su.shape, "torch.int32")
    elif case_name == "block_table":
        lse, bt = _aliases_last_lse_element(rt, lse.shape, bt.shape, "torch.int32")
    bias = {}
    if case_name == "additive_bias":
        # The reduce must never write LSE over a bias it (or a later replay)
        # reads. The alias is otherwise a valid bias (float32, long enough for
        # the broadcast strides), so only the overlap check can refuse it.
        problem = dataclasses.replace(problem, use_additive_bias=True)
        lse, storage = _aliases_last_lse_element(
            rt, lse.shape, (4096,), "torch.float32"
        )
        bias = dict(additive_bias=storage, additive_bias_sq_stride=0)
    out = an.empty_device(rt, case.q.shape, case.dtype)
    aliased = ("cu_seqlens_q", "seqused_k", "block_table", "additive_bias")
    # The 3D lane gets the scratch-buffer stub; the refusal comes before any launch.
    with U.workspace_allocator(rt) if backend == "3d" else contextlib.nullcontext():
        with pytest.raises(
            ValueError,
            match=f"lse overlaps {case_name}" if case_name in aliased else "lse",
        ):
            run_unified_attention_torch(
                problem=problem,
                q=dev.q,
                k=dev.k,
                v=dev.v,
                out=out,
                cu_seqlens_q=cu,
                seqused_k=su,
                softmax_scale=1.0 / math.sqrt(case.q.shape[-1]),
                block_table=bt,
                softcap=0.0,
                backend=backend,
                lse=lse,
                **bias,
            )
    assert np.isnan(
        an.read(out, case.dtype)
    ).all(), "a kernel ran for a malformed lse buffer"


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _PREFILL_SHAPES)
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_output_and_lse(rt, backend, dtype, d, q_lens, kv_lens):
    """Arch-native routing. On gfx942 bf16 D256 this is the shape the 4-warp
    lean body serves with LSE off (with LSE on it takes the default builder)."""
    case = _case2d(q_lens, kv_lens, d=d, dtype=dtype)
    if backend == "auto":
        assert case.problem.select_path() == "2d", "auto would not route this to 2D"
    lse = _nan_lse(rt, case.problem)
    out = _run2d(rt, case, lse=lse, softcap=0.0, backend=backend)
    off = _run2d_lse_off(rt, case, softcap=0.0, backend=backend)
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(d), 0.0)
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _PREFILL_SHAPES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_tiled_2d_lse_one_query_head_per_kv_head(rt, dtype, q_lens, kv_lens):
    """Hq == Hkv (8:8): NQK = 1, so every CTA row is its own token. backend is
    'tiled' because auto sends the ragged 8:8 shape to split-KV."""
    case = _case2d(q_lens, kv_lens, hq=8, hkv=8, dtype=dtype)
    lse = _nan_lse(rt, case.problem)
    out = _run2d(rt, case, lse=lse, softcap=0.0, backend="tiled")
    off = _run2d_lse_off(rt, case, softcap=0.0, backend="tiled")
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("window", [64, 4096])
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_lse_with_sliding_window(rt, backend, window, dtype):
    """D128 sliding window: on gfx942 the LSE-off route is the 4-warp GQA body,
    so this is where the LSE-on fallback to the default builder is exercised. Seq 0
    (256 queries over 5000 keys) is cut by both windows; seq 1 (64/64) only by 64."""
    q_lens, kv_lens = [256, 64], [5000, 64]
    case = _case2d(q_lens, kv_lens, dtype=dtype, sliding_window=window)
    lse = _nan_lse(rt, case.problem)
    out = _run2d(rt, case, lse=lse, softcap=0.0, backend=backend)
    off = _run2d_lse_off(rt, case, softcap=0.0, backend=backend)
    o_ref, lse_ref = _paged_reference(
        case, q_lens, kv_lens, 1.0 / math.sqrt(128), 0.0, window
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_lse_with_softcap(rt, backend):
    """Softcap is applied before the max, in log2 units: LSE must include it."""
    q_lens, kv_lens = [37, 300, 1], [37, 812, 900]
    case = _case2d(q_lens, kv_lens, softcap=30.0)
    lse = _nan_lse(rt, case.problem)
    out = _run2d(rt, case, lse=lse, softcap=30.0, backend=backend)
    off = _run2d_lse_off(rt, case, softcap=30.0, backend=backend)
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(128), 30.0)
    _assert_2d_result(out, off, lse, o_ref, lse_ref, "bf16")


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", [([37, 300, 1], [37, 812, 900]), ([5], [3])])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_pinned_branch_writes_lse(rt, branch, d, dtype, q_lens, kv_lens):
    """Every epilogue branch of the arch's tiled 2D builder, forced by an
    explicit tuning spec. Branch B (mfma32) is reachable only this way."""
    case = _case2d(q_lens, kv_lens, d=d, dtype=dtype)
    out, lse, off = _branch_pair(rt, case, branch)
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(d), 0.0)
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


def _token_major_view(head_major, q_lens, heads):
    """Reorder a flat head-major LSE buffer to [total_q, Hq, 1]."""
    rows, start = [], 0
    for q_len in q_lens:
        block = head_major[start * heads : (start + q_len) * heads]
        rows.append(block.reshape(heads, q_len).T)
        start += q_len
    return np.concatenate(rows)[..., None]


@pytest.mark.skipif(_ARCH != "gfx950", reason="head-major LSE is gfx950-only")
@pytest.mark.gpu
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [([256, 256], [256, 256]), ([37, 300, 1], [37, 812, 900]), ([5], [3])],
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_gfx950_head_major_lse(rt, branch, dtype, q_lens, kv_lens):
    """lse_layout="head_major" writes each sequence as [Hq, q_len]; for equal
    lengths that is the [B, Hq, S, 1] SDPA stats tensor. O is unchanged."""
    case = _case2d(q_lens, kv_lens, dtype=dtype)
    problem = case.problem
    spec = _branch_spec(problem, branch)
    spec = dataclasses.replace(
        spec,
        kernel_spec=dataclasses.replace(spec.kernel_spec, lse_layout="head_major"),
    )
    assert spec.kernel_name().endswith("_lsehm")
    lse = _nan_lse(rt, problem)  # same element count; the kernel writes it flat
    out = _run2d(rt, case, lse=lse, softcap=0.0, backend="tiled", tuning_spec=spec)
    off_problem = dataclasses.replace(problem, emit_lse=False)
    off = _run2d(
        rt,
        case,
        problem=off_problem,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(off_problem, branch),
    )
    o_ref, lse_ref = _paged_reference(
        case, q_lens, kv_lens, 1.0 / math.sqrt(case.q.shape[-1]), 0.0
    )
    hq = problem.num_query_heads
    flat = lse.read().reshape(-1)
    got = _token_major_view(flat, q_lens, hq)
    _assert_2d_result(out, off, got, o_ref, lse_ref, dtype)
    if len(set(q_lens)) == 1:
        bhs1 = flat.reshape(len(q_lens), hq, q_lens[0], 1)
        ref = lse_ref.reshape(len(q_lens), q_lens[0], hq, 1).transpose(0, 2, 1, 3)
        finite = np.isfinite(ref)
        np.testing.assert_allclose(
            bhs1[finite], ref[finite], atol=_LSE_ATOL, rtol=_LSE_RTOL
        )


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("branch", ["transposed32", "mfma32", "striped16"])
def test_dispatcher_2d_lse_end_to_end(rt, branch):
    """Request -> pinned 2D tuning spec -> bind_torch(tensors with 'lse') ->
    launch. (The generic unified_2d candidate has no Torch binding; executable
    dispatch goes through the tuning candidates, as for split-KV.)"""
    from dispatch.attention import AttentionRequest, dispatch_attention

    arch = _arch()
    q_lens, kv_lens = [256, 256], [256, 700]  # total_q = batch * seqlen_q
    case = _case2d(q_lens, kv_lens)
    dev = _upload(rt, case)

    def request(emit_lse):
        return AttentionRequest(
            batch=2,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=256,
            seqlen_k=700,
            hdim_q=128,
            hdim_v=128,
            arch=arch,
            mask_type=2,  # bottom-right causal, as in _paged_reference
            dtype="bf16",
            emit_lse=emit_lse,
            algorithm="unified_tuning",
            spec_id=f"{arch}_u2d_{_BRANCH_VARIANT[arch][branch]}",
        )

    on = dispatch_attention(request(True))
    off = dispatch_attention(request(False))
    assert on.spec.path == "2d"
    assert _epilogue_branch(on.spec.kernel_spec) == branch
    assert on.spec.kernel_spec.emit_lse is True
    assert off.spec.kernel_spec.emit_lse is False
    tensors = dict(
        q=dev.q,
        k=dev.k,
        v=dev.v,
        cu_seqlens_q=dev.cu,
        seqused_k=dev.su,
        block_table=dev.bt,
    )
    out = an.empty_device(rt, case.q.shape, "bf16")
    lse = _nan_lse(rt, case.problem)
    off_out = an.empty_device(rt, case.q.shape, "bf16")
    _on_cuda(*tensors.values(), out, lse, off_out)
    on.bind_torch({**tensors, "out": out, "lse": lse}).launch()
    off.bind_torch({**tensors, "out": off_out}).launch()
    o_ref, lse_ref = _paged_reference(case, q_lens, kv_lens, 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(
        an.read(out, "bf16"), an.read(off_out, "bf16"), lse, o_ref, lse_ref, "bf16"
    )


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,score", [("fp16", -20.0), ("bf16", -200.0)])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_2d_lse_masked_leading_tile_very_negative_scores(rt, branch, dtype, score):
    """Sliding window 64 over 256 tokens: some rows of each CTA see a fully
    masked first KV tile. Every visible score is `score` (natural log). Every
    row sees keys, so every LSE must be finite (about score + ln 64) and O must
    match the reference, on every epilogue branch."""
    case = _case2d([256], [256], dtype=dtype, sliding_window=64)
    a = 1.0 if dtype == "fp16" else 4.0
    case.q[...] = a
    case.k_cache[...] = score / (a * math.sqrt(128))
    case.k_cache[...] = an.round_to(case.k_cache, dtype)
    out, lse, off = _branch_pair(rt, case, branch)
    o_ref, lse_ref = _paged_reference(
        case, [256], [256], 1.0 / math.sqrt(128), 0.0, sliding_window=64
    )
    assert np.isfinite(lse_ref).all(), "the case must give every row a key"
    assert np.isfinite(lse.read()).all(), "a row with visible keys got LSE -inf/NaN"
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


_NAN_ROW, _NAN_HEAD = 100, 5


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_2d_lse_nan_query_row_stays_nan(rt, branch, dtype):
    """A NaN in one query row gives NaN O and NaN LSE for that (row, head)
    only. Its softmax sum l is NaN; !(l == 0) keeps row_has_mass true, so
    lse_store propagates NaN instead of writing the keyless -inf."""
    case = _case2d([256], [256], dtype=dtype)
    case.q[_NAN_ROW, _NAN_HEAD] = np.nan
    problem = case.problem
    off_problem = dataclasses.replace(problem, emit_lse=False)
    lse = _nan_lse(rt, problem, fill=np.inf)
    out = _run2d(
        rt,
        case,
        lse=lse,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(problem, branch),
    )
    off = _run2d(
        rt,
        case,
        problem=off_problem,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_branch_spec(off_problem, branch),
    )
    o_ref, lse_ref = _paged_reference(
        case, [256], [256], 1.0 / math.sqrt(case.q.shape[-1]), 0.0
    )
    lse = lse.read()
    assert not np.isposinf(lse).any(), "an LSE row was left unwritten"
    assert np.array_equal(out, off, equal_nan=True)
    assert np.isnan(lse[_NAN_ROW, _NAN_HEAD]).all(), "NaN row got a non-NaN LSE"
    assert np.isnan(out[_NAN_ROW, _NAN_HEAD]).all(), "NaN row was hidden in O"
    rest = np.ones((problem.total_q, problem.num_query_heads), dtype=bool)
    rest[_NAN_ROW, _NAN_HEAD] = False
    assert np.isfinite(lse[..., 0][rest]).all(), "NaN leaked into another LSE"
    assert np.isfinite(out[rest]).all(), "NaN leaked into another (row, head)"
    assert _max_abs(out[rest], o_ref[rest]) < _tolerance(dtype)
    np.testing.assert_allclose(
        lse[..., 0][rest], lse_ref[..., 0][rest], atol=_LSE_ATOL, rtol=_LSE_RTOL
    )


# (variant id, geometry suffix, knobs, epilogue branch). Per arch, so every
# collected case runs: gfx950 has no q-major grid.
_ROW_MAP_VARIANTS = {
    "gfx942": [
        (
            "qmajor_transposed32",
            "transposed_x8_nw1_mw32_t2xb_llvm",
            {"use_q_major_grid": True},
            "transposed32",
        ),
        (
            "qmajor_transposed32_s1",
            "transposed_x8_nw1_mw32_t2xb_llvm",
            {"use_q_major_grid": True, "use_transposed_scalar_state": True},
            "transposed32",
        ),
        (
            "qmajor_striped16",
            "narrow_nw1_mw16_t1xb_llvm",
            {"use_q_major_grid": True},
            "striped16",
        ),
        ("striped16_mw32", "narrow_nw1_mw32_t1xb_llvm", {}, "striped16"),
        ("striped16_mw32_nw2", "narrow_nw2_mw32_t1xb_llvm", {}, "striped16"),
    ],
    "gfx950": [
        ("striped16_mw32", "narrow_nw1_mw32_t1xb_llvm", {}, "striped16"),
        ("striped16_mw32_nw2", "narrow_nw2_mw32_t1xb_llvm", {}, "striped16"),
    ],
}


def _variant_spec(problem, geometry, knobs, branch):
    from dispatch.attention.unified_rules import canonicalize_tuning_spec

    arch = _arch()
    if arch == "gfx942":
        from dispatch.attention.gfx942_unified import (
            GFX942_TUNING_VARIANTS as variants,
        )
    else:
        from dispatch.attention.gfx950_unified import (
            GFX950_TUNING_VARIANTS as variants,
        )
    spec_id = f"{arch}_u2d_{geometry}"
    variant = next(v for v in variants if v.spec_id == spec_id)
    spec, why = canonicalize_tuning_spec(problem, variant, knobs)
    assert spec is not None, f"{spec_id} {knobs}: {why}"
    ks = spec.kernel_spec
    assert _epilogue_branch(ks) == branch, f"{spec_id} selects {_epilogue_branch(ks)}"
    for name, value in knobs.items():
        assert getattr(ks, name) == value, f"{spec_id}: {name} did not stick"
    assert ks.emit_lse is problem.emit_lse
    return spec


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", [([37, 300, 1], [37, 812, 900]), ([5], [3])])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "vid,geometry,knobs,branch",
    _ROW_MAP_VARIANTS.get(_arch(), []),
    ids=[v[0] for v in _ROW_MAP_VARIANTS.get(_arch(), [])],
)
def test_row_mapping_variant_writes_lse(
    rt, vid, geometry, knobs, branch, dtype, q_lens, kv_lens
):
    """Swapped grid, two atoms per warp and a second warp change which lane
    and slot owns each (token, head). A wrong mapping writes the right values
    to the wrong index, which only a reference comparison sees. The ragged
    shape spans three sequences; [5]/[3] has keyless rows 0-1."""
    case = _case2d(q_lens, kv_lens, dtype=dtype)
    problem = case.problem
    off_problem = dataclasses.replace(problem, emit_lse=False)
    lse = _nan_lse(rt, problem)
    out = _run2d(
        rt,
        case,
        lse=lse,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_variant_spec(problem, geometry, knobs, branch),
    )
    off = _run2d(
        rt,
        case,
        problem=off_problem,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=_variant_spec(off_problem, geometry, knobs, branch),
    )
    o_ref, lse_ref = _paged_reference(
        case, q_lens, kv_lens, 1.0 / math.sqrt(case.q.shape[-1]), 0.0
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


@requires_cdna3_or_4
@pytest.mark.gpu
@pytest.mark.parametrize(
    "q_lens,kv_lens,branch",
    [
        ([37, 300, 1], [37, 812, 900], "transposed32"),
        ([5], [3], "striped16"),
    ],
)
def test_tiled_2d_lse_with_additive_bias(rt, q_lens, kv_lens, branch):
    """Additive bias is added to the log2 score before the running max, so LSE
    must include it. gfx950 default routing: the ragged shape takes branch A
    (16 slots: bias turns scalar state off), 5/3 takes C and keeps two keyless
    rows. gfx942 has no bias kernel: the tiled 2D gate must refuse it."""
    case = _case(q_lens, kv_lens)
    problem = dataclasses.replace(case.problem, use_additive_bias=True)
    case = dataclasses.replace(case, problem=problem)
    arch = _arch()
    ok, why = au.supports_native_unified_attention_tiled(problem, arch)
    if arch == "gfx942":
        assert not ok and "only implemented on gfx950" in why, why
        return
    assert ok, f"the tiled 2D gate refuses LSE + bias: {why}"
    assert _epilogue_branch(au._tiled_spec_from_problem(problem, arch)) == branch
    bias_host = np.random.default_rng(1).standard_normal(
        (len(q_lens), 32, problem.max_seqlen_q, problem.max_seqlen_k), dtype=np.float32
    )
    bias = U.StorageArray(rt, bias_host)
    bias_args = dict(
        additive_bias=bias,
        additive_bias_batch_stride=bias.stride(0),
        additive_bias_head_stride=bias.stride(1),
        additive_bias_sq_stride=bias.stride(2),
    )
    dev = _upload(rt, case)
    lse = _nan_lse(rt, problem)
    out = _run2d(rt, case, lse=lse, softcap=0.0, backend="tiled", dev=dev, **bias_args)
    off = _run2d(
        rt,
        case,
        problem=dataclasses.replace(problem, emit_lse=False),
        lse=None,
        softcap=0.0,
        backend="tiled",
        dev=dev,
        **bias_args,
    )
    o_ref, lse_ref = _paged_reference(
        case, q_lens, kv_lens, 1.0 / math.sqrt(128), 0.0, additive_bias=bias_host
    )
    _assert_2d_result(out, off, lse, o_ref, lse_ref, "bf16")
