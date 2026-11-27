# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unified paged attention LSE output -- split-KV (3D) reduce and tiled 2D
epilogues on gfx942/gfx950 -- vs an independent FP64 numpy paged reference.

The launches run on ``rocke.runtime`` device arrays. Only the gfx942 2D graph
regression needs torch: graph replay captures through torch, so it serves torch
tensors only."""

import dataclasses
import itertools
import math
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from kernels.common import attention_unified as au
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from rocke.numeric import storage
from rocke.numeric.compare import Tolerance, assert_within_tolerance
from rocke.runtime.device_array import DeviceArray, host_values
from rocke.runtime.hip_module import get_device_arch
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.host_arrays import HostArray
from rocke.tests.lanes import TICKET, import_or_skip_until

from ._attention_paged_reference import (
    PAGED_OUTPUT_ATOL,
    PAGED_OUTPUT_RTOL,
    paged_attention_reference_with_lse,
)
from ._attention_tolerance import LSE_TOLERANCE, attention_tolerance
from ._attention_unified_support import random_half


def _arch():
    arch = get_device_arch()
    return arch if arch in ("gfx942", "gfx950") else None


@pytest.fixture
def cdna3_or_4():
    current_device_arch_in(("gfx942", "gfx950"), "the unified LSE numeric lane")


# The split-KV O bound. On the decode shapes max |O| is only about 0.1-0.3, so
# the output-dtype bound above would pass one misaddressed K or V page or a
# dropped key tile; the paged output bound (PAGED_OUTPUT_ATOL/RTOL) holds there.
# The PV product takes the probabilities in the KV dtype, so its error grows
# with the values it averages: on the short sequences, where a row averages a
# few keys and |O| reaches the values' own size, the absolute bound is 1/128 of
# the largest |O| instead (still far below what a misplaced page moves O by
# there; test_the_split_kv_bound_fails_one_misplaced_page).
_SPLIT_KV_O_ATOL_PER_MAX_O = 2**-7


def _paged_reference(data, scale, softcap, sliding_window=0, additive_bias=None):
    """FP64 O ``[total_q, Hq, D]`` and natural-log LSE ``[total_q, Hq, 1]`` (as
    float32) with the unified bottom-right causal mask (key k visible to query p
    iff k <= diag, diag = p + kv_len - q_len) and, when ``sliding_window`` > 0,
    diag - k < sliding_window. Rows with no key get O = 0 and LSE = -inf.
    ``additive_bias`` is ``[seq, q_head, q_row, key]``, added after the softcap,
    as the kernels do."""
    outs, lses = [], []
    starts = itertools.accumulate([0, *data.q_lens])
    for i, (ql, kl, start) in enumerate(zip(data.q_lens, data.kv_lens, starts)):
        out, lse = paged_attention_reference_with_lse(
            data.q[start : start + ql],
            data.k_cache,
            data.v_cache,
            kl,
            data.block_table[i],
            scale,
            sliding_window=sliding_window,
            softcap=softcap,
            score_bias=(
                None if additive_bias is None else additive_bias[i, :, :ql, :kl]
            ),
        )
        outs.append(out)
        lses.append(lse)
    return np.concatenate(outs), np.concatenate(lses)[..., None]


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
    """The problem and its host data: Q/K/V float32 values exact in ``dtype``,
    shuffled physical pages (catches any block-table indexing shortcut). A test
    may change the values before it launches."""
    rng = np.random.default_rng(seed)
    total_q, num_seqs = sum(q_lens), len(q_lens)
    max_blocks = max(1, max((kl + bs - 1) // bs for kl in kv_lens))
    num_blocks = num_seqs * max_blocks + 3
    data = SimpleNamespace(
        q=random_half(rng, (total_q, hq, d), dtype),
        k_cache=random_half(rng, (num_blocks, bs, hkv, d), dtype),
        v_cache=random_half(rng, (num_blocks, bs, hkv, d), dtype),
        block_table=rng.permutation(num_blocks)[: num_seqs * max_blocks]
        .reshape(num_seqs, max_blocks)
        .astype(np.int32),
        q_lens=list(q_lens),
        kv_lens=list(kv_lens),
        dtype=dtype,
    )
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
    return problem, data


def _device(values, dtype):
    return DeviceArray(storage.encode(values, dtype), dtype)


def _device_inputs(data):
    """Q/K/V and the metadata as device arrays, uploaded from ``data`` now."""
    return {
        "q": _device(data.q, data.dtype),
        "k": _device(data.k_cache, data.dtype),
        "v": _device(data.v_cache, data.dtype),
        "cu_seqlens_q": DeviceArray(
            np.array([0, *itertools.accumulate(data.q_lens)], np.int32)
        ),
        "seqused_k": DeviceArray(np.array(data.kv_lens, np.int32)),
        "block_table": DeviceArray(data.block_table),
    }


def _new_out(data, fill=np.nan):
    """An output array in the data's dtype holding ``fill`` (NaN by default, so
    an element left unwritten fails a comparison)."""
    return _device(np.full(data.q.shape, fill, np.float32), data.dtype)


#: The output fill of the NaN-query-row tests: finite and far from any output,
#: so an O row the kernel does not store keeps a value the NaN check sees. A NaN
#: fill would pass as the expected NaN.
_FINITE_OUT_FILL = 1024.0


def _run(
    problem,
    data,
    *,
    lse,
    softcap,
    backend="3d",
    tuning_spec=None,
    out_fill=np.nan,
    **launch_options,
):
    """Launch and return O as float32; ``lse`` is the caller's LSE array (or
    None), ``out_fill`` the output's contents before the launch."""
    out = _new_out(data, out_fill)
    run_unified_attention_torch(
        problem=problem,
        **_device_inputs(data),
        out=out,
        softmax_scale=1.0 / math.sqrt(data.q.shape[-1]),
        softcap=softcap,
        backend=backend,
        tuning_spec=tuning_spec,
        lse=lse,
        **launch_options,
    )
    return storage.decode(out.values(), data.dtype)


def _lse_array(problem, fill):
    return DeviceArray(
        np.full((problem.total_q, problem.num_query_heads, 1), fill, np.float32)
    )


_SHAPES = [
    # (q_lens, kv_lens)               what it covers
    ([1, 1, 1, 1], [4096, 777, 1500, 2049]),  # decode, ragged KV, partial last page
    ([4, 2], [3000, 1029]),  # spec-decode
    ([64], [64]),  # s_q == s_kv
    ([5], [3]),  # bottom-right s_q > s_kv: rows 0,1 empty
    ([1, 1], [0, 600]),  # kv_len == 0: whole sequence empty
]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("q_lens,kv_lens", _SHAPES)
def test_split_kv_output_and_lse(cdna3_or_4, dtype, d, q_lens, kv_lens):
    problem, data = _case(q_lens, kv_lens, d=d, dtype=dtype)
    lse = _lse_array(problem, np.nan)
    out = _run(problem, data, lse=lse, softcap=0.0)
    off = _run(
        dataclasses.replace(problem, emit_lse=False), data, lse=None, softcap=0.0
    )
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(d), 0.0)
    np.testing.assert_array_equal(out, off, err_msg="LSE-on changed O")
    _assert_matches_reference(out, lse.values(), o_ref, lse_ref)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_split_kv_lse_nan_query_row_stays_nan(cdna3_or_4, dtype):
    """The split-KV reduce follows the tiled 2D rule: a NaN query row gives
    NaN O and NaN LSE for that (token, head) only, never the keyless -inf.
    Its merged sum is NaN, and !(l == 0) keeps row_has_mass true."""
    q_lens, kv_lens = [1, 1, 1, 1], [4096, 777, 1500, 2049]
    nan_token, nan_head = 1, 5
    problem, data = _case(q_lens, kv_lens, dtype=dtype)
    data.q[nan_token, nan_head] = np.nan
    lse = _lse_array(problem, np.inf)
    out = _run(problem, data, lse=lse, softcap=0.0, out_fill=_FINITE_OUT_FILL)
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(data.q.shape[-1]), 0.0)
    _assert_only_the_nan_row_is_nan(
        out, lse.values(), o_ref, lse_ref, (nan_token, nan_head), _assert_split_kv_o
    )


def _assert_only_the_nan_row_is_nan(out, lse, o_ref, lse_ref, nan_row, assert_o):
    """NaN O and LSE at ``nan_row`` (token, head) only; every other row is
    written, finite and matches the reference: its O by ``assert_o(out,
    o_ref)``, its LSE within the fp32 LSE bound."""
    assert not np.isposinf(lse).any(), "an LSE row was left unwritten"
    assert np.isnan(lse[nan_row]).all(), "NaN row got a non-NaN LSE"
    assert np.isnan(out[nan_row]).all(), "NaN row was hidden in O"
    rest = np.ones(lse.shape[:2], bool)
    rest[nan_row] = False
    assert np.isfinite(lse[..., 0][rest]).all(), "NaN leaked into another LSE"
    assert np.isfinite(out[rest]).all(), "NaN leaked into another (row, head)"
    assert_o(out[rest], o_ref[rest])
    assert_within_tolerance(
        lse[..., 0][rest], lse_ref[..., 0][rest], LSE_TOLERANCE, context="LSE"
    )


@pytest.mark.gpu
def test_split_kv_lse_with_softcap(cdna3_or_4):
    q_lens, kv_lens = [1, 1], [2048, 900]
    problem, data = _case(q_lens, kv_lens, softcap=30.0)
    lse = _lse_array(problem, np.nan)
    _run(problem, data, lse=lse, softcap=30.0)
    _, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 30.0)
    _assert_split_kv_lse(lse.values(), lse_ref)


def _assert_split_kv_o(out, o_ref):
    """O matches the reference element-wise within the split-KV bound."""
    largest = float(np.abs(o_ref).max(initial=0.0))
    atol = max(PAGED_OUTPUT_ATOL, _SPLIT_KV_O_ATOL_PER_MAX_O * largest)
    assert_within_tolerance(
        out, o_ref, Tolerance(atol=atol, rtol=PAGED_OUTPUT_RTOL), context="O"
    )


def _assert_split_kv_lse(lse, lse_ref):
    """Every LSE row is written, the keyless rows are -inf, and the rest match
    the reference within the fp32 LSE bound."""
    assert not np.isnan(lse).any(), "an LSE row was left unwritten"
    np.testing.assert_array_equal(np.isneginf(lse), np.isneginf(lse_ref))
    finite = np.isfinite(lse_ref)
    assert_within_tolerance(lse[finite], lse_ref[finite], LSE_TOLERANCE, context="LSE")


def _assert_matches_reference(out, lse, o_ref, lse_ref):
    assert np.isfinite(out).all(), "non-finite O"
    _assert_split_kv_o(out, o_ref)
    _assert_split_kv_lse(lse, lse_ref)


# ---------------------------------------------------------------------------
# Tiled 2D LSE (gfx942 / gfx950 build_unified_attention_2d_tiled)
# ---------------------------------------------------------------------------

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
    ok, why = au.supports_native_unified_attention_tiled(case[0], _arch())
    assert ok, f"the tiled 2D gate refuses this LSE problem: {why}"
    return case


def _run2d(problem, data, *, lse, softcap, backend, tuning_spec=None, out_fill=np.nan):
    """One 2D launch. The split-KV runner is patched to fail, so a pass proves
    the tiled 2D path wrote O (and LSE)."""
    with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
        return _run(
            problem,
            data,
            lse=lse,
            softcap=softcap,
            backend=backend,
            tuning_spec=tuning_spec,
            out_fill=out_fill,
        )


def _run2d_lse_off(problem, data, *, softcap, backend):
    """The LSE-off launch of the kernel the LSE-on request runs. With LSE on,
    the gfx942 4-warp cohorts fall back to the default builder; force that
    here so the outputs compare one kernel with and without its LSE store
    (a no-op on gfx950)."""
    off = dataclasses.replace(problem, emit_lse=False)
    with mock.patch.object(au, "_gfx942_4warp_eligible", lambda problem, arch: False):
        return _run2d(off, data, lse=None, softcap=softcap, backend=backend)


def _assert_2d_o(out, o_ref, dtype):
    """O matches the reference within the output dtype's bound."""
    assert_within_tolerance(out, o_ref, attention_tolerance(dtype), context="O")


def _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype):
    assert not np.isnan(lse).any(), "an LSE row was left unwritten"
    np.testing.assert_array_equal(out, off, err_msg="LSE-on changed O")
    assert np.isfinite(out).all(), "non-finite O"
    _assert_2d_o(out, o_ref, dtype)
    keyless = np.isneginf(lse_ref[..., 0])  # [total_q, Hq]
    assert (out[keyless] == 0).all(), "a row with no visible key has O != 0"
    np.testing.assert_array_equal(
        np.isneginf(lse),
        np.isneginf(lse_ref),
        err_msg="the -inf (keyless) LSE rows differ from the reference",
    )
    assert not np.isposinf(lse).any(), "+inf LSE"
    finite = np.isfinite(lse_ref)
    assert_within_tolerance(lse[finite], lse_ref[finite], LSE_TOLERANCE, context="LSE")


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


# Decode with ragged KV and spec-decode: the shapes the split-KV path serves.
_DECODE_SHAPES = [([1, 1, 1, 1], [4096, 777, 1500, 2049]), ([4, 2], [3000, 1029])]


@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _DECODE_SHAPES)
def test_split_kv_lse_with_one_query_head_per_kv_head(cdna3_or_4, q_lens, kv_lens):
    """qpk = 1 (Hq == Hkv): a different query-head-to-row layout than GQA."""
    problem, data = _case(q_lens, kv_lens, hq=8, hkv=8)
    lse = _lse_array(problem, np.nan)
    out = _run(problem, data, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0)
    _assert_matches_reference(out, lse.values(), o_ref, lse_ref)


@pytest.mark.gpu
@pytest.mark.parametrize("window", [128, 1000])
@pytest.mark.parametrize("q_lens,kv_lens", _DECODE_SHAPES)
def test_split_kv_lse_with_sliding_window(cdna3_or_4, q_lens, kv_lens, window):
    """Sliding window empties the segments left of the window for every row."""
    problem, data = _case(q_lens, kv_lens, sliding_window=window)
    lse = _lse_array(problem, np.nan)
    out = _run(problem, data, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0, window)
    _assert_matches_reference(out, lse.values(), o_ref, lse_ref)


_PINNED_SPLIT_KV_SPEC = {
    "gfx942": "gfx942_u3d_splitkv_seg16_t1xb",
    "gfx950": "gfx950_u3d_splitkv_seg16_t1xb",
}


@pytest.mark.gpu
def test_pinned_split_kv_spec_end_to_end(cdna3_or_4):
    """The dispatcher route: a request pinned to a split-KV tuning spec with an
    explicit segment count, bound to real device arrays and launched."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    q_lens, kv_lens = [1, 1, 1, 1], [4096, 777, 1500, 2049]
    problem, data = _case(q_lens, kv_lens)
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
    out, lse = _new_out(data), _lse_array(problem, np.nan)
    result.bind_torch({**_device_inputs(data), "out": out, "lse": lse}).launch()
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0)
    _assert_matches_reference(
        storage.decode(out.values(), "bf16"),
        lse.values(),
        o_ref,
        lse_ref,
    )


# (q_len, kv_len): the last KV tile holds only keys row 0 cannot see (bottom-right
# causal: row 0 sees kv_len - q_len + 1 keys), for any tile of 16, 32 or 64; 5/3
# also leaves rows 0 and 1 with no visible key at all.
_MASKED_TILE_SHAPES = [(4, 17), (4, 33), (4, 65), (5, 3)]


def _masked_tile_case(q_len, kv_len):
    """One sequence whose every visible logit is -200: Q=2, K=-12.5, D=64,
    scale 1/8."""
    problem, data = _case([q_len], [kv_len], d=64)
    data.q[...] = 2.0
    data.k_cache[...] = -12.5
    return problem, data


@pytest.mark.gpu
@pytest.mark.parametrize("emit_lse", [True, False])
@pytest.mark.parametrize("q_len,kv_len", _MASKED_TILE_SHAPES)
def test_masked_segment_does_not_underflow_the_merge(
    cdna3_or_4, q_len, kv_len, emit_lse
):
    """A segment whose tiles are all masked for a row stores max=0, sum=0. With
    every visible logit at -200 (Q=2, K=-12.5, D=64, scale 1/8), letting that
    max set the merge max underflows every real segment: O became 0 and LSE
    -inf for rows that do see keys. Keyless rows must stay O=0, LSE=-inf."""
    problem, data = _masked_tile_case(q_len, kv_len)
    problem = dataclasses.replace(problem, emit_lse=emit_lse)
    lse = _lse_array(problem, np.nan) if emit_lse else None
    out = _run(problem, data, lse=lse, softcap=0.0)
    o_ref, lse_ref = _paged_reference(data, 0.125, 0.0)
    assert np.isfinite(out).all(), "the merge produced a non-finite O"
    _assert_split_kv_o(out, o_ref)
    if emit_lse:
        assert np.isfinite(lse_ref).any(), "the case has no row with visible keys"
        _assert_split_kv_lse(lse.values(), lse_ref)


@pytest.mark.gpu
def test_auto_backend_writes_lse_on_a_2d_routed_shape(cdna3_or_4):
    """A shape the runner used to refuse: backend='auto' routes it to tiled
    2D, which now writes LSE itself (the split-KV runner is patched to fail)."""
    problem, data = _case2d([256], [256])
    assert problem.select_path() == "2d"
    lse = _lse_array(problem, np.nan)
    out = _run2d(problem, data, lse=lse, softcap=0.0, backend="auto")
    off = _run2d_lse_off(problem, data, softcap=0.0, backend="auto")
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, "bf16")


def _forbid_tiled_2d(*_args, **_kwargs):
    raise AssertionError("an LSE request the tiled 2D gate declines reached it")


# D256 prefill at block size 32: shapes gfx942's tiled 2D gate declines with LSE
# on, so they run split-KV.
_D256_PREFILL_SHAPES = [([2048], [2048]), ([37, 300, 1], [37, 812, 900])]


@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _D256_PREFILL_SHAPES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_auto_hands_lse_to_split_kv_when_tiled_2d_declines(
    cdna3_or_4, dtype, q_lens, kv_lens
):
    """gfx942 D256 prefill at block size 32 routes to 2D, but the tiled 2D
    gate declines it with LSE on (LDS budget). backend='auto' runs split-KV,
    which writes LSE (the tiled 2D launcher is patched to fail)."""
    if _arch() != "gfx942":
        pytest.skip("the tiled 2D LDS gate declines this shape on gfx942 only")
    problem, data = _case(q_lens, kv_lens, d=256, dtype=dtype, bs=32)
    assert problem.select_path() == "2d"
    ok, why = au.supports_native_unified_attention_tiled(problem, "gfx942")
    assert not ok and "LDS budget" in why
    lse = _lse_array(problem, np.nan)
    with mock.patch.object(au, "_get_2d_launcher", _forbid_tiled_2d):
        out = _run(problem, data, lse=lse, softcap=0.0, backend="auto")
    off = _run(
        dataclasses.replace(problem, emit_lse=False), data, lse=None, softcap=0.0
    )
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(256), 0.0)
    np.testing.assert_array_equal(out, off, err_msg="LSE-on changed the split-KV O")
    _assert_matches_reference(out, lse.values(), o_ref, lse_ref)


# Every shape a split-KV O check above runs: (q_lens, kv_lens, how its case is
# built).
_SPLIT_KV_CHECKED_CASES = [
    *(
        pytest.param(list(q_lens), list(kv_lens), _case, id=f"shape-{q_lens}-{kv_lens}")
        for q_lens, kv_lens in dict.fromkeys(
            (tuple(q_lens), tuple(kv_lens))
            for q_lens, kv_lens in _SHAPES + _DECODE_SHAPES
        )
    ),
    *(
        pytest.param(
            [q_len],
            [kv_len],
            lambda q_lens, kv_lens: _masked_tile_case(q_lens[0], kv_lens[0]),
            id=f"masked-tile-{q_len}-{kv_len}",
        )
        for q_len, kv_len in _MASKED_TILE_SHAPES
    ),
    *(
        pytest.param(
            q_lens,
            kv_lens,
            lambda q_lens, kv_lens: _case(q_lens, kv_lens, d=256, bs=32),
            id=f"d256-prefill-{q_lens}-{kv_lens}",
        )
        for q_lens, kv_lens in _D256_PREFILL_SHAPES
    ),
]


@pytest.mark.parametrize("q_lens,kv_lens,make_case", _SPLIT_KV_CHECKED_CASES)
def test_the_split_kv_bound_fails_one_misplaced_page(q_lens, kv_lens, make_case):
    """The split-KV O bound is tight enough to fail a launch that read one
    page of the longest sequence from the wrong place: on every shape the
    split-KV O checks run, the reference with that page misplaced differs from
    the true one by more than the bound."""
    _, data = make_case(q_lens, kv_lens)
    scale = 1.0 / math.sqrt(data.q.shape[-1])
    o_ref, _ = _paged_reference(data, scale, 0.0)
    longest = int(np.argmax(kv_lens))
    block_size = data.k_cache.shape[1]
    pages_used = {
        int(page)
        for seq, kv_len in enumerate(kv_lens)
        for page in data.block_table[seq, : -(-kv_len // block_size)]
    }
    spare = next(
        (page for page in range(data.k_cache.shape[0]) if page not in pages_used),
        None,
    )
    assert spare is not None, "the case's cache has no page the sequences leave free"
    block_table = data.block_table.copy()
    block_table[longest, 0] = spare
    misplaced, _ = _paged_reference(
        SimpleNamespace(**{**vars(data), "block_table": block_table}), scale, 0.0
    )
    with pytest.raises(AssertionError):
        _assert_split_kv_o(misplaced, o_ref)


@pytest.mark.gpu
def test_cached_lse_off_2d_graph_never_serves_an_lse_request(cdna3_or_4):
    """Regression: the gfx942 2D graph fast path replays a cache hit before any
    support gate, keyed on _cheap_2d_sig (no emit_lse) + buffer ids. An LSE-off
    graph captured on these exact buffers must not be replayed for an LSE-on
    request: the replay would write O and leave lse unwritten.

    Graph replay captures through torch, so it serves torch tensors only and
    this test needs torch. On gfx942 the test first proves the hazard is live
    (the LSE-off launch captures a graph whose key the LSE-on request would
    hit). Then the LSE-on call must write O and LSE, match the reference, and
    add no graph. gfx950 has no 2D graph path, so there only the
    served-and-correct half runs."""
    torch = import_or_skip_until("torch", TICKET)
    problem, data = _case2d([256], [256])
    off_problem = dataclasses.replace(problem, emit_lse=False)

    def tensor(values, dtype=None):
        """A CUDA tensor of ``values``: bf16 when ``dtype`` is None (from its
        bits), else the numpy array as is."""
        if dtype is not None:
            return torch.from_numpy(np.ascontiguousarray(values)).cuda()
        bits = storage.encode(values, data.dtype).view(np.int16)
        return torch.from_numpy(bits).cuda().view(torch.bfloat16)

    out = torch.empty((*data.q.shape,), dtype=torch.bfloat16, device="cuda")
    args = dict(
        q=tensor(data.q),
        k=tensor(data.k_cache),
        v=tensor(data.v_cache),
        out=out,
        cu_seqlens_q=tensor(np.array([0, 256], np.int32), np.int32),
        seqused_k=tensor(np.array([256], np.int32), np.int32),
        softmax_scale=1.0 / math.sqrt(128),
        block_table=tensor(data.block_table, np.int32),
        softcap=0.0,
        backend="auto",
    )
    before = set(au._2D_GRAPHS)
    run_unified_attention_torch(problem=off_problem, **args)  # warm the 2D graph
    captured = set(au._2D_GRAPHS) - before
    run_unified_attention_torch(problem=off_problem, **args)  # cache hit: replay
    torch.cuda.synchronize()
    if _arch() == "gfx942":
        assert len(captured) == 1, "LSE-off launch did not capture a 2D graph"
        (key,) = captured
        sig = au._cheap_2d_sig(problem)  # the LSE-on request's signature
        assert key[: len(sig)] == sig and id(out) in key[len(sig) :]
        assert set(au._2D_GRAPHS) - before == captured, "second launch was not a replay"
    else:
        assert not captured, "2D graph replay is expected only on gfx942"
    off_out = storage.decode(host_values(out), "bf16")
    graphs = set(au._2D_GRAPHS)
    # A replay writes ``out``, never ``lse``: poison both in place (same
    # tensors, so the captured graph key still matches).
    out.fill_(float("nan"))
    lse = torch.full(
        (problem.total_q, problem.num_query_heads, 1), float("nan"), device="cuda"
    )
    with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
        run_unified_attention_torch(problem=problem, lse=lse, **args)
    torch.cuda.synchronize()
    assert set(au._2D_GRAPHS) == graphs, "an LSE request captured a 2D graph"
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(
        storage.decode(host_values(out), "bf16"),
        off_out,
        host_values(lse),
        o_ref,
        lse_ref,
        "bf16",
    )


def _aliases_last_lse_element(shape, like_shape):
    """An LSE array of ``shape`` and an array of ``like_shape`` whose first
    element shares memory with the LSE's last element: the smallest possible
    overlap. Both view one 4-byte buffer (the overlap check reads addresses
    only, and it runs before any other check of the aliased input)."""
    lse_size = math.prod(shape)
    pool = DeviceArray(
        np.full(lse_size - 1 + math.prod(like_shape), np.nan, np.float32)
    )
    return pool.view(shape, offset=0), pool.view(like_shape, offset=lse_size - 1)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["3d", "tiled"])
@pytest.mark.parametrize(
    "case",
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
def test_malformed_lse_buffer_is_refused_before_launch(cdna3_or_4, case, backend):
    problem, data = _case([4, 2], [3000, 1029])
    tq, hq = problem.total_q, problem.num_query_heads
    inputs = _device_inputs(data)
    lse = _lse_array(problem, np.nan)
    if case == "rows":
        lse = DeviceArray(np.full((tq - 1, hq, 1), np.nan, np.float32))
    elif case == "heads":
        lse = DeviceArray(np.full((tq, hq - 1, 1), np.nan, np.float32))
    elif case == "dtype":
        lse = DeviceArray(np.full((tq, hq, 1), np.nan, np.float16))
    elif case == "strided":
        lse = DeviceArray(np.full((tq, hq, 2), np.nan, np.float32), shape=(tq, hq, 1))
    elif case == "host":
        lse = HostArray((tq, hq, 1), "fp32", device=None)
    elif case in ("cu_seqlens_q", "seqused_k", "block_table"):
        lse, inputs[case] = _aliases_last_lse_element((tq, hq, 1), inputs[case].shape)
    bias = {}
    if case == "additive_bias":
        # The reduce must never write LSE over a bias it (or a later replay)
        # reads. The alias is otherwise a valid bias (float32, long enough for
        # the broadcast strides), so only the overlap check can refuse it.
        problem = dataclasses.replace(problem, use_additive_bias=True)
        lse, storage_ = _aliases_last_lse_element((tq, hq, 1), (4096,))
        bias = {"additive_bias": storage_, "additive_bias_sq_stride": 0}
    out = _new_out(data)
    aliased = ("cu_seqlens_q", "seqused_k", "block_table", "additive_bias")
    with pytest.raises(
        ValueError, match=f"lse overlaps {case}" if case in aliased else "lse"
    ):
        run_unified_attention_torch(
            problem=problem,
            **inputs,
            out=out,
            softmax_scale=1.0 / math.sqrt(data.q.shape[-1]),
            softcap=0.0,
            backend=backend,
            lse=lse,
            **bias,
        )
    assert np.isnan(
        storage.decode(out.values(), "bf16")
    ).all(), "a kernel ran for a malformed lse buffer"


@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _PREFILL_SHAPES)
@pytest.mark.parametrize("d", [64, 128, 256])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_output_and_lse(cdna3_or_4, backend, dtype, d, q_lens, kv_lens):
    """Arch-native routing. On gfx942 bf16 D256 this is the shape the 4-warp
    lean body serves with LSE off (with LSE on it takes the default builder)."""
    problem, data = _case2d(q_lens, kv_lens, d=d, dtype=dtype)
    if backend == "auto":
        assert problem.select_path() == "2d", "auto would not route this to 2D"
    lse = _lse_array(problem, np.nan)
    out = _run2d(problem, data, lse=lse, softcap=0.0, backend=backend)
    off = _run2d_lse_off(problem, data, softcap=0.0, backend=backend)
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(d), 0.0)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", _PREFILL_SHAPES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_tiled_2d_lse_one_query_head_per_kv_head(cdna3_or_4, dtype, q_lens, kv_lens):
    """Hq == Hkv (8:8): NQK = 1, so every CTA row is its own token. backend is
    'tiled' because auto sends the ragged 8:8 shape to split-KV."""
    problem, data = _case2d(q_lens, kv_lens, hq=8, hkv=8, dtype=dtype)
    lse = _lse_array(problem, np.nan)
    out = _run2d(problem, data, lse=lse, softcap=0.0, backend="tiled")
    off = _run2d_lse_off(problem, data, softcap=0.0, backend="tiled")
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("window", [64, 4096])
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_lse_with_sliding_window(cdna3_or_4, backend, window, dtype):
    """D128 sliding window: on gfx942 the LSE-off route is the 4-warp GQA body,
    so this is where the LSE-on fallback to the default builder is exercised. Seq 0
    (256 queries over 5000 keys) is cut by both windows; seq 1 (64/64) only by 64."""
    q_lens, kv_lens = [256, 64], [5000, 64]
    problem, data = _case2d(q_lens, kv_lens, dtype=dtype, sliding_window=window)
    lse = _lse_array(problem, np.nan)
    out = _run2d(problem, data, lse=lse, softcap=0.0, backend=backend)
    off = _run2d_lse_off(problem, data, softcap=0.0, backend=backend)
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0, window)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_tiled_2d_lse_with_softcap(cdna3_or_4, backend):
    """Softcap is applied before the max, in log2 units: LSE must include it."""
    q_lens, kv_lens = [37, 300, 1], [37, 812, 900]
    problem, data = _case2d(q_lens, kv_lens, softcap=30.0)
    lse = _lse_array(problem, np.nan)
    out = _run2d(problem, data, lse=lse, softcap=30.0, backend=backend)
    off = _run2d_lse_off(problem, data, softcap=30.0, backend=backend)
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 30.0)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, "bf16")


def _run2d_on_off(problem, data, *, spec_for, lse, out_fill=np.nan):
    """The LSE-on launch with ``spec_for(problem)`` and the LSE-off launch with
    ``spec_for`` of the LSE-off problem, both on the tiled 2D path, each output
    starting as ``out_fill``."""
    off_problem = dataclasses.replace(problem, emit_lse=False)
    out = _run2d(
        problem,
        data,
        lse=lse,
        softcap=0.0,
        backend="tiled",
        tuning_spec=spec_for(problem),
        out_fill=out_fill,
    )
    off = _run2d(
        off_problem,
        data,
        lse=None,
        softcap=0.0,
        backend="tiled",
        tuning_spec=spec_for(off_problem),
        out_fill=out_fill,
    )
    return out, off


@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", [([37, 300, 1], [37, 812, 900]), ([5], [3])])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_pinned_branch_writes_lse(cdna3_or_4, branch, d, dtype, q_lens, kv_lens):
    """Every epilogue branch of the arch's tiled 2D builder, forced by an
    explicit tuning spec. Branch B (mfma32) is reachable only this way."""
    problem, data = _case2d(q_lens, kv_lens, d=d, dtype=dtype)
    lse = _lse_array(problem, np.nan)
    out, off = _run2d_on_off(
        problem, data, spec_for=lambda p: _branch_spec(p, branch), lse=lse
    )
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(d), 0.0)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, dtype)


def _token_major_view(head_major, q_lens, heads):
    """Reorder a flat head-major LSE buffer to [total_q, Hq, 1]."""
    rows, start = [], 0
    for q_len in q_lens:
        block = head_major[start * heads : (start + q_len) * heads]
        rows.append(block.reshape(heads, q_len).T)
        start += q_len
    return np.concatenate(rows)[..., None]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [([256, 256], [256, 256]), ([37, 300, 1], [37, 812, 900]), ([5], [3])],
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_gfx950_head_major_lse(branch, dtype, q_lens, kv_lens):
    """lse_layout="head_major" writes each sequence as [Hq, q_len]; for equal
    lengths that is the [B, Hq, S, 1] SDPA stats tensor. O is unchanged."""
    current_device_arch_in(("gfx950",), "head-major LSE (gfx950-only)")
    problem, data = _case2d(q_lens, kv_lens, dtype=dtype)
    spec = _branch_spec(problem, branch)
    spec = dataclasses.replace(
        spec,
        kernel_spec=dataclasses.replace(spec.kernel_spec, lse_layout="head_major"),
    )
    assert spec.kernel_name().endswith("_lsehm")
    lse = _lse_array(problem, np.nan)  # same element count; the kernel writes it flat
    out, off = _run2d_on_off(
        problem,
        data,
        spec_for=lambda p: spec if p.emit_lse else _branch_spec(p, branch),
        lse=lse,
    )
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(data.q.shape[-1]), 0.0)
    hq = problem.num_query_heads
    written = lse.values()
    got = _token_major_view(written.reshape(-1), q_lens, hq)
    _assert_2d_result(out, off, got, o_ref, lse_ref, dtype)
    if len(set(q_lens)) == 1:
        bhs1 = written.reshape(len(q_lens), hq, q_lens[0], 1)
        ref = lse_ref.reshape(len(q_lens), q_lens[0], hq, 1).transpose(0, 2, 1, 3)
        finite = np.isfinite(ref)
        assert_within_tolerance(bhs1[finite], ref[finite], LSE_TOLERANCE, context="LSE")


@pytest.mark.gpu
@pytest.mark.parametrize("branch", ["transposed32", "mfma32", "striped16"])
def test_dispatcher_2d_lse_end_to_end(cdna3_or_4, branch):
    """Request -> pinned 2D tuning spec -> bind_torch(arrays with 'lse') ->
    launch. (The generic unified_2d candidate has no binding; executable
    dispatch goes through the tuning candidates, as for split-KV.)"""
    from dispatch.attention import AttentionRequest, dispatch_attention

    arch = _arch()
    q_lens, kv_lens = [256, 256], [256, 700]  # total_q = batch * seqlen_q
    problem, data = _case2d(q_lens, kv_lens)

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
    inputs = _device_inputs(data)
    out, lse, off_out = _new_out(data), _lse_array(problem, np.nan), _new_out(data)
    on.bind_torch({**inputs, "out": out, "lse": lse}).launch()
    off.bind_torch({**inputs, "out": off_out}).launch()
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(128), 0.0)
    _assert_2d_result(
        storage.decode(out.values(), "bf16"),
        storage.decode(off_out.values(), "bf16"),
        lse.values(),
        o_ref,
        lse_ref,
        "bf16",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,score", [("fp16", -20.0), ("bf16", -200.0)])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_2d_lse_masked_leading_tile_very_negative_scores(
    cdna3_or_4, branch, dtype, score
):
    """Sliding window 64 over 256 tokens: some rows of each CTA see a fully
    masked first KV tile. Every visible score is `score` (natural log). Every
    row sees keys, so every LSE must be finite (about score + ln 64) and O must
    match the reference, on every epilogue branch."""
    problem, data = _case2d([256], [256], dtype=dtype, sliding_window=64)
    a = 1.0 if dtype == "fp16" else 4.0
    data.q[...] = a
    # The cached K is the value the kernel reads: round it to the dtype.
    data.k_cache[...] = storage.round_to(score / (a * math.sqrt(128)), dtype)
    lse = _lse_array(problem, np.nan)
    out, off = _run2d_on_off(
        problem, data, spec_for=lambda p: _branch_spec(p, branch), lse=lse
    )
    o_ref, lse_ref = _paged_reference(
        data, 1.0 / math.sqrt(128), 0.0, sliding_window=64
    )
    lse = lse.values()
    assert np.isfinite(lse_ref).all(), "the case must give every row a key"
    assert np.isfinite(lse).all(), "a row with visible keys got LSE -inf/NaN"
    _assert_2d_result(out, off, lse, o_ref, lse_ref, dtype)


_NAN_ROW, _NAN_HEAD = 100, 5


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("branch", _BRANCHES)
def test_2d_lse_nan_query_row_stays_nan(cdna3_or_4, branch, dtype):
    """A NaN in one query row gives NaN O and NaN LSE for that (row, head)
    only. Its softmax sum l is NaN; !(l == 0) keeps row_has_mass true, so
    lse_store propagates NaN instead of writing the keyless -inf."""
    problem, data = _case2d([256], [256], dtype=dtype)
    data.q[_NAN_ROW, _NAN_HEAD] = np.nan
    lse = _lse_array(problem, np.inf)
    out, off = _run2d_on_off(
        problem,
        data,
        spec_for=lambda p: _branch_spec(p, branch),
        lse=lse,
        out_fill=_FINITE_OUT_FILL,
    )
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(data.q.shape[-1]), 0.0)
    np.testing.assert_array_equal(out, off)  # NaN positions must match too
    _assert_only_the_nan_row_is_nan(
        out,
        lse.values(),
        o_ref,
        lse_ref,
        (_NAN_ROW, _NAN_HEAD),
        lambda got, want: _assert_2d_o(got, want, dtype),
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


@pytest.mark.gpu
@pytest.mark.parametrize("q_lens,kv_lens", [([37, 300, 1], [37, 812, 900]), ([5], [3])])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "vid,geometry,knobs,branch",
    _ROW_MAP_VARIANTS.get(_arch(), []),
    ids=[v[0] for v in _ROW_MAP_VARIANTS.get(_arch(), [])],
)
def test_row_mapping_variant_writes_lse(
    cdna3_or_4, vid, geometry, knobs, branch, dtype, q_lens, kv_lens
):
    """Swapped grid, two atoms per warp and a second warp change which lane
    and slot owns each (token, head). A wrong mapping writes the right values
    to the wrong index, which only a reference comparison sees. The ragged
    shape spans three sequences; [5]/[3] has keyless rows 0-1."""
    problem, data = _case2d(q_lens, kv_lens, dtype=dtype)
    lse = _lse_array(problem, np.nan)
    out, off = _run2d_on_off(
        problem,
        data,
        spec_for=lambda p: _variant_spec(p, geometry, knobs, branch),
        lse=lse,
    )
    o_ref, lse_ref = _paged_reference(data, 1.0 / math.sqrt(data.q.shape[-1]), 0.0)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "q_lens,kv_lens,branch",
    [
        ([37, 300, 1], [37, 812, 900], "transposed32"),
        ([5], [3], "striped16"),
    ],
)
def test_tiled_2d_lse_with_additive_bias(cdna3_or_4, q_lens, kv_lens, branch):
    """Additive bias is added to the log2 score before the running max, so LSE
    must include it. gfx950 default routing: the ragged shape takes branch A
    (16 slots: bias turns scalar state off), 5/3 takes C and keeps two keyless
    rows. gfx942 has no bias kernel: the tiled 2D gate must refuse it."""
    problem, data = _case(q_lens, kv_lens)
    problem = dataclasses.replace(problem, use_additive_bias=True)
    arch = _arch()
    ok, why = au.supports_native_unified_attention_tiled(problem, arch)
    if arch == "gfx942":
        assert not ok and "only implemented on gfx950" in why, why
        return
    assert ok, f"the tiled 2D gate refuses LSE + bias: {why}"
    assert _epilogue_branch(au._tiled_spec_from_problem(problem, arch)) == branch
    bias = np.random.default_rng(1).standard_normal(
        (len(q_lens), 32, problem.max_seqlen_q, problem.max_seqlen_k), np.float32
    )
    _, heads_stride, rows_stride, _ = (stride // 4 for stride in bias.strides)
    bias_args = {
        "additive_bias": DeviceArray(bias),
        "additive_bias_batch_stride": bias.strides[0] // 4,
        "additive_bias_head_stride": heads_stride,
        "additive_bias_sq_stride": rows_stride,
    }
    scale = 1.0 / math.sqrt(128)

    def launch(prob, lse):
        with mock.patch.object(au, "_run_3d_tiled", _forbid_split_kv):
            return _run(prob, data, lse=lse, softcap=0.0, backend="tiled", **bias_args)

    lse = _lse_array(problem, np.nan)
    out = launch(problem, lse)
    off = launch(dataclasses.replace(problem, emit_lse=False), None)
    o_ref, lse_ref = _paged_reference(data, scale, 0.0, additive_bias=bias)
    _assert_2d_result(out, off, lse.values(), o_ref, lse_ref, "bf16")
