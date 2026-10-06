# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Non-default masks on gfx942 unified (paged) attention: numeric parity + wiring.

The gfx942 sibling of ``test_attention_unified_mask_gfx950_numeric.py``. The mask is
the cuDNN band ``(left = sliding_window, right = right_bound)`` around the diagonal
``d = context_len + query_pos``, with ``causal_top_left`` choosing the alignment:

* top-left (``causal_top_left``): ``context_len = 0`` (query ``p`` attends keys
  ``<= p``), versus the paged bottom-right default (``kv_len - q_len``). The ALiBi /
  QQ-bias offsets follow ``context_len``, so they move too.
* no mask (``right_bound=-1``): full attention.
* non-causal bands: a window with an unbounded or lookahead right bound.

A row whose band holds no key outputs exactly 0.

The oracle is ``builders.gfx942...parity_unified_attention.ref_paged_attn``. The
GPU classes are ``gpu``-marked and self-skip off a gfx942 device; the CPU class pins
the arch and checks kernel identity only. Rows aimed at one specific gfx942 body
assert its routing predicate first, so a routing change cannot silently move a row
off the body it is meant to cover.

Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_unified_mask_gfx942_numeric.py
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")

import kernels.common.attention_unified as au


def _device_arch() -> str:
    try:
        from rocke.runtime.hip_module import get_device_arch

        return (get_device_arch() or "").lower()
    except Exception:  # noqa: BLE001 -- no HIP runtime means "not gfx942"
        return ""


def _gpu_ready() -> bool:
    return bool(torch.cuda.is_available()) and "gfx942" in _device_arch()


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)


def _harness():
    import builders.gfx942.attention.prefill.parity_unified_attention as P

    return P


def _tol(dtype: str) -> float:
    return 2e-2 if dtype == "fp16" else 4e-2


def _max_abs(out, ref) -> float:
    return (out.float() - ref.float()).abs().max().item()


def _scenarios():
    try:
        P = _harness()
        return P.topleft_scenarios() + P.nomask_scenarios() + P.band_scenarios()
    except Exception:  # noqa: BLE001 -- harness deps missing: collect empty
        return []


def _scenario(name):
    return next(s for s in _scenarios() if s.name == name)


class _PinArch:
    """Pin ``_RESOLVED_ATTENTION_ARCH`` (memoized process-wide) to ``arch``."""

    def __init__(self, arch: str):
        self.arch = arch

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = self.arch
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


def _run(s, data, path):
    out, _ = _harness().run_dispatch(s, data, backend=path, warmup=0, attempts=1)
    return out


def _skip_if_path_unsupported_for_default_mask(s, data, path):
    """Skip a forced path only when it cannot run this shape even with the default
    mask (a geometry gap, not a mask gap). A mask-only rejection is a failure."""
    if path == "auto":
        return
    P = _harness()
    default = P.problem_of(s, data, causal_top_left=False, right_bound=0)
    fn = (
        au.supports_native_unified_attention_tiled
        if path == "2d"
        else au.supports_native_unified_attention_3d_tiled
    )
    ok, why = fn(default)
    if not ok:
        pytest.skip(f"{path} does not run this shape even causal: {why}")


# ---------------------------------------------------------------------------
# GPU: the harness mask sets on every path
# ---------------------------------------------------------------------------


class TestMaskNumeric:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d", "auto"])
    @pytest.mark.parametrize("scenario", _scenarios(), ids=lambda s: s.name)
    def test_matches_reference(self, scenario, path):
        P = _harness()
        data = P.make_inputs(scenario)
        _skip_if_path_unsupported_for_default_mask(scenario, data, path)
        ref = P.run_reference(scenario, data)
        out = _run(scenario, data, path)
        assert not torch.isnan(out).any(), "NaN in kernel output"
        err = _max_abs(out, ref)
        assert err < _tol(scenario.dtype), f"{scenario.name}/{path}: max_abs={err:.3e}"

    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "name",
        [
            "tl_q_lt_kv",
            "tl_q_gt_kv",
            "tl_alibi",
            "tl_qq_bias",
            "nm_equal",
            "nm_q_lt_kv",
            "nm_mixed",
            "bd_br_left_only_w64",
            "bd_br_lookahead_r16",
            "bd_tl_left_only_w64",
            "bd_tl_two_sided_w64_r32",
        ],
    )
    def test_default_mask_kernel_does_not_match(self, name):
        """Negative control: the same request forced onto the default bottom-right
        causal mask (window kept) must NOT match the oracle."""
        P = _harness()
        s = _scenario(name)
        data = P.make_inputs(s)
        ref = P.run_reference(s, data)
        default = dataclasses.replace(s, causal_top_left=False, right_bound=0)
        out = _run(default, data, "2d")
        # A NaN row (e.g. an empty bottom-right window row on a body that breaks
        # the zero-row contract) is a mismatch too; that bug is caught by
        # test_matches_reference on the default-mask rows, not here.
        err = _max_abs(torch.nan_to_num(out.float(), nan=float("inf")), ref)
        assert err > 1e-2, f"default-mask kernel unexpectedly matches ({err})"

    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d"])
    def test_top_left_moves_the_qq_bias_offset(self, path):
        """With top-left, ``context_len`` itself is 0, so the QQ-bias column offset
        (``k - context_len``) follows it. A kernel that moved only the causal bound
        and kept the bias on ``kv_len - q_len`` must not match. (The ALiBi offset is
        a per-row constant, which softmax cancels, so only QQ-bias discriminates.)"""
        P = _harness()
        q_len, kv_len = 200, 700
        s = dataclasses.replace(
            _scenario("tl_qq_bias"),
            seq_lens=((q_len, kv_len),),
            seqlen_q=q_len,
            seqlen_k=kv_len,
            batch=1,
        )
        data = P.make_inputs(s)
        _skip_if_path_unsupported_for_default_mask(s, data, path)
        data["qq_bias"] = data["qq_bias"] * 10.0  # O(1) bias: a clear discriminator
        ref = P.run_reference(s, data)
        out = _run(s, data, path)
        err = _max_abs(out, ref)
        assert err < _tol(s.dtype), f"tl qq_bias/{path}: max_abs={err:.3e}"
        # Wrong oracle: top-left mask, bias column at k - (kv_len - q_len).
        shift = kv_len - q_len
        wrong_qq = torch.zeros_like(data["qq_bias"])
        wrong_qq[:, shift:] = data["qq_bias"][:, : wrong_qq.shape[1] - shift]
        wrong = P.run_reference(s, dict(data, qq_bias=wrong_qq))
        assert _max_abs(ref, wrong) > 10 * _tol(s.dtype), "discriminator too weak"
        assert _max_abs(out, wrong) > 10 * _tol(s.dtype), (
            "kernel applies the QQ-bias at the bottom-right offset"
        )


class TestTopLeftDecode:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d", "auto"])
    @pytest.mark.parametrize("name", ["tl_decode_q1", "tl_d256_decode_q1"])
    def test_decode_attends_only_key_zero(self, name, path):
        """At seqlen_q=1 top-left lets the one query row see only key 0, so the
        output is exactly V[0] of each sequence, and perturbing every later key /
        value must not change it."""
        P = _harness()
        s = _scenario(name)
        data = P.make_inputs(s)
        _skip_if_path_unsupported_for_default_mask(s, data, path)
        out = _run(s, data, path)
        group = s.heads // s.kv_heads
        v0 = data["value_cache"][data["block_tables"][:, 0].long(), 0]
        want = v0.repeat_interleave(group, dim=1).float()
        err = _max_abs(out, want)
        assert err < _tol(s.dtype), f"{name}/{path}: max_abs vs V[0]={err:.3e}"
        vc = data["value_cache"].clone()
        kc = data["key_cache"].clone()
        for i, kv_len in enumerate(data["kv_lens_list"]):
            n_blocks = (kv_len + s.block_size - 1) // s.block_size
            blocks = data["block_tables"][i, :n_blocks].long()
            vc[blocks] += 1.0
            kc[blocks] += 1.0
            vc[blocks[0], 0] = data["value_cache"][blocks[0], 0]
            kc[blocks[0], 0] = data["key_cache"][blocks[0], 0]
        out2 = _run(s, dict(data, value_cache=vc, key_cache=kc), path)
        assert _max_abs(out2, out) < _tol(s.dtype)


class TestEmptyRows:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d"])
    @pytest.mark.parametrize(
        "name", ["bd_br_causal_w100", "bd_tl_causal_w100", "bd_br_lookahead_r16"]
    )
    def test_a_row_with_no_visible_key_is_exactly_zero(self, name, path):
        P = _harness()
        s = _scenario(name)
        data = P.make_inputs(s)
        _skip_if_path_unsupported_for_default_mask(s, data, path)
        out = _run(s, data, path)
        assert not torch.isnan(out).any()
        total, empty_found = 0, 0
        for qlen, kvlen in s.seq_lens:
            for p in range(qlen):
                d = p + (0 if s.causal_top_left else kvlen - qlen)
                hi = (
                    kvlen - 1
                    if s.right_bound < 0
                    else min(d + s.right_bound, kvlen - 1)
                )
                lo = 0 if not s.sliding_window else max(0, d - s.sliding_window + 1)
                if lo > hi:
                    empty_found += 1
                    assert out[total + p].abs().max().item() == 0.0, (qlen, kvlen, p)
            total += qlen
        assert empty_found > 0, "scenario has no empty rows; the test checks nothing"

    @requires_gfx942_gpu
    @pytest.mark.gpu
    def test_d256_lean_bottom_right_q_gt_kv_rows_are_zero(self):
        """Default mask (bottom-right causal, no window) at Sq > Sk on the D256 lean
        4-warp body: rows with ``kv_len - q_len + p < 0`` see no key and must be
        exactly 0 (not NaN from a CTA that runs no KV tile, nor a uniform average
        of masked V); the other rows match the reference. The last sequence's
        partial q-block also covers the Q-load clamp."""
        P = _harness()
        s = P._mask_shape(
            "d256_br_q_gt_kv",
            [(256, 1024), (900, 300)],
            group="body",
            head_size=256,
            dtype="bf16",
        )
        data = P.make_inputs(s)
        assert au._d256_gfx942_fast(P.problem_of(s, data)), "left the D256 lean cohort"
        ref = P.run_reference(s, data)
        out = _run(s, data, "2d")
        assert not torch.isnan(out).any(), "NaN in kernel output"
        total, empty_found = 0, 0
        for qlen, kvlen in s.seq_lens:
            for p in range(qlen):
                if kvlen - qlen + p < 0:
                    empty_found += 1
                    assert out[total + p].abs().max().item() == 0.0, (qlen, kvlen, p)
            total += qlen
        assert empty_found > 0, "shape has no empty rows; the test checks nothing"
        err = _max_abs(out, ref)
        assert err < _tol(s.dtype), f"d256 lean br q>kv: max_abs={err:.3e}"


class TestNoMaskVisibility:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d"])
    def test_future_keys_change_the_output(self, path):
        """No-mask means a query sees keys AFTER its own position: perturbing the
        last key must change early query rows (a causal kernel would not)."""
        P = _harness()
        s = _scenario("nm_equal")
        data = P.make_inputs(s)
        _skip_if_path_unsupported_for_default_mask(s, data, path)
        out = _run(s, data, path)
        last = data["kv_lens_list"][0] - 1
        blk = int(data["block_tables"][0, last // s.block_size])
        vc = data["value_cache"].clone()
        vc[blk, last % s.block_size] += 5.0
        out2 = _run(s, dict(data, value_cache=vc), path)
        assert _max_abs(out2[0:4], out[0:4]) > 1e-3


# ---------------------------------------------------------------------------
# GPU: one row per gfx942 body, predicate asserted first
# ---------------------------------------------------------------------------

# (causal_top_left, right_bound) -- no window.
_MASKS = {
    "top_left": (True, 0),
    "no_mask": (False, -1),
    "lookahead": (False, 16),
    "tl_lookahead": (True, 16),
}
# (causal_top_left, right_bound) -- applied together with the body's window.
_WINDOW_MASKS = {
    "tl_causal_window": (True, 0),
    "left_only": (False, -1),
    "tl_left_only": (True, -1),
    "two_sided": (True, 32),
    "br_two_sided": (False, 32),
}

# (id, predicate or None, backend, Shape kwargs). ``sliding_window`` in kwargs
# selects the window mask table.
_BODIES = [
    # fp16 transposed-x8 flash (the "dense_pipe" cohort). GQA at max_seqlen_q <= 768
    # takes the small-Q narrow carve-out instead, so keep Sq > 768.
    (
        "fp16_d128_flash",
        "_enable_gfx942_fp16_flash",
        "2d",
        dict(seq_lens=[(1024, 2048), (1024, 512)], dtype="fp16"),
    ),
    (
        "fp16_d64_flash_mha",
        "_enable_gfx942_fp16_flash",
        "2d",
        dict(
            seq_lens=[(1024, 2048), (1024, 512)],
            dtype="fp16",
            head_size=64,
            heads=8,
            kv_heads=8,
        ),
    ),
    (
        "bf16_d128_flash",
        "_enable_gfx942_bf16_flash",
        "2d",
        dict(seq_lens=[(1024, 2048), (1024, 512)], dtype="bf16"),
    ),
    (
        "fp16_d128_small_q_narrow",
        "_enable_gfx942_small_q_narrow",
        "2d",
        dict(seq_lens=[(256, 512), (512, 128)], dtype="fp16"),
    ),
    # D256 4-warp lean prefill (bs 16 and 32).
    (
        "bf16_d256_lean_bs16",
        "_d256_gfx942_fast",
        "2d",
        dict(seq_lens=[(256, 1024), (1024, 384)], dtype="bf16", head_size=256),
    ),
    (
        "bf16_d256_lean_bs32",
        "_d256_gfx942_fast",
        "2d",
        dict(
            seq_lens=[(1024, 384), (256, 1024)],
            dtype="bf16",
            head_size=256,
            block_size=32,
        ),
    ),
    # D256 bf16 decode cohort (all_decode -> 3D split-KV).
    (
        "bf16_d256_decode",
        "_d256_decode_cohort",
        "3d",
        dict(
            seq_lens=[(1, 1024), (1, 513), (1, 64), (1, 2048)],
            dtype="bf16",
            head_size=256,
        ),
    ),
    # D128 decode on both the 2D and 3D bodies.
    (
        "bf16_d128_decode_3d",
        None,
        "3d",
        dict(seq_lens=[(1, 512), (1, 777)], dtype="bf16"),
    ),
    (
        "bf16_d128_decode_2d",
        None,
        "2d",
        dict(seq_lens=[(1, 512), (1, 777)], dtype="bf16"),
    ),
    # Tiled 2D / 3D varlen, one Sq<Sk and one Sq>Sk.
    (
        "bf16_d128_varlen_2d",
        None,
        "2d",
        dict(seq_lens=[(64, 320), (384, 128)], dtype="bf16"),
    ),
    (
        "bf16_d128_varlen_3d",
        None,
        "3d",
        dict(seq_lens=[(64, 320), (384, 128)], dtype="bf16"),
    ),
    # D128 sliding-window 4-warp body.
    (
        "bf16_d128_swa",
        "_d128_gfx942_swa_fast",
        "2d",
        dict(seq_lens=[(512, 1024), (1024, 512)], dtype="bf16", sliding_window=128),
    ),
    (
        "fp16_d128_swa",
        "_d128_gfx942_swa_fast",
        "2d",
        dict(seq_lens=[(512, 1024), (1024, 512)], dtype="fp16", sliding_window=256),
    ),
]


# The 4-warp GQA bodies implement only right_bound == 0 (causal, either alignment):
# a lookahead / unbounded request must route OFF them (asserted), and still match.
_CAUSAL_ONLY_BODIES = {"_d256_gfx942_fast", "_d128_gfx942_swa_fast"}


def _body_rows():
    rows = []
    for bid, pred, backend, kw in _BODIES:
        table = _WINDOW_MASKS if kw.get("sliding_window") else _MASKS
        for mask, (tl, rb) in table.items():
            rows.append(pytest.param(pred, backend, kw, tl, rb, id=f"{bid}-{mask}"))
    return rows


class TestMaskBodies:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("pred,backend,kw,tl,rb", _body_rows())
    def test_body_matches_reference(self, pred, backend, kw, tl, rb):
        P = _harness()
        kw = dict(kw)
        s = P._mask_shape(
            "body",
            kw.pop("seq_lens"),
            group="body",
            causal_top_left=tl,
            right_bound=rb,
            **kw,
        )
        data = P.make_inputs(s)
        if pred is not None:
            problem = P.problem_of(s, data)
            if pred in _CAUSAL_ONLY_BODIES and rb != 0:
                assert not getattr(au, pred)(problem), (
                    f"right_bound={rb} routed onto the causal-only {pred} body"
                )
                # Off the 4-warp body, whatever production picks must be right
                # (a forced 2D can exceed the gfx942 LDS budget at D256 bs32).
                backend = "auto"
            else:
                assert getattr(au, pred)(problem), f"problem left the {pred} cohort"
        ref = P.run_reference(s, data)
        out = _run(s, data, backend)
        assert not torch.isnan(out).any(), "NaN in kernel output"
        err = _max_abs(out, ref)
        assert err < _tol(s.dtype), f"{pred}/{backend}: max_abs={err:.3e}"
        # Discriminator: the default-mask reference must be far from this one,
        # and the kernel far from it.
        ref_br = P.run_reference(
            dataclasses.replace(s, causal_top_left=False, right_bound=0), data
        )
        if _max_abs(ref, ref_br) > 10 * _tol(s.dtype):
            assert _max_abs(out, ref_br) > 10 * _tol(s.dtype), (
                "kernel matches the default bottom-right reference"
            )


# ---------------------------------------------------------------------------
# GPU: scalar body and dispatch end-to-end
# ---------------------------------------------------------------------------

_SCALAR_MASKS = {
    "no_mask": dict(causal_top_left=False, right_bound=-1),
    "top_left": dict(causal_top_left=True, right_bound=0),
    "lookahead": dict(causal_top_left=False, right_bound=16),
    "tl_lookahead": dict(causal_top_left=True, right_bound=16),
}


class TestMaskScalar:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("mask", list(_SCALAR_MASKS))
    @pytest.mark.parametrize(
        "kw",
        [
            dict(seq_lens=[(40, 100), (60, 20), (30, 30)]),
            dict(seq_lens=[(40, 100), (60, 20)], use_sinks=True),
            dict(seq_lens=[(40, 100), (60, 20)], softcap=30.0),
            dict(seq_lens=[(40, 100), (60, 20)], sliding_window=16),
        ],
        ids=["mixed", "sinks", "softcap", "window"],
    )
    def test_scalar_matches_reference(self, kw, mask):
        P = _harness()
        kw = dict(kw)
        s = P._mask_shape(
            "scalar",
            kw.pop("seq_lens"),
            group="scalar",
            heads=4,
            **_SCALAR_MASKS[mask],
            **kw,
        )
        data = P.make_inputs(s)
        ref = P.run_reference(s, data)
        out = _run(s, data, "scalar")
        assert not torch.isnan(out).any()
        err = _max_abs(out, ref)
        assert err < _tol("fp16"), f"scalar {kw}/{mask}: max_abs={err:.3e}"


# (id, AttentionRequest kwargs). Dispatch -> _problem(req) -> unified launch.
_DISPATCH_CASES = [
    (
        "tl_bf16_d128_prefill",
        dict(seqlen_q=256, seqlen_k=1024, mask_type="TOP_LEFT_CAUSAL"),
    ),
    (
        "tl_bf16_d128_sq>sk",
        dict(seqlen_q=1024, seqlen_k=256, mask_type="TOP_LEFT_CAUSAL"),
    ),
    (
        "tl_bf16_d256_decode",
        dict(
            batch=4, seqlen_q=1, seqlen_k=1024, hdim_q=256, mask_type="TOP_LEFT_CAUSAL"
        ),
    ),
    (
        "tl_fp16_d128_prefill",
        dict(seqlen_q=1024, seqlen_k=2048, dtype="fp16", mask_type="TOP_LEFT_CAUSAL"),
    ),
    ("nm_bf16_d128_prefill", dict(seqlen_q=256, seqlen_k=1024, mask_type="NO_MASK")),
    (
        "band_tl_w128_r16",
        dict(
            seqlen_q=512,
            seqlen_k=1024,
            mask_type="NO_MASK",
            sliding_window=128,
            right_bound=16,
        ),
    ),
    (
        "band_br_w128_r16",
        dict(
            seqlen_q=512,
            seqlen_k=1024,
            mask_type="NO_MASK",
            sliding_window=128,
            right_bound=16,
            diagonal_alignment=1,
        ),
    ),
    (
        "lookahead_tl_r16",
        dict(seqlen_q=512, seqlen_k=1024, mask_type="NO_MASK", right_bound=16),
    ),
]


class TestDispatchEndToEnd:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "kw", [c[1] for c in _DISPATCH_CASES], ids=[c[0] for c in _DISPATCH_CASES]
    )
    def test_dispatch_band_end_to_end(self, kw):
        from dispatch.attention import (
            AttentionMaskType,
            AttentionRequest,
            _problem,
            dispatch_attention,
        )

        P = _harness()
        kw = dict(kw)
        kw["mask_type"] = getattr(AttentionMaskType, kw["mask_type"])
        base = dict(
            batch=2, nhead_q=16, nhead_k=2, hdim_q=128, dtype="bf16", kv_block_size=16
        )
        base.update(kw)
        base["hdim_v"] = base["hdim_q"]
        req = AttentionRequest(arch="gfx942", **base)
        res = dispatch_attention(req)
        problem = _problem(req)
        assert not problem.default_mask, "request lost its band"
        path = getattr(res.spec, "path", "")
        assert path in ("2d", "3d"), f"auto picked {res.candidate.name}, not unified"
        s = P._mask_shape(
            "dispatch",
            [(int(req.seqlen_q), int(req.seqlen_k))] * int(req.batch),
            group="dispatch",
            dtype=problem.dtype,
            heads=problem.num_query_heads,
            kv_heads=problem.num_kv_heads,
            head_size=problem.head_size,
            block_size=problem.block_size,
            sliding_window=problem.sliding_window,
            causal_top_left=problem.causal_top_left,
            right_bound=problem.right_bound,
        )
        data = P.make_inputs(s)
        out = _run(s, data, path)
        ref = P.run_reference(s, data)
        err = _max_abs(out, ref)
        assert err < _tol(s.dtype), f"dispatch {res.candidate.name}/{path}: {err:.3e}"


# Levers that derive a bound from ``context_len`` on the gfx942 fp16 flash body.
# Combinations the spec validator rejects are skipped; the base must run.
_LEVER_VARIANTS = {
    "base": {},
    "causal_mask_phase_split": dict(use_causal_mask_phase_split=True),
}
# (causal_top_left, right_bound, sliding_window)
_LEVER_MASKS = {
    "top_left": (True, 0, 0),
    "no_mask": (False, -1, 0),
    "lookahead": (False, 16, 0),
    "tl_lookahead": (True, 16, 0),
    "window_left_only": (False, -1, 96),
    "window_two_sided": (True, 32, 96),
}


class TestMask2DLevers:
    @requires_gfx942_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("mask", list(_LEVER_MASKS))
    @pytest.mark.parametrize("variant", list(_LEVER_VARIANTS))
    def test_lever_matches_reference(self, variant, mask, monkeypatch):
        P = _harness()
        tl, rb, sw = _LEVER_MASKS[mask]
        # q > kv rows exercise the top-left in-prefix bound past the last KV tile.
        s = P._mask_shape(
            f"lever_{mask}",
            [(1024, 2048), (1024, 512)],
            group="lever",
            dtype="fp16",
            causal_top_left=tl,
            right_bound=rb,
            sliding_window=sw,
        )
        data = P.make_inputs(s)
        problem = P.problem_of(s, data)
        assert au._enable_gfx942_fp16_flash(problem), "left the fp16 flash cohort"
        orig = au._tiled_spec_from_problem
        over = _LEVER_VARIANTS[variant]
        try:
            dataclasses.replace(orig(problem), **over)
        except ValueError as exc:
            pytest.skip(f"lever combination invalid: {exc}")
        built = {}

        def patched(p):
            spec = dataclasses.replace(orig(p), **over)
            built["spec"] = spec
            return spec

        monkeypatch.setattr(au, "_tiled_spec_from_problem", patched)
        # The 2D caches key on the problem, not the spec: isolate them so the
        # lever variant is really compiled (restored by monkeypatch afterwards).
        for cache in (
            "_ATTN_TILED_CACHE",
            "_2D_LAUNCHERS",
            "_2D_LAUNCH_META",
            "_2D_GRAPHS",
            "_2D_GRAPH_REFS",
        ):
            monkeypatch.setattr(au, cache, {})
        ref = P.run_reference(s, data)
        out = _run(s, data, "2d")
        spec = built["spec"]
        for knob, want in over.items():
            assert getattr(spec, knob) == want, f"{knob} not applied"
        assert spec.causal_top_left == tl and spec.right_bound == rb
        assert not torch.isnan(out).any(), "NaN in kernel output"
        err = _max_abs(out, ref)
        assert err < _tol(s.dtype), f"{mask}/{variant}: max_abs={err:.3e}"


# ---------------------------------------------------------------------------
# CPU: kernel identity on the gfx942 spec builders
# ---------------------------------------------------------------------------


def _problem(q_lens, kv_lens, **kw):
    base = dict(
        total_q=int(sum(q_lens)),
        num_seqs=len(q_lens),
        num_query_heads=16,
        num_kv_heads=2,
        head_size=128,
        block_size=16,
        max_seqlen_q=int(max(q_lens)),
        max_seqlen_k=int(max(kv_lens)),
        dtype="bf16",
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


# Each lands on a different gfx942 2D spec branch of ``_tiled_spec_from_problem``.
_SPEC_ROWS = [
    ("bf16_d128", dict(dtype="bf16")),
    ("fp16_d128", dict(dtype="fp16")),
    ("bf16_d256_4warp", dict(dtype="bf16", head_size=256)),
    ("bf16_d128_swa", dict(dtype="bf16", sliding_window=128)),
]
# (causal_top_left, right_bound, expected name tokens)
_NAME_MASKS = [
    (True, 0, {"tl"}),
    (False, -1, {"rbu"}),
    (False, 16, {"rb16"}),
    (True, 16, {"tl", "rb16"}),
]


class TestKernelIdentity:
    @pytest.mark.parametrize("tl,rb,tokens", _NAME_MASKS)
    @pytest.mark.parametrize("label,kw", _SPEC_ROWS, ids=[r[0] for r in _SPEC_ROWS])
    def test_2d_spec_name_carries_mask_tokens(self, label, kw, tl, rb, tokens):
        with _PinArch("gfx942"):
            off = au._tiled_spec_from_problem(_problem([256], [1024], **kw))
            on = au._tiled_spec_from_problem(
                _problem([256], [1024], causal_top_left=tl, right_bound=rb, **kw)
            )
        assert on.causal_top_left is tl and on.right_bound == rb, label
        parts = on.kernel_name().split("_")
        for t in tokens:
            assert t in parts, (t, on.kernel_name())
        for t in ("tl", "rbu", "rb16"):
            assert t not in off.kernel_name().split("_"), off.kernel_name()
        assert on.kernel_name() != off.kernel_name()

    @pytest.mark.parametrize("tl,rb,tokens", _NAME_MASKS)
    @pytest.mark.parametrize("d", [128, 256])
    def test_3d_spec_name_carries_mask_tokens(self, d, tl, rb, tokens):
        with _PinArch("gfx942"):
            off = au._tiled_3d_spec_from_problem(
                _problem([1] * 4, [1024] * 4, head_size=d)
            )
            on = au._tiled_3d_spec_from_problem(
                _problem(
                    [1] * 4, [1024] * 4, head_size=d, causal_top_left=tl, right_bound=rb
                )
            )
        parts = on.kernel_name().split("_")
        for t in tokens:
            assert t in parts, (t, on.kernel_name())
        assert on.kernel_name() != off.kernel_name()

    def test_mask_fields_default_to_bottom_right_causal(self):
        from kernels.gfx942.attention_tiled_2d import UnifiedAttention2DTiledSpec
        from kernels.gfx942.attention_tiled_3d import UnifiedAttention3DTiledSpec

        for cls in (UnifiedAttention2DTiledSpec, UnifiedAttention3DTiledSpec):
            fields = {f.name: f for f in dataclasses.fields(cls)}
            assert fields["causal_top_left"].default is False, cls.__name__
            assert fields["right_bound"].default == 0, cls.__name__

    @pytest.mark.parametrize("tl,rb", [(True, 0), (False, -1), (True, 16)])
    def test_cache_keys_split_on_mask(self, tl, rb):
        with _PinArch("gfx942"):
            on = _problem([256], [1024], causal_top_left=tl, right_bound=rb)
            off = _problem([256], [1024])
            assert au._tiled_cache_key(on) != au._tiled_cache_key(off)
            assert au._tiled_3d_cache_key(on) != au._tiled_3d_cache_key(off)
            assert au._cache_key(on) != au._cache_key(off)

    @pytest.mark.parametrize("fn", ["tiled", "3d"])
    def test_gfx942_supports_the_band(self, fn):
        f = {
            "tiled": au.supports_native_unified_attention_tiled,
            "3d": au.supports_native_unified_attention_3d_tiled,
        }[fn]
        with _PinArch("gfx942"):
            for tl, rb, sw in [(True, 0, 0), (False, -1, 0), (True, 32, 128)]:
                p = _problem(
                    [1] * 4 if fn == "3d" else [256],
                    [1024] * 4 if fn == "3d" else [1024],
                    causal_top_left=tl,
                    right_bound=rb,
                    sliding_window=sw,
                )
                ok, why = f(p)
                assert ok, (tl, rb, sw, why)

    @pytest.mark.parametrize("bias", ["use_alibi", "use_qq_bias"])
    def test_right_bound_rejected_with_bias(self, bias):
        with _PinArch("gfx942"):
            p = _problem([256], [1024], right_bound=16, **{bias: True})
            ok, why = au.supports_native_unified_attention_tiled(p)
        assert not ok and "ALiBi" in why, why


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
