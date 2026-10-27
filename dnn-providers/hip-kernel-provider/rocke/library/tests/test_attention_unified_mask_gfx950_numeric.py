# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU numeric parity for the non-default masks on gfx950 unified (paged) attention.

The mask is the cuDNN band ``(left = sliding_window, right = right_bound)`` around the
diagonal, with ``causal_top_left`` choosing the alignment:

* top-left (``causal_top_left``): query ``p`` of each sequence attends keys ``<= p``
  (``context_len = 0``), versus the paged bottom-right default (``kv_len - q_len``).
* no mask (``right_bound=-1``): full attention. The unified kernels used to be
  unconditionally causal.
* non-causal bands: a window with an unbounded or lookahead right bound.

The oracle is ``parity_unified_attention.ref_paged_attn``. Every test is
``gpu``-marked. The tiled-kernel tests self-skip off a gfx950 device; the scalar
kernel is arch-neutral, so its tests run on any GPU, compiled for that GPU.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName.lower()


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"
)

requires_any_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU with ROCm torch"
)


def _harness():
    import builders.gfx950.attention.prefill.parity_unified_attention as P

    return P


def _tol(dtype) -> float:
    return 2e-2 if dtype == torch.float16 else 4e-2


def _max_abs(out, ref) -> float:
    return (out.float() - ref.float()).abs().max().item()


def _scenarios():
    try:
        P = _harness()
        return P.topleft_scenarios() + P.nomask_scenarios() + P.band_scenarios()
    except Exception:  # harness needs its optional deps; tests then collect empty
        return []


def _scenario(name):
    return next(s for s in _scenarios() if s.name == name)


class TestMaskNumeric:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d", "auto"])
    @pytest.mark.parametrize("scenario", _scenarios(), ids=lambda s: s.name)
    def test_matches_reference(self, scenario, path):
        P = _harness()
        data = P.make_inputs(scenario)
        ref = P.run_reference(scenario, data)
        out, _ = P._run_rocke(scenario, data, path=path, warmup=0, attempts=1)
        torch.cuda.synchronize()
        assert not torch.isnan(out).any(), "NaN in kernel output"
        err = _max_abs(out, ref)
        assert err < _tol(scenario.dtype), f"{scenario.name}/{path}: max_abs={err:.3e}"

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "name",
        [
            "tl_q_lt_kv",
            "tl_q_gt_kv",
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
        scenario = _scenario(name)
        data = P.make_inputs(scenario)
        ref = P.run_reference(scenario, data)
        default = dataclasses.replace(scenario, causal_top_left=False, right_bound=0)
        out, _ = P._run_rocke(default, data, path="2d", warmup=0, attempts=1)
        torch.cuda.synchronize()
        err = _max_abs(out, ref)
        assert err > 1e-2, f"default-mask kernel unexpectedly matches ({err})"


class TestDispatchedMixedCohort:
    """request -> dispatch -> launch on a cohort whose batch maxima are EQUAL.

    ``seqlen_q`` / ``seqlen_k`` of the request are the batch maxima for the paged
    unified kernels, which take ``context_len`` per sequence from ``cu_seqlens_q`` /
    ``seq_lens`` at runtime. The ``mixed`` cohort below has maxima 700 / 700 yet holds
    (700, 200) and (200, 700) sequences, so folding top-left into bottom-right because
    the maxima match silently gives those sequences the wrong diagonal. The other
    tests here build ``UnifiedAttentionProblem`` directly and never go through the
    request normalization, which is how that shortcut went unnoticed.
    """

    _MIXED = [(512, 512), (700, 200), (200, 700), (64, 64)]

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["tiled", "auto"])
    @pytest.mark.parametrize(
        "name,request_kw,band",
        [
            ("top_left", dict(mask_type=1), dict(causal_top_left=True, right_bound=0)),
            (
                "top_left_window",
                dict(mask_type=1, sliding_window=64),
                dict(causal_top_left=True, right_bound=0, sliding_window=64),
            ),
            (
                "left_only_window_trio",
                dict(mask_type=0, sliding_window=64),
                dict(causal_top_left=True, right_bound=-1, sliding_window=64),
            ),
            (
                "two_sided_window_trio",
                dict(mask_type=0, sliding_window=96, right_bound=32),
                dict(causal_top_left=True, right_bound=32, sliding_window=96),
            ),
        ],
    )
    def test_request_with_equal_batch_maxima_keeps_top_left(
        self, name, request_kw, band, path
    ):
        import dataclasses

        from dispatch.attention import AttentionRequest, _problem, dispatch_attention
        from kernels import run_unified_attention_torch

        P = _harness()
        scenario = P.Scenario(
            name=f"mixed_{name}",
            seq_lens=self._MIXED,
            num_query_heads=16,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            num_blocks=4096,
            sliding_window=band.get("sliding_window") or None,
            causal_top_left=band["causal_top_left"],
            right_bound=band["right_bound"],
        )
        data = P.make_inputs(scenario)
        req = AttentionRequest(
            batch=len(self._MIXED),
            nhead_q=16,
            nhead_k=2,
            seqlen_q=data["max_query_len"],
            seqlen_k=data["max_kv_len"],
            hdim_q=128,
            hdim_v=128,
            arch="gfx950",
            dtype="fp16",
            **request_kw,
        )
        assert req.seqlen_q == req.seqlen_k == 700, "the maxima must be equal"
        problem = _problem(req)
        assert dispatch_attention(req).spec.causal_top_left
        assert problem.causal_top_left == band["causal_top_left"]
        assert problem.right_bound == band["right_bound"]
        # The request carries ``batch * seqlen_q`` rows; the launch has the real count.
        problem = dataclasses.replace(problem, total_q=data["query"].shape[0])

        def launch(prob):
            out = torch.empty_like(data["query"])
            run_unified_attention_torch(
                problem=prob,
                q=data["query"],
                k=data["key_cache"],
                v=data["value_cache"],
                out=out,
                cu_seqlens_q=data["cu_q"],
                seqused_k=data["kv_lens"],
                softmax_scale=data["scale"],
                block_table=data["block_tables"],
                softcap=0.0,
                backend=path,
            )
            torch.cuda.synchronize()
            return out

        ref = P.run_reference(scenario, data)
        out = launch(problem)
        assert not torch.isnan(out).any()
        err = _max_abs(out, ref)
        assert err < _tol(torch.float16), f"{name}/{path}: max_abs={err:.3e}"
        # Negative control: the bottom-right body (what the equal-maxima shortcut
        # selected) must NOT match, or this test could not have caught the bug.
        wrong = launch(dataclasses.replace(problem, causal_top_left=False))
        assert _max_abs(wrong, ref) > 1e-2, "bottom-right body unexpectedly matches"


class TestTopLeftDecode:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d", "auto"])
    @pytest.mark.parametrize("name", ["tl_decode_q1", "tl_d256_decode_q1"])
    def test_decode_attends_only_key_zero(self, name, path):
        """At seqlen_q=1 top-left lets the one query row see only key 0, so the KV
        cache is ignored: the output is exactly V[0] of each sequence. This is the
        definition of top-left, not a bug (bottom-right would attend all keys)."""
        P = _harness()
        s = _scenario(name)
        data = P.make_inputs(s)
        out, _ = P._run_rocke(s, data, path=path, warmup=0, attempts=1)
        torch.cuda.synchronize()
        group = s.num_query_heads // s.num_kv_heads
        v0 = data["value_cache"][data["block_tables"][:, 0].long(), 0]  # [seq, kvh, d]
        want = v0.repeat_interleave(group, dim=1).float()
        err = _max_abs(out, want)
        assert err < _tol(s.dtype), f"{name}/{path}: max_abs={err:.3e}"
        # Perturbing every later key/value must not change the result.
        vc = data["value_cache"].clone()
        kc = data["key_cache"].clone()
        for i, kv_len in enumerate(data["kv_lens_list"]):
            n_blocks = (kv_len + s.block_size - 1) // s.block_size
            blocks = data["block_tables"][i, :n_blocks]
            vc[blocks.long()] += 1.0
            kc[blocks.long()] += 1.0
            vc[blocks[0].long(), 0] = data["value_cache"][blocks[0].long(), 0]
            kc[blocks[0].long(), 0] = data["key_cache"][blocks[0].long(), 0]
        data2 = dict(data, value_cache=vc, key_cache=kc)
        out2, _ = P._run_rocke(s, data2, path=path, warmup=0, attempts=1)
        torch.cuda.synchronize()
        assert _max_abs(out2, out) < _tol(s.dtype)


class TestEmptyRows:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d"])
    @pytest.mark.parametrize(
        "name", ["bd_br_causal_w100", "bd_tl_causal_w100", "bd_br_lookahead_r16"]
    )
    def test_a_row_with_no_visible_key_is_exactly_zero(self, name, path):
        """The contract for a row whose band holds no key (e.g. ``q > kv`` at
        bottom-right, or top-left rows past ``kv_len`` with a window): all-zero output,
        never NaN or stale data."""
        P = _harness()
        s = _scenario(name)
        data = P.make_inputs(s)
        out, _ = P._run_rocke(s, data, path=path, warmup=0, attempts=1)
        torch.cuda.synchronize()
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


class TestNoMaskVisibility:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d"])
    def test_future_keys_change_the_output(self, path):
        """No-mask means a query sees keys AFTER its own position: perturbing the
        last key must change early query rows (a causal kernel would not)."""
        P = _harness()
        s = _scenario("nm_equal")
        data = P.make_inputs(s)
        out, _ = P._run_rocke(s, data, path=path, warmup=0, attempts=1)
        kv_len = data["kv_lens_list"][0]
        last = kv_len - 1
        blk = int(data["block_tables"][0, last // s.block_size])
        vc = data["value_cache"].clone()
        vc[blk, last % s.block_size] += 5.0
        out2, _ = P._run_rocke(
            s, dict(data, value_cache=vc), path=path, warmup=0, attempts=1
        )
        torch.cuda.synchronize()
        first_rows = slice(0, 4)  # sequence 0, earliest queries
        assert _max_abs(out2[first_rows], out[first_rows]) > 1e-3


# (q_len, kv_len, left, right, top_left). ``left`` = 0 means unbounded.
_ONE_HOT_BANDS = {
    "causal_br": (40, 100, 0, 0, False),
    "causal_w16_br": (40, 100, 16, 0, False),
    "causal_w16_tl": (40, 100, 16, 0, True),
    "left_only_w16_tl": (40, 100, 16, -1, True),
    "left_only_w16_br": (40, 100, 16, -1, False),
    "two_sided_w8_r4_tl": (40, 100, 8, 4, True),
    "two_sided_w8_r4_br": (40, 100, 8, 4, False),
    "lookahead_r4_tl": (40, 100, 0, 4, True),
    "lookahead_r4_br": (40, 100, 0, 4, False),
    "full": (40, 100, 0, -1, False),
    "tl_empty_rows_w16": (110, 100, 16, 0, True),
}


class TestBandIndexTruthTable:
    """Index-level check of the band inequalities on the real kernels.

    With ``q = 0`` every visible score is 0, so the softmax is uniform over the
    visible keys; with ``V[k] = e_k`` (one-hot) the output row is ``1/n`` exactly at
    the visible key indices. The expected set is the literal definition
    ``d - k < left`` and ``k <= d + right`` (``d = p`` top-left, ``p + kv - q``
    bottom-right), so ``left = W`` sees W keys counting the diagonal and any
    off-by-one in a kernel shows up as a wrong index, not a tolerance miss.
    """

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("path", ["2d", "3d"])
    @pytest.mark.parametrize("name", list(_ONE_HOT_BANDS))
    def test_visible_key_set_is_exact(self, name, path):
        P = _harness()
        q_len, kv_len, left, right, top_left = _ONE_HOT_BANDS[name]
        head = 128
        s = P.Scenario(
            name=name,
            seq_lens=[(q_len, kv_len)],
            num_query_heads=2,
            num_kv_heads=1,
            head_size=head,
            block_size=16,
            dtype=torch.float16,
            sliding_window=left or None,
            num_blocks=16,
            causal_top_left=top_left,
            right_bound=right,
        )
        data = P.make_inputs(s)
        blocks = (kv_len + s.block_size - 1) // s.block_size
        data["block_tables"] = torch.arange(
            blocks, dtype=torch.int32, device="cuda"
        ).view(1, blocks)
        data["query"].zero_()
        vc = torch.zeros_like(data["value_cache"])
        for k in range(kv_len):
            vc[k // s.block_size, k % s.block_size, 0, k] = 1.0
        data["value_cache"] = vc
        out, _ = P._run_rocke(s, data, path=path, warmup=0, attempts=1)
        torch.cuda.synchronize()
        expected = torch.zeros(q_len, 2, head, dtype=torch.float32)
        for p in range(q_len):
            d = p if top_left else p + kv_len - q_len
            visible = [
                k
                for k in range(kv_len)
                if (left == 0 or d - k < left) and (right < 0 or k <= d + right)
            ]
            if visible:
                expected[p, :, visible] = 1.0 / len(visible)
        err = (out.float().cpu() - expected).abs().max().item()
        assert err < 2e-3, f"{name}/{path}: max_abs={err:.3e}"


_SCALAR_MASKS = {
    "no_mask": dict(causal_top_left=False, right_bound=-1),
    "top_left": dict(causal_top_left=True, right_bound=0),
    "lookahead": dict(causal_top_left=False, right_bound=16),
    "tl_lookahead": dict(causal_top_left=True, right_bound=16),
}


class TestMaskScalar:
    """The scalar kernel against the oracle, on whatever GPU is present.

    The ``gqa_40_8`` case maps 40 query heads onto 8 KV heads, a ratio of 5: the
    kernel's head -> KV-head magic division then has a multiplier other than 1, which
    a power-of-two ratio never exercises.
    """

    @requires_any_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("mask", list(_SCALAR_MASKS))
    @pytest.mark.parametrize(
        "kw",
        [
            dict(seq_lens=[(40, 100), (60, 20), (30, 30)]),
            dict(seq_lens=[(40, 100), (60, 20)], use_sinks=True),
            dict(seq_lens=[(40, 100), (60, 20)], softcap=30.0),
            dict(seq_lens=[(40, 100), (60, 20)], sliding_window=16),
            dict(seq_lens=[(40, 100), (60, 20)], num_query_heads=40, num_kv_heads=8),
        ],
        ids=["mixed", "sinks", "softcap", "window", "gqa_40_8"],
    )
    def test_scalar_matches_reference(self, kw, mask):
        from kernels import UnifiedAttentionProblem, run_unified_attention_torch

        P = _harness()
        band = _SCALAR_MASKS[mask]
        fields = dict(
            num_query_heads=4,
            num_kv_heads=2,
            head_size=128,
            block_size=16,
            dtype=torch.float16,
            num_blocks=256,
            **band,
        )
        fields.update(kw)
        s = P.Scenario(name="scalar", **fields)
        data = P.make_inputs(s)
        ref = P.run_reference(s, data)
        q = data["query"]
        problem = UnifiedAttentionProblem(
            total_q=q.shape[0],
            num_seqs=len(s.seq_lens),
            num_query_heads=s.num_query_heads,
            num_kv_heads=s.num_kv_heads,
            head_size=s.head_size,
            block_size=s.block_size,
            max_seqlen_q=data["max_query_len"],
            max_seqlen_k=data["max_kv_len"],
            dtype="fp16",
            sliding_window=s.sliding_window or 0,
            softcap=float(s.softcap),
            use_sinks=s.use_sinks,
            **band,
        )
        out = torch.empty_like(q)
        run_unified_attention_torch(
            problem=problem,
            q=q,
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_q"],
            seqused_k=data["kv_lens"],
            softmax_scale=data["scale"],
            block_table=data["block_tables"],
            softcap=float(s.softcap),
            sinks=data["sinks"],
            backend="scalar",
        )
        torch.cuda.synchronize()
        assert not torch.isnan(out).any()
        err = _max_abs(out, ref)
        assert err < _tol(torch.float16), f"scalar {kw}/{mask}: max_abs={err:.3e}"


# Transposed-32x32 softmax levers that each derive a causal bound from
# ``context_len``: the (default-on) mask_once + mask_limit stack, the hoisted
# invariants, the full-tile peel, and the per-element fallback. Combinations the
# spec validator rejects are skipped, but at least the base must run.
_LEVER_VARIANTS = {
    "base": {},
    "hoist": dict(use_transposed_invariant_hoist=True),
    "mask_phase_split": dict(use_mask_phase_split=True),
    "hoist_mask_phase_split": dict(
        use_transposed_invariant_hoist=True, use_mask_phase_split=True
    ),
    "no_mask_limit": dict(use_transposed_mask_limit=False),
    "no_mask_limit_hoist": dict(
        use_transposed_mask_limit=False, use_transposed_invariant_hoist=True
    ),
    # The window / right-bound compare on the transposed per-element path (the
    # default for window specs is the 16x16 path, so force the 32x32 transposed one).
    "transposed_per_element": dict(
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
        use_transposed_scalar_state=False,
        use_transposed_mask_once=False,
        use_transposed_mask_limit=False,
    ),
    "per_element_fallback": dict(
        use_transposed_mask_limit=False,
        use_transposed_mask_once=False,
        use_transposed_scalar_state=False,
    ),
}

_LEVER_SCENARIOS = {
    # bf16 GQA64x8 prefill: NQK=8, combo (transposed 32x32) cohort.
    "gqa64x8_mixed": dict(
        seq_lens=[(1024, 1024), (900, 300), (300, 900), (257, 129)],
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bfloat16",
    ),
    "gqa64x8_q_gt_kv": dict(
        seq_lens=[(2048, 512), (700, 100)],
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bfloat16",
    ),
}

# (causal_top_left, right_bound, sliding_window)
_MASKS = {
    "top_left": (True, 0, 0),
    "no_mask": (False, -1, 0),
    "lookahead": (False, 16, 0),
    "tl_lookahead": (True, 16, 0),
    "window_left_only": (False, -1, 96),
    "window_two_sided": (True, 32, 96),
}


class TestMask2DLevers:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("mask", list(_MASKS))
    @pytest.mark.parametrize("variant", list(_LEVER_VARIANTS))
    @pytest.mark.parametrize("scn", list(_LEVER_SCENARIOS))
    def test_lever_matches_reference(self, scn, variant, mask, monkeypatch):
        import kernels.common.attention_unified as au

        P = _harness()
        kw = dict(_LEVER_SCENARIOS[scn])
        kw["dtype"] = getattr(torch, kw["dtype"])
        scenario = P.Scenario(
            name=f"{mask}_lever_{scn}",
            num_blocks=4096,
            head_size=128,
            block_size=16,
            causal_top_left=_MASKS[mask][0],
            right_bound=_MASKS[mask][1],
            sliding_window=_MASKS[mask][2] or None,
            **kw,
        )
        orig = au._tiled_spec_from_problem
        over = _LEVER_VARIANTS[variant]
        built = {}

        def patched(problem, arch):
            spec = dataclasses.replace(orig(problem, arch), **over)
            built["spec"] = spec
            return spec

        monkeypatch.setattr(au, "_tiled_spec_from_problem", patched)
        data = P.make_inputs(scenario)
        ref = P.run_reference(scenario, data)
        if variant != "base":
            try:
                dataclasses.replace(orig(_problem_of(scenario, data), "gfx950"), **over)
            except ValueError as exc:
                pytest.skip(f"lever combination invalid: {exc}")
        out, _ = P._run_rocke(scenario, data, path="2d", warmup=0, attempts=1)
        torch.cuda.synchronize()
        spec = built["spec"]
        for knob, want in over.items():
            assert getattr(spec, knob) == want, f"{knob} not applied"
        assert spec.causal_top_left == _MASKS[mask][0]
        assert spec.right_bound == _MASKS[mask][1]
        assert not torch.isnan(out).any(), "NaN in kernel output"
        err = _max_abs(out, ref)
        assert err < _tol(scenario.dtype), f"{mask}/{scn}/{variant}: max_abs={err:.3e}"


def _problem_of(scenario, data):
    from kernels import UnifiedAttentionProblem

    q = data["query"]
    return UnifiedAttentionProblem(
        total_q=q.shape[0],
        num_seqs=len(scenario.seq_lens),
        num_query_heads=scenario.num_query_heads,
        num_kv_heads=scenario.num_kv_heads,
        head_size=scenario.head_size,
        block_size=scenario.block_size,
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        dtype="fp16" if q.dtype is torch.float16 else "bf16",
        sliding_window=scenario.sliding_window or 0,
        causal_top_left=scenario.causal_top_left,
        right_bound=scenario.right_bound,
        num_cus=120,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
