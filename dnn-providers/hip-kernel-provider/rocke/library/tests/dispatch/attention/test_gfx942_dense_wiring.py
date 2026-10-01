# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx942 ``attention_dense`` dispatcher wiring.

Required by ``library/dispatch/AGENTS.md`` step 4. Covers:
  - the candidate is registered and discoverable, with the right spec_id/algorithm
  - OPT-IN ONLY: nothing short of pinning ``algorithm`` + ``spec_id`` selects it,
    despite priority 3 outranking every other attention candidate
  - selection returns an ``AttentionTuningSpec`` (``path="dense"``) whose
    ``tuning_id`` replays; ``auto`` is the default spec
  - routing on gfx942, and rejection of every out-of-scope request
  - the default spec turns the persistent grid on once there is enough work;
    ``persistent`` and ``waves_per_eu`` are knobs of the candidate
  - both gfx942 dense grids read ``batch`` / ``seqlen_q`` / ``seqlen_kv`` and both
    head counts as runtime kernel params, so those fields drop out of
    ``kernel_name()`` and the cache key and the dispatched signature includes them.
    The persistent grid adds a fast-division magic/shift pair per work-decode
    divisor, and keys its decode order by the resolved (not raw 'auto') value.
"""

from __future__ import annotations

import dataclasses
import unittest
from itertools import islice

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    AttentionTuningSpec,
    attention_candidates,
    attention_tuning_spec,
    dispatch_attention,
    iter_registered_attention_combos,
    tuning_spec_with_knobs,
)
from kernels.common.attention_dense_spec import (
    AttentionDenseSpec,
    attention_dense_cache_key,
)
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    attention_dense_runtime_args,
    attention_dense_signature,
    build_attention_dense,
    supports_attention_dense,
)
from rocke.helpers.transforms import calculate_magic_numbers

_NAME = "attention_gfx942_dense"
_SPEC_ID = "gfx942_dense"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=128,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        mask_type=1,
        algorithm="attention_dense",
        spec_id=_SPEC_ID,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _persistent_spec(
    req: AttentionRequest, decode: str | None = None
) -> Gfx942AttentionDenseSpec:
    """The dispatched persistent spec for ``req``, its decode order pinned to
    ``decode`` when given. Pinned on the spec, not as a knob: the knob refuses an
    order that only restates what ``auto`` resolves to, which is exactly the
    identity the cache-key tests compare."""
    spec = _spec(req, persistent=True)
    return spec if decode is None else dataclasses.replace(spec, persist_decode=decode)


def _spec(req: AttentionRequest, **knobs) -> Gfx942AttentionDenseSpec:
    """The candidate's kernel spec for ``req``, with ``knobs`` applied."""
    if knobs:
        return tuning_spec_with_knobs(req, _SPEC_ID, knobs).kernel_spec
    return attention_tuning_spec(req, _SPEC_ID).kernel_spec


def _candidate():
    return next(c for c in attention_candidates() if c.name == _NAME)


class _Gfx942Arch:
    """Pin _RESOLVED_ATTENTION_ARCH so routing does not depend on the host GPU."""

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx942"
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestGfx942DenseRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        self.assertIn(_NAME, [c.name for c in attention_candidates()])

    def test_spec_id_and_algorithm(self):
        c = _candidate()
        self.assertEqual(c.spec_id, _SPEC_ID)
        self.assertEqual(c.algorithm, "attention_dense")
        self.assertTrue(c.opt_in)

    def test_priority_outranks_every_other_candidate(self):
        """Documents WHY the opt-in gate matters: nothing else holds this arm back."""
        c = _candidate()
        others = [o for o in attention_candidates() if o.name != _NAME]
        self.assertTrue(all(c.priority <= o.priority for o in others))


class TestGfx942DenseOptIn(unittest.TestCase):
    def test_unpinned_requests_never_select_it(self):
        with _Gfx942Arch():
            for kw in (
                dict(algorithm="auto", spec_id="auto"),
                dict(algorithm="auto", spec_id=_SPEC_ID),
                dict(algorithm="attention_dense", spec_id="auto"),
            ):
                with self.subTest(**kw):
                    ok, why = _candidate().admits(_req(**kw))
                    self.assertFalse(ok)
                    self.assertIn("opt-in", why)
            routed = dispatch_attention(_req(algorithm="auto", spec_id="auto"))
            self.assertNotEqual(routed.candidate.name, _NAME)

    def test_routes_on_explicit_pin(self):
        with _Gfx942Arch():
            r = dispatch_attention(_req())
            self.assertEqual(r.candidate.name, _NAME)
            self.assertIsInstance(r.spec, AttentionTuningSpec)
            self.assertEqual(r.spec.path, "dense")
            self.assertIsInstance(r.spec.kernel_spec, Gfx942AttentionDenseSpec)
            self.assertIn("gfx942", r.spec.kernel_name())
            self.assertNotEqual(r.grid, (0, 0, 0))

    def test_tuning_id_replays_and_unknown_ids_are_refused(self):
        with _Gfx942Arch():
            req = _req(seqlen_q=8192, seqlen_k=8192)
            specs = [
                s
                for _c, s in islice(
                    iter_registered_attention_combos(req, candidate_prefix=_NAME), 12
                )
            ]
            for spec in specs:
                with self.subTest(tuning_id=spec.tuning_id):
                    got = dispatch_attention(
                        _req(
                            seqlen_q=8192,
                            seqlen_k=8192,
                            tuning_id=spec.tuning_id,
                        )
                    )
                    self.assertEqual(got.spec, spec)
            ok, why = _candidate().admits(_req(tuning_id="other_wpe2@00"))
            self.assertFalse(ok)
            self.assertIn("tuning_id", why)


class TestGfx942DenseSupportGates(unittest.TestCase):
    """Arch, dtype and feature rejections are the declared ``Capability``'s job;
    only what capability cannot express as data stays in the predicate. Each test
    below asserts which of the two turned the request down, so a gate silently
    migrating between them is a failure rather than a rename."""

    def test_rejects_non_gfx942_arch(self):
        ok, why = _candidate().admits(_req(arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("capability", why)
        self.assertIn("gfx942", why)

    def test_rejects_unsupported_dtype(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(dtype="fp8"))
            self.assertFalse(ok)
            self.assertIn("capability", why)
            self.assertIn("fp8", why)

    def test_admits_sliding_window(self):
        with _Gfx942Arch():
            ok, _ = _candidate().admits(_req(sliding_window=64))
            self.assertTrue(ok)

    def test_rejects_sinks(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(use_sinks=True))
            self.assertFalse(ok)
            self.assertIn("capability", why)
            self.assertIn("sinks", why)

    def test_rejects_ragged_sequence_length(self):
        """The default spec sets ragged=True for any non-256-multiple
        self-attention length. The kernel must decline, not select-then-fail.
        Capability cannot see this one: it is a property of the BUILT spec, so
        it stays in the predicate."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(seqlen_q=1000, seqlen_k=1000))
            self.assertFalse(ok)
            self.assertNotIn("capability", why)
            self.assertIn("ragged", why)


class TestGfx942BottomRightSafety(unittest.TestCase):
    def test_moving_bottom_right_declines_at_capability(self):
        for mask_type in (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2):
            with self.subTest(mask_type=mask_type), _Gfx942Arch():
                ok, why = _candidate().admits(
                    _req(seqlen_q=2048, seqlen_k=4096, mask_type=mask_type)
                )
                self.assertFalse(ok)
                self.assertIn("capability", why)
                self.assertIn("causal_bottom_right", why)
                with self.assertRaisesRegex(ValueError, "causal_bottom_right"):
                    _spec(_req(seqlen_q=2048, seqlen_k=4096, mask_type=mask_type))

    def test_concrete_support_rejects_shared_bottom_right_spec(self):
        common = dict(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=4096,
            num_query_heads=128,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            causal_bottom_right=True,
            dtype="bf16",
        )
        spec = AttentionDenseSpec(**common)
        ok, why = supports_attention_dense(spec, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("causal_bottom_right", why)
        with self.assertRaisesRegex(ValueError, "causal_bottom_right"):
            Gfx942AttentionDenseSpec(**common)

    def test_equal_length_bottom_right_preserves_persistent_policy(self):
        common = dict(seqlen_q=8192, seqlen_k=8192)
        mask_pairs = (
            (AttentionMaskType.TOP_LEFT_CAUSAL, 2),
            (1, AttentionMaskType.BOTTOM_RIGHT_CAUSAL),
        )
        with _Gfx942Arch():
            for top_left, bottom_right in mask_pairs:
                with self.subTest(top_left=top_left, bottom_right=bottom_right):
                    top_left_spec = _spec(_req(mask_type=top_left, **common))
                    bottom_right_req = _req(mask_type=bottom_right, **common)
                    bottom_right_spec = _spec(bottom_right_req)
                    self.assertEqual(bottom_right_spec, top_left_spec)
                    self.assertFalse(bottom_right_spec.causal_bottom_right)
                    self.assertTrue(bottom_right_spec.persistent)
                    ok, why = _candidate().admits(bottom_right_req)
                    self.assertTrue(ok, why)


class TestGfx942DensePersistent(unittest.TestCase):
    def test_default_spec_turns_persistent_on_for_large_sq(self):
        """The default spec turns the persistent grid-stride variant on once
        there is enough work to fill the grid -- the large-Sq prefill regime."""
        with _Gfx942Arch():
            self.assertTrue(_spec(_req(seqlen_q=8192, seqlen_k=8192)).persistent)
            self.assertFalse(_spec(_req(nhead_q=8, nhead_k=8)).persistent)

    def test_persistent_is_a_knob_of_the_candidate(self):
        """Persistence is reached through the knob space (and its tuning ids),
        never silently downgraded to a default-grid kernel."""
        with _Gfx942Arch():
            req = _req(nhead_q=8, nhead_k=8)
            self.assertTrue(_spec(req, persistent=True).persistent)
            self.assertFalse(_spec(_req(), persistent=False).persistent)


class TestGfx942DenseWavesPerEu(unittest.TestCase):
    def test_shipped_policy_and_knob_override(self):
        self.assertEqual(_spec(_req()).waves_per_eu, 2)
        self.assertEqual(_spec(_req(hdim_q=64, hdim_v=64)).waves_per_eu, 4)
        overridden = _spec(_req(), waves_per_eu=3)
        self.assertEqual(overridden.waves_per_eu, 3)
        self.assertIn("wpe3", overridden.kernel_name())
        self.assertEqual(
            build_attention_dense(overridden, arch="gfx942").attrs["waves_per_eu"],
            3,
        )

    def test_invalid_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "waves_per_eu"):
            _spec(_req(), waves_per_eu=9)

    def _swept_waves(self, level: str) -> set[int]:
        # The full level streams the whole knob product; the WPE loop runs
        # inside each knob set, so the first one already carries every WPE.
        return {
            spec.kernel_spec.waves_per_eu
            for _candidate, spec in islice(
                iter_registered_attention_combos(
                    _req(), candidate_prefix=_NAME, sweep_level=level
                ),
                64,
            )
        }

    def test_production_and_full_sweeps_expand_wpe(self):
        self.assertEqual(self._swept_waves("production"), {2, 4})
        self.assertEqual(self._swept_waves("full"), {1, 2, 3, 4})


class TestGfx942DenseSpecIdentity(unittest.TestCase):
    def test_kernel_name_follows_the_runtime_shape_contract(self):
        """Both gfx942 dense grids take batch, both seqlens and both head counts as
        kernel params, so one name AND one cache key cover every batch and head
        config, and the dispatched signature carries the shape. The persistent
        grid appends its work decode's fast-division pairs. The baked control is
        gfx950 persistent dense, which still bakes heads and seqlens into the
        symbol: it proves the token regexes below can see a baked shape."""
        shape_args = [
            "batch",
            "seqlen_q",
            "seqlen_kv",
            "num_query_heads",
            "num_kv_heads",
        ]
        ptrs_and_scale = ["q_ptr", "k_ptr", "v_ptr", "o_ptr", "scale"]
        shapes = [
            {"batch": b, "nhead_q": hq, "nhead_k": hk}
            for b in (1, 2, 4)
            for hq, hk in ((128, 8), (40, 8), (32, 32))
        ]
        with _Gfx942Arch():
            for persistent, tail in (
                (False, []),
                # Every shape here resolves 'auto' to qb_major (asserted below).
                (
                    True,
                    [
                        "batch_magic",
                        "batch_shift",
                        "num_query_heads_magic",
                        "num_query_heads_shift",
                        "gqa_magic",
                        "gqa_shift",
                    ],
                ),
            ):
                with self.subTest(persistent=persistent):
                    specs = [_spec(_req(**kw), persistent=persistent) for kw in shapes]
                    self.assertTrue(all(s.persistent == persistent for s in specs))
                    self.assertTrue(all(s.runtime_shape for s in specs))
                    self.assertEqual(len({s.kernel_name() for s in specs}), 1)
                    self.assertEqual(
                        len(
                            {attention_dense_cache_key(s, arch="gfx942") for s in specs}
                        ),
                        1,
                    )
                    name = specs[0].kernel_name()
                    self.assertNotRegex(name, r"_b\d+")
                    self.assertNotRegex(name, r"_(hq|kv)\d+")
                    self.assertNotRegex(name, r"_s[qk]\d+")
                    if persistent:
                        self.assertEqual(
                            {s.resolved_persist_decode for s in specs}, {"qb_major"}
                        )
                        self.assertIn("persist304", name)
                    names = [p["name"] for p in attention_dense_signature(specs[0])]
                    self.assertEqual(names, ptrs_and_scale + shape_args + tail)
                    self.assertEqual(
                        names[len(ptrs_and_scale) :],
                        list(specs[0].runtime_kernarg_fields),
                    )

        # Baked control on gfx950: the persistent body there declares no shape
        # params, so heads and seqlens stay in the symbol and split identity.
        from kernels.gfx950.attention_dense import (
            Gfx950AttentionDenseSpec,
        )
        from kernels.gfx950.attention_dense import (
            attention_dense_signature as gfx950_signature,
        )

        baked = [
            attention_tuning_spec(
                _req(arch="gfx950", algorithm="auto", spec_id="auto", **kw),
                "gfx950_dense_persist",
            ).kernel_spec
            for kw in shapes
        ]
        self.assertTrue(all(isinstance(s, Gfx950AttentionDenseSpec) for s in baked))
        self.assertTrue(all(s.persistent and not s.runtime_shape for s in baked))
        by_heads: dict[tuple[int, int], set[str]] = {}
        for kw, s in zip(shapes, baked):
            name = s.kernel_name()
            self.assertRegex(name, r"_hq\d+_kv\d+_")
            self.assertRegex(name, r"_sq\d+_sk\d+_")
            by_heads.setdefault((kw["nhead_q"], kw["nhead_k"]), set()).add(name)
        # Head configs never share a symbol (gfx950's own decode policy may split
        # further by batch, which is not what this control is about).
        name_sets = list(by_heads.values())
        self.assertEqual(len(name_sets), 3)
        for i, a in enumerate(name_sets):
            for other in name_sets[i + 1 :]:
                self.assertFalse(a & other)
        self.assertEqual(
            [p["name"] for p in gfx950_signature(baked[0])], ptrs_and_scale
        )

    def test_support_implies_the_dispatched_spec_builds(self):
        """The dispatch-level half of the supports/build contract: the spec the
        dispatcher actually selects is exactly what the builder emits."""
        with _Gfx942Arch():
            req = _req()
            self.assertTrue(_candidate().admits(req)[0])
            selected = dispatch_attention(req).spec
            kd = selected.build("gfx942")
            self.assertEqual(kd.name, selected.kernel_name())


def _fastdiv(n: int, magic_i32: int, shift: int) -> int:
    """Host model of the persistent decode's divide: (umulhi(n, magic) + n) >> shift,
    with the magic read back from its two's-complement i32 kernarg."""
    return (((n * (magic_i32 & 0xFFFFFFFF)) >> 32) + n) >> shift


class TestGfx942DensePersistentRuntimeShape(unittest.TestCase):
    """The persistent grid's runtime-shape ABI: the fast-division kernargs its work
    decode reads, the cache identity of its decode order, and the dividend bound."""

    # Per resolved decode order, the divisors in kernarg order.
    _DIVISORS = {
        "hkv_major": ("batch", "gqa", "nqb"),
        "qb_major": ("batch", "num_query_heads", "gqa"),
    }

    def test_signature_and_runtime_args_carry_the_fastdiv_pairs(self):
        """Every decode divisor gets a magic/shift kernarg pair, in the order the
        built body declares them, holding CK's magic numbers for this launch's
        divisor -- and those numbers really divide every work-item index. Odd
        batch, gqa=5 and nqb=24 keep each divisor off the power-of-two path, where
        magic is 1 and a wrong multiplier would go unnoticed."""
        base = dict(
            batch=3,
            nhead_q=40,
            nhead_k=8,
            seqlen_q=6144,
            seqlen_k=6144,
        )
        with _Gfx942Arch():
            for decode, divisors in self._DIVISORS.items():
                with self.subTest(decode=decode):
                    req = _req(**base)
                    self.assertTrue(_candidate().admits(req)[0])
                    spec = _persistent_spec(req, decode)
                    self.assertTrue(spec.persistent and spec.runtime_shape)
                    self.assertEqual(spec.resolved_persist_decode, decode)
                    self.assertEqual(spec.fastdiv_divisors, divisors)
                    fastdiv_fields = [
                        f"{d}_{part}" for d in divisors for part in ("magic", "shift")
                    ]
                    self.assertEqual(
                        list(spec.runtime_kernarg_fields),
                        list(spec.runtime_param_fields) + fastdiv_fields,
                    )

                    sig = [p["name"] for p in attention_dense_signature(spec)]
                    self.assertEqual(sig[5:], list(spec.runtime_kernarg_fields))
                    kd = build_attention_dense(spec, arch="gfx942")
                    self.assertEqual([p.name for p in kd.params], sig)

                    args = attention_dense_runtime_args(spec)
                    self.assertEqual(list(args), list(spec.runtime_kernarg_fields))
                    for f in spec.runtime_param_fields:
                        self.assertEqual(args[f], getattr(spec, f))
                    nqb = spec.seqlen_q // spec.block_m
                    value = {
                        "batch": 3,
                        "num_query_heads": 40,
                        "gqa": 5,
                        "nqb": nqb,
                    }
                    self.assertEqual(nqb, 24)
                    work = nqb * 40 * 3
                    for d in divisors:
                        magic, shift = args[f"{d}_magic"], args[f"{d}_shift"]
                        self.assertTrue(-(2**31) <= magic < 2**31, d)
                        self.assertEqual(
                            (magic & 0xFFFFFFFF, shift),
                            calculate_magic_numbers(value[d]),
                            d,
                        )
                        self.assertNotEqual(magic, 1, d)  # not the pow2 path
                        for n in range(work):
                            self.assertEqual(
                                _fastdiv(n, magic, shift), n // value[d], (d, n)
                            )

    def test_default_grid_has_no_fastdiv_kernargs(self):
        with _Gfx942Arch():
            spec = _spec(_req(), persistent=False)
            self.assertFalse(spec.persistent)
            self.assertEqual(spec.fastdiv_divisors, ())
            self.assertEqual(spec.runtime_kernarg_fields, spec.runtime_param_fields)
            self.assertEqual(
                list(attention_dense_runtime_args(spec)),
                list(spec.runtime_param_fields),
            )

    def test_auto_decode_is_keyed_by_its_resolved_order(self):
        """With the shape out of the key, raw 'auto' would let two shapes that
        resolve to different decode bodies share one cache slot. The key and the
        name carry the resolved order instead: shapes that resolve alike share
        both, shapes that resolve differently split both, and an explicit order
        lands in the same slot as the 'auto' that resolves to it."""
        common = dict(seqlen_q=8192, seqlen_k=8192)
        groups = {
            # gqa*nqb*B < 2*NP (608), or gqa == 1
            "qb_major": [
                dict(batch=1, nhead_q=128, nhead_k=8),
                dict(batch=1, nhead_q=40, nhead_k=8),
                dict(batch=4, nhead_q=32, nhead_k=32),
            ],
            # gqa > 1 and gqa*nqb*B >= 2*NP
            "hkv_major": [
                dict(batch=2, nhead_q=128, nhead_k=8),
                dict(batch=4, nhead_q=128, nhead_k=8),
                dict(batch=4, nhead_q=40, nhead_k=8),
            ],
        }
        keys, names = {}, {}
        with _Gfx942Arch():
            for decode, shapes in groups.items():
                with self.subTest(decode=decode):
                    specs = [_persistent_spec(_req(**common, **kw)) for kw in shapes]
                    self.assertTrue(all(s.persist_decode == "auto" for s in specs))
                    self.assertEqual(
                        {s.resolved_persist_decode for s in specs}, {decode}
                    )
                    group_keys = {
                        attention_dense_cache_key(s, arch="gfx942") for s in specs
                    }
                    group_names = {s.kernel_name() for s in specs}
                    self.assertEqual(len(group_keys), 1)
                    self.assertEqual(len(group_names), 1)
                    keys[decode], names[decode] = group_keys.pop(), group_names.pop()

                    explicit = _persistent_spec(_req(**common, **shapes[0]), decode)
                    self.assertEqual(explicit.persist_decode, decode)
                    self.assertEqual(
                        attention_dense_cache_key(explicit, arch="gfx942"),
                        keys[decode],
                    )
                    self.assertEqual(explicit.kernel_name(), names[decode])
            self.assertNotEqual(keys["qb_major"], keys["hkv_major"])
            self.assertNotEqual(names["qb_major"], names["hkv_major"])
            self.assertIn("hkvmaj", names["hkv_major"])
            self.assertNotIn("hkvmaj", names["qb_major"])

    def test_work_item_count_at_the_fastdiv_limit_raises(self):
        """The decode's fast division is exact only below 2**30; past it the
        quotient is silently wrong, so the launcher must refuse W = nqb*Hq*B >=
        2**30 rather than pack kernargs. The default grid has no such decode."""
        common = dict(
            seqlen_q=256,
            seqlen_kv=256,
            num_query_heads=1,
            num_kv_heads=1,
            head_size=128,
            causal=True,
            dtype="bf16",
            num_persistent=304,
        )
        below = Gfx942AttentionDenseSpec(batch=2**30 - 1, persistent=True, **common)
        self.assertEqual(below.seqlen_q // below.block_m, 1)  # W == batch
        self.assertEqual(attention_dense_runtime_args(below)["batch"], 2**30 - 1)
        at = Gfx942AttentionDenseSpec(batch=2**30, persistent=True, **common)
        with self.assertRaisesRegex(ValueError, r"2\*\*30"):
            attention_dense_runtime_args(at)
        default_grid = Gfx942AttentionDenseSpec(batch=2**30, persistent=False, **common)
        self.assertEqual(attention_dense_runtime_args(default_grid)["batch"], 2**30)


class TestGfx942SlidingWindow(unittest.TestCase):
    """Sliding-window pass-through and capability tests, mirroring gfx950's suite."""

    def test_sliding_window_zero_by_default(self):
        with _Gfx942Arch():
            self.assertEqual(_spec(_req()).sliding_window, 0)

    def test_sliding_window_passes_through_to_spec(self):
        with _Gfx942Arch():
            self.assertEqual(_spec(_req(sliding_window=128)).sliding_window, 128)

    def test_sliding_window_appears_in_kernel_name(self):
        with _Gfx942Arch():
            self.assertIn("swa256", _spec(_req(sliding_window=256)).kernel_name())

    def test_different_window_sizes(self):
        with _Gfx942Arch():
            for window in (64, 128, 256):
                spec = _spec(_req(sliding_window=window))
                self.assertEqual(spec.sliding_window, window)

    def test_sliding_window_in_supports_features(self):
        self.assertIn("sliding_window", _candidate().capability.supports_features)

    def test_sliding_window_requires_causal(self):
        """sliding_window without causal is rejected by the spec validator."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(sliding_window=128, mask_type=0))
            self.assertFalse(ok)
            self.assertNotIn("capability", why)


if __name__ == "__main__":
    unittest.main()
