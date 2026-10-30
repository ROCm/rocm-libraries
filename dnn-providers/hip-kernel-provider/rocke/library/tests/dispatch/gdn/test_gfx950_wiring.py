# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GDN decode dispatch wiring: the ``problem -> spec`` direction.

CPU-only by construction -- it builds specs and reads the registry, never
compiling or launching. That is deliberate: a family whose only tests need a
GPU contributes nothing on a CPU CI machine, so the selection logic is covered
here and the numeric behaviour is covered separately by the on-device test.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict
from itertools import product

import numpy as np

from dispatch.gdn import (
    GDN_REGISTRY,
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
    gdn_candidates,
    gdn_sweep_space,
    request_errors,
)
from dispatch.gdn.axes import AUTO_DEFAULTS, AUTO_STATE_ROW
from dispatch.gdn.gfx950 import (
    ARCH,
    CONFIGURED_TILES,
    CONFIGURED_TILES_BY_GATE,
    KDA_CONFIGURED_TILES,
    spec_id_for,
)
from kernels.gfx950.gdn_decode import (
    STATE_DTYPES,
    gdn_decode_grid,
    gdn_decode_signature,
    is_valid_spec,
)
from rocke.dispatch.core import PinRefused

_TILE = lambda s: (s.num_warps, s.warp_threads_k, s.blocks_per_v_dim)  # noqa: E731


def _req(batch: int, **kw) -> GdnDecodeRequest:
    kw.setdefault("arch", ARCH)
    return GdnDecodeRequest(batch=batch, **kw)


def _candidates_of(gate_kind: str):
    """The registered candidates of one gate kind, in registry order."""
    ids = {spec_id_for(tile, gate_kind) for tile in CONFIGURED_TILES_BY_GATE[gate_kind]}
    return [c for c in gdn_candidates() if c.spec_id in ids]


class TestRegistration(unittest.TestCase):
    def test_every_configured_tile_is_registered(self):
        # GDN's tiles first, then KDA's, each in its gate kind's fallback order.
        ids = tuple(c.spec_id for c in gdn_candidates())
        gdn = tuple(f"nw{nw}_wtk{wtk}_bpv{bpv}" for nw, wtk, bpv in CONFIGURED_TILES)
        self.assertEqual(ids[: len(gdn)], gdn)

    def test_every_configured_tile_is_registered_for_kda_too(self):
        ids = tuple(c.spec_id for c in gdn_candidates())[len(CONFIGURED_TILES) :]
        self.assertEqual(
            ids,
            tuple(
                f"kda_nw{nw}_wtk{wtk}_bpv{bpv}" for nw, wtk, bpv in KDA_CONFIGURED_TILES
            ),
        )
        self.assertEqual(set(KDA_CONFIGURED_TILES), set(CONFIGURED_TILES))

    def test_registry_family_is_consistent(self):
        for cand in gdn_candidates():
            self.assertEqual(cand.family, GDN_REGISTRY.family)


_NT = {"state_load_hint": "streaming", "state_store_hint": "streaming"}
_DPP = {**_NT, "dpp_reduce": True}
_FUSED_KNOBS = {"conv_once": True, "out_lds": True}
# The shipped auto table, (gate_kind, fused, state row) -> (tile, knobs). A
# default change edits this, AUTO_DEFAULTS and the ALGORITHM.md §4.6 table.
_SHIPPED = {
    ("gdn", False, "bf16"): ((2, 16, 8), _DPP),
    ("gdn", False, "f32"): ((2, 16, 16), {**_DPP, "interleave_cols": True}),
    ("gdn", True, "bf16"): (
        (4, 16, 1),
        {**_DPP, **_FUSED_KNOBS, "norm_gate_once": True, "stream_rows": True},
    ),
    ("gdn", True, "f32"): (
        (8, 16, 1),
        {
            **_DPP,
            **_FUSED_KNOBS,
            "interleave_cols": True,
            "stream_rows": True,
            "waves_per_eu": 4,
        },
    ),
    ("kda", False, "bf16"): ((1, 16, 8), {**_DPP, "waves_per_eu": 4}),
    ("kda", False, "f32"): (
        (1, 16, 16),
        {**_DPP, "interleave_cols": True, "stream_rows": True},
    ),
    ("kda", True, "bf16"): ((4, 8, 1), {**_DPP, **_FUSED_KNOBS}),
    ("kda", True, "f32"): (
        (4, 8, 1),
        {**_DPP, **_FUSED_KNOBS, "interleave_cols": True, "stream_rows": True},
    ),
}
_BATCHES = (1, 2, 4, 5, 8, 16, 32, 33, 64, 128, 129, 256)


def _case_req(batch, gate_kind, fused, state_dtype, heads=(16, 16), **kw):
    return _req(
        batch,
        gate_kind=gate_kind,
        num_k_heads=heads[0],
        num_v_heads=heads[1],
        state_dtype=state_dtype,
        fuse_conv=fused,
        fuse_out_norm=fused,
        **kw,
    )


class TestStaticSelection(unittest.TestCase):
    """``auto`` runs one static tile and knob stack per case (gate kind x
    fused x state width), never a batch or work winner.

    The shipped table is pinned as a literal, not as ``AUTO_DEFAULTS``: a test
    that follows the constant cannot notice a default moving. Changing a value
    here changes what every gfx950 ``auto`` user of that case runs, so it needs
    a cold-memory measurement of the case in the same change.
    """

    # Known limitation (see CONFIGURED_TILES): an illegal case tile falls back
    # to the first legal tile in priority order, at its base configuration.
    # Pinned so a change is deliberate.
    _FALLBACK_TILE = (1, 1, 1)

    def test_auto_defaults_are_the_shipped_table(self):
        self.assertEqual(
            {case: (d.tile, dict(d.knobs)) for case, d in AUTO_DEFAULTS.items()},
            _SHIPPED,
        )

    def test_f16_state_runs_the_bf16_row(self):
        self.assertEqual(set(AUTO_STATE_ROW), set(STATE_DTYPES))
        self.assertEqual(AUTO_STATE_ROW["f16"], "bf16")
        for (gate_kind, fused, row), _ in _SHIPPED.items():
            if row != "bf16":
                continue
            with self.subTest(gate=gate_kind, fused=fused):
                bf16 = dispatch_gdn_decode(_case_req(8, gate_kind, fused, "bf16"))
                f16 = dispatch_gdn_decode(_case_req(8, gate_kind, fused, "f16"))
                self.assertEqual(f16.candidate.spec_id, bf16.candidate.spec_id)
                self.assertEqual(f16.spec.knobs, bf16.spec.knobs)
                self.assertEqual(f16.spec.tuning_id, bf16.spec.tuning_id)
                self.assertEqual(f16.spec.kernel_spec.state_dtype, "f16")

    def test_every_case_runs_its_tile_and_knobs_at_every_batch(self):
        # Sharded-head deployments see head counts other than the default
        # geometry, so cover them; fuse_conv needs Hk == Hv.
        plain_heads = ((2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (16, 16))
        fused_heads = ((4, 4), (8, 8), (16, 16), (32, 32))
        for (gate_kind, fused, row), (tile, knobs) in _SHIPPED.items():
            for heads in fused_heads if fused else plain_heads:
                picks = set()
                for batch in _BATCHES:
                    with self.subTest(
                        case=(gate_kind, fused, row), heads=heads, b=batch
                    ):
                        got = dispatch_gdn_decode(
                            _case_req(batch, gate_kind, fused, row, heads)
                        )
                        kernel = got.spec.kernel_spec
                        self.assertEqual(_TILE(kernel), tile)
                        self.assertEqual(
                            got.candidate.spec_id, spec_id_for(tile, gate_kind)
                        )
                        self.assertEqual(dict(got.spec.knobs), knobs)
                        for name, value in knobs.items():
                            self.assertEqual(getattr(kernel, name), value, name)
                        self.assertEqual(kernel.gate_kind, gate_kind)
                        ok, why = is_valid_spec(kernel, arch=ARCH)
                        self.assertTrue(ok, why)
                        # What ran replays from the request the result carries.
                        self.assertEqual(
                            dispatch_gdn_decode(got.request).spec, got.spec
                        )
                        picks.add(got.spec.tuning_id)
                # No batch dependence: one configuration for every batch.
                self.assertEqual(len(picks), 1, (gate_kind, fused, row, heads))

    def test_waves_per_eu_is_dropped_off_the_measured_head_dims(self):
        # The row's waves_per_eu was measured at DK = DV = 128 only; elsewhere
        # it caps the registers below what the state needs and the kernel
        # spills, so auto runs the rest of the row without it.
        knobs = _SHIPPED[("gdn", True, "f32")][1]
        self.assertIn("waves_per_eu", knobs)
        want = {k: v for k, v in knobs.items() if k != "waves_per_eu"}
        for dk, dv in ((256, 128), (256, 256), (128, 256)):
            with self.subTest(dk=dk, dv=dv):
                got = dispatch_gdn_decode(
                    _case_req(8, "gdn", True, "f32", head_k_dim=dk, head_v_dim=dv)
                )
                self.assertEqual(dict(got.spec.knobs), want)
                self.assertEqual(got.spec.kernel_spec.waves_per_eu, 0)

    def test_one_fused_flag_drops_the_knobs_it_cannot_read(self):
        for (gate_kind, fused, row), (tile, knobs) in _SHIPPED.items():
            if not fused:
                continue
            for conv, norm in ((True, False), (False, True)):
                with self.subTest(case=(gate_kind, row), conv=conv, norm=norm):
                    got = dispatch_gdn_decode(
                        _req(
                            8,
                            gate_kind=gate_kind,
                            num_k_heads=16,
                            num_v_heads=16,
                            state_dtype=row,
                            fuse_conv=conv,
                            fuse_out_norm=norm,
                        )
                    )
                    self.assertEqual(_TILE(got.spec.kernel_spec), tile)
                    dropped = (
                        {"norm_gate_once", "out_lds"}
                        if conv
                        else {"conv_once", "norm_gate_once"}
                    )
                    want = {k: v for k, v in knobs.items() if k not in dropped}
                    self.assertEqual(dict(got.spec.knobs), want)

    def test_a_spec_id_pin_of_the_auto_tile_runs_its_base(self):
        """Pins keep their meaning: a bare ``spec_id`` is the tile's base
        configuration (every knob off), not the case's stack."""
        for (gate_kind, fused, row), (tile, _) in _SHIPPED.items():
            with self.subTest(case=(gate_kind, fused, row)):
                req = _case_req(8, gate_kind, fused, row)
                pinned = dispatch_gdn_decode(
                    _case_req(
                        8, gate_kind, fused, row, spec_id=spec_id_for(tile, gate_kind)
                    )
                )
                self.assertEqual(pinned.spec.knobs, ())
                self.assertNotEqual(pinned.spec, dispatch_gdn_decode(req).spec)

    def test_auto_is_the_case_stack_or_the_documented_fallback(self):
        """Over a broad request domain: where the case's tile is legal, auto
        runs it with the case's stack (the validator never refuses a stack;
        the state hints are in scope everywhere, so knobs are never empty);
        where it is not, auto runs the first legal tile of the gate kind's
        priority order at its base configuration."""
        from dispatch.gdn.gfx950 import CONFIGURED_TILES_BY_GATE, make_spec

        fused_modes = ((False, False), (True, False), (False, True), (True, True))
        gdn_dims = tuple((d, d) for d in (64, 96, 128, 192, 256)) + (
            (128, 16),
            (128, 32),
            (128, 64),
            (128, 256),
            (256, 128),
        )
        for gate_kind, dims in (("gdn", gdn_dims), ("kda", ((128, 128),))):
            for (dk, dv), state_dtype, (conv, norm), heads, batch in product(
                dims,
                ("bf16", "f16", "f32"),
                fused_modes,
                ((4, 4), (16, 16), (16, 32), (48, 96)),
                (1, 64),
            ):
                req = _req(
                    batch,
                    gate_kind=gate_kind,
                    num_k_heads=heads[0],
                    num_v_heads=heads[1],
                    head_k_dim=dk,
                    head_v_dim=dv,
                    state_dtype=state_dtype,
                    fuse_conv=conv,
                    fuse_out_norm=norm,
                )
                if request_errors(req):
                    continue
                case = (gate_kind, conv or norm, AUTO_STATE_ROW[state_dtype])
                tile = _SHIPPED[case][0]
                with self.subTest(req=req):
                    got = dispatch_gdn_decode(req)
                    if is_valid_spec(make_spec(req, tile), arch=ARCH)[0]:
                        self.assertEqual(_TILE(got.spec.kernel_spec), tile)
                        self.assertTrue(got.spec.knobs)
                    else:
                        first = next(
                            t
                            for t in CONFIGURED_TILES_BY_GATE[gate_kind]
                            if is_valid_spec(make_spec(req, t), arch=ARCH)[0]
                        )
                        self.assertEqual(_TILE(got.spec.kernel_spec), first)
                        self.assertEqual(got.spec.knobs, ())

    def test_supported_geometry_falls_back_when_the_case_tile_is_invalid(self):
        for head_k_dim in (64, 192):
            for state_dtype in ("bf16", "f32"):
                for fused in (False, True):
                    for batch in (1, 256):
                        with self.subTest(
                            head_k_dim=head_k_dim,
                            state=state_dtype,
                            fused=fused,
                            batch=batch,
                        ):
                            result = dispatch_gdn_decode(
                                _case_req(
                                    batch,
                                    "gdn",
                                    fused,
                                    state_dtype,
                                    head_k_dim=head_k_dim,
                                )
                            )
                            kernel = result.spec.kernel_spec
                            ok, why = is_valid_spec(kernel, arch=ARCH)
                            self.assertTrue(ok, why)
                            self.assertEqual(kernel.head_k_dim, head_k_dim)
                            self.assertEqual(_TILE(kernel), self._FALLBACK_TILE)
                            self.assertEqual(result.spec.knobs, ())

    def test_plain_gdn_with_a_narrow_value_dim_falls_back_to_the_fused_tile(self):
        # (2,16,8) needs head_v_dim / 8 to be a multiple of 8, so DV 16 and 32
        # are illegal; the first legal tile of the GDN priority order is then
        # another case's auto tile.
        for head_v_dim in (16, 32):
            with self.subTest(head_v_dim=head_v_dim):
                result = dispatch_gdn_decode(
                    _case_req(8, "gdn", False, "bf16", head_v_dim=head_v_dim)
                )
                self.assertEqual(_TILE(result.spec.kernel_spec), (4, 16, 1))
                self.assertEqual(result.spec.knobs, ())


class TestRequestRejection(unittest.TestCase):
    def test_other_arch_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(8, arch="gfx942"))
        self.assertIn("gfx942", str(ctx.exception))

    def test_head_ratio_must_divide(self):
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(8, num_v_heads=33))
        self.assertIn("multiple", str(ctx.exception))

    def test_non_positive_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(_req(0))

    def test_unsupported_dtype_is_rejected(self):
        with self.assertRaises(ValueError):
            dispatch_gdn_decode(_req(8, dtype="fp8"))

    def test_kda_d128_is_admitted(self):
        result = dispatch_gdn_decode(
            _req(
                1,
                gate_kind="kda",
                num_k_heads=32,
                num_v_heads=32,
                head_k_dim=128,
                head_v_dim=128,
            )
        )
        self.assertEqual(result.spec.kernel_spec.gate_kind, "kda")
        self.assertEqual(
            (result.spec.kernel_spec.head_k_dim, result.spec.kernel_spec.head_v_dim),
            (128, 128),
        )

    def test_kda_non_d128_is_loudly_scoped_out(self):
        for head_k_dim, head_v_dim in ((64, 128), (128, 64)):
            with self.subTest(head_k_dim=head_k_dim, head_v_dim=head_v_dim):
                with self.assertRaises(ValueError) as ctx:
                    dispatch_gdn_decode(
                        _req(
                            1,
                            gate_kind="kda",
                            num_k_heads=32,
                            num_v_heads=32,
                            head_k_dim=head_k_dim,
                            head_v_dim=head_v_dim,
                        )
                    )
                self.assertIn("NOT_YET_IMPLEMENTED", str(ctx.exception))
                self.assertIn("128", str(ctx.exception))

    def test_gdn_d64_remains_supported(self):
        result = dispatch_gdn_decode(_req(1, head_k_dim=64))
        self.assertEqual(result.spec.kernel_spec.gate_kind, "gdn")
        self.assertEqual(result.spec.kernel_spec.head_k_dim, 64)


class TestMalformedRequestFields(unittest.TestCase):
    """A wrong-typed request field is rejected, never coerced into a spec.

    ``bool("false")`` is True and ``int(True)`` is 1, so a dispatcher that
    converted these fields would hand back a DIFFERENT valid kernel (the
    ``_l2`` one for ``use_qk_l2norm="false"``) instead of an error.
    """

    def _assert_rejected(self, field, message, **kw):
        req = _req(8, **kw)
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(req)
        self.assertIn(field, str(ctx.exception))
        self.assertIn(message, str(ctx.exception))
        self.assertEqual(dispatch_gdn_decode_all(req), ())

    def test_non_bool_l2norm_is_rejected(self):
        for value in ("false", "true", None, 0, 1):
            for gate_kind in ("gdn", "kda"):
                with self.subTest(value=value, gate_kind=gate_kind):
                    self._assert_rejected(
                        "use_qk_l2norm",
                        "must be a bool",
                        use_qk_l2norm=value,
                        gate_kind=gate_kind,
                        num_k_heads=32,
                    )

    def test_non_int_head_fields_are_rejected(self):
        # num_v_heads 32 is a multiple of True / 4.0, so only the type rule
        # can catch these.
        for field in ("num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim"):
            for value in (True, 4.0):
                with self.subTest(field=field, value=value):
                    kw = {"num_k_heads": 4, "num_v_heads": 32, field: value}
                    if field == "num_v_heads":
                        kw["num_k_heads"] = 1
                    self._assert_rejected(field, "must be an int", **kw)

    def test_non_str_gate_kind_is_rejected(self):
        for value in (None, 1, b"gdn", ("gdn",)):
            with self.subTest(value=value):
                self._assert_rejected("gate_kind", "unsupported", gate_kind=value)


class TestSpecIdPin(unittest.TestCase):
    def test_pin_selects_the_exact_registered_tile(self):
        result = dispatch_gdn_decode(
            _req(256, algorithm="warp_tiled", spec_id="nw4_wtk16_bpv8")
        )
        self.assertEqual(result.candidate.spec_id, "nw4_wtk16_bpv8")
        self.assertEqual(_TILE(result.spec.kernel_spec), (4, 16, 8))

    def test_every_legal_gdn_pin_is_reachable_at_any_batch(self):
        for candidate in _candidates_of("gdn"):
            req = _req(64, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
            if not candidate.admits(req)[0]:
                continue
            with self.subTest(spec_id=candidate.spec_id):
                got = dispatch_gdn_decode(req)
                self.assertEqual(got.candidate.spec_id, candidate.spec_id)

    def test_every_legal_kda_pin_is_reachable_with_kda_gate(self):
        for candidate in _candidates_of("kda"):
            req = GdnDecodeRequest(
                batch=64,
                arch=ARCH,
                spec_id=candidate.spec_id,
                gate_kind="kda",
                num_k_heads=32,
                num_v_heads=32,
            )
            if not candidate.admits(req)[0]:
                continue
            with self.subTest(spec_id=candidate.spec_id):
                got = dispatch_gdn_decode(req)
                self.assertEqual(got.candidate.spec_id, candidate.spec_id)
                self.assertEqual(got.spec.kernel_spec.gate_kind, "kda")

    def test_a_kda_pin_cannot_serve_gdn(self):
        with self.assertRaises(PinRefused):
            dispatch_gdn_decode(
                GdnDecodeRequest(
                    batch=64, arch=ARCH, spec_id="kda_nw4_wtk16_bpv4", gate_kind="gdn"
                )
            )

    def test_algorithm_pin_is_honoured_and_an_unknown_one_is_rejected(self):
        # The `algorithm` pin is a separate selector from `spec_id` above, and
        # until this test nothing exercised it: a tuner re-measuring the table,
        # or anyone bisecting a routing regression, forces the algorithm rather
        # than the tile. The family ships one algorithm today, so the case that
        # would silently rot is the REJECTION -- a pin nobody serves must fail
        # loudly instead of falling through to the tuned default.
        got = dispatch_gdn_decode(_req(64, algorithm="warp_tiled"))
        self.assertEqual(got.candidate.algorithm, "warp_tiled")
        with self.assertRaises(ValueError) as ctx:
            dispatch_gdn_decode(_req(64, algorithm="no_such_algorithm"))
        self.assertIn("algorithm", str(ctx.exception))


class TestRegistryContract(unittest.TestCase):
    def test_the_family_refuses_an_unbuildable_candidate(self):
        """`require_build=True` is what stops a candidate being selectable but
        not compilable -- it fails at registration instead of at launch. The
        sibling family asserts this (tests/dispatch/kda); without the assertion
        the flag could be dropped and nothing would go red."""
        self.assertTrue(GDN_REGISTRY.require_build)
        for candidate in gdn_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertIsNotNone(candidate.build)


class TestDtypeCoverage(unittest.TestCase):
    def test_kernel_and_capability_name_the_same_dtypes(self):
        """The dtype set is declared once by the kernel and re-exported by
        dispatch. This pins the CONTENT as well as the sharing: narrowing the
        tuple would otherwise silently shrink coverage -- every test still
        passes, there are just fewer of them -- which is the quiet direction
        the re-export was meant to prevent."""
        from kernels.gfx950.gdn_decode import GDN_DTYPES

        self.assertEqual(set(GDN_DTYPES), {"bf16", "f16"})
        for candidate in gdn_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertEqual(set(candidate.capability.dtypes), set(GDN_DTYPES))
        for dtype in GDN_DTYPES:
            with self.subTest(dtype=dtype):
                got = dispatch_gdn_decode(_req(16, dtype=dtype))
                self.assertEqual(got.spec.kernel_spec.dtype, dtype)


class TestLaunchGeometry(unittest.TestCase):
    def test_grid_and_block_track_the_selected_spec(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                got = dispatch_gdn_decode(_req(batch))
                self.assertEqual(got.grid, gdn_decode_grid(batch, got.spec.kernel_spec))
                self.assertEqual(got.block, (got.spec.kernel_spec.block_size, 1, 1))

    def test_dtype_aliases_normalize(self):
        a = dispatch_gdn_decode(_req(16, dtype="bfloat16"))
        b = dispatch_gdn_decode(_req(16, dtype="bf16"))
        self.assertEqual(
            a.spec.kernel_spec.kernel_name(), b.spec.kernel_spec.kernel_name()
        )


class TestKernelIdentity(unittest.TestCase):
    def test_same_request_gives_a_stable_cache_key(self):
        a = dispatch_gdn_decode(_req(16)).kernel_id
        b = dispatch_gdn_decode(_req(16)).kernel_id
        self.assertEqual(a.spec_hash, b.spec_hash)
        self.assertEqual(a.compile_key, b.compile_key)

    def test_different_tiles_do_not_share_a_cache_key(self):
        seen = set()
        for spec_id in ("nw1_wtk8_bpv1", "nw2_wtk8_bpv2", "nw4_wtk16_bpv8"):
            result = dispatch_gdn_decode(
                _req(16, algorithm="warp_tiled", spec_id=spec_id)
            )
            self.assertNotIn(result.kernel_id.compile_key, seen)
            seen.add(result.kernel_id.compile_key)

    def test_spec_hash_covers_the_tile(self):
        from rocke.dispatch.core import stable_json_hash

        a = dispatch_gdn_decode(
            _req(16, algorithm="warp_tiled", spec_id="nw1_wtk8_bpv1")
        ).spec.kernel_spec
        b = dispatch_gdn_decode(
            _req(16, algorithm="warp_tiled", spec_id="nw2_wtk8_bpv2")
        ).spec.kernel_spec
        self.assertNotEqual(
            stable_json_hash(asdict(a), n=16), stable_json_hash(asdict(b), n=16)
        )


class TestSweepSpace(unittest.TestCase):
    def test_sweep_space_is_non_empty_and_valid(self):
        specs = gdn_sweep_space(_req(16))
        self.assertTrue(specs)
        for spec in specs:
            ok, why = is_valid_spec(spec.kernel_spec, arch=ARCH)
            self.assertTrue(ok, why)

    def test_sweep_space_of_a_bad_request_is_empty(self):
        self.assertEqual(gdn_sweep_space(_req(8, num_v_heads=33)), ())


class TestDispatchResultContract(unittest.TestCase):
    """The result must be sufficient to drive a launch on its own.

    A caller should not need to reach back into the kernel module for the
    signature or the grid; if the result disagrees with the spec it carries,
    kernel arguments would be packed against one layout and the kernel compiled
    against another.
    """

    def test_build_returns_the_kernel_the_spec_names(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                result = dispatch_gdn_decode(_req(batch))
                kernel = result.build()
                self.assertEqual(kernel.name, result.spec.kernel_spec.kernel_name())

    def test_signature_matches_the_spec(self):
        for batch in (1, 16, 64, 256):
            with self.subTest(batch=batch):
                result = dispatch_gdn_decode(_req(batch))
                self.assertEqual(
                    tuple(result.signature),
                    tuple(gdn_decode_signature(result.spec.kernel_spec)),
                )

    def test_compile_key_names_arch_and_abi(self):
        kid = dispatch_gdn_decode(_req(16)).kernel_id
        self.assertIn(ARCH, kid.compile_key)
        self.assertIn("rocke-gdn-decode", kid.compile_key)


if __name__ == "__main__":
    unittest.main()


class TestGateKindWiring(unittest.TestCase):
    """The request carries gate_kind through to the spec, GDN by default."""

    def test_request_defaults_to_the_gdn_gate(self):
        self.assertEqual(_req(1).gate_kind, "gdn")
        self.assertEqual(dispatch_gdn_decode(_req(1)).spec.kernel_spec.gate_kind, "gdn")

    def test_kda_request_selects_a_kda_spec(self):
        req = GdnDecodeRequest(batch=4, arch=ARCH, gate_kind="kda")
        spec = dispatch_gdn_decode(req).spec.kernel_spec
        self.assertEqual(spec.gate_kind, "kda")
        self.assertIn("kda", spec.kernel_name())

    def test_gate_kind_reaches_the_compile_key(self):
        # Two requests differing only in gate_kind select different kernels, so
        # they must not collapse onto one compile-cache entry.
        gdn = dispatch_gdn_decode(GdnDecodeRequest(batch=4, arch=ARCH))
        kda = dispatch_gdn_decode(GdnDecodeRequest(batch=4, arch=ARCH, gate_kind="kda"))
        self.assertNotEqual(gdn.kernel_id.compile_key, kda.kernel_id.compile_key)

    def test_unknown_gate_kind_is_rejected(self):
        errors = request_errors(GdnDecodeRequest(batch=1, arch=ARCH, gate_kind="mamba"))
        self.assertTrue(any("gate_kind" in e for e in errors), errors)


class TestGdnAndKdaTileNamespaces(unittest.TestCase):
    """The shared registry keeps GDN and KDA selectable identities disjoint."""

    def test_kda_spec_ids_cannot_collide_with_gdn_spec_ids(self):
        gdn_ids = {spec_id_for(tile, "gdn") for tile in CONFIGURED_TILES}
        kda_ids = {spec_id_for(tile, "kda") for tile in KDA_CONFIGURED_TILES}
        self.assertFalse(gdn_ids & kda_ids)
        registered = [candidate.spec_id for candidate in gdn_candidates()]
        self.assertEqual(len(registered), len(set(registered)))
        self.assertEqual(set(registered), gdn_ids | kda_ids)


class TestStateDtypeDispatch(unittest.TestCase):
    """An f32 recurrent state is reachable from the request for both gate
    kinds. Which tile and knobs it runs is the f32 row of its case
    (``TestStaticSelection``)."""

    def test_every_f32_spelling_reaches_the_spec_for_both_gates(self):
        for gate_kind in ("gdn", "kda"):
            for spelling in ("f32", "fp32", "float32", " F32 "):
                with self.subTest(gate_kind=gate_kind, spelling=spelling):
                    got = dispatch_gdn_decode(
                        _req(16, gate_kind=gate_kind, state_dtype=spelling)
                    )
                    self.assertEqual(got.spec.kernel_spec.state_dtype, "f32")
                    self.assertEqual(got.spec.kernel_spec.dtype, "bf16")
                    self.assertIn("_stf32_", got.spec.kernel_spec.kernel_name())
                    self.assertIn(
                        ("state", "ptr<f32, global>"), _sig_types(got.signature)
                    )

    def test_f32_is_a_state_dtype_only(self):
        for spelling in ("f32", "float32"):
            with self.subTest(spelling=spelling):
                errors = request_errors(_req(16, dtype=spelling))
                self.assertTrue(any("unsupported dtype" in e for e in errors), errors)
                with self.assertRaises(ValueError):
                    dispatch_gdn_decode(_req(16, dtype=spelling))

    def test_unknown_state_dtype_is_rejected(self):
        errors = request_errors(_req(16, state_dtype="f64"))
        self.assertTrue(any("state_dtype" in e for e in errors), errors)


def _sig_types(signature):
    return {(p["name"], p["type"]) for p in signature}


def _fused_req(batch: int = 8, **kw) -> GdnDecodeRequest:
    base = dict(
        num_k_heads=16,
        num_v_heads=16,
        fuse_conv=True,
        fuse_out_norm=True,
    )
    base.update(kw)
    return _req(batch, **base)


class TestFusedDispatch(unittest.TestCase):
    """fuse_conv / fuse_out_norm requests: the flags reach the spec and the
    launch ABI, auto runs the fused case's BPV=1 tile, and only BPV=1 tiles
    serve. The fused tiles are pinned as literals in ``_SHIPPED``.
    """

    _MODES = ((True, False), (False, True), (True, True))

    def test_auto_selects_the_fused_tile_for_every_mode(self):
        for gate_kind in ("gdn", "kda"):
            for state_dtype in ("bf16", "f16", "f32"):
                for conv, norm in self._MODES:
                    for batch in (1, 8, 128, 4096):
                        with self.subTest(
                            gate=gate_kind,
                            state=state_dtype,
                            conv=conv,
                            norm=norm,
                            batch=batch,
                        ):
                            got = dispatch_gdn_decode(
                                _fused_req(
                                    batch,
                                    gate_kind=gate_kind,
                                    state_dtype=state_dtype,
                                    fuse_conv=conv,
                                    fuse_out_norm=norm,
                                )
                            )
                            spec = got.spec.kernel_spec
                            row = AUTO_STATE_ROW[state_dtype]
                            tile = _SHIPPED[(gate_kind, True, row)][0]
                            self.assertEqual(_TILE(spec), tile)
                            self.assertEqual(
                                got.candidate.spec_id, spec_id_for(tile, gate_kind)
                            )
                            self.assertEqual(
                                (spec.fuse_conv, spec.fuse_out_norm), (conv, norm)
                            )
                            self.assertEqual(spec.gate_kind, gate_kind)
                            self.assertEqual(spec.state_dtype, state_dtype)
                            ok, why = is_valid_spec(spec, arch=ARCH)
                            self.assertTrue(ok, why)

    def test_fused_auto_ignores_head_count_and_ratio(self):
        # fuse_out_norm alone allows Hv > Hk (the GDN target's layout).
        for gate_kind in ("gdn", "kda"):
            for num_k_heads, num_v_heads in ((4, 4), (8, 32), (16, 32), (32, 32)):
                with self.subTest(gate=gate_kind, hk=num_k_heads, hv=num_v_heads):
                    got = dispatch_gdn_decode(
                        _fused_req(
                            16,
                            gate_kind=gate_kind,
                            num_k_heads=num_k_heads,
                            num_v_heads=num_v_heads,
                            fuse_conv=False,
                        )
                    )
                    self.assertEqual(
                        _TILE(got.spec.kernel_spec),
                        _SHIPPED[(gate_kind, True, "bf16")][0],
                    )

    def test_fused_flags_reach_the_launch_abi(self):
        """The DispatchResult's signature is the fused kernel ABI, so a caller
        packing arguments from it passes the fused tensors."""
        for conv, norm in self._MODES:
            with self.subTest(conv=conv, norm=norm):
                got = dispatch_gdn_decode(
                    _fused_req(gate_kind="kda", fuse_conv=conv, fuse_out_norm=norm)
                )
                names = {p["name"] for p in got.signature}
                self.assertEqual(
                    {"mixed_qkv", "conv_state", "conv_weight", "qkv_stride"} <= names,
                    conv,
                )
                self.assertEqual("query" in names, not conv)
                self.assertEqual(
                    {"out_gate", "norm_weight", "og_stride", "norm_eps"} <= names, norm
                )
                self.assertEqual(
                    tuple(got.signature),
                    tuple(gdn_decode_signature(got.spec.kernel_spec)),
                )
                self.assertEqual(got.grid, gdn_decode_grid(8, got.spec.kernel_spec))
                self.assertEqual(got.build().name, got.spec.kernel_spec.kernel_name())

    def test_each_mode_has_its_own_compile_key(self):
        for gate_kind in ("gdn", "kda"):
            keys = set()
            for conv, norm in ((False, False),) + self._MODES:
                # The same tile in every mode, so only the flags differ.
                got = dispatch_gdn_decode(
                    _fused_req(
                        8,
                        gate_kind=gate_kind,
                        fuse_conv=conv,
                        fuse_out_norm=norm,
                        algorithm="warp_tiled",
                        spec_id=(
                            "kda_nw4_wtk16_bpv1"
                            if gate_kind == "kda"
                            else "nw4_wtk16_bpv1"
                        ),
                    )
                )
                keys.add(got.kernel_id.compile_key)
            self.assertEqual(len(keys), 4, gate_kind)

    def test_bpv_above_one_cannot_serve_a_fused_request(self):
        for spec_id, gate_kind in (
            ("nw2_wtk16_bpv8", "gdn"),
            ("nw4_wtk16_bpv4", "gdn"),
            ("kda_nw4_wtk16_bpv4", "kda"),
            ("kda_nw1_wtk16_bpv4", "kda"),
        ):
            with self.subTest(spec_id=spec_id):
                with self.assertRaises(ValueError) as ctx:
                    dispatch_gdn_decode(
                        _fused_req(gate_kind=gate_kind, spec_id=spec_id)
                    )
                self.assertIn("blocks_per_v_dim", str(ctx.exception))

    def test_bpv1_pins_serve_a_fused_request(self):
        for spec_id, gate_kind, tile in (
            ("nw2_wtk16_bpv1", "gdn", (2, 16, 1)),
            ("nw8_wtk8_bpv1", "gdn", (8, 8, 1)),
            ("kda_nw2_wtk16_bpv1", "kda", (2, 16, 1)),
            ("kda_nw4_wtk16_bpv1", "kda", (4, 16, 1)),
        ):
            with self.subTest(spec_id=spec_id):
                got = dispatch_gdn_decode(
                    _fused_req(gate_kind=gate_kind, spec_id=spec_id)
                )
                self.assertEqual(got.candidate.spec_id, spec_id)
                self.assertEqual(_TILE(got.spec.kernel_spec), tile)
                kernel = got.spec.kernel_spec
                self.assertTrue(kernel.fuse_conv and kernel.fuse_out_norm)

    def test_fuse_conv_with_grouped_heads_is_a_request_error(self):
        """Dispatch and the kernel validator report one shared rule, so the
        request-level rejection cannot outlive the kernel limit."""
        from dispatch.gdn.gfx950 import make_spec
        from kernels.gfx950.gdn_decode import fuse_conv_heads_error

        rule = fuse_conv_heads_error(8, 16)
        self.assertIn("NOT_YET_IMPLEMENTED", rule)
        self.assertIsNone(fuse_conv_heads_error(16, 16))
        for gate_kind in ("gdn", "kda"):
            req = _fused_req(gate_kind=gate_kind, num_k_heads=8, num_v_heads=16)
            with self.subTest(gate=gate_kind):
                self.assertIn(rule, request_errors(req))
                self.assertEqual(
                    is_valid_spec(make_spec(req, (4, 16, 1)), arch=ARCH),
                    (False, rule),
                )
                with self.assertRaises(ValueError):
                    dispatch_gdn_decode(req)
                self.assertEqual(dispatch_gdn_decode_all(req), ())

    def test_the_kda_fused_candidate_cannot_serve_gdn(self):
        with self.assertRaises(PinRefused) as raised:
            dispatch_gdn_decode(
                _fused_req(gate_kind="gdn", spec_id="kda_nw4_wtk16_bpv1")
            )
        self.assertIn("serves the 'kda' gate", str(raised.exception))


class TestRequestFieldTypes(unittest.TestCase):
    """The ABI-changing request flags must be real bools and the dims real
    ints: a request is rejected, never coerced into a different one
    (``batch=2.5`` would launch 2 sequences, ``fuse_conv="false"`` the fused
    kernel)."""

    _INT_FIELDS = (
        "batch",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
    )

    def test_non_int_dims_are_request_errors(self):
        for field in self._INT_FIELDS:
            for bad in (2.5, 16.0, True, "4", None):
                for gate_kind in ("gdn", "kda"):
                    with self.subTest(field=field, value=bad, gate=gate_kind):
                        req = _fused_req(gate_kind=gate_kind, **{field: bad})
                        errors = request_errors(req)
                        self.assertIn(f"{field} must be an int, got {bad!r}", errors)
                        with self.assertRaises(ValueError):
                            dispatch_gdn_decode(req)
                        self.assertEqual(dispatch_gdn_decode_all(req), ())

    def test_int_dims_reach_the_spec_and_grid_unchanged(self):
        for fused in (False, True):
            req = _req(
                3,
                num_k_heads=16,
                num_v_heads=16,
                head_k_dim=128,
                head_v_dim=64,
                fuse_conv=fused,
                fuse_out_norm=fused,
            )
            with self.subTest(fused=fused):
                self.assertEqual(request_errors(req), [])
                got = dispatch_gdn_decode(req)
                spec = got.spec.kernel_spec
                self.assertEqual(
                    (
                        spec.num_k_heads,
                        spec.num_v_heads,
                        spec.head_k_dim,
                        spec.head_v_dim,
                    ),
                    (16, 16, 128, 64),
                )
                self.assertEqual(got.grid, gdn_decode_grid(3, spec))

    def test_non_bool_flags_are_request_errors(self):
        for field in ("fuse_conv", "fuse_out_norm", "use_qk_l2norm"):
            for bad in ("false", "True", 1, 0, None):
                with self.subTest(field=field, value=bad):
                    req = _fused_req(**{field: bad})
                    errors = request_errors(req)
                    self.assertTrue(
                        any(e.startswith(f"{field} must be a bool") for e in errors),
                        errors,
                    )
                    with self.assertRaises(ValueError):
                        dispatch_gdn_decode(req)
                    self.assertEqual(dispatch_gdn_decode_all(req), ())

    def test_bool_flags_are_accepted(self):
        for value in (False, True):
            with self.subTest(value=value):
                self.assertEqual(
                    request_errors(
                        _fused_req(
                            fuse_conv=value, fuse_out_norm=value, use_qk_l2norm=value
                        )
                    ),
                    [],
                )


class TestNumpyIntegralFields(unittest.TestCase):
    """A numpy integer is an integer: ``batch=np.int64(4)`` must pass the
    request check, dispatch end to end (the identity payload is JSON-hashed,
    so a raw np.int64 would raise ``TypeError`` there), and produce the same
    identity as the same value as a plain ``int``. ``np.bool_`` stays a
    non-bool: it is not a flag."""

    _INT_FIELDS = (
        "batch",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
    )

    def test_numpy_ints_dispatch_and_hash_like_plain_ints(self):
        plain_defaults = dict(
            batch=4,
            num_k_heads=16,
            num_v_heads=32,
            head_k_dim=128,
            head_v_dim=128,
        )
        for npint in (np.int32, np.int64):
            for field in self._INT_FIELDS:
                # Independent all-Python reference; the numpy request replaces
                # one field with a np scalar, the reference stays plain end to
                # end.
                ref = dispatch_gdn_decode(_req(**plain_defaults))
                np_defaults = dict(plain_defaults)
                np_defaults[field] = npint(plain_defaults[field])
                with self.subTest(field=field, npint=npint.__name__):
                    req = GdnDecodeRequest(arch=ARCH, **np_defaults)
                    self.assertEqual(request_errors(req), [])
                    got = dispatch_gdn_decode(req)
                    self.assertEqual(
                        got.kernel_id.request_hash, ref.kernel_id.request_hash
                    )
                    self.assertEqual(got.kernel_id.spec_hash, ref.kernel_id.spec_hash)
                    self.assertEqual(got.grid, ref.grid)

    def test_numpy_ints_reach_the_spec_as_plain_ints(self):
        req = GdnDecodeRequest(
            arch=ARCH,
            batch=np.int64(3),
            num_k_heads=np.int64(16),
            num_v_heads=np.int64(16),
            head_k_dim=np.int64(128),
            head_v_dim=np.int64(64),
        )
        self.assertEqual(request_errors(req), [])
        got = dispatch_gdn_decode(req)
        for value in (
            got.spec.kernel_spec.num_k_heads,
            got.spec.kernel_spec.num_v_heads,
            got.spec.kernel_spec.head_k_dim,
            got.spec.kernel_spec.head_v_dim,
        ):
            self.assertIs(type(value), int)

    def test_np_bool_flag_is_not_a_bool(self):
        req = _fused_req(use_qk_l2norm=np.bool_(True))
        self.assertTrue(
            any(
                e.startswith("use_qk_l2norm must be a bool")
                for e in request_errors(req)
            )
        )
