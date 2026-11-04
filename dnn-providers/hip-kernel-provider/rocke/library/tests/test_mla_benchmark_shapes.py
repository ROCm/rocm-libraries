# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the MLA benchmark shape loader and the gfx942 prefill driver.

The failure these guard against is specific and has already happened once: the
MLA shape files shipped as a specification that the ndjson prefill harness
parsed to **zero** shapes, silently. A loader that returns an empty list on a
schema change would put the harness straight back there, so "parses at least the
declared shapes" is asserted rather than assumed.

CPU only -- resolving a shape through dispatch needs no GPU.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmarks.common.mla_shapes import (
    MLAGeometry,
    filter_shapes,
    load_mla_shapes,
)

_PREFILL_SHAPES = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "gfx942"
    / "attention"
    / "prefill"
    / "mla_prefill_shapes.json"
)


class TestLoader(unittest.TestCase):
    def test_the_shipped_prefill_file_parses(self):
        shapes = load_mla_shapes(_PREFILL_SHAPES)
        self.assertTrue(shapes, "the gfx942 MLA prefill shape file parsed to nothing")

    def test_every_declared_shape_is_loaded(self):
        # Against the file itself, so adding a shape to the JSON without the
        # loader picking it up is a failure rather than a quiet omission.
        doc = json.loads(_PREFILL_SHAPES.read_text())
        declared = sum(len(m.get("shapes", [])) for m in doc["models"])
        self.assertEqual(len(load_mla_shapes(_PREFILL_SHAPES)), declared)

    def test_both_families_are_present(self):
        regimes = {s.regime for s in load_mla_shapes(_PREFILL_SHAPES)}
        # Family 1 alone cannot measure this design: every full_prompt shape has
        # S_q == S_k, so the chunked family is the only place S_q != S_k.
        self.assertEqual(regimes, {"full_prompt", "chunked"})

    def test_both_head_counts_are_present(self):
        heads = {s.num_query_heads for s in load_mla_shapes(_PREFILL_SHAPES)}
        self.assertEqual(heads, {64, 128})

    def test_model_defaults_are_merged_into_each_shape(self):
        for shape in load_mla_shapes(_PREFILL_SHAPES):
            with self.subTest(shape=shape.label):
                self.assertGreater(shape.num_query_heads, 0)
                self.assertGreater(shape.block_size, 0)

    def test_geometry_is_the_canonical_mla_geometry(self):
        shapes = load_mla_shapes(_PREFILL_SHAPES)
        self.assertEqual(shapes[0].geometry, MLAGeometry())
        self.assertEqual(shapes[0].geometry.head_dim_qk, 192)

    def test_total_q_is_packed_not_rectangular(self):
        for shape in load_mla_shapes(_PREFILL_SHAPES):
            with self.subTest(shape=shape.label):
                self.assertEqual(shape.total_q, shape.batch * shape.seqlen_q)

    def test_dtype_is_bf16_only(self):
        # fp8 is a later-arch concern and is excluded from gfx942 entirely.
        self.assertEqual({s.dtype for s in load_mla_shapes(_PREFILL_SHAPES)}, {"bf16"})

    def test_rejects_a_file_with_no_shapes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text(json.dumps({"models": [{"model": "x", "shapes": []}]}))
            with self.assertRaises(ValueError):
                load_mla_shapes(path)

    def test_rejects_an_unknown_geometry_key(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                json.dumps(
                    {
                        "mla_geometry": {"qk_nope_dimension": 128},
                        "models": [{"model": "x", "shapes": []}],
                    }
                )
            )
            with self.assertRaises(ValueError) as ctx:
                load_mla_shapes(path)
            self.assertIn("qk_nope_dimension", str(ctx.exception))

    def test_filters(self):
        shapes = load_mla_shapes(_PREFILL_SHAPES)
        chunked = filter_shapes(shapes, regime="chunked")
        self.assertTrue(chunked)
        self.assertTrue(all(s.regime == "chunked" for s in chunked))
        self.assertEqual(len(filter_shapes(shapes, limit=3)), 3)
        kimi = filter_shapes(shapes, model="kimi")
        self.assertTrue(kimi)
        self.assertTrue(all("kimi" in s.model.lower() for s in kimi))


class TestEveryShapeDispatches(unittest.TestCase):
    """Every declared benchmark shape must be servable by a real candidate."""

    def test_all_shapes_resolve_on_gfx942(self):
        from dispatch.mla import dispatch_mla

        for shape in load_mla_shapes(_PREFILL_SHAPES):
            with self.subTest(shape=shape.signature):
                result = dispatch_mla(shape.to_request(arch="gfx942"))
                self.assertEqual(result.candidate.name, "mla_prefill_fwd_gfx942")

    def test_absorbed_query_reaches_the_dispatched_spec(self):
        from dispatch.mla import dispatch_mla

        for shape in load_mla_shapes(_PREFILL_SHAPES):
            with self.subTest(shape=shape.signature):
                req = shape.to_request(arch="gfx942", absorbed_query=True)
                self.assertTrue(req.absorbed_query)
                if shape.num_query_heads % 2:
                    # Single-head layout: no absorbed kernel.
                    with self.assertRaises(ValueError):
                        dispatch_mla(req)
                    continue
                spec = dispatch_mla(req).spec
                self.assertTrue(spec.absorbed_query)
                self.assertTrue(spec.fwd_kernel_name().endswith("_aq"))

    def test_resolve_mode_honours_absorbed_query(self):
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        for shape in load_mla_shapes(_PREFILL_SHAPES):
            if shape.num_query_heads % 2:
                continue
            with self.subTest(shape=shape.signature):
                fused = drv._resolve(shape, arch="gfx942", build=False)
                absorbed = drv._resolve(
                    shape, arch="gfx942", build=False, absorbed_query=True
                )
                self.assertFalse(fused["kernel"].endswith("_aq"))
                self.assertTrue(absorbed["kernel"].endswith("_aq"))

    def test_grid_follows_the_packed_block_numbering(self):
        from dispatch.mla import dispatch_mla

        for shape in load_mla_shapes(_PREFILL_SHAPES):
            with self.subTest(shape=shape.signature):
                result = dispatch_mla(shape.to_request(arch="gfx942"))
                expected = shape.total_q // result.spec.block_q + shape.batch
                heads = shape.num_query_heads // result.spec.heads_per_wg
                self.assertEqual(result.grid, (expected, heads, 1))


class TestSpecOverrideParsing(unittest.TestCase):
    """``--spec-override`` values are parsed by the spec field's type."""

    @staticmethod
    def _parse(*pairs):
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        return drv.parse_spec_overrides(pairs)

    def test_int_fields_parse_to_ints_and_the_spec_is_admitted(self):
        import dataclasses

        from dispatch.mla import MLARequest, dispatch_mla
        from kernels.mla.mla_prefill_gfx942 import supports_mla_prefill

        overrides = self._parse("num_xcds=0", "q_tiles_per_wg=2")
        self.assertEqual(overrides, {"num_xcds": 0, "q_tiles_per_wg": 2})
        self.assertIs(type(overrides["num_xcds"]), int)
        base = dispatch_mla(
            MLARequest(
                num_heads=128,
                total_q=64,
                num_seqs=2,
                max_seqlen_k=96,
                arch="gfx942",
                num_cus=304,
                num_kv_blocks=12,
            )
        ).spec
        spec = dataclasses.replace(base, **overrides)
        self.assertEqual(supports_mla_prefill(spec, arch="gfx942"), (True, ""))
        # q_tiles_per_wg=1: the two-heads layout with one 16-row tile per head.
        # "1" must stay the int 1; parsed as True it fails the exact-int check.
        one = self._parse("q_tiles_per_wg=1", "num_warps=2", "block_q=16")
        self.assertEqual(one, {"q_tiles_per_wg": 1, "num_warps": 2, "block_q": 16})
        self.assertIs(type(one["q_tiles_per_wg"]), int)
        spec = dataclasses.replace(base, **one)
        self.assertEqual(supports_mla_prefill(spec, arch="gfx942"), (True, ""))

    def test_bool_field_parses_true_false(self):
        self.assertEqual(self._parse("agpr_acc=true"), {"agpr_acc": True})
        self.assertEqual(self._parse("agpr_acc=FALSE"), {"agpr_acc": False})

    def test_bad_values_are_rejected_with_the_field_named(self):
        for pair, needle in (
            ("num_xcds=abc", "num_xcds is an int"),
            ("agpr_acc=1", "agpr_acc is a bool"),
            ("no_such_field=1", "not an MlaPrefillSpec field"),
            ("absorbed_query=true", "use --absorbed-query"),
            ("num_xcds", "expected KEY=VAL"),
        ):
            with self.subTest(pair=pair):
                with self.assertRaises(ValueError) as ctx:
                    self._parse(pair)
                self.assertIn(needle, str(ctx.exception))

    def test_cli_rejects_a_bad_override(self):
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        with self.assertRaises(SystemExit):
            drv.main(["--mode", "resolve", "--spec-override", "agpr_acc=1"])


class TestDriver(unittest.TestCase):
    def test_resolve_mode_reports_every_shape(self):
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        self.assertEqual(drv.main(["--mode", "resolve"]), 0)

    def test_list_shapes_mode(self):
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        self.assertEqual(drv.main(["--list-shapes", "--limit", "2"]), 0)

    def test_unknown_mode_is_rejected(self):
        # ``--mode time`` used to raise NotImplementedError and was asserted on
        # here; it is now fully implemented. Exercising it needs a GPU, aiter,
        # and minutes of wall time, which this suite -- CPU-only by contract,
        # see the module docstring -- will not do. What is still worth pinning
        # is that the mode list is validated rather than silently ignored.
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        with self.assertRaises(SystemExit):
            drv.main(["--mode", "nope"])

    def test_unsupported_arch_is_reported_not_raised(self):
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        # One bad shape must not hide the rest, so _resolve reports instead of
        # raising. On a non-gfx942 arch every shape is unsupported.
        self.assertEqual(drv.main(["--arch", "gfx950", "--limit", "1"]), 1)

    def test_request_carries_the_allocated_pool(self):
        # The driver allocates shape.num_kv_blocks pages; dispatch refuses 0.
        for shape in load_mla_shapes(_PREFILL_SHAPES):
            with self.subTest(shape=shape.signature):
                req = shape.to_request(arch="gfx942")
                pages = -(-shape.seqlen_k // shape.block_size)
                self.assertEqual(req.num_kv_blocks, shape.batch * pages)
                self.assertEqual(
                    shape.to_request(arch="gfx942", num_kv_blocks=7).num_kv_blocks, 7
                )

    def test_gate_heads_cover_both_waves_of_a_workgroup(self):
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        for h in (5, 6, 16, 64, 128):
            with self.subTest(num_heads=h):
                heads = drv._gate_heads(h)
                self.assertEqual(heads, sorted(set(heads)))
                self.assertTrue(all(0 <= x < h for x in heads))
                self.assertIn(h - 1, heads)
                for x in heads:
                    if x % 2 == 0 and x + 1 < h:
                        self.assertIn(x + 1, heads)
        self.assertEqual(drv._gate_heads(128), [0, 1, 32, 33, 64, 65, 96, 97, 127])

    def test_gate_applies_the_verifier_bars(self):
        # The rocke arm is held to out abs + out max_abs/peak + lse; the aiter
        # arm (no lse) to out abs + rel. Each failure below passes the 4e-2
        # absolute bar on its own.
        try:
            import torch
        except ImportError:
            self.skipTest("torch required")
        import numpy as np

        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        rows = slice(0, 4)
        ref_out = np.full((4, 2, 8), 0.1, dtype=np.float32)
        ref_lse = np.zeros((4, 2), dtype=np.float32)
        refs = [(rows, ref_out, ref_lse)]

        def gate(out_err, lse_err, with_lse=True):
            out = torch.from_numpy(ref_out + out_err)
            lse = torch.from_numpy(ref_lse + lse_err) if with_lse else None
            return drv._gate(out, lse, refs)

        self.assertTrue(gate(1e-3, 1e-3)["ok"])
        # 3e-2 abs: under 4e-2, but 0.3 of the 0.1 peak.
        rel = gate(3e-2, 1e-3)
        self.assertFalse(rel["ok"])
        self.assertAlmostEqual(rel["max_rel_peak_out"], 0.3, places=5)
        self.assertFalse(gate(3e-2, 0.0, with_lse=False)["ok"])
        self.assertTrue(gate(1e-3, 0.0, with_lse=False)["ok"])
        # lse 2e-2: under 4e-2, over LSE_ATOL.
        self.assertFalse(gate(1e-3, 2e-2)["ok"])
        self.assertEqual(
            (drv.OUT_REL_PEAK_TOL, drv.LSE_ATOL, drv.BF16_ATTN_ATOL),
            (2e-2, 1.7e-2, 4e-2),
        )

    def test_overridden_spec_is_readmitted(self):
        # --spec-override swaps the dispatched spec; admission must run again
        # and refuse as a status, not build a spec dispatch would never ship.
        from benchmarks.gfx942.attention.prefill import (
            benchmark_mla_prefill_live as drv,
        )

        shape = load_mla_shapes(_PREFILL_SHAPES)[0]
        with self.assertRaises(drv.SpecRefused) as ctx:
            drv._build_rocke_kernel(
                shape, arch="gfx942", spec_overrides={"binary_search_iters": 16}
            )
        self.assertIn("binary_search_iters", str(ctx.exception))
        with self.assertRaises(drv.SpecRefused):
            drv._build_rocke_kernel(
                shape, arch="gfx942", spec_overrides={"block_q": 24}
            )


if __name__ == "__main__":
    unittest.main()
