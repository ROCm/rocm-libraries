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

    def test_grid_follows_the_packed_block_numbering(self):
        from dispatch.mla import dispatch_mla

        for shape in load_mla_shapes(_PREFILL_SHAPES):
            with self.subTest(shape=shape.signature):
                result = dispatch_mla(shape.to_request(arch="gfx942"))
                expected = shape.total_q // result.spec.block_q + shape.batch
                self.assertEqual(result.grid, (expected, shape.num_query_heads, 1))


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


if __name__ == "__main__":
    unittest.main()
