# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the dispatcher-driven fused-MoE fp8 benchmark."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rocke.benchmark.moe.fused_mega_fp8_dispatch import (
    DEFAULT_TOKENS,
    ORACLE_MAX_TOKENS,
    ExpertWeights,
    MoeShape,
    Routing,
    ShapeRecord,
    TokenLayout,
    _dispatched_block_count,
    check_tokens,
    grid_bound_warnings,
    oracle_tokens,
    plan_stage_grids,
    write_json,
)
from rocke.dispatch.families.moe import dispatch_moe_plan


def _plan(tokens: int):
    return dispatch_moe_plan(MoeShape(tokens).request(arch="gfx950", dtype="fp8"))


class TestDefaultTokenSet(unittest.TestCase):
    def test_default_tokens_cover_both_routes_and_both_tile_m_bands(self):
        """The point of the harness: if this set stops spanning the routes, the
        benchmark stops being able to see a routing regression."""
        launches = {t: len(_plan(t)) for t in DEFAULT_TOKENS}
        self.assertIn(1, set(launches.values()), launches)
        self.assertIn(2, set(launches.values()), launches)
        tile_ms = {_plan(t)[0].spec.tile_m for t in DEFAULT_TOKENS}
        self.assertEqual(tile_ms, {16, 32})

    def test_band_interior_selects_one_kernel_for_both_token_counts(self):
        """T=32 and T=64 sit in the same band, so they must select the same
        kernels and share a compile."""
        a, b = _plan(32), _plan(64)
        self.assertEqual([r.candidate.name for r in a], [r.candidate.name for r in b])
        self.assertEqual(
            [r.kernel_id.compile_key for r in a],
            [r.kernel_id.compile_key for r in b],
        )

    def test_split_stages_agree_on_tile_m(self):
        for tokens in DEFAULT_TOKENS:
            plan = _plan(tokens)
            if len(plan) == 2:
                with self.subTest(tokens=tokens):
                    self.assertEqual(plan[0].spec.tile_m, plan[1].spec.tile_m)


class TestGrids(unittest.TestCase):
    def test_block_axis_comes_from_the_routing_and_the_rest_from_dispatch(self):
        shape = MoeShape(64)
        plan = _plan(64)
        layout = TokenLayout(Routing(shape, seed=7), shape, plan[0].spec.tile_m)
        grids = plan_stage_grids(plan, layout)
        self.assertEqual(len(grids), len(plan))
        for grid, result in zip(grids, plan):
            self.assertTrue(all(x > 0 for x in grid), grid)
            self.assertEqual(grid[1], layout.num_m_blocks)
            self.assertEqual((grid[0], grid[2]), (result.grid[0], result.grid[2]))

    def test_a_routing_over_the_dispatched_bound_is_reported(self):
        """The warning exists to catch the bound regressing into an estimate,
        so it is checked by forcing the condition rather than by finding a
        routing that triggers it -- no routing should."""
        shape = MoeShape(64)
        plan = _plan(64)
        layout = TokenLayout(Routing(shape, seed=7), shape, plan[0].spec.tile_m)
        self.assertEqual(grid_bound_warnings(plan, layout), ())
        layout.num_m_blocks = max(_dispatched_block_count(r) for r in plan) + 1
        warnings = grid_bound_warnings(plan, layout)
        self.assertEqual(len(warnings), len(plan))
        self.assertIn("dispatched grid bound", warnings[0])

    def test_uneven_routing_at_512_tokens_stays_under_the_bound(self):
        """The T=512 band is where the average-vs-bound gap used to bite: the
        even split predicted 128 blocks for a routing that needs 189."""
        shape = MoeShape(512)
        plan = _plan(512)
        layout = TokenLayout(Routing(shape, seed=11939), shape, plan[0].spec.tile_m)
        self.assertGreater(layout.num_m_blocks, 128)
        for result in plan:
            with self.subTest(candidate=result.candidate.name):
                self.assertGreaterEqual(
                    _dispatched_block_count(result), layout.num_m_blocks
                )
        self.assertEqual(grid_bound_warnings(plan, layout), ())


class TestLayout(unittest.TestCase):
    def test_every_routed_slot_gets_a_padded_row(self):
        shape = MoeShape(32)
        routing = Routing(shape, seed=3)
        layout = TokenLayout(routing, shape, 16)
        self.assertTrue((layout.slot_row >= 0).all())
        rows = layout.slot_row.reshape(-1)
        self.assertEqual(len(set(rows.tolist())), rows.size)
        for t in range(shape.num_tokens):
            for slot in range(shape.top_k):
                row = int(layout.slot_row[t, slot])
                expert = int(routing.topk_ids[t, slot])
                self.assertEqual(int(layout.sorted_token_ids[row]), t)
                self.assertEqual(
                    int(layout.block_expert_ids[row // layout.tile_m]), expert
                )


class TestOracleSampling(unittest.TestCase):
    """The sampled check must be the same arithmetic as the full one, or a
    large-T verdict would not mean what the small-T verdict means."""

    def test_sampled_oracle_matches_the_full_oracle(self):
        shape = MoeShape(8, hidden=128, intermediate=128, num_experts=4, top_k=2)
        weights = ExpertWeights(shape, seed=5)
        layout = TokenLayout(Routing(shape, seed=5), shape, 16)
        full = oracle_tokens(weights, layout, np.arange(shape.num_tokens))
        picked = np.array([1, 5])
        sampled = oracle_tokens(weights, layout, picked)
        np.testing.assert_allclose(sampled, full[picked], rtol=0, atol=0)


class TestCheckSelection(unittest.TestCase):
    def test_auto_uses_every_token_up_to_the_oracle_limit(self):
        tokens, label = check_tokens(
            MoeShape(ORACLE_MAX_TOKENS), mode="auto", sample=8, seed=1
        )
        self.assertEqual(len(tokens), ORACLE_MAX_TOKENS)
        self.assertEqual(label, "oracle/all")

    def test_auto_samples_above_the_limit(self):
        tokens, label = check_tokens(MoeShape(512), mode="auto", sample=8, seed=1)
        self.assertEqual(len(tokens), 8)
        self.assertEqual(label, "oracle/sample(8)")
        self.assertTrue(set(tokens.tolist()) <= set(range(512)))

    def test_none_skips_the_oracle(self):
        tokens, label = check_tokens(MoeShape(512), mode="none", sample=8, seed=1)
        self.assertEqual(len(tokens), 0)
        self.assertEqual(label, "none")


class TestJsonRecord(unittest.TestCase):
    def test_schema_and_shape_records(self):
        shape = MoeShape(1)
        record = ShapeRecord(shape=shape, route="fused", launches=1, verdict="pass")
        out = Path(tempfile.mkdtemp(prefix="rocke_moe_bench_")) / "moe.json"
        write_json(out, config={"arch": "gfx950"}, records=[record])
        doc = json.loads(out.read_text())
        self.assertEqual(
            doc["schema"], "ck.dsl.benchmark.moe.fused_mega_fp8_dispatch/v1"
        )
        self.assertEqual(doc["shapes"][0]["shape"]["num_tokens"], 1)
        self.assertEqual(doc["shapes"][0]["route"], "fused")


if __name__ == "__main__":
    unittest.main()
