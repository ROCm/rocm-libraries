# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The search engine over attention's real tuning spaces.

Family neutrality on production axes: a candidate's ``search_space`` gives
a :class:`~rocke.dispatch.tuning.search.KnobSearchSpace` whose keys are the
candidate's own ``config_key``\\ s, every production spec is reachable from
its knobs, and a searched winner pins back through dispatch unchanged.
Compile-free: the "backend" scores tuning ids with a fixed hash.
"""

from __future__ import annotations

import hashlib
import random
import unittest

from dispatch.attention import ATTENTION_EXECUTION_REGISTRY, AttentionRequest
from rocke.dispatch.core import pin_to_spec
from rocke.dispatch.tuning import sweep_level
from rocke.dispatch.tuning.search import (
    Budget,
    CompileOutcome,
    KnobSearchSpace,
    SpecBackend,
    TuningDB,
    tune_candidate,
)

CANDIDATES = (
    "attention_gfx950_dense_grid",
    "attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm",
)


def _req(**kw) -> AttentionRequest:
    base = {
        "batch": 1,
        "nhead_q": 32,
        "nhead_k": 8,
        "seqlen_q": 1024,
        "seqlen_k": 1024,
        "hdim_q": 128,
        "hdim_v": 128,
        "arch": "gfx950",
        "dtype": "bf16",
        "mask_type": 1,
    }
    base.update(kw)
    return AttentionRequest(**base)


def _fake_time(spec) -> float:
    h = hashlib.sha256(spec.tuning_id.encode()).hexdigest()
    return 1.0 + int(h[:6], 16) / 0xFFFFFF


def _backend(space):
    return SpecBackend(
        space,
        compile=lambda spec: CompileOutcome(True, artifact=spec),
        check=lambda spec, art: (True, "ok"),
        measure=lambda spec, art, fid: [_fake_time(art)] * fid.reps,
    )


class TestAttentionSearch(unittest.TestCase):
    def _candidate(self, name):
        return ATTENTION_EXECUTION_REGISTRY.get(name)

    def test_keys_are_the_candidates_config_keys(self):
        req = _req()
        for name in CANDIDATES:
            cand = self._candidate(name)
            with self.subTest(candidate=name):
                ks, base = cand.search_space(req)
                space = KnobSearchSpace(ks, base)
                self.assertGreater(space.dims, 1)
                genomes = space.stratified(random.Random(0), 24)
                self.assertGreater(len(genomes), 4)
                for g in genomes:
                    key = space.key(g)
                    spec = space.spec(key)
                    self.assertEqual(spec.config_key, key)
                    pinned = pin_to_spec(req, cand, spec)
                    self.assertEqual(cand.select_spec(pinned).tuning_id, spec.tuning_id)

    def test_every_production_spec_is_reachable(self):
        req = _req()
        for name in CANDIDATES:
            cand = self._candidate(name)
            with self.subTest(candidate=name):
                ks, base = cand.search_space(req)
                space = KnobSearchSpace(ks, base)
                with sweep_level("production"):
                    specs = list(cand.sweep_space(pin_to_spec(req, cand, None)))
                self.assertGreater(len(specs), 1)
                for spec in specs:
                    g = space.encode(dict(spec.knobs))
                    self.assertEqual(space.key(g), spec.config_key, dict(spec.knobs))

    def test_choices_axes_have_one_level_per_config(self):
        """A ``choices`` axis lists the base value; its level must merge
        with ``()`` (it used to sit next to it as a duplicate, so a climber
        could never step up from the base), and numeric ones stay ordinal."""
        req = _req()
        cand = self._candidate(CANDIDATES[0])
        ks, base = cand.search_space(req)
        space = KnobSearchSpace(ks, base)
        g0 = space.encode({})
        k0 = space.key(g0)
        for i, p in enumerate(space.params):
            with self.subTest(param=p.name):
                keys = []
                for j in range(p.size):
                    g = g0[:i] + (j,) + g0[i + 1 :]
                    k = space.key(g)
                    if k is not None:
                        keys.append(k)
                self.assertEqual(len(keys), len(set(keys)), p.name)
        self.assertNotIn(
            k0, [space.key(m) for m in space.moves(g0, ordinal_radius=None)]
        )
        by = {p.name: p for p in space.params}
        self.assertEqual(by["block_n"].kind, "ordinal")

    def test_searched_winner_pins_back(self):
        req = _req()
        cand = self._candidate(CANDIDATES[0])
        db = TuningDB()
        res, rec = tune_candidate(
            cand,
            req,
            _backend,
            db=db,
            strategy="portfolio",
            budget=Budget(evaluations=24),
            confirm=0,
        )
        self.assertIsNotNone(rec)
        self.assertLessEqual(res.evaluations, 24)
        self.assertEqual(res.observations[0].trial.origin, "warm:dispatcher")
        spec = cand.select_spec(db.pin(req, rec))
        self.assertEqual(spec.tuning_id, rec.tuning_id)
        self.assertAlmostEqual(_fake_time(spec), rec.value)


if __name__ == "__main__":
    unittest.main()
