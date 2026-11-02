# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for ``TraitSpec.tdm_early_refill`` on the gfx1250 TDM ping-pong.

The trait refills an LDS half as soon as every wave has finished reading it: the
pair owning the half waits on the split barrier it just signalled and issues the
fill of the tile two ahead. Two failure modes compile cleanly:

* **A hang.** Every wave must wait exactly once per barrier phase. The owning
  pair waits before its refill, every other wave at the top of the next tile;
  a missing or doubled wait parks the workgroup. Hence the narrow gate.
* **A silent race.** The refill overwrites the half just read. It must come
  after a barrier wait (all waves have drained their reads) and no LDS read may
  sit between the signal and that refill.

These are IR-shape assertions on the lowered kernel; they need no GPU.
"""

from __future__ import annotations

import os
import sys
import re
import unittest

# pytest.ini uses --import-mode=importlib, which does not put this directory on
# sys.path: without the insert the sibling import below only resolves when an
# earlier-collected module happened to add it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_gemm_split_barrier import (
    _LDS_LOAD,
    _SIGNAL,
    _WAIT,
    _loop_body,
    _lower,
    _spec,
)  # noqa: E402

ARCH = "gfx1250"
_TDM = "tensor.load.to.lds"
_ON = dict(split_barrier=True, tdm_alternate_issuers=True)


def _calls(lines):
    """Instruction lines only (labels and blank lines dropped)."""
    return [l for l in lines if l.strip() and not re.match(r"[\w.]+:", l)]


class TestGemmTdmEarlyRefillValidity(unittest.TestCase):
    def _valid(self, spec):
        from rocke.instances.common.gemm_universal import is_valid_spec

        return is_valid_spec(spec, arch=ARCH)

    def test_admitted_with_split_barrier_and_alternating_issuers(self):
        ok, why = self._valid(_spec(tdm_early_refill=True, **_ON))
        self.assertTrue(ok, why)

    def test_rejected_without_its_prerequisites(self):
        for label, kw in (
            ("no split barrier", dict(tdm_alternate_issuers=True)),
            ("no alternating issuers", dict(split_barrier=True)),
            ("neither", {}),
        ):
            with self.subTest(label):
                ok, why = self._valid(_spec(tdm_early_refill=True, **kw))
                self.assertFalse(ok, f"{label} admitted")
                self.assertIn("tdm_early_refill", why)


class TestGemmTdmEarlyRefillEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = _lower(_spec(**_ON))
        cls.on = _lower(_spec(tdm_early_refill=True, **_ON))
        cls.body = _calls(_loop_body(cls.on))

    def test_default_is_unchanged(self):
        self.assertEqual(self.off, _lower(_spec(tdm_early_refill=False, **_ON)))
        self.assertNotEqual(self.off, self.on)

    def test_loop_runs_two_tiles_per_iteration(self):
        """Two signals per iteration, one per K-tile, and two refills each."""
        self.assertEqual(sum(_SIGNAL in l for l in self.body), 2)
        self.assertEqual(sum(_TDM in l for l in self.body), 4)

    def test_every_refill_follows_a_barrier_wait(self):
        """The refill overwrites the half just read: only after the barrier."""
        for i, l in enumerate(self.body):
            if _TDM in l:
                self.assertIn(_WAIT, self.body[i - 1], "refill without a wait first")

    def test_no_lds_read_between_signal_and_refill(self):
        sigs = [i for i, l in enumerate(self.body) if _SIGNAL in l]
        tdms = [i for i, l in enumerate(self.body) if _TDM in l]
        for s in sigs:
            nxt = next(t for t in tdms if t > s)
            self.assertFalse(
                any(_LDS_LOAD.search(l) for l in self.body[s + 1 : nxt]),
                "LDS read between the signal and the refill of that half races it",
            )

    def test_refill_waits_come_after_the_descriptor_arithmetic(self):
        """The wait sits directly before the TDM issue, not ahead of its setup."""
        for i, l in enumerate(self.body):
            if _TDM in l:
                self.assertIn(_WAIT, self.body[i - 1])
                self.assertNotIn(_WAIT, self.body[i - 2])

    def test_tile_count_covers_a_partial_last_tile(self):
        """The loop covers all of K, rounding the tile count up before making it even.

        The tile count is evaluated from the emitted i32 arithmetic for each K:
        it feeds the ``srem ..., 2`` that pads it to an even count. Rounding it
        down dropped the partial last tile when K = 2m * tile_k + r, 0 < r < tile_k.
        """
        tile_k = _spec().tile.tile_k
        defs = {}
        for l in self.on.splitlines():
            m = re.match(
                r"\s*(%[\w.]+) = (add|sub|mul|sdiv|srem)(?: nsw)? i32 (\S+), (\S+)$", l
            )
            if m:
                defs[m.group(1)] = (m.group(2), m.group(3).rstrip(","), m.group(4))
        evens = [n for n, (op, _, rhs) in defs.items() if op == "srem" and rhs == "2"]
        self.assertEqual(len(evens), 1, "expected one srem-by-2 (the even-count pad)")
        n_tiles = defs[evens[0]][1]

        def value(x, K):
            if x == "%K":
                return K
            if re.fullmatch(r"-?\d+", x):
                return int(x)
            op, a, c = defs[x]
            a, c = value(a, K), value(c, K)
            if op in ("sdiv", "srem"):
                q = abs(a) // abs(c) * (1 if (a >= 0) == (c >= 0) else -1)
                return q if op == "sdiv" else a - q * c
            return {"add": a + c, "sub": a - c, "mul": a * c}[op]

        for tiles in (1, 2, 3, 4, 5):
            for r in (0, 1, tile_k // 2, tile_k - 1):
                K = tiles * tile_k + r
                with self.subTest(K=K):
                    n = value(n_tiles, K)
                    n_even = n + n % 2
                    self.assertGreaterEqual(
                        n * tile_k, K, "the last partial tile is skipped"
                    )
                    self.assertLess(
                        (n - 1) * tile_k, K, "a whole tile past K is counted"
                    )
                    self.assertGreaterEqual(n_even * tile_k, K)


if __name__ == "__main__":
    unittest.main()
