# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for ``TraitSpec.tdm_alternate_issuers`` on the gfx1250 TDM ping-pong.

The trait moves the TDM issue of each LDS half to its own pair of waves
(waves ``2p`` / ``2p + 1`` fill half ``p``). It fails silently in two ways:

* **A missing or doubled fill.** If the issuer predicate does not track the
  half being filled, a tile is issued by no wave (stale LDS) or by two (a
  wasted copy). Hence: in the loop, the issuer compare is fed by the parity of
  the half being refilled, and the prologue still fills half 0 from waves 0/1.
* **An inexact wait.** The TENSORcnt waits assume each issuer has at most one
  fill outstanding, which holds only in the depth-2 ping-pong. Hence the
  narrow gate.

These are IR-shape assertions on the lowered kernel; they need no GPU.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

from test_gemm_split_barrier import _loop_body, _lower, _spec

ARCH = "gfx1250"
_TDM = "call void @llvm.amdgcn.tensor.load.to.lds("
_ISSUER_CMP = re.compile(r"%(\w+) = icmp eq i32 %\w+, (%\w+|\d+)$")


def _defs(ll: str) -> dict:
    """SSA name -> its defining instruction text, for every ``%x = ...`` line."""
    out = {}
    for line in ll.splitlines():
        m = re.match(r"\s*%([\w.]+) = (.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _is_wave_one(op: str, defs: dict) -> bool:
    """Wave 1 appears as a literal ``1`` or as the folded ``add 0, 1``."""
    return op == "1" or defs.get(op.lstrip("%"), "").endswith("add nsw i32 0, 1")


def _issuer_operands(lines) -> list:
    """The right-hand operand of each icmp that guards a TDM issue, in order."""
    ops, pending = [], None
    for line in lines:
        m = _ISSUER_CMP.search(line.strip())
        if m:
            pending = m.group(2)
        elif _TDM in line and pending is not None:
            ops.append(pending)
            pending = None
    return ops


class TestGemmTdmAlternateIssuersValidity(unittest.TestCase):
    def _valid(self, spec):
        from rocke.instances.common.gemm_universal import is_valid_spec

        return is_valid_spec(spec, arch=ARCH)

    def test_admitted_on_the_depth2_pingpong(self):
        for split in (False, True):
            with self.subTest(split_barrier=split):
                ok, why = self._valid(
                    _spec(tdm_alternate_issuers=True, split_barrier=split)
                )
                self.assertTrue(ok, why)

    def test_rejected_outside_the_pingpong(self):
        for label, kw in (
            ("depth 1", dict(tdm_depth=1)),
            ("depth 3", dict(tdm_depth=3, lds_k_pad=16)),
            ("no tdm", dict(tdm=False, tdm_depth=1)),
        ):
            with self.subTest(label):
                spec = _spec(tdm_alternate_issuers=True, **kw)
                if label == "depth 3":
                    spec = replace(spec, tile=replace(spec.tile, tile_k=128))
                ok, why = self._valid(spec)
                self.assertFalse(ok, f"{label} admitted")
                self.assertIn("tdm_alternate_issuers", why)

    def test_rejected_below_four_waves(self):
        spec = _spec(tdm_alternate_issuers=True)
        spec = replace(
            spec,
            tile=replace(spec.tile, tile_m=64, tile_n=64, warp_m=1, warp_n=2),
            block_size=2 * spec.wave_size,
        )
        ok, why = self._valid(spec)
        self.assertFalse(ok)
        self.assertIn("tdm_alternate_issuers", why)


class TestGemmTdmAlternateIssuersEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = _lower(_spec(split_barrier=True))
        cls.on = _lower(_spec(split_barrier=True, tdm_alternate_issuers=True))

    def test_default_is_unchanged(self):
        self.assertEqual(
            self.off, _lower(_spec(split_barrier=True, tdm_alternate_issuers=False))
        )
        self.assertNotEqual(self.off, self.on)

    def test_default_issuers_are_waves_0_and_1(self):
        body = _loop_body(self.off)
        defs = _defs(self.off)
        ops = _issuer_operands(body)
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0], "0")
        self.assertTrue(_is_wave_one(ops[1], defs), ops[1])

    def test_loop_issuers_follow_the_refilled_half(self):
        """A = 2 * (1 - par), B = A + 1: the half being refilled, not the one read."""
        body = _loop_body(self.on)
        defs = _defs(self.on)
        ops = _issuer_operands(body)
        self.assertEqual(len(ops), 2, "one A and one B issue per iteration")
        a_def = defs[ops[0].lstrip("%")]
        m = re.match(r"mul nsw i32 %([\w.]+), 2$", a_def)
        self.assertIsNotNone(m, f"A issuer is not 2 * parity: {a_def}")
        self.assertRegex(defs[m.group(1)], r"sub nsw i32 1, %par$")
        self.assertEqual(
            defs[ops[1].lstrip("%")], f"add nsw i32 {ops[0]}, 1", "B issuer is not A + 1"
        )

    def test_prologue_fills_half_zero_from_waves_0_and_1(self):
        lines = self.on.splitlines()
        start = next(i for i, l in enumerate(lines) if re.match(r"for\.body\.\d+:", l))
        defs = _defs(self.on)
        ops = _issuer_operands(lines[:start])
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0], "0")
        self.assertTrue(_is_wave_one(ops[1], defs), ops[1])


if __name__ == "__main__":
    unittest.main()
