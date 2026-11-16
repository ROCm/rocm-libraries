# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests: every gfx1250 TDM K-loop covers a partial last K tile.

K is a kernel argument, so a K that is not a multiple of tile_k only shows up at
run time: a loop that counts its tiles rounding down skips the partial last tile
and silently leaves that K tail out of C (the early-refill loop once did). TDM
clips the partial tile to K, so a loop only has to reach it.

Each test lowers one K-loop and evaluates its emitted i32 arithmetic for a range
of K: the K loop's bounds and step (``%k0``), plus the tiles each loop handles
outside it. The loop must process ceil(K / tile_k) tiles; the early-refill loop
pads the count to an even number with one zero-filled tile past K. The ring also
clamps its prologue fills to the last tile origin, which must be the start of the
partial tile, and never before it: with K = 0 (no K tile) the fills start at
the first K element with nothing to move, not one tile before A and B.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

# pytest.ini uses --import-mode=importlib, which does not put this directory on
# sys.path: without the insert the sibling import below only resolves when an
# earlier-collected module happened to add it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_gemm_split_barrier import _lower, _ring_spec, _spec  # noqa: E402

from rocke.tests.llvm_ir import KernelIR  # noqa: E402


class _Kernel(KernelIR):
    """One lowered kernel, read through :class:`KernelIR`, plus its K loop:
    the i32 arithmetic that bounds and steps ``%k0`` evaluated for a given K."""

    _EVAL = {
        "add": lambda a, c: a + c,
        "sub": lambda a, c: a - c,
        "mul": lambda a, c: a * c,
        "llvm.smin": min,
        "llvm.smax": max,
    }

    def __init__(self, ll: str):
        super().__init__(ll)
        assert "%k0" in self.definitions, "no %k0 K loop"
        self.start, iv_next = self.incoming("%k0")
        assert self.opcode(iv_next) == "add", self.definitions[iv_next]
        iv, self.step = self.operands(iv_next)
        assert iv == "%k0", self.definitions[iv_next]
        # Selected by their text, not KernelIR.find: find asks every definition
        # for its opcode, and the kernel's inline-asm calls have no callee.
        (bound,) = [
            self.operands(n)[1]
            for n, d in self.definitions.items()
            if d.startswith("icmp slt i32 %k0,")
        ]
        self.bound = bound
        self.smins = [
            n
            for n, d in self.definitions.items()
            if d.startswith("call i32 @llvm.smin.i32(")
        ]

    def value(self, x: str, K: int) -> int:
        """The i32 value of ``x`` when the kernel runs with this K."""
        if x == "%K":
            return K
        if re.fullmatch(r"-?\d+", x):
            return int(x)
        op = self.opcode(x)
        a, c = (self.value(v, K) for v in self.operands(x))
        if op in ("sdiv", "srem"):
            q = abs(a) // abs(c) * (1 if (a >= 0) == (c >= 0) else -1)
            return q if op == "sdiv" else a - q * c
        return self._EVAL[op](a, c)

    def trips(self, K: int) -> int:
        lo, hi, step = (self.value(x, K) for x in (self.start, self.bound, self.step))
        return max(0, -(-(hi - lo) // step))


def _ks(tile_k):
    """K from one to five tiles, aligned and with every kind of remainder."""
    for tiles in (1, 2, 3, 4, 5):
        for r in (0, 1, tile_k // 2, tile_k - 1):
            K = tiles * tile_k + r
            yield K, -(-K // tile_k)


class TestTdmKLoopsCoverAPartialLastTile(unittest.TestCase):
    def _check(self, spec, tiles_of, *, pad_even=False):
        k = _Kernel(_lower(spec))
        tile_k = spec.tile.tile_k
        for K, need in _ks(tile_k):
            with self.subTest(K=K):
                got = tiles_of(k, K)
                want = need + need % 2 if pad_even else need
                self.assertEqual(got, want, f"K={K}: {got} tiles, need {want}")
        return k

    def test_ping_pong(self):
        """Depth 2: the loop computes tile i and loads tile i + 1; one more after."""
        for label, kw in (
            ("joint barrier", {}),
            ("split barrier", dict(split_barrier=True)),
            (
                "alternating issuers",
                dict(split_barrier=True, tdm_alternate_issuers=True),
            ),
        ):
            with self.subTest(label):
                self._check(_ring_spec(2, **kw), lambda k, K: k.trips(K) + 1)

    def test_ring(self):
        """Depth 3+: one tile per iteration; fills clamp to the last tile origin."""
        for depth in (3, 4):
            with self.subTest(depth=depth):
                spec = _ring_spec(depth)
                k = self._check(spec, lambda k, K: k.trips(K))
                tile_k = spec.tile.tile_k
                last = {k.operands(n)[1] for n in k.smins}
                self.assertEqual(len(last), 1, "fills clamp to one last tile origin")
                (last_origin,) = last
                for K, _ in _ks(tile_k):
                    with self.subTest(depth=depth, K=K):
                        o = k.value(last_origin, K)
                        self.assertLess(o, K, "the last fill starts past K")
                        self.assertGreaterEqual(
                            o + tile_k, K, "the last fill stops short of K"
                        )
                with self.subTest(depth=depth, K=0):
                    self.assertEqual(k.trips(0), 0)
                    self.assertEqual(
                        k.value(last_origin, 0), 0, "with K = 0 a fill starts before K"
                    )

    def test_early_refill(self):
        """Two tiles per iteration plus two after, over an even tile count."""
        spec = _spec(
            split_barrier=True, tdm_alternate_issuers=True, tdm_early_refill=True
        )
        self._check(spec, lambda k, K: 2 * k.trips(K) + 2, pad_even=True)


class TestTdmTileOriginIsI64(unittest.TestCase):
    def test_every_fill_address_is_an_i64_offset(self):
        """batch + row * K + k overflows i32 once an operand passes 2^31
        elements, and the GEP would sign-extend it to an address before the
        tensor; the offset must be computed, and indexed, in i64."""
        ll = _lower(_ring_spec(2))
        geps = [
            m.group(1)
            for m in re.finditer(
                r"%gaddr_gep\S* = getelementptr \S+, ptr addrspace\(1\) \S+, (\S+) ", ll
            )
        ]
        self.assertTrue(geps, "no TDM fill address")
        self.assertEqual(set(geps), {"i64"})

    def test_a_batched_fill_offset_is_i64_from_the_start(self):
        """batch * stride must be formed in i64 and be what the fills add, not
        an i32 product widened after it may already have wrapped."""
        from dataclasses import replace

        ll = _lower(replace(_ring_spec(2), batched=True))
        ir = KernelIR(ll)
        fills = re.findall(
            r"getelementptr \S+, ptr addrspace\(1\) %(\w+), i64 (%[\w.]+)", ll
        )
        self.assertTrue(fills, "no TDM fill address")
        seen = set()
        for operand, index in fills:
            with self.subTest(operand=operand, index=index):
                self.assertEqual(
                    ir.opcode(index), "add", "the fill offset is batch + row/k"
                )
                batch = ir.operands(index)[0]
                self.assertEqual(
                    ir.opcode(batch), "mul", "the batch term is an i64 product"
                )
                self.assertTrue(
                    ir.definitions[batch].startswith("mul nsw i64 ")
                    or ir.definitions[batch].startswith("mul i64 "),
                    ir.definitions[batch],
                )
                factors = ir.operands(batch)
                widened = {
                    ir.operands(f)[0]
                    for f in factors
                    if f in ir.definitions and ir.opcode(f) == "zext"
                }
                stride = f"%stride_{operand.lower()}"
                self.assertIn(stride, widened, "the stride is widened before *")
                seen.add(operand)
        self.assertEqual(seen, {"A", "B"})


class TestTdmKLoopIgnoresTheUnrollExperiment(unittest.TestCase):
    def test_unroll_env_does_not_replace_the_tdm_loop(self):
        """``ROCKE_EXP_UNROLL_K`` unrolls a loop with no TENSORcnt waits; a TDM
        kernel must keep its own loop (and the name that says so)."""
        from unittest import mock

        spec = _ring_spec(2, split_barrier=True)
        with mock.patch.dict(os.environ, {"ROCKE_EXP_UNROLL_K": "1024"}):
            ll = _lower(spec)
        self.assertIn("s.wait.tensorcnt", ll)
        self.assertIn("%k0 = phi", ll)


if __name__ == "__main__":
    unittest.main()
