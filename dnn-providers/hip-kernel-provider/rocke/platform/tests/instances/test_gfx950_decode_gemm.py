# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the gfx950 decode GEMM builder (``instances/gfx950/decode_gemm.py``).

They pin the emitted-IR contract on CPU: every spec the validity check accepts
builds and lowers with the Python engine, the IR carries the instructions each
family is built from (16x16x32 or 4x4x4 bf16 MFMAs, cache-policy weight loads,
the split-K atomics and fences), the kernel name is a stable, unique function of
the spec, and every single knob move from each problem's default is either valid
and builds or is refused with a reason. A numpy model of the smallm work split
checks that the workgroups, waves and lanes cover every (row, K) element of W
exactly once and reproduce ``X @ W.T``. On-GPU numerics are outside this file.

Run:  PYTHONPATH=python python3 -m pytest tests/instances/test_gfx950_decode_gemm.py
"""

from __future__ import annotations

import os
import re
import unittest
from dataclasses import MISSING, fields, replace

import numpy as np
from rocke.instances.gfx950.decode_gemm import (
    DecodeGemmSpec,
    build_decode_gemm,
    decode_gemm_launch,
    decode_gemm_signature,
    default_spec,
    gemv_plan,
    is_valid_spec,
    knob_moves,
    skinny_plan,
    smallm_plan,
)

# Decode-shaped problems (M, N, K): small M against a wide or a deep weight.
PROBLEMS = [(m, 2304, 8192) for m in (1, 16, 17, 32)] + [
    (m, 8192, 1024) for m in (1, 16, 17, 32)
]
_MFMA16 = "llvm.amdgcn.mfma.f32.16x16x32.bf16("
_MFMA4 = "llvm.amdgcn.mfma.f32.4x4x4bf16.1k("
_BLOAD = "call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32("
_SCHED = "llvm.amdgcn.sched.barrier("


def _lower(kernel) -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    flavor = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22")
    return _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)


def _calls(ll: str, needle: str) -> int:
    return sum(
        1
        for line in ll.splitlines()
        if needle in line and not line.lstrip().startswith("declare")
    )


class TestValidity(unittest.TestCase):
    def test_defaults_are_valid(self):
        for m, n, k in PROBLEMS:
            spec = default_spec(m, n, k)
            ok, why = is_valid_spec(spec)
            self.assertTrue(ok, f"{spec}: {why}")

    def test_default_grids_fill_whole_waves_of_workgroups(self):
        for m, n, k in PROBLEMS:
            launch = decode_gemm_launch(default_spec(m, n, k))
            self.assertEqual(launch["grid"][0] % 256, 0, (m, n, k, launch))

    def test_refusals_carry_a_reason(self):
        bad = [
            DecodeGemmSpec(16, 2304, 8192),  # X of all of K does not fit LDS
            DecodeGemmSpec(16, 2300, 8192),  # N not a multiple of 16
            DecodeGemmSpec(80, 2304, 8192, x_stage="global"),  # more than four X groups
            DecodeGemmSpec(16, 2304, 8192, split_k=16, epi="direct"),
            DecodeGemmSpec(
                16, 2304, 8192, n_waves=9, k_waves=2, split_k=16
            ),  # 18 waves
            DecodeGemmSpec(16, 2304, 8192, split_k=3),
            DecodeGemmSpec(16, 2304, 8192, family="gemv"),  # gemv keeps M <= 8
            DecodeGemmSpec(1, 2304, 8192, family="gemv", w_cache=4),
            DecodeGemmSpec(1, 2304, 8000, family="gemv"),  # K not a multiple of 512
        ]
        for spec in bad:
            ok, why = is_valid_spec(spec)
            self.assertFalse(ok, spec)
            self.assertTrue(why and why != "ok", spec)
            with self.assertRaises(ValueError):
                build_decode_gemm(spec)

    def test_other_arch_refused(self):
        self.assertFalse(is_valid_spec(default_spec(1, 2304, 8192), arch="gfx942")[0])

    def test_only_the_problem_is_required(self):
        required = [
            f.name
            for f in fields(DecodeGemmSpec)
            if f.default is MISSING and f.default_factory is MISSING
        ]
        self.assertEqual(required, ["M", "N", "K"])


class TestKnobMoves(unittest.TestCase):
    def test_every_single_move_from_each_default_builds_or_is_refused(self):
        for m, n, k in PROBLEMS:
            base = default_spec(m, n, k)
            moves = knob_moves(base)
            self.assertGreater(len(moves), 10)
            n_valid = 0
            for spec in moves:
                ok, why = is_valid_spec(spec)
                if not ok:
                    self.assertTrue(why, spec)
                    continue
                n_valid += 1
                ll = _lower(build_decode_gemm(spec))
                self.assertIn("define amdgpu_kernel", ll, spec.kernel_name())
            self.assertGreater(n_valid, len(moves) // 3, (m, n, k))

    def test_moves_reach_every_relevant_knob(self):
        base = default_spec(16, 2304, 8192)
        moved = {
            f.name
            for s in knob_moves(base)
            for f in fields(s)
            if getattr(s, f.name) != getattr(base, f.name)
        }
        self.assertEqual(moved, set(base.relevant_fields()[3:]))


class TestNames(unittest.TestCase):
    def test_name_is_stable_and_unique(self):
        seen = {}
        for m, n, k in PROBLEMS:
            base = default_spec(m, n, k)
            for spec in [base] + [s for s in knob_moves(base) if is_valid_spec(s)[0]]:
                name = spec.kernel_name()
                self.assertEqual(name, replace(spec).kernel_name())
                self.assertRegex(name, r"^[A-Za-z0-9_]+$")
                if name in seen:
                    self.assertEqual(seen[name], spec, name)
                seen[name] = spec
        self.assertEqual(DecodeGemmSpec(16, 2304, 8192, n_waves=9, split_k=16).kernel_name(),
                         "dg_smallm_m16n2304k8192_nw9_sk16")  # fmt: skip

    def test_knobs_of_the_other_family_are_ignored(self):
        a = DecodeGemmSpec(1, 2304, 8192, family="gemv", waves=9)
        b = replace(a, n_waves=4, split_k=8)
        self.assertEqual(a.kernel_name(), b.kernel_name())
        self.assertEqual(_lower(build_decode_gemm(a)), _lower(build_decode_gemm(b)))


class TestIR(unittest.TestCase):
    def test_smallm_ir(self):
        spec = DecodeGemmSpec(32, 2304, 8192, n_waves=9, split_k=16)
        ll = _lower(build_decode_gemm(spec))
        p = smallm_plan(spec)
        self.assertEqual(_calls(ll, _MFMA16), p["S"] * spec.row_groups * p["MG"])
        finisher_reads = -(
            -spec.M * p["block_n"] // 4 // p["threads"]
        )  # 4 floats per thread per pass
        self.assertEqual(_calls(ll, _BLOAD), p["S"] * spec.row_groups + finisher_reads)
        self.assertEqual(
            len(re.findall(r"global\.load\.lds\(", ll)) - 1, -(-32 // p["waves"])
        )
        self.assertIn("atomicrmw fadd", ll)
        self.assertIn('fence syncscope("agent") release', ll)
        self.assertIn('fence syncscope("agent") acquire', ll)
        self.assertEqual(decode_gemm_launch(spec)["grid"], (256, 1, 1))
        self.assertEqual(decode_gemm_launch(spec)["block"], (576, 1, 1))

    def test_weight_loads_carry_the_cache_bits(self):
        for bits in (0, 2, 3, 18):
            spec = DecodeGemmSpec(16, 8192, 1024, n_waves=2, w_cache=bits)
            ll = _lower(build_decode_gemm(spec))
            loads = [ln for ln in ll.splitlines() if _BLOAD in ln]
            self.assertTrue(loads)
            self.assertTrue(
                all(ln.rstrip().endswith(f"i32 {bits})") for ln in loads), bits
            )

    def test_split_k_reduction_modes(self):
        base = DecodeGemmSpec(16, 2304, 8192, n_waves=9, split_k=16)
        atomic_fence = _lower(build_decode_gemm(base))
        slots = _lower(build_decode_gemm(replace(base, reduce="slots")))
        relaxed = _lower(build_decode_gemm(replace(base, sk_sync="relaxed")))
        self.assertIn("atomicrmw fadd", atomic_fence)
        self.assertNotIn("atomicrmw fadd", slots)  # plain partial stores, one ticket
        self.assertEqual(slots.count("atomicrmw add"), 1)
        self.assertNotIn("fence", relaxed)  # vmcnt waits and sc1 partial loads instead
        sc1_loads = [
            ln
            for ln in relaxed.splitlines()
            if "buffer.load" in ln and ln.rstrip().endswith("i32 16)")
        ]
        self.assertTrue(sc1_loads)
        self.assertEqual(
            decode_gemm_launch(replace(base, reduce="slots"))["ws_bytes"],
            16 * 16 * 2304 * 4,
        )
        # xcdrr (the reference that trusts round-robin dispatch): partners tile +
        # ks * tiles (16 tiles: one XCD), an agent-scope returning ticket (sc0, no
        # sc1), plain partial stores and loads, no fence.
        trusting = replace(base, sk_sync="xcdrr", wg_order="tile")
        self.assertTrue(is_valid_spec(trusting)[0], is_valid_spec(trusting))
        rr = _lower(build_decode_gemm(trusting))
        self.assertNotIn("fence", rr)
        self.assertNotIn("atomicrmw add", rr)
        self.assertNotIn("s_getreg", rr)
        tickets = [ln for ln in rr.splitlines() if "global_atomic_add $0" in ln]
        self.assertEqual(len(tickets), 1)
        self.assertIn("sc0", tickets[0])
        self.assertNotIn("sc1", tickets[0])
        loads = [ln for ln in rr.splitlines() if "buffer.load.v4i32" in ln]
        self.assertTrue(loads)
        self.assertFalse([ln for ln in loads if ln.rstrip().endswith("i32 16)")])
        # Refused: partners spread over XCDs, no split, 145 row blocks (not % 8).
        odd = DecodeGemmSpec(16, 2320, 8192, n_waves=9, split_k=16)
        for bad in (
            replace(trusting, wg_order="split"),
            replace(trusting, wg_order="skew"),
            replace(trusting, split_k=1, x_stage="global", pf=8),
            replace(odd, sk_sync="xcdrr", wg_order="tile"),
        ):
            self.assertTrue(is_valid_spec(replace(bad, sk_sync="relaxed"))[0])
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())
        moves = knob_moves(replace(trusting, sk_sync="xcd"))
        self.assertNotIn("xcdrr", {s.sk_sync for s in moves})
        self.assertNotIn("skew", {s.wg_order for s in moves})

    def test_xcd_recomputes_the_block_when_a_partner_ran_off_the_finishers_xcc(self):
        # xcd: plain partial stores, an agent-scope ticket that adds
        # 1 + (xcc << 6) + (xcc^2 << 16), and two finisher bodies: plain slot
        # loads when the sums show every partner on the finisher's XCC, else a
        # whole-K recompute of the row block from W and X (no slot reads).
        base = DecodeGemmSpec(
            16, 2304, 8192, n_waves=9, split_k=16, reduce="slots", sk_sync="xcd"
        )
        for order in ("tile", "skew", "split"):
            spec = replace(base, wg_order=order)
            self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
            ll = _lower(build_decode_gemm(spec))
            self.assertNotIn("fence", ll)
            self.assertNotIn("atomicrmw add", ll)
            self.assertIn("s_getreg", ll)
            tickets = [ln for ln in ll.splitlines() if "global_atomic_add $0" in ln]
            self.assertEqual(len(tickets), 1)
            self.assertNotIn("sc1", tickets[0])
            calls = [
                ln.rstrip()
                for ln in ll.splitlines()
                if "call" in ln and "declare" not in ln
            ]
            partial = [ln for ln in calls if "buffer.store.v4i32" in ln]
            self.assertTrue(partial)
            self.assertTrue(all(ln.endswith(", i32 0, i32 0)") for ln in partial))
            loads = [ln for ln in calls if "buffer.load.v4i32" in ln]
            self.assertFalse([ln for ln in loads if ln.endswith("i32 16)")])
            slot_reads = [ln for ln in loads if ln.endswith(", i32 0, i32 0)")]
            self.assertEqual(len(slot_reads), 16 * -(-16 * 144 // 4 // 576))
            # Recompute: 9 row groups (one per wave), 8 steps x (W + X) per iteration.
            again = [ln for ln in loads if re.search(r", i32 %add\d+, i32 0\)$", ln)]
            self.assertEqual(len(again), 9 * 8 * 2)
        # The co-location test: count, XCC-id sum and square sum fields.
        self.assertRegex(ll, r"lshr i32 %\S+, 16")
        self.assertRegex(ll, r"and i32 %\S+, 1023")
        self.assertRegex(ll, r"and i32 %\S+, 63")
        for bad in (
            replace(base, split_k=1, x_stage="global", pf=8),
            replace(base, split_k=64),
        ):
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())
        self.assertFalse(
            is_valid_spec(replace(base, sk_sync="fence", wg_order="skew", split_k=1))[0]
        )

    def test_interleaved_w_ring_keeps_the_loads_and_spreads_the_banks(self):
        base = DecodeGemmSpec(
            17, 8192, 1024, n_waves=2, preload=1, lpf=2, w_stage="lds", w_piece=256
        )
        for piece, pitch in ((256, 1056), (128, 1088)):
            spec = replace(base, w_piece=piece, w_ilv=1)
            self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
            p = smallm_plan(spec)
            self.assertEqual(p["w_pitch"], pitch)
            ll, ll0 = (
                _lower(build_decode_gemm(s)) for s in (spec, replace(spec, w_ilv=0))
            )
            for needle in ("buffer.load.lds", "load <4 x i32>, ptr addrspace(3)"):
                self.assertEqual(_calls(ll, needle), _calls(ll0, needle), needle)
            # The layout the builder uses: DMA lane l of block j holds row j R + l % R,
            # 16 B chunk l / R at byte 16 l; the A operand of lane (r, g) at step t is
            # chunk 4 t + g of row r, read at (r / R) pitch + ((4 t + g) R + r % R) 16.
            R = p["w_rows"]
            held = {}
            for j in range(16 // R):
                for lane in range(64):
                    held[j * pitch + 16 * lane] = (j * R + lane % R, lane // R)
                    rows = {(j * R + q % R) for q in range(64)}
                    self.assertEqual(len(rows), R)  # R whole-line runs per DMA
            for t in range(piece // 32):
                for g in range(4):
                    offs = []
                    for r in range(16):
                        off = (r // R) * pitch + ((4 * t + g) * R + r % R) * 16
                        self.assertEqual(held[off], (r, 4 * t + g))
                        offs.append(off % 256 // 16)
                    self.assertEqual(len(set(offs)), 16)  # 16 distinct 16-B bank groups
        for bad in (
            replace(base, w_ilv=1, w_piece=512),
            DecodeGemmSpec(16, 8192, 1024, n_waves=2, w_ilv=1),
        ):
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())

    def test_acc_split_alternates_two_accumulator_chains(self):
        direct = DecodeGemmSpec(16, 8192, 1024, n_waves=2, preload=1)
        lds = replace(direct, w_stage="lds", w_piece=128, w_ring=4, lds_swz=1, lpf=1)
        for spec0 in (direct, lds):
            spec = replace(spec0, acc_split=2)
            self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
            ll, ll0 = (_lower(build_decode_gemm(s)) for s in (spec, spec0))
            self.assertEqual(_calls(ll, _MFMA16), _calls(ll0, _MFMA16))
            self.assertEqual(_calls(ll, "fadd <4 x float>"), _calls(ll0, "fadd <4 x float>") + 1)
            res, accs = [], []
            for ln in ll.splitlines():
                if _MFMA16 in ln and "=" in ln:
                    res.append(ln.split("=")[0].strip())
                    accs.append(re.findall(r"<4 x float> (%[\w.]+)", ln)[-1])
            self.assertEqual(accs[0], accs[1])  # both chains start from zero
            self.assertNotIn(accs[0], res)
            for k in range(2, len(res)):  # step k accumulates onto step k - 2
                self.assertEqual(accs[k], res[k - 2], (spec.kernel_name(), k))
            self.assertNotEqual(spec.kernel_name(), spec0.kernel_name())
        self.assertTrue(is_valid_spec(replace(direct, M=32, acc_split=2))[0])
        for bad in (replace(direct, M=48, acc_split=2), replace(direct, M=32, row_groups=2, acc_split=2),
                    replace(direct, acc_split=3)):  # fmt: skip
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())

    def test_two_step_tail_stages_the_last_piece_in_halves(self):
        spec0 = DecodeGemmSpec(16, 8192, 1024, n_waves=2, preload=1, lpf=1,
                               w_stage="lds", w_piece=128, w_ring=4, lds_swz=1)  # fmt: skip
        spec = replace(spec0, w_tail=1)
        self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
        p, p0 = smallm_plan(spec), smallm_plan(spec0)
        self.assertEqual(p["w_lds"] - p0["w_lds"], p["waves"] * p["w_dmas"] * 1024)
        ll, ll0 = (_lower(build_decode_gemm(s)) for s in (spec, spec0))
        for needle in ("buffer.load.lds", "load <4 x i32>, ptr addrspace(3)", _MFMA16):
            self.assertEqual(_calls(ll, needle), _calls(ll0, needle), needle)
        waits = re.findall(r"s\.waitcnt\(i32 (\d+)\)", ll)
        waits0 = re.findall(r"s\.waitcnt\(i32 (\d+)\)", ll0)
        vm = 0xF70  # vmcnt(n), n < 16, with expcnt and lgkmcnt left open
        self.assertEqual(waits[-2:], [str(vm + p["w_dmas"] // 2), str(vm)])
        self.assertEqual(len(waits), len(waits0) + 1)
        # Half layout: DMA d writes lane l's 16 B at byte 1024 d + 16 l, which is chunk slot
        # l % 8 of dense 128 B row r = 8 d + l // 8, and fetches chunk (l % 8) ^ ((r >> 1) & 7);
        # the operand lane (r, g) of half step tt reads chunk 4 tt + g of row r.
        held = {}
        for d in range(2):
            for lane in range(64):
                r, s = 8 * d + lane // 8, lane % 8
                held[1024 * d + 16 * lane] = (r, s ^ ((r >> 1) & 7))
        for tt in range(2):
            for g in range(4):
                offs = [r * 128 + 16 * ((4 * tt + g) ^ ((r >> 1) & 7)) for r in range(16)]
                self.assertEqual([held[o] for o in offs], [(r, 4 * tt + g) for r in range(16)])
                self.assertEqual(len({o % 256 // 16 for o in offs}), 16)
        for bad in (replace(spec, w_piece=256, w_ring=2), replace(spec, lds_swz=0), replace(spec, w_stage="direct")):
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())

    def test_xor_swizzle_packs_the_rows_and_keeps_the_loads(self):
        base = DecodeGemmSpec(
            24, 8192, 1024, n_waves=2, preload=1, lpf=1, w_stage="lds", w_piece=128, w_ring=4
        )
        qkv = DecodeGemmSpec(16, 2304, 8192, n_waves=5, block_rows=72, split_k=8, k_waves=2,
                             pf=8, reduce="slots", sk_sync="xcd", wg_order="tile")  # fmt: skip
        for spec0 in (base, replace(base, w_piece=256, w_ring=2), replace(base, M=1), qkv):
            spec = replace(spec0, lds_swz=1)
            self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
            p, p0 = smallm_plan(spec), smallm_plan(spec0)
            self.assertEqual(p["x_pitch"], p["k_wg"] * 2)
            self.assertEqual(p["x_lds"], p0["x_lds"] - 16 * spec.M)
            if spec.w_stage == "lds":
                self.assertEqual(p["w_pitch"], 1024)
                self.assertEqual(p0["w_lds"] - p["w_lds"], 16 * p["ring"] * p["waves"] * p["w_dmas"])
            ll, ll0 = (_lower(build_decode_gemm(s)) for s in (spec, spec0))
            for needle in ("buffer.load.lds", "global.load.lds(", "load <4 x i32>, ptr addrspace(3)", _MFMA16):
                self.assertEqual(_calls(ll, needle), _calls(ll0, needle), (spec.kernel_name(), needle))
            self.assertIn(" xor i32 ", ll)
            self.assertNotEqual(spec.kernel_name(), spec0.kernel_name())
        # The layout: DMA d writes lane l's 16 B linearly at byte 1024 d + 16 l, which is
        # chunk slot s of dense row r, and fetches chunk s ^ (r & 15) (low 4 bits) of row
        # r; the operand lane (r, g) of step t reads chunk 4 t + g of row r at
        # r pitch + 16 ((4 t + g) ^ r) (low 4 bits).
        for row_bytes in (256, 512, 1024, 2048):
            ch = row_bytes // 16
            held = {}
            for d in range(16 * row_bytes // 1024):
                for lane in range(64):
                    byte = 1024 * d + 16 * lane
                    r, s = byte // row_bytes, byte % row_bytes // 16
                    held[byte] = (r, (s & ~15) | ((s & 15) ^ (r & 15)))
            for t in range(ch // 4):
                for g in range(4):
                    offs = []
                    for r in range(16):
                        c = 4 * t + g
                        off = r * row_bytes + 16 * ((c & ~15) | ((c & 15) ^ r))
                        self.assertEqual(held[off], (r, c), (row_bytes, t, g, r))
                        offs.append(off % 256 // 16)
                    self.assertEqual(len(set(offs)), 16)
        for bad in (
            replace(base, lds_swz=1, w_ilv=1),
            replace(base, lds_swz=1, x_stream=1, w_piece=256),
            replace(base, lds_swz=1, x_stage="global", w_stage="direct", w_piece=512),
            replace(base, lds_swz=2),
        ):
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())

    def test_x_per_piece_interleaves_x_with_w_and_cycles_a_ring(self):
        ring4 = DecodeGemmSpec(24, 8192, 1024, n_waves=2, preload=1, lpf=1, w_stage="lds",
                            w_piece=128, w_ring=4, lds_swz=1)  # fmt: skip
        dmas, ds = "global.load.lds(", "load <4 x i32>, ptr addrspace(3)"
        for spec0, x_ring in ((ring4, 0), (ring4, 5), (replace(ring4, M=1), 0), (replace(ring4, M=17, w_piece=256, w_ring=2), 3)):
            spec = replace(spec0, x_stream=2, x_ring=x_ring)
            self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
            p = smallm_plan(spec)
            P, nd = p["pieces"], -(-spec.M // (512 // spec.w_piece))
            self.assertEqual((p["x_phases"], p["x_dmas"], p["x_slots"]), (P, nd, x_ring or P))
            self.assertEqual(p["x_lds"], (x_ring or P) * nd * 1024)
            ll, ll0 = (_lower(build_decode_gemm(s)) for s in (spec, spec0))
            self.assertEqual(_calls(ll, dmas), P * -(-nd // 2))  # every wave the same count per phase
            self.assertEqual(_calls(ll, "buffer.load.lds"), _calls(ll0, "buffer.load.lds"))
            self.assertEqual(_calls(ll, ds), _calls(ll0, ds))
            self.assertEqual(_calls(ll, _MFMA16), _calls(ll0, _MFMA16))
            self.assertEqual(_calls(ll, "s.barrier"), _calls(ll0, "s.barrier") - 1 + P)  # one per piece
            self.assertIn(f"dg_x{(x_ring or P) - 1}", ll)
            self.assertNotIn(f"dg_x{(x_ring or P)}", ll)
        # Piece 0 waits for X(0) and W(0): X(1..3) (3 DMAs each) and W(1..3) (4 each) may stay in flight.
        self.assertIn("s.waitcnt(i32 20341)", _lower(build_decode_gemm(replace(ring4, x_stream=2))))
        # M = 64 with ring 4 fits only through the X ring.
        m64 = replace(ring4, M=64, x_stream=2)
        self.assertFalse(is_valid_spec(m64)[0])
        self.assertTrue(is_valid_spec(replace(m64, x_ring=5))[0], is_valid_spec(replace(m64, x_ring=5)))
        for bad in (
            replace(ring4, x_stream=2, lds_swz=0),
            replace(ring4, x_stream=2, x_ring=4),  # not above the W ring
            replace(ring4, x_stream=2, x_ring=9),  # more slots than pieces
            replace(ring4, x_ring=5),  # x_ring without x_stream=2
            replace(ring4, x_stream=2, M=16, N=2304, K=8192, n_waves=5, k_waves=2, split_k=8),
        ):
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())

    def test_x_stream_waits_for_one_phase_at_a_time(self):
        base = DecodeGemmSpec(
            17, 8192, 1024, n_waves=2, preload=1, lpf=2, w_stage="lds", w_piece=256
        )
        spec = replace(base, x_stream=1)
        self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
        p = smallm_plan(spec)
        self.assertEqual(p["x_phases"], 2)
        self.assertEqual(p["x_lds"], 2 * 17 * 1040)
        ll, ll0 = (_lower(build_decode_gemm(s)) for s in (spec, base))
        dmas = "global.load.lds("
        self.assertEqual(_calls(ll, dmas), 2 * -(-17 // 2))  # 9 per phase on 2 waves
        self.assertEqual(_calls(ll0, dmas), -(-34 // 2))
        self.assertEqual(_calls(ll, "s.barrier"), _calls(ll0, "s.barrier") + 1)
        self.assertIn("dg_x1", ll)
        # vmcnt of piece 0: W piece 1 (8) and X phase 1 (8) may still be in flight.
        self.assertIn("s.waitcnt(i32 20336)", ll)  # vmcnt(16), lgkmcnt any
        for bad in (
            replace(base, x_stream=1, w_stage="direct", w_piece=512),
            replace(spec, M=16, N=2304, K=8192, n_waves=5, k_waves=2, split_k=8),
        ):
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())

    def test_shared_finish_spins_on_a_counting_ticket(self):
        base = DecodeGemmSpec(
            16, 2304, 8192, n_waves=5, block_rows=72, split_k=8, k_waves=2, pf=8,
            reduce="slots", sk_sync="relaxed", wg_order="tile",
        )  # fmt: skip
        spec = replace(base, sk_finish="shared")
        self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
        self.assertEqual(decode_gemm_launch(spec)["cnt_ints"], 2 * 32)
        ll = _lower(build_decode_gemm(spec))
        tickets = [ln for ln in ll.splitlines() if "global_atomic_add $0" in ln]
        self.assertEqual(len(tickets), 1)
        self.assertNotIn("atomicrmw add", ll)
        self.assertNotIn("store i32 0, ptr addrspace(1)", ll)  # no ticket reset
        self.assertIn("poll", ll)
        # Each partner reads 1 / 8 of the block from all 8 slots: 16 x 72 / 4 / 8
        # = 36 vectors, one pass of loads; the last-finisher reads it all.
        sc1 = [
            ln
            for ln in ll.splitlines()
            if "buffer.load.v4i32" in ln and ln.rstrip().endswith("i32 16)")
        ]
        self.assertEqual(len(sc1), 8)
        # shared_sc: the same kernel with the ticket polled at system scope (sc0 sc1).
        sc_spec = replace(spec, sk_finish="shared_sc")
        sc = _lower(build_decode_gemm(sc_spec)).replace(sc_spec.kernel_name(), spec.kernel_name()).splitlines()
        diff = [(a, c) for a, c in zip(ll.splitlines(), sc) if a != c]
        self.assertEqual(len(sc), len(ll.splitlines()))
        self.assertTrue(diff)
        self.assertTrue(all(a.replace("i32 16)", "i32 17)") == c for a, c in diff), diff[:2])
        for bad in (
            replace(spec, sk_sync="xcd"),
            replace(spec, reduce="atomic"),
            replace(spec, sk_finish="all"),
        ):
            self.assertFalse(is_valid_spec(bad)[0], bad.kernel_name())
        big = replace(spec, block_rows=36, n_waves=3)  # 64 blocks x 8 = 512 WGs
        self.assertFalse(is_valid_spec(big)[0], big.kernel_name())

    def test_block_rows_masks_rows_and_sets_the_grid(self):
        spec = DecodeGemmSpec(
            32, 2304, 8192, n_waves=3, block_rows=36, split_k=4, k_waves=4
        )
        self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
        self.assertEqual(decode_gemm_launch(spec)["grid"], (256, 1, 1))
        self.assertEqual(smallm_plan(spec)["vec"], 4)
        ll = _lower(build_decode_gemm(spec))
        self.assertIn(_MFMA16, ll)
        odd = replace(
            spec, block_rows=18, n_waves=2, split_k=2, k_waves=8, x_stage="global"
        )
        self.assertTrue(is_valid_spec(odd)[0], is_valid_spec(odd))
        self.assertEqual(smallm_plan(odd)["vec"], 2)
        self.assertIn("define amdgpu_kernel", _lower(build_decode_gemm(odd)))
        self.assertFalse(is_valid_spec(replace(spec, block_rows=49))[0])
        self.assertFalse(
            is_valid_spec(
                replace(spec, block_rows=18, epi="direct", split_k=1, k_waves=1)
            )[0]
        )

    def test_epilogue_waits_for_every_wave_before_reusing_the_x_lds(self):
        # The LDS layout may put the epilogue tile over the staged X: a barrier
        # must separate the last MFMA (last X read) from the first tile store.
        def barrier_between(ll):
            body = [
                ln for ln in ll.splitlines() if not ln.lstrip().startswith("declare")
            ]
            last_mfma = max(i for i, ln in enumerate(body) if _MFMA16 in ln)
            first_store = next(
                i
                for i, ln in enumerate(body)
                if i > last_mfma and "store <4 x float>" in ln and "addrspace(3)" in ln
            )
            return any("s.barrier" in ln for ln in body[last_mfma:first_store])

        for spec in (
            DecodeGemmSpec(16, 2304, 8192, n_waves=9, split_k=16),
            DecodeGemmSpec(16, 8192, 1024, n_waves=2, k_waves=2),
        ):
            self.assertTrue(
                barrier_between(_lower(build_decode_gemm(spec))), spec.kernel_name()
            )

    def test_lds_staged_weights(self):
        spec = DecodeGemmSpec(
            16, 8192, 1024, n_waves=2, w_stage="lds", w_piece=512, w_ring=2
        )
        self.assertTrue(is_valid_spec(spec)[0], is_valid_spec(spec))
        kern = build_decode_gemm(spec)
        ll = _lower(kern)
        p = smallm_plan(spec)
        self.assertEqual(
            (p["pieces"], p["ring"], p["w_dmas"], p["w_rows"]), (2, 2, 16, 1)
        )
        dmas = [
            ln
            for ln in ll.splitlines()
            if "raw.ptr.buffer.load.lds(" in ln and "declare" not in ln
        ]
        self.assertEqual(len(dmas), p["pieces"] * p["w_dmas"])
        self.assertTrue(
            all("i32 0, i32 2)" in ln for ln in dmas)
        )  # nt on the weight DMAs
        self.assertTrue(
            all("!alias.scope" in ln for ln in dmas)
        )  # one scope per ring slot
        self.assertEqual(_calls(ll, _BLOAD), 0)  # no direct weight loads
        self.assertEqual(_calls(ll, _MFMA16), p["S"] * spec.row_groups * p["MG"])
        self.assertTrue(kern.attrs.get("lds_alias_scopes"))
        self.assertEqual(p["w_lds"], 2 * 2 * 16 * 1040)
        # two rows per DMA (256-element pieces) and four (128): still valid and built
        for piece in (256, 128):
            odd = replace(spec, w_piece=piece)
            self.assertTrue(is_valid_spec(odd)[0], piece)
            self.assertIn(_MFMA16, _lower(build_decode_gemm(odd)))
        self.assertFalse(is_valid_spec(replace(spec, x_stage="global"))[0])
        self.assertFalse(is_valid_spec(replace(spec, w_piece=384))[0])

    def test_no_split_no_atomics(self):
        ll = _lower(build_decode_gemm(DecodeGemmSpec(16, 8192, 1024, n_waves=2)))
        self.assertNotIn("atomicrmw", ll)
        self.assertNotIn("fence", ll)
        self.assertEqual(
            decode_gemm_launch(DecodeGemmSpec(16, 8192, 1024, n_waves=2))["ws_bytes"], 0
        )

    def test_gemv_ir(self):
        spec = DecodeGemmSpec(2, 2304, 8192, family="gemv", waves=9)
        ll = _lower(build_decode_gemm(spec))
        p = gemv_plan(spec)
        self.assertEqual(p["rounds"], 1)
        self.assertEqual(_calls(ll, _MFMA4), 2 * spec.M * p["C"])
        self.assertEqual(_calls(ll, _BLOAD), p["C"])
        self.assertEqual(decode_gemm_launch(spec)["lds_bytes"], 2 * 8192 * 2)

    def test_gemv_idle_waves_run_the_body_without_a_branch(self):
        # With some waves idle (WE < W) a branch around the body let LLVM sink the
        # prologue's weight loads past the X staging wait, so vmcnt(n) no longer
        # covered the X DMA. Every wave now runs the body: same control flow as
        # the all-waves-compute twin (weff=-1).
        for waves, ytile in ((12, 2), (16, 3)):
            spec = DecodeGemmSpec(1, 8192, 1024, family="gemv", waves=waves, ytile=ytile)
            twin = replace(spec, weff=-1)
            p, pt = gemv_plan(spec), gemv_plan(twin)
            self.assertLess(p["WE"], p["W"])
            self.assertEqual((pt["WE"], pt["rounds"]), (pt["W"], p["rounds"]))

            def branches(s):
                return sum(1 for ln in _lower(build_decode_gemm(s)).splitlines() if ln.strip().startswith("br "))

            self.assertEqual(branches(spec), branches(twin), (waves, ytile))
            ll = _lower(build_decode_gemm(spec))
            self.assertEqual(_calls(ll, "global.load.lds("), -(-2 // waves))
            self.assertIn("s.waitcnt(", ll)

    def test_gemv_load_window_stays_within_the_register_budget(self):
        # 4 x window chunks x (ytile + X rows from global) VGPRs of loads in flight.
        inc = DecodeGemmSpec(1, 2304, 8192, family="gemv", waves=9, stage="global")
        self.assertEqual(gemv_plan(inc)["load_vgprs"], 128)
        self.assertTrue(is_valid_spec(inc)[0])
        two = replace(inc, M=2)
        self.assertEqual(gemv_plan(two)["load_vgprs"], 192)
        ok, why = is_valid_spec(two)
        self.assertFalse(ok)
        self.assertIn("VGPRs of loads", why)
        for good in (replace(two, stage="dma", pf=8), replace(two, pf=4), replace(inc, M=8, stage="dma", pf=8)):
            self.assertTrue(is_valid_spec(good)[0], (good.kernel_name(), is_valid_spec(good)))
        self.assertEqual(gemv_plan(replace(two, stage="dma", pf=8))["load_vgprs"], 32)

    def test_window_bounds_the_loads_in_flight(self):
        # pf=4: four steps of weight loads ahead, then each step issues step s + 4 before its MFMA.
        full = _lower(build_decode_gemm(DecodeGemmSpec(16, 8192, 1024, n_waves=2)))
        win = _lower(build_decode_gemm(DecodeGemmSpec(16, 8192, 1024, n_waves=2, pf=4)))
        self.assertEqual(_calls(full, _BLOAD), _calls(win, _BLOAD))

        def order(ll):
            body = [
                ln for ln in ll.splitlines() if not ln.lstrip().startswith("declare")
            ]
            return "".join(
                "L" if _BLOAD in ln else "M"
                for ln in body
                if _MFMA16 in ln or _BLOAD in ln
            )

        self.assertEqual(order(win)[:9], "LLLLLMLML")
        self.assertEqual(order(full)[:33], "L" * 32 + "M")

    def test_pin_fences_the_steps_of_the_window_prologue(self):
        # A scheduling fence after each prologue step keeps an early step's load
        # from sinking behind later steps. pin=1 places one with X from global in
        # two 16-row X groups (one W and two X loads per step); pin=2 after every
        # step, also with one X group and with X in LDS (the one-shot window's 32
        # W loads, between the fences around the X staging); pin=0 has none.
        def order(spec):
            ll = _lower(build_decode_gemm(spec))
            body = [
                ln for ln in ll.splitlines() if not ln.lstrip().startswith("declare")
            ]
            marks = ((_BLOAD, "L"), (_MFMA16, "M"), (_SCHED, "S"))
            return "".join(c for ln in body for needle, c in marks if needle in ln)

        knobs = {
            "n_waves": 1,
            "block_rows": 9,
            "k_waves": 8,
            "x_stage": "global",
            "pf": 8,
        }
        two = DecodeGemmSpec(32, 2304, 8192, **knobs)
        one = DecodeGemmSpec(16, 2304, 8192, **knobs)
        shot = DecodeGemmSpec(16, 8192, 1024, n_waves=2)
        self.assertTrue(order(two).startswith("LLLS" * 9 + "MMS"))
        self.assertTrue(order(one).startswith("LL" * 9 + "SMS"))
        self.assertTrue(order(replace(one, pin=2)).startswith("LLS" * 9 + "MS"))
        self.assertTrue(order(shot).startswith("S" + "L" * 32 + "S"))
        self.assertTrue(order(replace(shot, pin=2)).startswith("S" + "LS" * 32 + "S"))
        self.assertNotIn("S", order(replace(shot, pin=0)))
        for spec in (
            replace(default_spec(1, 2304, 8192), pin=2),
            replace(shot, w_stage="lds", pin=2),
        ):
            self.assertFalse(is_valid_spec(spec)[0], spec.kernel_name())

    def test_preload_marks_the_arguments_inreg(self):
        base = DecodeGemmSpec(16, 8192, 1024, n_waves=2)
        on = build_decode_gemm(replace(base, preload=1))
        define = next(
            ln
            for ln in _lower(on).splitlines()
            if ln.startswith("define amdgpu_kernel")
        )
        self.assertEqual(define.count(" inreg"), 5)
        self.assertTrue(on.attrs.get("kernarg_preload"))
        self.assertNotIn(" inreg", _lower(build_decode_gemm(base)))

    def test_direct_levers_leave_the_llvm_ir_alone(self):
        from rocke.core.ir_serialize import serialize

        base = DecodeGemmSpec(16, 8192, 1024, n_waves=2)
        lev = replace(base, dlevers="loadfirst,storewait")
        self.assertIn("dlevers", serialize(build_decode_gemm(lev)))
        self.assertNotIn("dlevers", serialize(build_decode_gemm(base)))
        strip = re.compile(r"dg_smallm\w*")
        self.assertEqual(
            strip.sub("K", _lower(build_decode_gemm(lev))),
            strip.sub("K", _lower(build_decode_gemm(base))),
        )

    def test_lowering_is_deterministic(self):
        spec = DecodeGemmSpec(17, 8192, 1024, n_waves=2, k_waves=2)
        self.assertEqual(
            _lower(build_decode_gemm(spec)), _lower(build_decode_gemm(spec))
        )

    def test_signature(self):
        sig = decode_gemm_signature(default_spec(1, 2304, 8192))
        self.assertEqual([p["name"] for p in sig], ["W", "X", "Y", "WS", "CNT"])


class TestSmallmModel(unittest.TestCase):
    """A numpy model of the smallm decomposition: grid -> (tile, split), wave ->
    (n-wave, k-wave), lane -> 16x16x32 operand slots, step -> 32 K elements."""

    def _model(self, spec, X, W):
        p = smallm_plan(spec)
        M, N, K = spec.M, spec.N, spec.K
        mp = 16 * p["MG"]
        Xp = np.zeros((mp, K), np.float64)
        Xp[:M] = X
        Y = np.zeros((M, N), np.float64)
        touched = np.zeros((N, K), np.int32)
        for bid in range(p["grid"]):
            if spec.split_k == 1:
                tile, ks = bid, 0
            elif spec.wg_order == "split":
                tile, ks = divmod(bid, spec.split_k)
            else:
                ks, tile = divmod(bid, p["n_tiles"])
                if spec.wg_order == "skew" and ks == spec.split_k - 1:
                    tile = (tile + 1) % p["n_tiles"]
            for wave in range(p["waves"]):
                nw, kw = wave % spec.n_waves, wave // spec.n_waves
                kbase = ks * p["k_wg"] + kw * p["kw_len"]
                for rg in range(spec.row_groups):
                    a = np.zeros((16, p["kw_len"]))
                    loc = (
                        nw * p["rows_w"] + rg * 16 + np.arange(16)
                    )  # rows in the block
                    rows = tile * p["block_rows"] + loc
                    ok = (rows < N) & (loc < p["block_rows"])
                    for lane in range(64):
                        r, g = lane % 16, lane // 16
                        for s in range(p["S"]):
                            k = kbase + s * 32 + 8 * g
                            if ok[r]:
                                a[r, s * 32 + 8 * g : s * 32 + 8 * g + 8] = W[
                                    rows[r], k : k + 8
                                ]
                                touched[rows[r], k : k + 8] += 1
                    d = a @ Xp[:, kbase : kbase + p["kw_len"]].T  # [16 rows, mp]
                    Y[:, rows[ok]] += d[ok, :M].T
        return Y, touched

    def test_decomposition_covers_w_once_and_matches(self):
        rng = np.random.default_rng(3)
        for spec in (
            DecodeGemmSpec(17, 96, 1024, n_waves=3, split_k=2),
            DecodeGemmSpec(
                5, 64, 2048, row_groups=2, k_waves=2, split_k=2, wg_order="tile"
            ),
            DecodeGemmSpec(1, 48, 512, n_waves=1),
            DecodeGemmSpec(33, 80, 512, n_waves=2, k_waves=2, x_stage="global"),
            # CU-exact row blocks: 36 and 18 rows per workgroup inside 48 / 32 wave rows
            DecodeGemmSpec(16, 144, 1024, n_waves=3, block_rows=36, split_k=2),
            DecodeGemmSpec(
                3, 160, 512, n_waves=2, block_rows=18, k_waves=2, reduce="slots"
            ),
        ):
            self.assertTrue(is_valid_spec(spec)[0], spec)
            X = rng.standard_normal((spec.M, spec.K))
            W = rng.standard_normal((spec.N, spec.K))
            Y, touched = self._model(spec, X, W)
            self.assertTrue((touched == 1).all(), spec)
            np.testing.assert_allclose(Y, X @ W.T, rtol=1e-9, atol=1e-9)


def _skinny(m, n, k, **kw):
    return DecodeGemmSpec(m, n, k, family="skinny", **kw)


class TestSkinny(unittest.TestCase):
    """Family "skinny": wave-private stage rings, X and W streamed per stage."""

    def test_specs_plans_and_refusals(self):
        cases = [  # spec, LDS bytes, grid, threads
            (_skinny(1, 8192, 1024, k_waves=2), 40960, 256, 128),
            (_skinny(16, 8192, 1024, block_k=128, stages=2, k_waves=2), 49152, 256, 128),
            (_skinny(17, 8192, 1024, block_m=24, k_waves=4), 114688, 256, 256),
            (_skinny(4, 2304, 8192, block_m=8, block_n=16, stages=6, k_waves=8), 147456, 144, 512),
            (_skinny(1, 8192, 1024, k_waves=2, xcd=1), 40960, 256, 128),
        ]
        for spec, lds, grid, threads in cases:
            self.assertEqual(is_valid_spec(spec), (True, "ok"), spec)
            launch = decode_gemm_launch(spec)
            self.assertEqual(
                (launch["lds_bytes"], launch["grid"][0], launch["block"][0]), (lds, grid, threads)
            )
            self.assertTrue(spec.kernel_name().startswith("dg_skinny_"))
        for bad in (
            _skinny(9, 8192, 1024, block_m=8),  # M above block_m
            _skinny(1, 8192, 1024, block_n=24),
            _skinny(1, 8192, 1024, block_k=32),
            _skinny(1, 8192, 1056),  # K not a multiple of k_waves * block_k
            _skinny(1, 2304, 8192, block_n=16, stages=12, k_waves=8),  # LDS
            _skinny(1, 2304, 1024, block_n=64, xcd=1),  # 36 tiles do not split over 8 XCDs
        ):
            ok, why = is_valid_spec(bad)
            self.assertFalse(ok, bad)
            self.assertTrue(why.startswith("skinny"), why)

    def test_ir_streams_stages_without_a_barrier_in_the_k_loop(self):
        spec = _skinny(1, 8192, 1024, k_waves=2)
        ll = _lower(build_decode_gemm(spec))
        dmas = [ln for ln in ll.splitlines() if "buffer.load.lds(" in ln and "declare" not in ln]
        # 8 stages x (1 X DMA + 4 W DMAs); W non-temporal, X default
        self.assertEqual(len(dmas), 40)
        self.assertEqual(sum(ln.rstrip().split("!alias")[0].rstrip(", ").endswith("i32 2)") for ln in dmas), 32)
        self.assertEqual(_calls(ll, "load <4 x i32>, ptr addrspace(3)"), 8 * 6 + 2)
        self.assertEqual(_calls(ll, _MFMA16), 32)
        waits = [int(w) for w in re.findall(r"s\.waitcnt\(i32 (\d+)\)", ll)]
        vm = [(w & 0xF) | (((w >> 14) & 3) << 4) for w in waits if (w >> 8) & 0xF == 15]
        self.assertEqual(vm, [15, 15, 15, 15, 15, 10, 5, 0])
        self.assertEqual(_calls(ll[: ll.rindex(_MFMA16)], "call void @llvm.amdgcn.s.barrier()"), 0)
        self.assertEqual(_calls(ll, "call void @llvm.amdgcn.s.barrier()"), 1)

    def test_model_covers_w_once_and_matches(self):
        """A numpy model of the stage DMAs (linear LDS writes of swizzled global
        chunks), the swizzled operand reads, the 16x16x32 lane layout and the
        k-wave sum reproduces ``X @ W.T``."""
        rng = np.random.default_rng(5)
        for spec in (
            _skinny(3, 64, 512, k_waves=2),
            _skinny(16, 32, 1024, block_k=128, stages=2, k_waves=2),
            _skinny(17, 64, 512, block_m=24, k_waves=4, stages=3),
            _skinny(5, 48, 1024, block_n=16, k_waves=4, stages=6),
        ):
            self.assertEqual(is_valid_spec(spec), (True, "ok"), spec)
            M, N, K = spec.M, spec.N, spec.K
            bm, bn, bk = skinny_plan(spec)["block_m"], spec.block_n, spec.block_k
            rb = 2 * bk
            C, R = rb // 16, 1024 // rb
            xor = (lambda r: (r >> 1) & 7) if rb == 128 else (lambda r: r & 15)
            X = rng.standard_normal((M, K))
            W = rng.standard_normal((N, K))
            Y = np.zeros((16 * -(-bm // 16), N))
            touched = np.zeros((N, K), np.int32)
            kw_len = K // spec.k_waves
            for tile in range(N // bn):
                for w in range(spec.k_waves):
                    for t in range(kw_len // bk):
                        lds = np.zeros((bm + bn, C, 8))
                        for d in range((bm + bn) // R):
                            for lane in range(64):
                                row, chunk = d * R + lane // C, lane % C
                                k0 = w * kw_len + t * bk + 8 * (chunk ^ xor(row))
                                if row < bm:
                                    if row < M:
                                        lds[row, chunk] = X[row, k0 : k0 + 8]
                                else:
                                    wr = tile * bn + row - bm
                                    lds[row, chunk] = W[wr, k0 : k0 + 8]
                                    touched[wr, k0 : k0 + 8] += 1
                        for ks in range(bk // 32):
                            for lane in range(64):
                                r16, g4 = lane % 16, lane // 16
                                for mg in range(-(-bm // 16)):
                                    xr = 16 * mg + r16
                                    if xr >= bm + bn:
                                        continue
                                    if xr >= bm:  # W rows of the stage: output rows past M
                                        continue
                                    xv = lds[xr, (g4 + 4 * ks) ^ xor(xr)]
                                    for rg in range(bn // 16):
                                        wrow = bm + 16 * rg
                                        for i in range(16):  # A row i of this lane's K slice
                                            wv = lds[wrow + i, (g4 + 4 * ks) ^ xor(wrow + i)]
                                            Y[xr, tile * bn + 16 * rg + i] += float(np.dot(wv, xv))
            self.assertTrue((touched == 1).all(), spec)
            np.testing.assert_allclose(Y[:M], X @ W.T, rtol=1e-9, atol=1e-9)


def _shared(m, n, k, **kw):
    kw = {"block_m": 64, "block_n": 64, "stages": 6, "m_waves": 4, "n_waves": 2, **kw}
    return _skinny(m, n, k, **kw)


class TestSkinnyShared(unittest.TestCase):
    """Family "skinny" with m_waves x n_waves waves sharing one stage ring."""

    def test_specs_plans_and_refusals(self):
        spec = _shared(128, 8192, 1024)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        launch = decode_gemm_launch(spec)
        self.assertEqual((launch["lds_bytes"], launch["grid"][0], launch["block"][0]), (98304, 256, 512))
        self.assertEqual(skinny_plan(spec)["wave_dmas"], 2)
        for bad in (
            _shared(128, 8192, 1024, k_waves=2),
            _shared(128, 8192, 1024, block_m=24),
            _shared(128, 8192, 1024, n_waves=4),  # 16 waves: 8 X DMAs do not split
            _shared(128, 8192, 1024, stages=1),
            _shared(128, 8192, 1024, stages=12),  # LDS
        ):
            ok, why = is_valid_spec(bad)
            self.assertFalse(ok, bad)
            self.assertTrue(why.startswith("skinny"), why)

    def test_ir_waits_then_barrier_then_refill_per_stage(self):
        ll = _lower(build_decode_gemm(_shared(128, 8192, 1024)))
        dmas = [ln for ln in ll.splitlines() if "buffer.load.lds(" in ln and "declare" not in ln]
        # 16 stages x (1 X DMA + 1 W DMA) per wave
        self.assertEqual(len(dmas), 32)
        self.assertEqual(sum(ln.rstrip().split("!alias")[0].rstrip(", ").endswith("i32 2)") for ln in dmas), 16)
        self.assertEqual(_calls(ll, "load <4 x i32>, ptr addrspace(3)"), 16 * 6)
        self.assertEqual(_calls(ll, _MFMA16), 64)
        self.assertEqual(_calls(ll, "call void @llvm.amdgcn.s.barrier()"), 16)
        waits = [int(w) for w in re.findall(r"s\.waitcnt\(i32 (\d+)\)", ll)]
        vm = [(w & 0xF) | (((w >> 14) & 3) << 4) for w in waits if (w >> 8) & 0xF == 15]
        self.assertEqual(vm, [8] * 12 + [6, 4, 2, 0])

    def test_model_covers_x_and_w_once_per_tile_and_matches(self):
        """A numpy model of the shared ring: wave ``w`` DMAs stage rows
        ``(j * waves + w) * R + lane // C``, wave (wm, wn) reads its rows of
        the tile, and the grid walks M tiles then column tiles."""
        rng = np.random.default_rng(6)
        for spec in (
            _shared(100, 128, 512),
            _shared(64, 64, 256, block_m=32, block_n=32, m_waves=2, n_waves=2, block_k=128, stages=2),
        ):
            self.assertEqual(is_valid_spec(spec), (True, "ok"), spec)
            p = skinny_plan(spec)
            M, N, K = spec.M, spec.N, spec.K
            bm, bn, bk = spec.block_m, spec.block_n, spec.block_k
            nw, WV = spec.n_waves, p["waves"]
            rb = 2 * bk
            C, R = rb // 16, 1024 // rb
            xor = (lambda r: (r >> 1) & 7) if rb == 128 else (lambda r: r & 15)
            X = rng.standard_normal((M, K))
            W = rng.standard_normal((N, K))
            Y = np.zeros((p["m_tiles"] * bm, N))
            for tile in range(p["grid"]):
                mt, nt = divmod(tile, N // bn)
                touched = np.zeros(bm + bn, np.int32)
                for t in range(K // bk):
                    lds = np.zeros((bm + bn, C, 8))
                    for w in range(WV):
                        for j in range(p["wave_dmas"]):
                            for lane in range(64):
                                row, chunk = (j * WV + w) * R + lane // C, lane % C
                                k0 = t * bk + 8 * (chunk ^ xor(row))
                                if row < bm:
                                    if mt * bm + row < M:
                                        lds[row, chunk] = X[mt * bm + row, k0 : k0 + 8]
                                else:
                                    lds[row, chunk] = W[nt * bn + row - bm, k0 : k0 + 8]
                                if t == 0 and chunk == 0:
                                    touched[row] += 1
                    for w in range(WV):
                        wm, wn = divmod(w, nw)
                        for ks in range(bk // 32):
                            for lane in range(64):
                                r16, g4 = lane % 16, lane // 16
                                for mg in range(p["MG"]):
                                    xr = wm * 16 * p["MG"] + 16 * mg + r16
                                    xv = lds[xr, (g4 + 4 * ks) ^ xor(xr)]
                                    for rg in range(p["RG"]):
                                        wrow = bm + wn * 16 * p["RG"] + 16 * rg
                                        for i in range(16):
                                            wv = lds[wrow + i, (g4 + 4 * ks) ^ xor(wrow + i)]
                                            n = nt * bn + wrow - bm + i
                                            Y[mt * bm + xr, n] += float(np.dot(wv, xv))
                self.assertTrue((touched == 1).all(), spec)
            np.testing.assert_allclose(Y[:M], X @ W.T, rtol=1e-9, atol=1e-9)

    def test_refill_after_reads(self):
        """refill=1: same DMAs, waits and barriers; each refill follows the
        step's operand reads instead of the barrier."""
        base = _lower(build_decode_gemm(_shared(128, 8192, 1024)))
        late = _lower(build_decode_gemm(_shared(128, 8192, 1024, refill=1)))
        for text in ("buffer.load.lds(", "call void @llvm.amdgcn.s.barrier()", _MFMA16):
            self.assertEqual(_calls(late, text), _calls(base, text), text)
        body = late.split("call void @llvm.amdgcn.s.barrier()")[1]
        self.assertLess(body.index("load <4 x i32>, ptr addrspace(3)"), body.index("buffer.load.lds("))
        body = base.split("call void @llvm.amdgcn.s.barrier()")[1]
        self.assertGreater(body.index("load <4 x i32>, ptr addrspace(3)"), body.index("buffer.load.lds("))


def _split(**kw):
    kw = {"block_m": 128, "block_n": 64, "stages": 4, "m_waves": 1, "n_waves": 4, "split_k": 7, **kw}
    return _skinny(128, 2304, 8192, **kw)


class TestSkinnySplit(unittest.TestCase):
    """Shared ring with split_k workgroups along K and an in-kernel reduce."""

    def test_specs_plans_and_refusals(self):
        spec = _split()
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        launch = decode_gemm_launch(spec)
        self.assertEqual((launch["lds_bytes"], launch["grid"][0], launch["block"][0]), (98304, 252, 256))
        self.assertEqual((launch["ws_bytes"], launch["cnt_ints"]), (7 * 128 * 2304 * 4, 36))
        self.assertEqual(skinny_plan(spec)["steps"], 19)
        self.assertEqual(decode_gemm_launch(_split(split_k=1))["ws_bytes"], 0)
        for bad in (
            _split(sk_sync="xcd"),
            _split(split_k=0),
            _split(split_k=65),  # a split under 2 steps
            _skinny(1, 2304, 8192, block_n=16, k_waves=8, split_k=2),  # private ring
            _skinny(1, 2304, 8192, block_n=16, k_waves=8, refill=1),
        ):
            ok, why = is_valid_spec(bad)
            self.assertFalse(ok, bad)
            self.assertTrue(why.startswith("skinny"), why)

    def test_splits_cover_every_step_once(self):
        for sk in (2, 3, 7, 8):
            steps = 8192 // 64
            ranges = [(s * steps // sk, (s + 1) * steps // sk) for s in range(sk)]
            longest = skinny_plan(_split(split_k=sk))["steps"]
            self.assertEqual(max(e - b for b, e in ranges), longest)
            covered = [k for b, e in ranges for k in range(b, e)]
            self.assertEqual(covered, list(range(steps)))

    def test_ir_ring_partials_ticket_and_finisher(self):
        ll = _lower(build_decode_gemm(_split()))
        # 19 stages x (4 X + 2 W DMAs) per wave, 8 X + 1 W operand reads per 32 K
        self.assertEqual(_calls(ll, "buffer.load.lds("), 19 * 6)
        self.assertEqual(_calls(ll, "load <4 x i32>, ptr addrspace(3)"), 19 * 2 * 9)
        self.assertEqual(_calls(ll, _MFMA16), 19 * 16)
        self.assertEqual(_calls(ll, "call void @llvm.amdgcn.s.barrier()"), 19 + 2)
        self.assertEqual(_calls(ll, 'fence syncscope("agent") release'), 1)
        self.assertEqual(_calls(ll, 'fence syncscope("agent") acquire'), 1)
        self.assertEqual(_calls(ll, "atomicrmw add"), 1)
        relaxed = _lower(build_decode_gemm(_split(sk_sync="relaxed")))
        self.assertEqual(_calls(relaxed, "fence syncscope"), 0)


class TestSkinnyM0Fill(unittest.TestCase):
    """m0fill: the M0 hazard slot before each LDS DMA."""

    def test_groups_stay_in_reach_and_operand(self):
        from rocke.instances.gfx950.decode_gemm_skinny import m0_groups

        self.assertEqual(m0_groups(5, 1, 1), ([0, 1, 1, 1, 1], [0, 0, 1024, 2048, 3072]))
        self.assertEqual(m0_groups(6, 1, 1), ([0, 1, 1, 1, 1, 5], [0, 0, 1024, 2048, 3072, 0]))
        self.assertEqual(m0_groups(4, 2, 2), ([0, 1, 1, 3], [0, 0, 2048, 0]))
        self.assertEqual(m0_groups(6, 4, 16), (list(range(6)), [0] * 6))

    def test_voffsets_stay_non_negative(self):
        """The immediate taken out of a DMA's voffset never exceeds the row
        offset of its first lane (every row of a later DMA in a group sits
        at least ``imm / 1024`` DMA rows into its operand)."""
        from rocke.instances.gfx950.decode_gemm_skinny import m0_groups

        for spec in (
            _skinny(1, 8192, 1024, block_m=8, k_waves=2),
            _skinny(24, 8192, 1024, block_m=24, k_waves=2),
            _skinny(16, 8192, 1024, block_m=16, block_k=128, stages=2, k_waves=2),
            _skinny(4, 2304, 8192, block_m=8, block_n=16, stages=6, k_waves=8),
            _shared(16, 2304, 8192, block_m=16, n_waves=2, m_waves=1, stages=4),
        ):
            p = skinny_plan(spec)
            wv = p["waves"] if p["waves"] > 1 and spec.k_waves == 1 else 1
            _, imm = m0_groups(p["wave_dmas"], wv, p["x_dmas"])
            for j, off in enumerate(imm):
                d = j * wv
                row0 = d * p["rows_dma"] - (spec.block_m or p["block_m"] if d >= p["x_dmas"] else 0)
                self.assertGreaterEqual(row0 * spec.K * 2, off, (spec, j))

    def test_ir_shares_m0_and_interleaves(self):
        base = _lower(build_decode_gemm(_skinny(1, 8192, 1024, block_m=8, k_waves=2)))
        one = _lower(build_decode_gemm(_skinny(1, 8192, 1024, block_m=8, k_waves=2, m0fill=1)))
        dmas = [ln for ln in one.splitlines() if "buffer.load.lds(" in ln and "declare" not in ln]
        self.assertEqual(len(dmas), _calls(base, "buffer.load.lds("))
        # 8 stages of 1 X + 4 W DMAs: the last three W DMAs ride on the first's M0
        imms = [int(re.search(r", i32 (\d+), i32 \d+\), !alias", ln).group(1)) for ln in dmas]
        self.assertEqual(imms, [0, 0, 1024, 2048, 3072] * 8)
        two = _lower(build_decode_gemm(_skinny(1, 8192, 1024, block_m=8, k_waves=2, m0fill=2)))
        self.assertGreater(_calls(two, "sched.group.barrier("), 0)
        self.assertEqual(_calls(two, _MFMA16), _calls(base, _MFMA16))
        self.assertFalse(is_valid_spec(_shared(128, 8192, 1024, m0fill=2))[0])
        self.assertTrue(is_valid_spec(_shared(128, 8192, 1024, m0fill=2, refill=1))[0])


def _ring144(m=128, **kw):
    kw = {"block_m": 128, "block_n": 144, "stages": 4, "m_waves": 2, "n_waves": 1, "split_k": 16,
          "sk_sync": "relaxed", **kw}
    return _skinny(m, 2304, 8192, **kw)


class TestSkinnySplitFinish(unittest.TestCase):
    """block_n=144 (16 qkv column tiles) and sk_finish 'epoch' / 'shared'."""

    def test_144_layouts(self):
        """A 144-row tile needs n_waves in (1, 3, 9) and the stage's X and W
        DMAs split evenly over the waves."""
        for kw in ({}, {"block_m": 64}, {"block_m": 32}, {"block_m": 48, "m_waves": 1, "n_waves": 3},
                   {"block_m": 64, "m_waves": 4, "block_k": 128, "stages": 2}):
            spec = _ring144(**kw)
            self.assertEqual(is_valid_spec(spec), (True, "ok"), kw)
        for kw in ({"block_m": 16, "m_waves": 1, "n_waves": 3}, {"n_waves": 2}, {"m_waves": 4}):
            self.assertFalse(is_valid_spec(_ring144(**kw))[0], kw)
        self.assertFalse(is_valid_spec(_skinny(128, 8192, 1024, block_m=128, block_n=144, m_waves=2))[0])
        p = skinny_plan(_ring144())
        self.assertEqual((p["tiles"], p["grid"], p["steps"]), (16, 256, 8))
        from rocke.instances.gfx950.decode_gemm import resolved_spec

        auto = _ring144(split_k=-1)
        self.assertEqual(resolved_spec(auto).split_k, 16)
        self.assertEqual(resolved_spec(replace(auto, cus=128)).split_k, 8)
        self.assertEqual(skinny_plan(_ring144(64, block_m=64))["grid"], 256)

    def test_finish_validity_and_launch(self):
        for fin in ("epoch", "shared"):
            self.assertEqual(is_valid_spec(_ring144(sk_finish=fin)), (True, "ok"))
            self.assertFalse(is_valid_spec(_split(sk_finish=fin))[0])  # split_k 7
            self.assertFalse(is_valid_spec(_ring144(sk_finish=fin, split_k=1))[0])
        self.assertTrue(is_valid_spec(_ring144(sk_finish="epoch", sk_sync="fence"))[0])
        self.assertFalse(is_valid_spec(_ring144(sk_finish="shared", sk_sync="fence"))[0])
        self.assertFalse(is_valid_spec(_ring144(sk_finish="shared", cus=128))[0])  # 256 workgroups
        self.assertTrue(is_valid_spec(_ring144(sk_finish="shared", cus=128, split_k=8))[0])
        self.assertFalse(is_valid_spec(_ring144(sk_finish="smeared"))[0])
        self.assertEqual(decode_gemm_launch(_ring144())["cnt_ints"], 16)
        self.assertEqual(decode_gemm_launch(_ring144(sk_finish="epoch"))["cnt_ints"], 16)
        self.assertEqual(decode_gemm_launch(_ring144(sk_finish="shared"))["cnt_ints"], 32)

    def test_ir_tails(self):
        reset = "store i32 0, ptr addrspace(1)"
        last = _lower(build_decode_gemm(_ring144()))
        epoch = _lower(build_decode_gemm(_ring144(sk_finish="epoch")))
        shared = _lower(build_decode_gemm(_ring144(sk_finish="shared")))
        self.assertEqual(_calls(last, reset), 1)
        self.assertEqual(_calls(epoch, reset), 0)
        self.assertEqual(_calls(shared, reset), 0)
        for ll in (last, epoch, shared):
            self.assertEqual(_calls(ll, _MFMA16), 8 * 4 * 9 * 2)  # 8 steps x 4 X groups x 9 W tiles x 2 K subs
        self.assertIn("s_sleep 1", shared)
        self.assertNotIn("s_sleep 1", epoch)
        # 128 x 144 / 4 = 4608 vectors over 128 threads x 16 slots: the last partner
        # reads 36 rounds, a shared partner its 1 / 16 (288 vectors, 3 rounds)
        self.assertEqual(_calls(last, _BLOAD), 36 * 16)
        self.assertEqual(_calls(shared, _BLOAD), 3 * 16)

    def test_shared_finish_covers_the_tile_once(self):
        """Partner ks sums vectors [ks * share, (ks + 1) * share) of the
        bm x bn / 4 tile: every vector exactly once."""
        for spec in (_ring144(), _ring144(32, block_m=32), _ring144(48, block_m=48, m_waves=1, n_waves=3, split_k=8)):
            p = skinny_plan(spec)
            per_row, total = spec.block_n // 4, p["block_m"] * spec.block_n // 4
            share = -(-total // spec.split_k)
            seen = [0] * total
            for ks in range(spec.split_k):
                for q in range(-(-share // p["threads"])):
                    for t in range(p["threads"]):
                        e_loc = t + q * p["threads"]
                        e = ks * share + e_loc
                        if e < total and e_loc < share:
                            seen[e] += 1
            self.assertEqual(seen, [1] * total)
            self.assertEqual(per_row * p["block_m"], total)


class TestPartition(unittest.TestCase):
    """Device fields cus and xccs: a compute partition's CU and XCC counts."""

    def test_defaults_keep_names(self):
        for spec in (
            _skinny(4, 8192, 1024, block_m=8, k_waves=2, xcd=1),
            _split(),
            DecodeGemmSpec(16, 2304, 8192, split_k=4, sk_sync="xcd"),
        ):
            same = replace(spec, cus=256, xccs=8)
            self.assertEqual(same.kernel_name(), spec.kernel_name())
            self.assertEqual(_lower(build_decode_gemm(same)), _lower(build_decode_gemm(spec)))
        self.assertIn("_cu128_xc4", replace(_split(), cus=128, xccs=4).kernel_name())

    def test_skinny_xcd_modulus(self):
        """xcd=1 maps workgroups to tiles one-to-one on any XCC count dividing
        the grid; each XCC's workgroups get one contiguous tile range."""
        spec = _split(split_k=1, xcd=1)
        self.assertFalse(is_valid_spec(spec)[0])  # 36 tiles over 8 XCCs
        for x in (4, 3, 2):
            s = replace(spec, xccs=x)
            self.assertEqual(is_valid_spec(s), (True, "ok"))
            grid = skinny_plan(s)["grid"]
            tiles = [(b % x) * (grid // x) + b // x for b in range(grid)]
            self.assertEqual(sorted(tiles), list(range(grid)))
            for c in range(x):
                got = sorted(tiles[c::x])
                self.assertEqual(got, list(range(got[0], got[0] + grid // x)))
            ll = _lower(build_decode_gemm(s))
            self.assertEqual(_calls(ll, _MFMA16), _calls(_lower(build_decode_gemm(replace(s, xcd=0))), _MFMA16))
        self.assertFalse(is_valid_spec(replace(spec, xccs=0))[0])
        self.assertFalse(is_valid_spec(replace(spec, xccs=17))[0])

    def test_skinny_auto_split(self):
        """split_k=-1: the most splits with tiles x splits <= cus."""
        from rocke.instances.gfx950.decode_gemm import resolved_spec

        auto = _split(split_k=-1)
        self.assertEqual(resolved_spec(auto).split_k, 7)  # 36 tiles on 256 CUs
        self.assertEqual(resolved_spec(replace(auto, cus=128)).split_k, 3)
        self.assertEqual(resolved_spec(replace(auto, cus=16)).split_k, 1)
        self.assertEqual(decode_gemm_launch(replace(auto, cus=128))["grid"][0], 108)
        self.assertEqual(skinny_plan(replace(auto, cus=128))["grid"], 108)
        name = auto.kernel_name()
        self.assertEqual(
            _lower(build_decode_gemm(auto)).replace(name, "K"),
            _lower(build_decode_gemm(_split())).replace(_split().kernel_name(), "K"),
        )
        # a private ring has no split-K
        self.assertFalse(is_valid_spec(_skinny(1, 2304, 8192, block_n=16, k_waves=8, split_k=-1))[0])

    def test_smallm_sync_guard(self):
        """sk_sync='auto' takes the xcd protocol only when wg_order='tile'
        co-locates the partners on xccs XCCs (36 row blocks: 4, not 8)."""
        from rocke.instances.gfx950.decode_gemm import resolved_spec

        spec = DecodeGemmSpec(16, 2304, 8192, n_waves=4, split_k=4, wg_order="tile", sk_sync="auto")
        self.assertEqual(smallm_plan(spec)["n_tiles"], 36)
        self.assertEqual(resolved_spec(spec).sk_sync, "relaxed")
        self.assertEqual(resolved_spec(replace(spec, xccs=4)).sk_sync, "xcd")
        self.assertEqual(resolved_spec(replace(spec, xccs=4, wg_order="split")).sk_sync, "relaxed")
        for s in (spec, replace(spec, xccs=4)):
            self.assertEqual(is_valid_spec(s), (True, "ok"))
            self.assertNotEqual(_lower(build_decode_gemm(s)), "")
        rr = replace(spec, sk_sync="xcdrr")
        self.assertFalse(is_valid_spec(rr)[0])
        self.assertTrue(is_valid_spec(replace(rr, xccs=4))[0])

    def test_shared_finish_grid_bound(self):
        spec = DecodeGemmSpec(
            16, 2304, 8192, n_waves=4, split_k=4, reduce="slots", sk_sync="relaxed", sk_finish="shared"
        )
        self.assertEqual(smallm_plan(spec)["grid"], 144)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        self.assertFalse(is_valid_spec(replace(spec, cus=128))[0])

    def test_default_spec_records_the_device(self):
        for M, N, K in PROBLEMS:
            spec = default_spec(M, N, K, cus=128, xccs=4)
            self.assertEqual(spec.cus, 128)
            if spec.family != "gemv":
                self.assertEqual(spec.xccs, 4)
            self.assertTrue(is_valid_spec(spec)[0], spec)


if __name__ == "__main__":
    unittest.main()
