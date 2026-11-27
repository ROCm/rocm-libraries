# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dropout of the gfx942 and gfx950 unified tiled 2D kernels, on the host.

The 16x16 bodies' PV A-operand map (rows and keys of each lane) runs through the
integer IR evaluator against the mask oracle (``rocke.numeric.philox_dropout``);
the transposed 32x32 bodies use the C-layout map the dense tests check. The spec
and problem refusals, the name, cache key, ABI, IR and the runner's launch
arguments are checked without a device.
"""

from __future__ import annotations

import dataclasses
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

from kernels.common import attention_unified as au
from kernels.common._attention_dropout import (
    DROPOUT_PARAMS,
    DropoutState,
    dropout_keep_run,
    dropout_query_counter,
    dropout_tile_counter,
)
from kernels.gfx942 import attention_tiled_2d as t2_942
from kernels.gfx950 import attention_tiled_2d as t2_950
from rocke.core.ir import I32, IRBuilder
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.numeric import philox_dropout as oracle
from rocke.tests.engines import lower_with_engine
from rocke.tests.lanes import BUILD_VARIANT, import_or_skip_until
from rocke.tests.host_arrays import HostArray
from rocke.tests.int_eval import evaluate_int_ops

M32 = 0xFFFFFFFF
_KERNELS = {"gfx942": t2_942, "gfx950": t2_950}
# The params the kernel appends for dropout, in ABI order.
PARAM_NAMES = [name for name, _ in DROPOUT_PARAMS] + [
    "dropout_max_seqlen_q",
    "dropout_max_seqlen_k",
]


class PvOperandCounterMapTest(unittest.TestCase):
    def test_a_operand_map_matches_oracle(self):
        # gfx942 has only the K=16 PV atom.
        for arch, pv_k_step in (("gfx950", 32), ("gfx950", 16), ("gfx942", 16)):
            with self.subTest(arch=arch, pv_k_step=pv_k_step):
                self._check(arch, pv_k_step)

    def _check(self, arch, pv_k_step):
        t2 = _KERNELS[arch]
        # Skv not a multiple of 4 or of the tile; the offset's low word wraps.
        B, Hq, Sq, Skv, p = 2, 3, 70, 203, 0.35
        seed, offset = 0x0123456789ABCDEF, (1 << 32) - 9
        T, ATOMS = 64, 2
        b = IRBuilder("drop_pv_map")
        names = ("seed_lo", "seed_hi", "off_lo", "off_hi", "thr", "z", "h", "q0")
        seed_lo, seed_hi, off_lo, off_hi, thr, z, h, q0 = (
            b.param(n, I32) for n in names
        )
        lane, tile = b.param("lane", I32), b.param("tile", I32)
        st = DropoutState(
            seed_lo, seed_hi, off_lo, off_hi, thr, scale=None, mad64=arch == "gfx950"
        )
        lane_rg = b.div(lane, b.const_i32(16))
        lane_col = b.mod(lane, b.const_i32(16))

        def slot_key(slot):
            if arch == "gfx942":
                return t2._pv_a_operand_key(b, lane_rg, slot)
            return t2._pv_a_operand_key(b, lane_rg, pv_k_step, slot)

        flat, rows = [], []
        for atom in range(ATOMS):
            # One query per A row: the test's row r is query q0 + r of head h.
            row = t2._pv_a_operand_row(b, lane_col, atom)
            rows.append(row)
            ctr = dropout_query_counter(
                b,
                st,
                batch=z,
                head=h,
                query=b.add(q0, row),
                num_heads=b.const_i32(Hq),
                seqlen_q=b.const_i32(Sq),
                seqlen_kv=b.const_i32(Skv),
            )
            tile_ctr = dropout_tile_counter(
                b, ctr, b.mul(tile, b.const_i32(T)), slot_key
            )
            for k in range(T // pv_k_step):
                for g in range(pv_k_step // 16):
                    flat += dropout_keep_run(
                        b,
                        st,
                        tile_ctr,
                        nsub=k,
                        g=g,
                        slot_key=slot_key,
                        sub_keys=pv_k_step,
                    )
        keys = [slot_key(slot) for slot in range(pv_k_step // 4)]
        mask = oracle.philox_keep_mask(B, Hq, Sq, Skv, p, seed, offset)
        checked = 0
        for zz, hh, qq in ((0, 0, 0), (1, 2, 37)):
            for ln in (0, 17, 46, 63):
                for t in range(-(-Skv // T)):
                    inputs = {
                        "seed_lo": seed & M32,
                        "seed_hi": seed >> 32,
                        "off_lo": offset & M32,
                        "off_hi": offset >> 32,
                        "thr": oracle.dropout_threshold(p) & M32,
                        "z": zz,
                        "h": hh,
                        "q0": qq,
                        "lane": ln,
                        "tile": t,
                    }
                    got = evaluate_int_ops(b, inputs, flat + rows + keys)
                    got_rows = got[len(flat) : len(flat) + ATOMS]
                    got_keys = got[len(flat) + ATOMS :]
                    i = 0
                    for atom in range(ATOMS):
                        for k in range(T // pv_k_step):
                            for slot in range(pv_k_step // 4):
                                key = t * T + k * pv_k_step + got_keys[slot]
                                flag = got[i]
                                i += 1
                                if key >= Skv:
                                    continue  # padded key: P is already 0 there
                                want = mask[zz, hh, qq + got_rows[atom], key]
                                self.assertEqual(
                                    bool(flag), bool(want), (zz, hh, ln, key)
                                )
                                checked += 1
        self.assertGreater(checked, 0)


def _problem(**kw):
    base = dict(
        total_q=361,
        num_seqs=3,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=128,
        block_size=16,
        max_seqlen_q=200,
        max_seqlen_k=700,
        dtype="bf16",
        num_kv_blocks=200,
        right_bound=-1,
    )
    base.update(kw)
    return au.UnifiedAttentionProblem(**base)


_GFX942_PREFILL = dict(
    total_q=1024,
    num_seqs=1,
    num_query_heads=32,
    num_kv_heads=8,
    block_size=64,
    max_seqlen_q=1024,
    max_seqlen_k=1024,
    num_kv_blocks=16,
)
# A problem per body: transposed 32x32, 16x16 with P in registers, 16x16 from
# LDS; on gfx942 also the transposed body with the sliced-K ring and the
# conflict-free V store, and the D256 register-PV body the D256 prefill rows
# take. gfx942 tiled 2D takes causal masks only.
_BODIES = {
    "gfx950": {
        "t32": dict(total_q=600, max_seqlen_q=300, max_seqlen_k=128),
        "regpv": {},
        "plds": dict(dtype="fp16"),
    },
    "gfx942": {
        "t32": dict(_GFX942_PREFILL, right_bound=0),
        "t32ring": dict(_GFX942_PREFILL, dtype="fp16", right_bound=0),
        "regpv": dict(right_bound=0),
        "regpv_d256": dict(
            head_size=256,
            total_q=2048,
            num_seqs=1,
            num_query_heads=16,
            max_seqlen_q=2048,
            max_seqlen_k=2048,
            num_kv_blocks=128,
            right_bound=0,
        ),
        "plds": dict(dtype="fp16", right_bound=0),
    },
}


def _body_of(spec):
    if spec.use_mfma_32x32 or getattr(spec, "use_mfma_32x32x8", False):
        return "t32ring" if getattr(spec, "use_k_sliced_ring", False) else "t32"
    if spec.use_register_pv:
        return "regpv_d256" if spec.head_size == 256 else "regpv"
    return "plds"


def _spec_for(arch, body, **kw):
    problem = _problem(**{**_BODIES[arch][body], **kw})
    spec = au._tiled_spec_from_problem(problem, arch)
    assert _body_of(spec) == body, (arch, body, _body_of(spec))
    return spec


def _arch_bodies():
    return [(arch, body) for arch in _BODIES for body in _BODIES[arch]]


class DropoutSpecTest(unittest.TestCase):
    def test_refusals(self):
        off = _spec_for("gfx950", "t32")
        cases = [
            (
                dict(use_transposed_qk_32x32=False),
                r"^dropout is not supported with the non-transposed 32x32 body$",
            ),
            (
                dict(use_grouped_kv2_softmax=True),
                r"^dropout is not supported with use_grouped_kv2_softmax$",
            ),
            (
                dict(kv_storage_dtype="fp8e4m3"),
                r"^dropout is not supported with fp8 KV storage$",
            ),
        ]
        for kw, msg in cases:
            with self.subTest(**kw):
                with self.assertRaisesRegex(ValueError, msg):
                    dataclasses.replace(off, dropout=True, **kw)
        with self.assertRaisesRegex(ValueError, "dropout must be a bool"):
            dataclasses.replace(off, dropout=1)
        # gfx942 needs no grouped-KV2 or fp8 dropout cases: its spec refuses both
        # with dropout off too.
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout is not supported with the non-transposed 32x32x8 body$",
        ):
            dataclasses.replace(
                _spec_for("gfx942", "t32"),
                use_transposed_qk_32x32=False,
                dropout=True,
            )

    def test_name_cache_key_and_abi(self):
        for arch, body in _arch_bodies():
            with self.subTest(arch=arch, body=body):
                self._check(arch, body)

    def _check(self, arch, body):
        t2 = _KERNELS[arch]
        on = _spec_for(arch, body, dropout=True, emit_lse=True)
        off = dataclasses.replace(on, dropout=False)
        self.assertNotIn("_do", off.kernel_name())
        self.assertRegex(on.kernel_name(), r"_do(_|$)")
        problem = _problem(**_BODIES[arch][body], emit_lse=True)
        self.assertNotEqual(
            au._tiled_cache_key(dataclasses.replace(problem, dropout=True), arch),
            au._tiled_cache_key(problem, arch),
        )
        n = len(PARAM_NAMES)
        sig_on = au._tiled_2d_signature(arch, "bf16", emit_lse=True, dropout=True)
        sig_off = au._tiled_2d_signature(arch, "bf16", emit_lse=True)
        self.assertEqual([a["name"] for a in sig_on[-n:]], PARAM_NAMES)
        self.assertEqual(sig_on[:-n], sig_off)
        kernel = t2.build_unified_attention_2d_tiled(on)
        self.assertEqual([p.name for p in kernel.params][-n:], PARAM_NAMES)
        ir_on = lower_kernel_to_llvm(kernel, arch=arch)
        ir_off = lower_kernel_to_llvm(
            t2.build_unified_attention_2d_tiled(off), arch=arch
        )
        self.assertNotIn("dropout", ir_off)
        # The mask reads the threshold and the epilogue the scale: a body that
        # lost either would leave the param unused.
        self.assertRegex(ir_on, r"icmp sgt i32 [^\n]*%dropout_threshold")
        self.assertRegex(ir_on, r"fmul [^\n]*%dropout_scale")
        # The row counters read the declared extents (declaration + a use).
        for name in ("dropout_max_seqlen_q", "dropout_max_seqlen_k"):
            self.assertGreaterEqual(ir_on.count(f"%{name}"), 2, name)


class DropoutCppEngineTest(unittest.TestCase):
    """The C++ builder serves these kernels without Python and must emit the
    same mask. Developer builds only: CI does not build the rocke_engine
    binding (its C++ check is the byte-identity gate's parity configs)."""

    def test_cpp_engine_emits_the_same_llvm_ir(self):
        engine = import_or_skip_until("rocke_engine", BUILD_VARIANT)
        # The gfx942 C twin has no sliced-K ring, and its transposed body
        # differs from Python on the dispatch's flash knobs with dropout off
        # too; the parity configs cover the transposed mask instead.
        cases = [
            (arch, body)
            for arch, body in _arch_bodies()
            if not (arch == "gfx942" and body.startswith("t32"))
        ]
        for arch, body in cases:
            with self.subTest(arch=arch, body=body):
                # The C++ spec has no right_bound: causal specs only.
                spec = _spec_for(arch, body, dropout=True, emit_lse=True, right_bound=0)
                expected = lower_with_engine(
                    _KERNELS[arch].build_unified_attention_2d_tiled(spec),
                    arch,
                    engine="python",
                )
                lower = getattr(engine, f"{arch}_attention_tiled_2d_lower_llvm")
                self.assertEqual(lower(asdict(spec), arch), expected)


class DropoutProblemTest(unittest.TestCase):
    """Paths without dropout refuse it rather than run without the mask."""

    def test_path_refusals(self):
        on = _problem(dropout=True)
        cases = [
            (
                au.supports_native_unified_attention_3d_tiled(on, "gfx950"),
                "dropout is not implemented on the split-KV attention path",
            ),
            (
                au.supports_native_unified_attention(on, "gfx950"),
                "dropout is not implemented on the scalar 2D attention path",
            ),
            (
                au.supports_native_unified_attention_tiled(on, "gfx1250"),
                "dropout is not implemented on gfx1250 tiled 2D attention",
            ),
            (
                au.supports_native_unified_attention_tiled(
                    _problem(dropout=True, use_fp8=True), "gfx950"
                ),
                "dropout is not implemented with FP8 K/V",
            ),
        ]
        for got, msg in cases:
            with self.subTest(msg=msg):
                self.assertEqual(got, (False, msg))
        for arch in _BODIES:
            with self.subTest(arch=arch):
                causal = _problem(dropout=True, right_bound=0)
                self.assertTrue(
                    au.supports_native_unified_attention_tiled(causal, arch)[0]
                )
        with self.assertRaisesRegex(ValueError, "dropout must be a bool"):
            _problem(dropout=1)

    def test_gfx942_4warp_builder_declines(self):
        # The 4-warp GQA builder has no dropout; its D256 cohort takes the
        # default tiled 2D builder instead.
        d256 = _problem(
            head_size=256,
            total_q=2048,
            num_seqs=1,
            num_query_heads=16,
            num_kv_heads=2,
            max_seqlen_q=2048,
            max_seqlen_k=2048,
            num_kv_blocks=128,
            right_bound=0,
        )
        self.assertIsNotNone(au._gfx942_4warp_route(d256, "gfx942"))
        on = dataclasses.replace(d256, dropout=True)
        self.assertIsNone(au._gfx942_4warp_route(on, "gfx942"))
        spec4 = dataclasses.replace(
            au._tiled_spec_from_problem(d256, "gfx942"), dropout=True
        )
        self.assertEqual(
            t2_942.supports_gfx942_4warp_gqa_spec(spec4),
            (False, au._GFX942_4WARP_DROPOUT_REFUSAL),
        )
        with self.assertRaisesRegex(
            NotImplementedError, "^dropout is not implemented on the gfx942 4-warp"
        ):
            t2_942.build_gfx942_4warp_gqa(spec4)
        self.assertEqual(au.select_auto_path(on, "gfx942"), "2d")
        spec = au._tiled_spec_from_problem(on, "gfx942")
        self.assertTrue(spec.dropout)


class DropoutLaunchBindingTest(unittest.TestCase):
    """The runner binds exactly the params of the kernel it builds."""

    def test_runner_binds_the_kernel_params(self):
        for arch, body in _arch_bodies():
            with self.subTest(arch=arch, body=body):
                self._check_binding(arch, body)

    def _check_binding(self, arch, body):
        problem = _problem(**_BODIES[arch][body], dropout=True, compile_backend="llvm")
        built, bound = [], []

        def compile_kernel(kernel, *, arch, **_):
            built.append(kernel)
            return SimpleNamespace(hsaco=b"", kernel_name=kernel.name)

        def launcher(*, signature, **_):
            names = [a["name"] for a in signature]
            return lambda vals, **__: bound.append((names, dict(vals)))

        hq, hkv, d = problem.num_query_heads, problem.num_kv_heads, problem.head_size
        dt = problem.dtype
        arrays = dict(
            q=HostArray((problem.total_q, hq, d), dt),
            k=HostArray((problem.num_kv_blocks, problem.block_size, hkv, d), dt),
            v=HostArray((problem.num_kv_blocks, problem.block_size, hkv, d), dt),
            out=HostArray((problem.total_q, hq, d), dt),
            cu_seqlens_q=HostArray((problem.num_seqs + 1,), "i32"),
            seqused_k=HostArray((problem.num_seqs,), "i32"),
            block_table=HostArray((problem.num_seqs, 64), "i32"),
        )
        with mock.patch.object(
            au, "_resolve_attention_arch", lambda: arch
        ), mock.patch.object(au, "compile_kernel", compile_kernel), mock.patch.object(
            au, "KernelLauncher", launcher
        ), mock.patch.object(
            au, "_2D_LAUNCHERS", {}
        ), mock.patch.object(
            au, "_ATTN_TILED_CACHE", {}
        ):
            with self.assertRaisesRegex(
                ValueError,
                "^dropout requires dropout_seed, dropout_offset and dropout_p$",
            ):
                au.run_unified_attention_torch(
                    problem=problem, softmax_scale=0.1, softcap=0.0, **arrays
                )
            au.run_unified_attention_torch(
                problem=problem,
                softmax_scale=0.1,
                softcap=0.0,
                dropout_seed=HostArray((1,), "i64"),
                dropout_offset=7,
                dropout_p=0.1,
                **arrays,
            )
        (kernel,) = built
        ((names, vals),) = bound
        self.assertEqual([p.name for p in kernel.params], names)
        self.assertLessEqual(set(names), set(vals))
        self.assertEqual({n for n in vals if n.startswith("dropout")}, set(PARAM_NAMES))
        self.assertEqual(vals["dropout_max_seqlen_q"], problem.max_seqlen_q)
        self.assertEqual(vals["dropout_max_seqlen_k"], problem.max_seqlen_k)


if __name__ == "__main__":
    unittest.main()
