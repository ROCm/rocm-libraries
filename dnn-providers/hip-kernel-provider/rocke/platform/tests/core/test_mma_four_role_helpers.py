# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression coverage for helpers that must distinguish MMA C from D."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rocke.core.arch import ArchTarget, LayoutMap, MmaCatalog, MmaOp
from rocke.core.ir import F32, I32, IRBuilder, VectorType
from rocke.core.isa.backend import Gfx11RdnaBackend
from rocke.helpers.atoms import (
    MfmaAtom,
    WmmaAtom,
    require_mma_recurrence,
    zero_mma_c,
)
from rocke.helpers.distribution import (
    WmmaTensor,
    require_wmma_recurrence,
    store_wmma_acc,
    wmma_mma,
)
from rocke.helpers.mfma_gemm_inner import (
    _require_recurrent_accumulator_contract,
    mfma_k_loop,
    mfma_k_loop_dynamic_K,
    validate_mfma_atom_in_catalog,
)
from rocke.instances.common.gemm_universal import _emit_zero_acc_op


class _Layout:
    def __init__(self, role: str):
        self.role = role
        self.slots = []

    def coord(self, _builder, _lane, slot):
        self.slots.append(slot)
        return slot, slot + 10


class _Atom:
    k = 1
    a_per_lane = 1
    b_per_lane = 1
    c_per_lane = 2
    d_per_lane = 3
    dtype_c = "i32"
    dtype_d = "f32"

    def __init__(self):
        self.c_map = _Layout("c")
        self.d_map = _Layout("d")

    def zero_acc(self, _builder):
        return "zero-c"

    def emit(self, _builder, _a, _b, _c):
        return "result-d"

    def a_layout(self, _arch):
        return _Layout("a")

    def b_layout(self, _arch):
        return _Layout("b")

    def c_layout(self, _arch):
        return self.c_map

    def d_layout(self, _arch):
        return self.d_map


class _StoreBuilder:
    def const_i32(self, value):
        return value

    def add(self, a, b):
        return a + b

    def vec_extract(self, value, slot):
        return value[slot]

    def cast_f32_to(self, value, _dtype):
        return value


class _Window:
    dtype = F32

    def __init__(self):
        self.stores = []

    def store_scalar(self, _builder, *indices, value, align=None):
        self.stores.append((indices, value, align))


class _EmitBlock:
    def __init__(self):
        self.lines = []

    def emit(self, line):
        self.lines.append(line)


class _Lowerer:
    def __init__(self):
        self.block = _EmitBlock()

    def _need(self, _key):
        pass

    def _operand(self, value):
        return value.name

    def _current(self):
        return self.block


class TestFourRoleAtomHelpers(unittest.TestCase):
    def test_catalog_validation_queries_c_and_d_independently(self):
        op = MmaOp(
            family="mma",
            a_dtype="fp16",
            b_dtype="fp16",
            c_dtype="i32",
            d_dtype="fp32",
            m=1,
            n=1,
            k=1,
            op_id="synthetic_mma",
        )
        target = SimpleNamespace(mma=MmaCatalog([op]))
        atom = SimpleNamespace(
            dtype_in="f16",
            dtype_c="i32",
            dtype_d="f32",
            m=1,
            n=1,
            k=1,
            name="synthetic_mma",
        )

        with patch("rocke.core.arch.ArchTarget.from_gfx", return_value=target):
            validate_mfma_atom_in_catalog(atom, "gfx-test", where="test")
            atom.dtype_c = "f32"
            with self.assertRaisesRegex(NotImplementedError, "not in the gfx-test"):
                validate_mfma_atom_in_catalog(atom, "gfx-test", where="test")

    def test_recurrent_mfma_helpers_reject_distinct_c_and_d(self):
        atom = _Atom()

        with self.assertRaisesRegex(ValueError, r"C=i32\[2\].*D=f32\[3\]"):
            mfma_k_loop(
                object(),
                K=1,
                atom=atom,
                load_a=lambda *_: None,
                load_b=lambda *_: None,
            )
        with self.assertRaisesRegex(ValueError, r"C=i32\[2\].*D=f32\[3\]"):
            mfma_k_loop_dynamic_K(
                object(),
                K_runtime=object(),
                atom=atom,
                load_a=lambda *_: None,
                load_b=lambda *_: None,
            )

    def test_equal_recurrent_contract_is_accepted(self):
        atom = SimpleNamespace(
            c_per_lane=4,
            d_per_lane=4,
            dtype_c="f32",
            dtype_d="f32",
        )
        _require_recurrent_accumulator_contract(atom, where="test")

    def test_wmma_recurrence_contract_rejects_distinct_c_and_d(self):
        atom = _Atom()

        with self.assertRaisesRegex(ValueError, r"test:.*C=i32\[2\].*D=f32\[3\]"):
            require_wmma_recurrence(atom, where="test")

        atom.c_per_lane = atom.d_per_lane
        atom.dtype_c = atom.dtype_d
        require_wmma_recurrence(atom, where="test")

    def test_wmma_loop_state_relabel_does_not_replace_builder_guard(self):
        atom = _Atom()
        carried_d_value = "loop-carried-d"

        # Attention builders serialize only the value and reconstruct the
        # wrapper as C in unpack(). That relabeling bypasses wmma_mma's
        # role-sensitive defense, so recurrent builders must validate the atom
        # once before constructing their scf.for_iter loop.
        relabeled = WmmaTensor(atom, "c", carried_d_value)
        self.assertEqual(relabeled.role, "c")
        with self.assertRaisesRegex(ValueError, r"attention:.*C=i32\[2\].*D=f32\[3\]"):
            require_wmma_recurrence(atom, where="attention")

    def test_gfx1151_attention_builders_validate_before_loop_unpack(self):
        from builders.gfx1151.attention import (
            fmha_blockn,
            fmha_multiwave,
            fmha_pipelined,
            fmha_regblocked,
            fmha_singlewave,
        )

        cases = (
            (
                fmha_blockn,
                fmha_blockn.build_wmma_fmha_blockn,
                fmha_blockn.BlockNCfg(head_size=64, num_query_heads=1),
            ),
            (
                fmha_multiwave,
                fmha_multiwave.build_wmma_fmha_multiwave,
                fmha_multiwave.MultiWaveCfg(head_size=64, num_query_heads=1),
            ),
            (
                fmha_pipelined,
                fmha_pipelined.build_wmma_fmha_pipelined,
                fmha_pipelined.PipelinedCfg(head_size=64, num_query_heads=1),
            ),
            (
                fmha_regblocked,
                fmha_regblocked.build_wmma_fmha_regblocked,
                fmha_regblocked.RegBlockedCfg(head_size=64, num_query_heads=1),
            ),
            (
                fmha_singlewave,
                fmha_singlewave.build_wmma_fmha_singlewave,
                fmha_singlewave.SingleWaveCfg(head_size=64, num_query_heads=1),
            ),
        )

        for module, build, cfg in cases:
            with self.subTest(module=module.__name__), patch.object(
                module.WmmaAtom, "f16_16x16x16", return_value=_Atom()
            ), self.assertRaisesRegex(ValueError, r"C=i32\[2\].*D=f32\[3\]"):
                build(cfg)

    def test_universal_gemm_zero_uses_c_and_rejects_incompatible_recurrence(self):
        builder = IRBuilder("four_role_universal_zero")
        equal = SimpleNamespace(
            c_dtype="i32", d_dtype="i32", c_frag_len=2, d_frag_len=2
        )
        zero = _emit_zero_acc_op(builder, equal)
        self.assertEqual(zero.type.count, 2)
        self.assertIs(zero.type.elem, I32)

        unequal = SimpleNamespace(
            c_dtype="i32", d_dtype="fp32", c_frag_len=2, d_frag_len=3
        )
        with self.assertRaisesRegex(ValueError, r"C=i32\[2\].*D=fp32\[3\]"):
            _emit_zero_acc_op(builder, unequal)

    def test_mma_op_recurrence_checks_layout_and_zero_uses_c(self):
        def shared_layout(*_args):
            return 0, 0

        equal = MmaOp(
            family="wmma",
            a_dtype="fp16",
            b_dtype="fp16",
            c_dtype="i32",
            d_dtype="i32",
            m=1,
            n=1,
            k=1,
            op_id="synthetic_equal",
            c_frag_len=2,
            d_frag_len=2,
            _c_layout=LayoutMap("c", 2, 32, shared_layout),
            _d_layout=LayoutMap("d", 2, 32, shared_layout),
        )
        require_mma_recurrence(equal, where="test")
        zero = zero_mma_c(IRBuilder("four_role_mma_op_zero"), equal)
        self.assertEqual(zero.type.count, 2)
        self.assertIs(zero.type.elem, I32)

        unequal_layout = MmaOp(
            **{
                **equal.__dict__,
                "op_id": "synthetic_unequal_layout",
                "_d_layout": LayoutMap("d", 2, 32, lambda *_args: (1, 1)),
            }
        )
        with self.assertRaisesRegex(ValueError, "C and D contracts differ"):
            require_mma_recurrence(unequal_layout, where="test")

    def test_deep_fused_recurrent_initializers_reject_layout_mismatch(self):
        from rocke.instances.common.deep_fused_conv_pool import (
            _zero_recurrent_mma_acc as common_zero,
        )
        from rocke.instances.gfx1151.deep_fused_conv_pool import (
            _zero_recurrent_mma_acc as gfx1151_zero,
        )

        def c_layout(*_args):
            return 0, 0

        op = MmaOp(
            family="wmma",
            a_dtype="fp16",
            b_dtype="fp16",
            c_dtype="fp32",
            d_dtype="fp32",
            m=16,
            n=16,
            k=16,
            op_id="synthetic_deep_fused_layout_mismatch",
            c_frag_len=8,
            d_frag_len=8,
            wave_size=32,
            _c_layout=LayoutMap("c", 8, 32, c_layout),
            _d_layout=LayoutMap("d", 8, 32, lambda *_args: (1, 1)),
        )

        for name, zero in (("common", common_zero), ("gfx1151", gfx1151_zero)):
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, rf"{name}:.*C and D contracts differ"
            ):
                zero(IRBuilder(f"deep_fused_{name}"), op, where=name)

    def test_all_generic_recurrent_resolvers_reject_layout_mismatch(self):
        from kernels.gfx1250 import _wmma_attention_common
        from rocke.helpers import mfma_attention
        from rocke.instances.common import (
            conv_implicit_gemm,
            conv_implicit_gemm_dgrad,
            conv_implicit_gemm_wgrad,
        )

        def c_layout(*_args):
            return 0, 0

        op = MmaOp(
            family="wmma",
            a_dtype="fp16",
            b_dtype="fp16",
            c_dtype="fp32",
            d_dtype="fp32",
            m=16,
            n=16,
            k=16,
            op_id="wmma_f32_16x16x16_f16",
            a_frag_len=16,
            b_frag_len=16,
            c_frag_len=8,
            d_frag_len=8,
            wave_size=32,
            _c_layout=LayoutMap("c", 8, 32, c_layout),
            _d_layout=LayoutMap("d", 8, 32, lambda *_args: (1, 1)),
        )
        target = SimpleNamespace(wave_size=32, mma=MmaCatalog([op]))
        spec = SimpleNamespace(
            data=SimpleNamespace(dtype_a="fp16", dtype_b="fp16"),
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        )

        resolvers = (
            (conv_implicit_gemm._resolve_conv_op, "implicit_gemm_conv"),
            (conv_implicit_gemm_dgrad._resolve_dgrad_op, "implicit_gemm_conv_dgrad"),
            (conv_implicit_gemm_wgrad._resolve_wgrad_op, "implicit_gemm_conv_wgrad"),
        )
        with patch("rocke.core.arch.ArchTarget.from_gfx", return_value=target):
            for resolve, where in resolvers:
                with self.subTest(where=where), self.assertRaisesRegex(
                    ValueError, rf"{where}:.*contracts differ"
                ):
                    resolve(spec, "gfx-test")

        gfx1250_op = MmaOp(
            **{
                **op.__dict__,
                "op_id": _wmma_attention_common.WMMA_OP_ID,
                "k": 32,
            }
        )
        gfx1250_target = SimpleNamespace(mma=MmaCatalog([gfx1250_op]))
        with patch(
            "rocke.core.arch.ArchTarget.from_gfx", return_value=gfx1250_target
        ), self.assertRaisesRegex(
            ValueError, "gfx1250 WMMA attention:.*contracts differ"
        ):
            _wmma_attention_common.resolve_wmma("gfx1250")

        with patch.object(
            mfma_attention, "require_mma_recurrence", side_effect=ValueError("guarded")
        ), self.assertRaisesRegex(ValueError, "guarded"):
            mfma_attention._wmma_attention_fwd_inner_body(
                object(),
                Q=None,
                K=None,
                V=None,
                O=None,
                head_size=16,
                seqlen_k=None,
                q_tile_base=None,
                head_idx=None,
                kv_head_idx=None,
                q_pos_base=None,
                stride_q_token=None,
                stride_q_head=None,
                stride_k_token=None,
                stride_k_head=None,
                stride_v_token=None,
                stride_v_head=None,
                stride_o_token=None,
                stride_o_head=None,
                scale_log2=None,
                dtype="f16",
                mask_mode="none",
                sliding_window=0,
                causal_ctx_offset=None,
                k_token_offset_elems=None,
                v_token_offset_elems=None,
                k_row_base_fn=None,
                v_row_base_fn=None,
                k_tile_start=None,
                k_tile_stop=None,
                extra_score_transform=None,
                extra_mask_predicate=None,
                extra_skip_predicate=None,
                k_block_iter_fn=None,
                v_scale=None,
                arch="gfx1151",
                target=ArchTarget.from_gfx("gfx1151"),
            )

    def test_zero_acc_uses_c_dtype_and_width(self):
        builder = IRBuilder("four_role_zero")
        mfma = MfmaAtom(1, 1, 1, 1, 1, 2, 5, "f16", "i32", "f32", "synthetic")
        wmma = WmmaAtom(1, 1, 1, 1, 1, 2, 5, "f16", "i32", "f32", "synthetic")

        for atom in (mfma, wmma):
            with self.subTest(atom=type(atom).__name__):
                zero = atom.zero_acc(builder)
                self.assertEqual(zero.type.count, 2)
                self.assertIs(zero.type.elem, I32)

    def test_wmma_tensor_transitions_from_c_to_d(self):
        atom = _Atom()
        c = WmmaTensor.zero_acc(object(), atom)
        self.assertEqual(c.role, "c")
        self.assertEqual(c.num_slots, 2)
        self.assertIs(c._layout(), atom.c_map)

        d = wmma_mma(
            object(),
            WmmaTensor(atom, "a", "a"),
            WmmaTensor(atom, "b", "b"),
            c,
        )
        self.assertEqual(d.role, "d")
        self.assertEqual(d.num_slots, 3)
        self.assertIs(d._layout(), atom.d_map)

        with self.assertRaisesRegex(ValueError, "C and D contracts differ"):
            wmma_mma(
                object(),
                WmmaTensor(atom, "a", "a"),
                WmmaTensor(atom, "b", "b"),
                d,
            )

    def test_store_uses_d_layout_and_d_slot_count(self):
        atom = _Atom()
        window = _Window()
        store_wmma_acc(_StoreBuilder(), window, atom, lane=0, acc=[1.0, 2.0, 3.0])

        self.assertEqual(atom.c_map.slots, [])
        self.assertEqual(atom.d_map.slots, [0, 1, 2])
        self.assertEqual(len(window.stores), 3)

    def test_wmma_lowering_types_c_argument_and_d_result_independently(self):
        lowerer = _Lowerer()
        op = SimpleNamespace(
            name="tile.wmma_f32_16x16x16_f16",
            operands=[
                SimpleNamespace(name="%a", type=VectorType(F32, 16)),
                SimpleNamespace(name="%b", type=VectorType(F32, 16)),
                SimpleNamespace(name="%c", type=VectorType(I32, 2)),
            ],
            result=SimpleNamespace(name="%d", type=VectorType(F32, 5)),
        )

        Gfx11RdnaBackend(ArchTarget.from_gfx("gfx1151")).emit_wmma(lowerer, op)

        call = lowerer.block.lines[-1]
        self.assertIn("%d = call <5 x float>", call)
        self.assertIn("<2 x i32> %c", call)


if __name__ == "__main__":
    unittest.main()
