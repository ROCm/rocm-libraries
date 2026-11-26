# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Philox dropout of the gfx942 and gfx950 dense kernels, on the host.

The lane-to-counter map of ``kernels.common._attention_dropout`` runs
through the integer IR evaluator against the SEM-015 oracle
(``rocke.numeric.philox_dropout``); the spec flag's refusals, name, ABI, IR and
launch arguments are checked without a device.
"""

from __future__ import annotations

import dataclasses
import re
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from kernels.common._attention_dropout import (
    DROPOUT_PARAMS,
    DropoutState,
    apply_dropout_tile,
    dropout_launch_args,
    dropout_query_counter,
)
from kernels.gfx942 import attention_dense as dense942
from kernels.gfx950 import attention_dense as dense950
from kernels.gfx942.attention_tiled_2d import _mfma_32x32_c_row as c_row942
from kernels.gfx950.attention_tiled_2d import _mfma_32x32_c_row as c_row950
from rocke.core.ir import I32, IRBuilder
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.numeric import philox_dropout as oracle
from rocke.tests.host_arrays import HostArray
from rocke.tests.int_eval import evaluate_int_ops

M32 = 0xFFFFFFFF
ROUND_FORMS = ((False, False), (False, True), (True, True))  # (xor3, mad64)
PARAM_NAMES = [name for name, _ in DROPOUT_PARAMS]
# The C-layout row map each dense body passes as ``slot_key``.
_C_ROW = {"gfx942": c_row942, "gfx950": c_row950}


class DropoutCounterMapTest(unittest.TestCase):
    def test_tile_map_matches_oracle(self):
        # Every Philox round form on each dense body's C layout; expected keys come
        # from that layout.
        for arch, c_row in _C_ROW.items():
            for xor3, mad64 in ROUND_FORMS:
                with self.subTest(arch=arch, xor3=xor3, mad64=mad64):
                    self._check_tile_map(c_row, xor3, mad64)

    def _check_tile_map(self, c_row, xor3, mad64):
        # Skv not a multiple of 4 or of the tile; the offset's low word wraps.
        B, Hq, Sq, Skv, p = 2, 3, 70, 203, 0.35
        seed, offset = 0x0123456789ABCDEF, (1 << 32) - 9
        BN, N_SUB, SUB_KEYS, SLOTS = 64, 2, 32, 16
        b = IRBuilder("drop_map")
        names = ("seed_lo", "seed_hi", "off_lo", "off_hi", "thr")
        seed_lo, seed_hi, off_lo, off_hi, thr = (b.param(n, I32) for n in names)
        z, h, q, lane, tile = (b.param(n, I32) for n in ("z", "h", "q", "lane", "tile"))
        st = DropoutState(
            seed_lo, seed_hi, off_lo, off_hi, thr, scale=None, xor3=xor3, mad64=mad64
        )
        row = dropout_query_counter(
            b,
            st,
            batch=z,
            head=h,
            query=q,
            num_heads=b.const_i32(Hq),
            seqlen_q=b.const_i32(Sq),
            seqlen_kv=b.const_i32(Skv),
        )
        layout = dict(
            tile_key0=b.mul(tile, b.const_i32(BN)),
            slot_key=lambda slot: c_row(b, lane, slot),
            sub_keys=SUB_KEYS,
        )
        one = b.const_f32(1.0)
        p_in = [[one] * SLOTS for _ in range(N_SUB)]
        p_out = apply_dropout_tile(b, st, row, p_in, **layout)
        slot_keys = [c_row(b, lane, slot) for slot in range(SLOTS)]
        flat = [v for sub in p_out for v in sub] + slot_keys
        mask = oracle.philox_keep_mask(B, Hq, Sq, Skv, p, seed, offset)
        checked = 0
        for zz in range(B):
            for hh in range(Hq):
                for qq in (0, 1, 31, Sq - 1):
                    for ln in (0, 33, 63):
                        for t in range((Skv + BN - 1) // BN):
                            inputs = {
                                "seed_lo": seed & M32,
                                "seed_hi": seed >> 32,
                                "off_lo": offset & M32,
                                "off_hi": offset >> 32,
                                "thr": oracle.dropout_threshold(p) & M32,
                                "z": zz,
                                "h": hh,
                                "q": qq,
                                "lane": ln,
                                "tile": t,
                            }
                            got = evaluate_int_ops(b, inputs, flat)
                            keys = got[N_SUB * SLOTS :]
                            for nsub in range(N_SUB):
                                for slot in range(SLOTS):
                                    key = t * BN + nsub * SUB_KEYS + keys[slot]
                                    if key >= Skv:
                                        continue  # padded key: P is already 0 there
                                    want = 1.0 if mask[zz, hh, qq, key] else 0.0
                                    self.assertEqual(
                                        got[nsub * SLOTS + slot],
                                        want,
                                        (zz, hh, qq, key),
                                    )
                                    checked += 1
        self.assertGreater(checked, 0)


_ARCHS = {
    "gfx942": (dense942, dense942.Gfx942AttentionDenseSpec),
    "gfx950": (dense950, dense950.Gfx950AttentionDenseSpec),
}


def _spec(arch, **kw):
    base = dict(
        batch=2,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=128,
        causal=False,
    )
    base.update(kw)
    return _ARCHS[arch][1](**base)


class DropoutSpecTest(unittest.TestCase):
    def test_refusals(self):
        for arch in _ARCHS:
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(
                    ValueError, r"^dropout is not supported with kv_segments > 1$"
                ):
                    _spec(arch, causal=True, dropout=True, kv_segments=2)
                with self.assertRaisesRegex(ValueError, "dropout must be a bool"):
                    _spec(arch, dropout=1)
        with self.assertRaisesRegex(
            ValueError, r"^dropout is not supported with fp8 KV storage$"
        ):
            _spec("gfx950", dropout=True, kv_storage_dtype="fp8e4m3")

    def test_training_layouts_admitted(self):
        # SEM-015 counts positions within a sequence over the declared extents,
        # which varlen and paged bodies already index by.
        _spec("gfx950", causal=True, varlen=True, dropout=True)
        for arch in _ARCHS:
            with self.subTest(arch=arch):
                _spec(
                    arch,
                    batch=1,
                    causal=True,
                    paged=True,
                    block_size=64,
                    num_kv_blocks=16,
                    dropout=True,
                )

    def test_name_and_abi(self):
        for arch, (mod, _) in _ARCHS.items():
            for persistent, head_size in ((False, 128), (True, 128), (False, 64)):
                with self.subTest(arch=arch, persistent=persistent, d=head_size):
                    self._check_name_and_abi(mod, arch, persistent, head_size)

    def _check_name_and_abi(self, mod, arch, persistent, head_size):
        off = _spec(arch, persistent=persistent, head_size=head_size)
        on = dataclasses.replace(off, dropout=True)
        self.assertNotIn("_do", off.kernel_name())
        self.assertRegex(on.kernel_name(), r"_do(_|$)")
        self.assertNotEqual(
            mod.attention_dense_cache_key(on, arch=arch),
            mod.attention_dense_cache_key(off, arch=arch),
        )
        sig = mod.attention_dense_signature(on)
        n = len(PARAM_NAMES)
        self.assertEqual([a["name"] for a in sig[-n:]], PARAM_NAMES)
        self.assertEqual(sig[:-n], mod.attention_dense_signature(off))
        kernel = mod.build_attention_dense(on)
        self.assertEqual([p.name for p in kernel.params][-n:], PARAM_NAMES)
        ir_on = lower_kernel_to_llvm(kernel, arch=arch)
        ir_off = lower_kernel_to_llvm(mod.build_attention_dense(off), arch=arch)
        self.assertNotIn("dropout", ir_off)
        # The mask reads the threshold and the epilogue the scale: a body that
        # lost either would leave the param unused.
        self.assertRegex(ir_on, r"icmp sgt i32 [^\n]*%dropout_threshold")
        self.assertRegex(ir_on, r"fmul [^\n]*%dropout_scale")
        # Seed and offset come from memory only behind a null check.
        for name in ("dropout_seed_ptr", "dropout_offset_ptr"):
            self.assertRegex(ir_on, rf"ptrtoint ptr addrspace\(1\) %{name} to i64")


class DropoutLaunchArgsTest(unittest.TestCase):
    def test_by_value_and_by_pointer(self):
        p = 0.2
        seed_arr = HostArray((1,), "i64")
        got = dropout_launch_args(
            True, dropout_seed=seed_arr, dropout_offset=(1 << 64) - 5, dropout_p=p
        )
        self.assertEqual(sorted(got), sorted(PARAM_NAMES))
        self.assertIs(got["dropout_seed_ptr"], seed_arr)
        self.assertIsNone(got["dropout_offset_ptr"])  # packed as a null pointer
        self.assertEqual(got["dropout_offset"], -5)  # the u64's i64 bits
        self.assertEqual(got["dropout_threshold"], oracle.dropout_threshold(p))
        self.assertEqual(got["dropout_scale"], oracle.dropout_scale(p))
        as_numpy = dropout_launch_args(
            True,
            dropout_seed=np.uint64(1 << 63),
            dropout_offset=np.int64(3),
            dropout_p=p,
        )
        self.assertEqual(
            (as_numpy["dropout_seed"], as_numpy["dropout_offset"]), (-(1 << 63), 3)
        )
        self.assertEqual(
            dropout_launch_args(
                False, dropout_seed=None, dropout_offset=None, dropout_p=None
            ),
            {},
        )

    def test_refusals(self):
        def call(enabled=True, seed=1, offset=2, p=0.1):
            return dropout_launch_args(
                enabled, dropout_seed=seed, dropout_offset=offset, dropout_p=p
            )

        cases = [
            (
                dict(enabled=False),
                r"^dropout_seed, dropout_offset, dropout_p provided but dropout is off$",
            ),
            (
                dict(offset=None),
                r"^dropout requires dropout_seed, dropout_offset and dropout_p$",
            ),
            (dict(p=0.0), r"^dropout_p must be in \(0, 1\), got 0\.0$"),
            (dict(p=1.0), r"^dropout_p must be in \(0, 1\), got 1\.0$"),
            (
                dict(seed=1 << 64),
                r"^dropout_seed=18446744073709551616 does not fit 64 bits$",
            ),
            (
                dict(seed=HostArray((1,), "i32")),
                re.escape(
                    "dropout_seed must be an int64 array of one element, got i32 of shape (1,)"
                ),
            ),
            (
                dict(offset=HostArray((2,), "i64")),
                re.escape(
                    "dropout_offset must be an int64 array of one element, got i64 of shape (2,)"
                ),
            ),
            (
                dict(offset=HostArray((1,), "i64", device=None)),
                r"^dropout_offset must be in device memory$",
            ),
        ]
        for kw, msg in cases:
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                with self.assertRaisesRegex(ValueError, msg):
                    call(**kw)
        with self.assertRaisesRegex(TypeError, "got bool"):
            call(seed=True)


class DropoutLaunchBindingTest(unittest.TestCase):
    """Each runner binds exactly the params of the kernel it builds."""

    def test_runner_binds_the_kernel_params(self):
        for arch, (mod, _) in _ARCHS.items():
            for persistent in (False, True):
                with self.subTest(arch=arch, persistent=persistent):
                    self._check_binding(mod, arch, persistent)

    def _check_binding(self, mod, arch, persistent):
        import rocke.helpers.compile as compile_mod
        import rocke.runtime as runtime_mod

        spec = _spec(arch, persistent=persistent, dropout=True)
        built, bound = [], []

        def compile_kernel(kernel, *, arch, **_):
            built.append(kernel)
            return SimpleNamespace(hsaco=b"", kernel_name=kernel.name)

        def launcher(*, signature, **_):
            names = [a["name"] for a in signature]
            return lambda vals, **__: bound.append((names, dict(vals)))

        b, s, hq, hkv, d = 2, 512, 8, 8, 128
        q = HostArray((b, s, hq, d), "bf16")
        kv = HostArray((b, s, hkv, d), "bf16")
        with mock.patch.object(
            compile_mod, "compile_kernel", compile_kernel
        ), mock.patch.object(
            runtime_mod, "KernelLauncher", launcher
        ), mock.patch.object(
            mod, "_DENSE_LAUNCHER_CACHE", {}
        ):
            mod.run_attention_dense_torch(
                spec=spec,
                q=q,
                k=kv,
                v=HostArray((b, s, hkv, d), "bf16"),
                out=HostArray((b, s, hq, d), "bf16"),
                scale=0.125,
                arch=arch,
                dropout_seed=HostArray((1,), "i64"),
                dropout_offset=7,
                dropout_p=0.1,
            )
        (kernel,) = built
        ((names, vals),) = bound
        self.assertEqual([p.name for p in kernel.params], names)
        self.assertLessEqual(set(names), set(vals))
        self.assertEqual({n for n in vals if n.startswith("dropout")}, set(PARAM_NAMES))


if __name__ == "__main__":
    unittest.main()
