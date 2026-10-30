# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ``vgpr_prefetch`` (gfx942 register-staged) body with the fused operand
prologue, the complex-B embedding, the batched form and 64-bit tile origins.

CPU checks: the validity scope (f32 K atoms, release atoms, batched,
``lds_swizzle_l``, ``wide_addr``, ``split_mode``), the launch rules
(``launch_shape_error`` / ``batched_launch_error``), the emitted IR of the new
paths (fragment read width, complex-B row addressing, batch / 64-bit bases,
the bit-level split) and the Python-engine-only refusal of the C++ converter.
A device test (skipped without a gfx942 GPU) runs every recipe, the complex
forms and the batched / 64-bit forms against an f64 reference.
"""

from __future__ import annotations

import re
import unittest
from dataclasses import replace

import pytest

from rocke.core import backend
from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    batched_launch_error,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
)

_X6 = dict(out_f32=True, src="f32", recipe="x6", split_at="use", psub=True)
_GEO = (128, 128, 64, 2, 2, 1, 16, 16, 16)


def _spec(geo=_GEO, mode=4, batched=False, **trait):
    base = dict(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        ab_double_buffer=mode != 4,
        chiplet_swizzle=True,
        lds_swizzle=True,
        lds_swizzle_r=0,
        vgpr_prefetch=mode,
    )
    base.update(trait)
    return UniversalGemmSpec(
        name="vpp",
        tile=TileSpec(*geo),
        trait=TraitSpec(**base),
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        batched=batched,
    )


def _pro(spec, **kw):
    return attach_prologue(spec, Prologue(**dict(_X6, **kw)))


def _ir(spec, arch="gfx942"):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(build_universal_gemm(spec, arch=arch), arch=arch)


def _lds_loads(ll):
    return re.findall(r"= load (<\d+ x \w+>), ptr addrspace\(3\)", ll)


class TestValidity(unittest.TestCase):
    def test_every_recipe_and_body_admitted(self):
        for recipe in ("cvt", "x3", "x4", "x6", "x9"):
            for mode, extra in (
                (3, {}),
                (4, {}),
                (4, dict(vp_unroll=1)),
                (4, dict(vp_tail_pack=3, vp_pair_reads=1, vp_mfma_first=1)),
                (1, {}),
                (2, {}),
            ):
                s = _pro(_spec(mode=mode, **extra), recipe=recipe)
                with self.subTest(recipe=recipe, mode=mode, extra=extra):
                    self.assertEqual(is_valid_spec(s, arch="gfx942"), (True, "ok"))

    def test_f32_k_atoms(self):
        # tile_k 32 of a 16x16x16 atom: one f32 K atom -> modes 3 / 4 refused
        g32 = (128, 128, 32, 2, 2, 1, 16, 16, 16)
        self.assertTrue(is_valid_spec(_spec(g32, mode=4), arch="gfx942")[0])
        self.assertFalse(is_valid_spec(_pro(_spec(g32, mode=4)), arch="gfx942")[0])
        # vp_release_atoms counts the f32 atoms: tile_k 64 holds 2 -> R must be 1
        s = _spec(vp_release_atoms=2)
        self.assertTrue(is_valid_spec(s, arch="gfx942")[0])
        self.assertFalse(
            is_valid_spec(_pro(_spec(vp_release_atoms=2)), arch="gfx942")[0]
        )
        g128 = (128, 128, 128, 2, 4, 1, 16, 16, 16)
        self.assertTrue(
            is_valid_spec(_pro(_spec(g128, vp_release_atoms=3)), arch="gfx942")[0]
        )

    def test_batched(self):
        self.assertTrue(is_valid_spec(_spec(batched=True), arch="gfx942")[0])
        self.assertTrue(is_valid_spec(_pro(_spec(batched=True)), arch="gfx942")[0])
        for bad in (dict(split_k=2), dict(active_tile_skip=True)):
            with self.subTest(bad=bad):
                self.assertFalse(
                    is_valid_spec(_spec(batched=True, **bad), arch="gfx942")[0]
                )

    def test_lds_swizzle_l(self):
        self.assertTrue(is_valid_spec(_spec(lds_swizzle_l=3), arch="gfx942")[0])
        cases = {
            "no vgpr_prefetch": dict(
                vgpr_prefetch=0, ab_double_buffer=True, lds_swizzle_l=3
            ),
            "no swizzle": dict(lds_swizzle=False, lds_swizzle_r=-1, lds_swizzle_l=3),
            "range": dict(lds_swizzle_l=5),
        }
        for what, tr in cases.items():
            with self.subTest(what=what):
                self.assertFalse(is_valid_spec(_spec(**tr), arch="gfx942")[0])
        # one group slot per row: tile_k 16 = 2^4 (mode 1: no K-atom rule)
        g16 = (128, 128, 16, 2, 2, 1, 16, 16, 16)
        ok, why = is_valid_spec(_spec(g16, mode=1, lds_swizzle_l=4), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("lds_swizzle_l", why)
        self.assertTrue(
            is_valid_spec(_spec(g16, mode=1, lds_swizzle_l=3), arch="gfx942")[0]
        )

    def test_wide_addr(self):
        for b in (False, True):
            self.assertTrue(
                is_valid_spec(_pro(_spec(batched=b, wide_addr=True)), arch="gfx942")[0]
            )
        refused = {
            "split-K": dict(wide_addr=True, split_k=2),
            "cshuffle": dict(
                mode=3, wide_addr=True, epilogue="cshuffle", ab_double_buffer=True
            ),
        }
        for what, tr in refused.items():
            with self.subTest(what=what):
                self.assertFalse(is_valid_spec(_spec(**tr), arch="gfx942")[0])
        s = attach_prologue(_spec(wide_addr=True), Prologue(**dict(_X6, epi="alpha")))
        self.assertFalse(is_valid_spec(s, arch="gfx942")[0])

    def test_cplx_b_builds_on_vp(self):
        s = _pro(_spec(), cplx_b=True, conj_a=True, conj_b=True)
        self.assertTrue(is_valid_spec(s, arch="gfx942")[0])
        self.assertIn("_cxab", build_universal_gemm(s, arch="gfx942").name)

    def test_split_mode_check(self):
        with self.assertRaises(ValueError):
            Prologue(**dict(_X6, recipe="cvt", split_mode="rnef")).check()
        with self.assertRaises(ValueError):
            Prologue(out_f32=True, split_mode="trunc").check()
        with self.assertRaises(ValueError):
            Prologue(**dict(_X6, split_mode="rtz")).check()
        Prologue(**dict(_X6, recipe="x3", split_mode="trunc")).check()
        self.assertTrue(
            Prologue(**dict(_X6, split_mode="trunc")).suffix().endswith("_smt")
        )


class TestLaunchRules(unittest.TestCase):
    def test_wide_skips_operand_bytes(self):
        M, N, K = 33024, 256, 65536  # A: 4.3 GB of 16-bit halves
        s = _pro(_spec())
        self.assertIsNotNone(launch_shape_error(s, M, N, K))
        self.assertIsNone(launch_shape_error(_pro(_spec(wide_addr=True)), M, N, K))
        # a tile's span must still fit the DMA range
        self.assertIsNotNone(
            launch_shape_error(_pro(_spec(wide_addr=True)), 128, 128, 2**24)
        )

    def test_cplx_rows(self):
        s = _pro(_spec(), cplx_b=True)
        self.assertIsNotNone(launch_shape_error(s, 128, 130, 256))  # N % tile_n
        self.assertIsNone(launch_shape_error(s, 128, 256, 256))
        self.assertIsNotNone(launch_shape_error(s, 128, 256, 254))  # K % 4

    def test_batched_rules(self):
        s = _pro(_spec(batched=True))
        M, N, K = 1024, 1024, 4096
        self.assertIsNone(batched_launch_error(s, M, N, K, 4, M * K, N * K, M * N))
        self.assertIsNotNone(
            batched_launch_error(s, M, N, K, 70000, M * K, N * K, M * N)
        )
        # (L - 1) * stride_a + M * K past 2^31 elements
        self.assertIsNotNone(batched_launch_error(s, M, N, K, 600, M * K, N * K, M * N))
        w = _pro(_spec(batched=True, wide_addr=True))
        self.assertIsNone(batched_launch_error(w, M, N, K, 600, M * K, N * K, M * N))
        self.assertIsNotNone(batched_launch_error(_pro(_spec()), M, N, K, 2, 0, 0, 0))


class TestEmission(unittest.TestCase):
    def test_f32_fragments_read_per_swizzle_group(self):
        # 16x16x16 atom, L = 2: an f32 fragment is two 4-half groups -> 2 x <4 x bf16>
        h16 = attach_prologue(_spec(), Prologue(out_f32=True))
        x6 = _pro(_spec())
        n16 = len(_lds_loads(_ir(h16)))
        l6 = _lds_loads(_ir(x6))
        self.assertEqual(set(l6), {"<4 x bfloat>"})
        self.assertEqual(len(l6), n16)  # twice as wide, half the K atoms
        # lds_swizzle_l = 3: one <8 x bf16> read per fragment, one 16-byte write per vector
        l3 = _lds_loads(_ir(_pro(_spec(lds_swizzle_l=3))))
        self.assertEqual(set(l3), {"<8 x bfloat>"})
        self.assertEqual(len(l3), n16 // 2)

    def test_modes_1_2_build(self):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                self.assertIn("vp", _ir(_pro(_spec(mode=mode))))

    def test_cplx_b_rows(self):
        def halvings(ll):
            return len(re.findall(r"sdiv i32 [^,]+, 2\b", ll))

        for pad_n in (False, True):
            plain = _ir(_pro(_spec(pad_n=pad_n)))
            cx = _ir(_pro(_spec(pad_n=pad_n), cplx_b=True))
            # complex row = B' row / 2: the tile's base row (block_n_off / 2), the
            # operand's rows (N / 2) and each of the 4 loaded rows per thread
            # (128 x 64 B tile / 8-half vectors / 256 threads)
            with self.subTest(pad_n=pad_n):
                self.assertEqual(halvings(cx) - halvings(plain), 2 + 4)
                self.assertIn("sdiv i32 %N, 2", cx)
                self.assertGreater(cx.count("select i1"), plain.count("select i1"))

    def test_batched_and_wide_bases(self):
        bat = _ir(_pro(_spec(batched=True)))
        for nm in ("stride_a", "stride_b", "stride_c"):
            # used in a batch-offset multiply, not only declared
            self.assertRegex(bat, rf"= mul (nsw )?i32 [^\n]*%{nm}\b")
        wide = _ir(_pro(_spec(batched=True, wide_addr=True)))
        self.assertIn("zext i32 %stride_a to i64", wide)
        self.assertIn("zext i32 %stride_c to i64", wide)
        self.assertIn(
            "_w64",
            build_universal_gemm(_pro(_spec(wide_addr=True)), arch="gfx942").name,
        )

    def test_split_mode_bits(self):
        rne = _ir(_pro(_spec()))
        tr = _ir(_pro(_spec(), split_mode="trunc"))
        rf = _ir(_pro(_spec(), split_mode="rnef"))
        self.assertIn("fptrunc", rne)
        self.assertNotIn("fptrunc", tr)
        self.assertNotIn("fptrunc", rf)
        self.assertIn("32767", rf)  # + 0x7FFF

    def test_defaults_unchanged(self):
        # the explicit group width equal to the geometry rule's (L = 2 for the
        # 4-half fragments of 16x16x16) emits the auto body (only the name differs)
        s = _spec()
        self.assertNotIn("swl", s.kernel_name())
        e = replace(s, trait=replace(s.trait, lds_swizzle_l=2))
        self.assertIn("swl2", e.kernel_name())
        a, b = _ir(s), _ir(e)
        self.assertEqual(a, b.replace(e.kernel_name(), s.kernel_name()))
        # and a different width changes the body
        f = replace(s, trait=replace(s.trait, lds_swizzle_l=3))
        self.assertNotEqual(a, _ir(f).replace(f.kernel_name(), s.kernel_name()))


class TestPhase2Rules(unittest.TestCase):
    """Rules added after review: swizzle group vs fragment, split-mode scope,
    stage="vgpr", the two-mode batch, i32 strides and the exact span bounds."""

    def test_swizzle_group_holds_a_fragment(self):
        # L = 1 splits the 4-half 16x16x16 fragment over two groups: refused with
        # and without a prologue (the plain read is one ds_read at one column)
        for pro in (None, "x6", "h16"):
            for mode in (1, 3, 4):
                s = _spec(mode=mode, lds_swizzle_l=1)
                if pro == "x6":
                    _pro(s)
                elif pro == "h16":
                    attach_prologue(s, Prologue(out_f32=True))
                with self.subTest(pro=pro, mode=mode):
                    ok, why = is_valid_spec(s, arch="gfx942")
                    self.assertFalse(ok)
                    self.assertIn("whole MFMA fragment", why)
        # 32x32x8: 4-half fragments too; L = 2 is the smallest admitted width
        g32 = (256, 256, 32, 2, 2, 1, 32, 32, 8)
        self.assertTrue(
            is_valid_spec(_spec(g32, mode=1, lds_swizzle_l=2), arch="gfx942")[0]
        )
        self.assertFalse(
            is_valid_spec(_spec(g32, mode=1, lds_swizzle_l=1), arch="gfx942")[0]
        )

    def test_split_mode_scope(self):
        for sm in ("trunc", "rnef"):
            with self.subTest(sm=sm):
                with self.assertRaises(ValueError):  # one part
                    Prologue(**dict(_X6, recipe="cvt", split_mode=sm)).check()
                Prologue(**dict(_X6, recipe="x3", split_mode=sm)).check()
                # f16 storage: the split would round to f16 -> refused, not ignored
                s = UniversalGemmSpec(
                    name="vpf",
                    tile=TileSpec(*_GEO),
                    trait=_spec().trait,
                    data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
                )
                _pro(s, split_mode=sm)
                ok, why = is_valid_spec(s, arch="gfx942")
                self.assertFalse(ok)
                self.assertIn("bf16", why)
                with self.assertRaises(ValueError):
                    build_universal_gemm(s, arch="gfx942")

    def test_rnef_keeps_nan_words(self):
        ll = _ir(_pro(_spec(), split_mode="rnef"))
        # (d & 0x7FFFFFFF) >= 0x7F800000 selects the unrounded word
        self.assertIn("and i32", ll)
        self.assertRegex(ll, r"icmp sge i32 [^\n]*, 2139095040")
        self.assertIn("2147483647", ll)
        # ... and the select is on the data path: one per split element more than trunc
        tr = _ir(_pro(_spec(), split_mode="trunc"))
        self.assertGreater(ll.count("select i1"), tr.count("select i1"))

    def test_stage_vgpr_refused_on_vp(self):
        s = attach_prologue(
            _spec(), Prologue(src="f32", recipe="cvt", out_f32=True, stage="vgpr")
        )
        ok, why = is_valid_spec(s, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("stage", why)
        with self.assertRaises(ValueError):
            build_universal_gemm(s, arch="gfx942")

    def test_pad_fast_tile_keeps_the_prologue(self):
        # the full-tile branch of pad_fast_tile used a copy of the spec without
        # the prologue: it rounded the f32 accumulators to bf16 into the f32 C
        for mode in (4, 1):
            s = attach_prologue(
                _spec(mode=mode, pad_m=True, pad_n=True, pad_fast_tile=True),
                Prologue(out_f32=True),
            )
            with self.subTest(mode=mode):
                self.assertTrue(is_valid_spec(s, arch="gfx942")[0])
                ll = _ir(s)
                self.assertNotIn("fptrunc", ll)
                self.assertNotIn("store bfloat", ll)

    def test_two_mode_batch_refused_on_vp(self):
        s = _pro(_spec(batched=True))
        object.__setattr__(s, "_batch2", True)
        ok, why = is_valid_spec(s, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("_batch2", why)

    def test_batched_bounds_exact(self):
        s = _pro(_spec(batched=True))
        M, N, K = 1024, 512, 1024  # tile multiples
        el_a, el_b, el_c = M * K, N * K, M * N
        # operand span (L - 1) * stride + rows * K: 2^31 - 1 admitted, 2^31 refused
        st = 2**31 - 1 - el_a
        self.assertIsNone(batched_launch_error(s, M, N, K, 2, st, el_b, el_c))
        self.assertIsNotNone(batched_launch_error(s, M, N, K, 2, st + 1, el_b, el_c))
        # C span (L - 1) * stride_c + M * N: 2^31 admitted, 2^31 + 1 refused
        stc = 2**31 - el_c
        self.assertIsNone(batched_launch_error(s, M, N, K, 2, el_a, el_b, stc))
        self.assertIsNotNone(batched_launch_error(s, M, N, K, 2, el_a, el_b, stc + 1))
        # i32 stride arguments, also in the 64-bit form
        w = _pro(_spec(batched=True, wide_addr=True))
        self.assertIsNone(batched_launch_error(w, M, N, K, 2, 2**31 - 1, el_b, el_c))
        for i in range(3):
            st3 = [el_a, el_b, el_c]
            st3[i] = 2**31
            with self.subTest(stride=i):
                self.assertIsNotNone(batched_launch_error(w, M, N, K, 2, *st3))
                self.assertIsNotNone(batched_launch_error(s, M, N, K, 2, *st3))
        # complex B: its N / 2 rows count for B's span
        c = _pro(_spec(batched=True), cplx_b=True)
        stb = 2**31 - 1 - (N // 2) * K
        self.assertIsNone(batched_launch_error(c, M, N, K, 2, el_a, stb, el_c))
        self.assertIsNotNone(batched_launch_error(s, M, N, K, 2, el_a, stb, el_c))

    def test_cplx_rows_in_operand_rule(self):
        # B' rows N * K * 2 = 2^32 bytes (refused), complex rows N / 2: 2^31 (admitted)
        M, N, K = 128, 2**17, 2**14
        self.assertIsNotNone(launch_shape_error(_pro(_spec()), M, N, K))
        self.assertIsNone(launch_shape_error(_pro(_spec(), cplx_b=True), M, N, K))


class TestConverters(unittest.TestCase):
    def test_converters_pass_the_mirrored_fields(self):
        # lds_swizzle_l and Prologue.split_mode are mirrored by the C++ engine
        # (gemm_spec.cpp / gemm_build_vp.cpp, gemm_prologue.cpp): converted
        self.assertNotIn("lds_swizzle_l", dict(backend._PYTHON_ONLY_GEMM_LEVERS))
        self.assertNotIn("split_mode", dict(backend._PYTHON_ONLY_PROLOGUE))
        s = _spec(vgpr_prefetch=0, ab_double_buffer=True, lds_swizzle_r=-1)
        d = backend.universal_gemm_spec_to_dict(
            replace(s, trait=replace(s.trait, lds_swizzle_l=3))
        )
        self.assertEqual(d["lds_swizzle_l"], 3)
        p = attach_prologue(
            _spec(vgpr_prefetch=0, ab_double_buffer=True, lds_swizzle_r=-1),
            Prologue(**dict(_X6, split_mode="trunc")),
        )
        self.assertEqual(
            backend.universal_gemm_spec_to_dict(p)["prologue"]["split_mode"], "trunc"
        )


# ------------------------------------------------------------------ device (gfx942)
def _device_arch():
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001 - no torch / no device
        return None


@pytest.mark.skipif(_device_arch() != "gfx942", reason="needs a gfx942 device")
@pytest.mark.parametrize(
    "recipe,cx,mode,batched,wide,extra",
    [
        ("x6", "plain", 4, False, False, {}),
        (
            "x6",
            "plain",
            4,
            True,
            False,
            dict(vp_tail_pack=3, vp_pair_reads=1, vp_mfma_first=1),
        ),
        ("x6", "cplx", 4, False, False, dict(vp_unroll=1)),
        ("x6", "conj", 3, True, False, {}),
        ("x3", "plain", 4, False, False, {}),
        ("x4", "cplx", 4, True, True, {}),
        ("x9", "plain", 4, False, False, dict(lds_swizzle_l=3)),
        ("cvt", "plain", 4, True, False, {}),
        ("h16", "plain", 4, True, True, {}),
        ("x6", "plain", 1, False, False, {}),
        ("x6", "plain", 4, False, False, dict(_split="trunc")),
        # ragged M / N / K (pad_m / pad_n / pad_k) with the batched, 64-bit and
        # complex forms (padded buffer ranges, the 64-bit epilogue's row bound)
        ("x6", "cplx", 4, True, True, dict(_ragged=True)),
        ("x6", "plain", 4, True, False, dict(_ragged=True)),
        ("h16", "plain", 4, False, True, dict(_ragged=True)),
        ("x3", "conj", 3, True, True, dict(_ragged=True)),
        # prologue + pad_fast_tile (the full-tile epilogue branch)
        ("h16", "plain", 4, False, False, dict(_ragged=True, pad_fast_tile=True)),
        ("x6", "cplx", 4, True, False, dict(_ragged=True, pad_fast_tile=True)),
    ],
)
def test_vp_prologue_device(recipe, cx, mode, batched, wide, extra):
    import torch

    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    extra = dict(extra)
    sm = extra.pop("_split", "rne")
    ragged = extra.pop("_ragged", False)
    if ragged:
        extra.update(pad_m=True, pad_n=True, pad_k=True)
    s = _spec(mode=mode, batched=batched, wide_addr=wide, **extra)
    if recipe == "h16":
        pro = Prologue(out_f32=True)
    else:
        kw = dict(_X6, recipe=recipe, split_mode=sm)
        if cx != "plain":
            kw["cplx_b"] = True
        if cx == "conj":
            kw.update(conj_a=True, conj_b=True)
        pro = Prologue(**kw)
    attach_prologue(s, pro)
    assert is_valid_spec(s, arch="gfx942")[0]
    kd = build_universal_gemm(s, arch="gfx942")
    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        if isinstance(prm.type, PtrType):
            sig.append(
                {
                    "name": prm.name,
                    "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>",
                }
            )
        else:
            sig.append({"name": prm.name, "type": str(nm)})
    art = compile_kernel(kd, arch="gfx942", capture_ir_text=False, backend="python")
    launch = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
    L = 2 if batched else 1
    f32 = recipe != "h16"
    # K in f32 words (16-bit elements for h16): 3 K tiles of the kernel's halves
    M, N, Kf = 256, 256, (96 if f32 else 192)
    if ragged:
        # rows / columns / K past the last whole tile (N even, K f32 even: complex)
        M, N, Kf = 200, 180, (84 if f32 else 168)
    Kh0 = 2 * Kf if f32 else Kf
    Nr0 = N // 2 if cx != "plain" else N
    assert launch_shape_error(s, M, N, Kh0) is None
    if batched:
        assert batched_launch_error(s, M, N, Kh0, L, M * Kh0, Nr0 * Kh0, M * N) is None
    g = torch.Generator(device="cuda")
    g.manual_seed(11)
    Nr = N // 2 if cx != "plain" else N
    A = torch.rand(L, M, Kf, device="cuda", generator=g) * 2 - 1
    B = torch.rand(L, Nr, Kf, device="cuda", generator=g) * 2 - 1
    if not f32:
        A, B = A.to(torch.bfloat16), B.to(torch.bfloat16)
    Kh = 2 * Kf if f32 else Kf
    D = torch.full((L, M, N), float("nan"), device="cuda")
    vals = {
        "A": A.view(torch.bfloat16) if f32 else A,
        "B": B.view(torch.bfloat16) if f32 else B,
        "C": D,
        "M": M,
        "N": N,
        "K": Kh,
        "stride_a": M * Kh,
        "stride_b": Nr * Kh,
        "stride_c": M * N,
    }
    launch(
        {p.name: vals[p.name] for p in kd.params},
        config=LaunchConfig(
            grid=(-(-N // s.tile.tile_n), -(-M // s.tile.tile_m), L),
            block=(s.block_size, 1, 1),
            fence=False,
        ),
    )
    torch.cuda.synchronize()
    A64, B64 = A.double(), B.double()
    if cx == "plain":
        ref = A64 @ B64.transpose(1, 2)
        mag = A64.abs() @ B64.abs().transpose(1, 2)
    else:
        Ac = torch.complex(A64[..., 0::2], A64[..., 1::2])
        Bc = torch.complex(B64[..., 0::2], B64[..., 1::2])
        if cx == "conj":
            Ac, Bc = Ac.conj(), Bc.conj()
        Dc = Ac @ Bc.transpose(1, 2)
        ref = torch.stack((Dc.real, Dc.imag), dim=-1).reshape(L, M, N)
        m = (A64[..., 0::2].abs() + A64[..., 1::2].abs()) @ (
            B64[..., 0::2].abs() + B64[..., 1::2].abs()
        ).transpose(1, 2)
        mag = torch.stack((m, m), dim=-1).reshape(L, M, N)
    rel = {"cvt": 2 * 2.0**-8, "x3": 2.0**-14, "x4": 2.0**-14}.get(
        recipe, 32 * 2.0**-24
    )
    if recipe == "h16":
        rel = 0.0
    tol = mag * (rel + 2 * Kf * 2.0**-24) + 1e-30
    err = ((D.double() - ref).abs() / tol).nan_to_num(nan=float("inf"))
    assert bool(torch.isfinite(D).all())
    assert float(err.max()) <= 1.0


@pytest.mark.skipif(_device_arch() != "gfx942", reason="needs a gfx942 device")
@pytest.mark.parametrize("split_mode", ["rne", "trunc", "rnef"])
@pytest.mark.parametrize("recipe", ["x3", "x6"])
def test_split_nan_encodings_device(split_mode, recipe):
    """NaN words (0xFFFFFFFF, 0x7FFF8000, a low-payload 0x7F800001) and +-Inf in A
    reach the output as NaN in every split mode; the other outputs stay finite."""
    import torch

    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    s = _pro(_spec(), recipe=recipe, split_mode=split_mode)
    kd = build_universal_gemm(s, arch="gfx942")
    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        if isinstance(prm.type, PtrType):
            sig.append(
                {
                    "name": prm.name,
                    "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>",
                }
            )
        else:
            sig.append({"name": prm.name, "type": str(nm)})
    art = compile_kernel(kd, arch="gfx942", capture_ir_text=False, backend="python")
    launch = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)
    M, N, Kf = 128, 128, 64
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    A = torch.rand(M, Kf, device="cuda", generator=g) * 2 - 1
    B = torch.rand(N, Kf, device="cuda", generator=g) * 2 - 1
    words = [-1, 0x7FFF8000, 0x7F800001, 0x7F800000, -8388608]  # NaN x3, +Inf, -Inf
    Ai = A.view(torch.int32)
    for r, w in enumerate(words):
        Ai[3 + 7 * r, 5 + r] = w
    D = torch.full((M, N), 0.0, device="cuda")
    vals = {"A": A.view(torch.bfloat16), "B": B.view(torch.bfloat16), "C": D}
    vals.update(M=M, N=N, K=2 * Kf)
    launch(
        {p.name: vals[p.name] for p in kd.params},
        config=LaunchConfig(grid=(1, 1, 1), block=(s.block_size, 1, 1), fence=False),
    )
    torch.cuda.synchronize()
    bad_rows = [3 + 7 * r for r in range(len(words))]
    for r in bad_rows:
        # every split recipe turns an Inf operand into NaN too (Inf - Inf residual)
        assert bool(torch.isnan(D[r]).all()), (split_mode, recipe, r)
    keep = torch.ones(M, dtype=torch.bool, device="cuda")
    keep[bad_rows] = False
    assert bool(torch.isfinite(D[keep]).all())
