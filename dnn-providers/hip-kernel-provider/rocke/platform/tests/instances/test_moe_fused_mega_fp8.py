# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU spec-validation tests for the fp8 single-launch MoE mega-kernel.

These pin the three ways an unbuildable fp8 mega spec must fail at build time
instead of inside comgr or at kernel load: an MFMA atom the arch's MMA catalog
does not list, a wave32 arch, and a tiling whose total LDS overruns the per-WG
budget; and that the hidden activation is requantized to the largest value of
the target's e4m3 encoding. On-device numeric verification lives under
``examples/gfx950``.
"""

from __future__ import annotations

import re
import unittest

import pytest

# Module level only because the parametrize below needs it at collection; the
# tests import the rest of rocke inside their bodies.
from rocke.tests.engines import ENGINES


def _spec(**kw):
    from rocke.instances.common.moe_fused_mega_fp8 import FusedMegaKernelSpecFp8

    kw.setdefault("name", "mega_fp8")
    return FusedMegaKernelSpecFp8(**kw)


class TestMoeFusedMegaFp8ArchGuards(unittest.TestCase):
    def test_rejects_hero_atom_on_gfx942(self):
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # The K=128 hero atom is the gfx950 fp8 16x16x128 MFMA, which the gfx942
        # MMA catalog does not list. The catalog guard refuses it before the
        # instruction can reach comgr as an uncatchable LLVM abort.
        with self.assertRaises(NotImplementedError) as cm:
            build_moe_fused_mega_gemm_fp8(_spec(), arch="gfx942")
        msg = str(cm.exception)
        self.assertIn("(fp8e4m3 16x16x128) is not in the gfx942 MMA catalog", msg)

    def test_rejects_hero_down_atom_on_gfx942(self):
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # down_k selects independently of gate_up_k, so a K=128 down atom under
        # a catalog gate/up atom is refused the same way.
        with self.assertRaises(NotImplementedError) as cm:
            build_moe_fused_mega_gemm_fp8(
                _spec(gate_up_k=32, down_k=128), arch="gfx942"
            )
        self.assertIn(
            "(fp8e4m3 16x16x128) is not in the gfx942 MMA catalog", str(cm.exception)
        )

    def test_accepts_catalog_atom_on_gfx942(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # The legacy K=32 path IS a gfx942 catalog shape, so it must still build.
        kd = build_moe_fused_mega_gemm_fp8(
            _spec(gate_up_k=32, down_k=32), arch="gfx942"
        )
        self.assertIn("atomicrmw", lower_kernel_to_llvm(kd, arch="gfx942"))

    def test_rejects_wave32_target(self):
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # Every lane map here is wave64 (the amax butterfly is a hardcoded
        # 6-stage xor over lanes 1..32), so a wave32 target is silently wrong.
        for arch in ("gfx1250", "gfx1151"):
            with self.subTest(arch=arch):
                with self.assertRaises(ValueError) as cm:
                    build_moe_fused_mega_gemm_fp8(_spec(), arch=arch)
                self.assertIn(
                    f"spec wave_size 64 != {arch} wave_size 32", str(cm.exception)
                )

    def test_gfx950_default_builds_through_the_catalog_guard(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.moe_fused_mega_fp8 import (
            build_moe_fused_mega_gemm_fp8,
        )

        # The K=128 hero atom is a gfx950 MMA catalog row, so the catalog guard
        # accepts the default spec there.
        ll = lower_kernel_to_llvm(
            build_moe_fused_mega_gemm_fp8(_spec(), arch="gfx950"), arch="gfx950"
        )
        self.assertIn("mfma.scale.f32.16x16x128.f8f6f4", ll)
        self.assertIn("atomicrmw", ll)


class TestMoeFusedMegaFp8LdsBudget(unittest.TestCase):
    def test_accounting_matches_emitted_pool(self):
        # ``BStage_smem`` is allocated unconditionally but only referenced under
        # ``use_dtla``; the packer dead-strips it otherwise, so the accounting
        # has to track that or it over-counts 32 KiB.
        from rocke.core import lower_llvm
        from rocke.instances.common._moe_fused_mega_lds import mega_lds_pool_bytes
        from rocke.instances.common.moe_fused_mega_fp8 import (
            _lds_allocs,
            build_moe_fused_mega_gemm_fp8,
        )

        cases = (
            {},
            {"use_dtla": False},
            {"gate_up_k": 32, "down_k": 32},
            {"tile_m": 32},
            {"tile_n_inter": 512, "warp_n": 4},
        )
        for kw in cases:
            with self.subTest(**kw):
                spec = _spec(**kw)
                kd = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
                low = lower_llvm._Lowerer(kd, arch="gfx950")
                low._collect_smem(kd.body)
                low._compute_smem_layout()
                self.assertEqual(
                    mega_lds_pool_bytes(_lds_allocs(spec)), low._smem_pool_size
                )

    def test_rejects_over_budget_spec(self):
        from rocke.core.arch import ArchTarget
        from rocke.instances.common._moe_fused_mega_lds import mega_lds_pool_bytes
        from rocke.instances.common.moe_fused_mega_fp8 import (
            _lds_allocs,
            build_moe_fused_mega_gemm_fp8,
        )

        # Widening the inter slice scales Hidden + the f32 amax scratch + the
        # DTLA landing zone together, past CDNA4's 160 KiB.
        spec = _spec(tile_m=32, tile_n_inter=2048, warp_n=16)
        total = mega_lds_pool_bytes(_lds_allocs(spec))
        cap = ArchTarget.from_gfx("gfx950").lds_capacity_bytes
        self.assertGreater(total, cap)
        with self.assertRaises(ValueError) as cm:
            build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
        msg = str(cm.exception)
        self.assertIn(str(total), msg)
        self.assertIn(str(cap), msg)
        self.assertIn("Hidden_smem", msg)


# (arch, gate/up and down MFMA K, the target's e4m3 bound, the other target's)
_REQUANT_BOUNDS = [("gfx942", 32, 240.0, 448.0), ("gfx950", 128, 448.0, 240.0)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "arch, atom_k, bound, other",
    _REQUANT_BOUNDS,
    ids=[arch for arch, *_ in _REQUANT_BOUNDS],
)
def test_requantizes_to_the_largest_value_of_the_target_encoding(
    arch, atom_k, bound, other, engine
):
    """The block scale is ``amax / bound``, so the block's largest value
    converts to the bound: 240 for gfx942's e4m3 FNUZ (448 would convert to
    NaN there; the conversion does not saturate), 448 for gfx950's OCP e4m3.
    Checked in the Python builder's kernel and in the C++ builder's, which
    emits the same IR."""
    from rocke.core.lower_llvm import llvm_f32_literal
    from rocke.instances.common.moe_fused_mega_fp8 import (
        build_moe_fused_mega_gemm_fp8,
    )
    from rocke.tests.engines import instance_llvm_from_engine

    def reciprocal_of(value):
        return f"= fdiv float {llvm_f32_literal(1.0)}, {llvm_f32_literal(value)}\n"

    text = instance_llvm_from_engine(
        _spec(gate_up_k=atom_k, down_k=atom_k),
        arch,
        engine=engine,
        build_kernel=build_moe_fused_mega_gemm_fp8,
        binding="moe_fused_mega_fp8_lower_llvm",
    )
    assert text.count(reciprocal_of(bound)) == 1
    assert reciprocal_of(other) not in text


@pytest.mark.parametrize("cadence", ["iglp1", "none", "sgb"])
def test_compiles_in_a_fresh_process_exactly_when_it_emits_iglp_opt_1(cadence):
    """A kernel that emits ``iglp_opt(1)`` is marked to compile in the fresh,
    memory-capped process (on a compiler without the fix the hint can
    allocate without bound); a kernel without it compiles in-process."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.helpers.compile import COMGR_FRESH_PROCESS_ATTR
    from rocke.instances.common.moe_fused_mega_fp8 import (
        build_moe_fused_mega_gemm_fp8,
    )

    kernel = build_moe_fused_mega_gemm_fp8(_spec(sched_cadence=cadence), "gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    emits_hint = "call void @llvm.amdgcn.iglp.opt(i32 1)" in text
    assert emits_hint is (cadence == "iglp1")
    assert kernel.attrs.get(COMGR_FRESH_PROCESS_ATTR, False) is emits_hint


# (arch, spec fields, whether the gate/up loop stages its weights through LDS)
_HINT_CASES = [
    ("gfx950", {}, True),
    ("gfx950", {"use_dtla": False}, False),
    # direct-to-LDS needs the K=128 atom: with K=32 the gate/up loop skips it
    ("gfx950", {"gate_up_k": 32, "down_k": 32}, False),
    ("gfx942", {"gate_up_k": 32, "down_k": 32}, False),
]
_HINT_CASE_IDS = ["gfx950-dtla", "gfx950-no-dtla", "gfx950-k32", "gfx942-k32"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "arch, fields, stages_through_lds", _HINT_CASES, ids=_HINT_CASE_IDS
)
def test_iglp_opt_1_sits_only_in_loop_bodies_that_read_lds(
    arch, fields, stages_through_lds, engine
):
    """Every block that carries ``iglp_opt(1)`` also reads LDS. On a compiler
    older than ROCm 7.11 the hint runs the compile out of memory in a
    scheduling region with fewer than 4 LDS reads; a gate/up loop without LDS
    staging has none. So the gate/up loop (induction variable ``kg_*``)
    carries the hint only when it stages its weights through LDS; every down
    loop (``dg_*``) carries it. This checks the IR before the LLVM middle end;
    whether every region then has 4 reads is what
    ``test_every_hint_configuration_compiles`` exercises on such a compiler.
    Checked in the Python builder's kernel and in the C++ builder's, which
    emits the same IR."""
    from rocke.instances.common.moe_fused_mega_fp8 import (
        build_moe_fused_mega_gemm_fp8,
    )
    from rocke.tests.engines import instance_llvm_from_engine

    text = instance_llvm_from_engine(
        _spec(sched_cadence="iglp1", **fields),
        arch,
        engine=engine,
        build_kernel=build_moe_fused_mega_gemm_fp8,
        binding="moe_fused_mega_fp8_lower_llvm",
    )
    hinted = [
        block
        for block in re.split(r"\n(?=[\w.]+:)", text)
        if "call void @llvm.amdgcn.iglp.opt(i32 1)" in block
    ]
    assert all(re.search(r"= load .*ptr addrspace\(3\)", b) for b in hinted)
    gate_up = [block for block in hinted if "%kg_" in block]
    down = [block for block in hinted if "%dg_" in block]
    assert len(gate_up) == int(stages_through_lds)
    assert down and len(gate_up) + len(down) == len(hinted)


@pytest.mark.parametrize(
    "arch, fields",
    [(arch, fields) for arch, fields, _ in _HINT_CASES],
    ids=_HINT_CASE_IDS,
)
def test_every_hint_configuration_compiles(arch, fields):
    """Each configuration compiles with the loaded compiler, in the
    memory-capped fresh process its ``iglp_opt(1)`` asks for. On a compiler
    without LLVM commit 576e1affab35 the configurations without LDS staging
    ran out of memory while their gate/up loop carried the hint; on a fixed
    compiler this checks that the cap leaves room for the kernel."""
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common.moe_fused_mega_fp8 import (
        build_moe_fused_mega_gemm_fp8,
    )
    from rocke.tests.comgr import require_comgr_target

    require_comgr_target(arch)
    kernel = build_moe_fused_mega_gemm_fp8(_spec(sched_cadence="iglp1", **fields), arch)
    artifact = compile_kernel(kernel, arch=arch, backend="python")
    assert artifact.hsaco[:4] == b"\x7fELF"


if __name__ == "__main__":
    unittest.main()
