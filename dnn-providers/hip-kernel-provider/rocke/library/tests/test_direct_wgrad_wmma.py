# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests for the wave32 WMMA dense direct wgrad (``conv_direct_wgrad_wmma``).

The spec validator and its reasons, the launch geometry (the run split and
the XCD-grouped 1-D grid), and the C++ engine against Python: validity,
reason, kernel name, launch geometry and LDS size over the whole gfx1250 AOT
grid and a set of rejected specs, and the serialized IR of a few specs (the
byte-identity gate covers the lowered IR of the parity configs). The numerics
live in ``test_direct_conv_correctness.py``.

Run:
    PYTHONPATH=rocke/platform/python:rocke/library python -m pytest \\
        rocke/library/tests/test_direct_wgrad_wmma.py -v
"""

from __future__ import annotations

import unittest

import pytest

from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.common.conv_direct_wgrad_wmma import (
    DirectWgradWmmaSpec,
    build_direct_wgrad_wmma,
    is_valid_direct_wgrad_wmma_spec,
)

# The shipped (AOT) knob set; waves and target_blocks vary per job.
_AOT = dict(
    tiles_k=2,
    tiles_c=2,
    stages=4,
    rows=2,
    prefetch=True,
    sched=True,
    pad=16,
    xcd=8,
    tdm=True,
)


def _problem(N=2, H=13, W=40, C=64, K=96, groups=1, KH=3, KW=None, PAD=1, **kw):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C // groups,
        kpg=K // groups,
        KH=KH,
        KW=KH if KW is None else KW,
        PAD=PAD,
        **{"stride": 1, "dtype": "bf16", **kw},
    )


def _reason(spec, arch="gfx1250"):
    return is_valid_direct_wgrad_wmma_spec(spec, arch)[1]


class TestWgradWmmaSpec(unittest.TestCase):
    def test_accepts_defaults_and_aot(self):
        self.assertEqual(_reason(DirectWgradWmmaSpec(problem=_problem())), "ok")
        spec = DirectWgradWmmaSpec(problem=_problem(), **_AOT)
        self.assertEqual(_reason(spec), "ok")
        self.assertEqual(spec.vgprs(), 416)

    def test_rejects(self):
        cases = (
            (dict(dil_h=2), {}, "no dilation"),
            (dict(groups=2), {}, "groups == 1 only"),
            (dict(dtype="fp32"), {}, "fp16/bf16 only"),
            (dict(stride=2), {}, "stride 1 only"),
            (dict(PAD=3), {}, "PAD <= K - 1"),
            (dict(KH=1, KW=3), {}, "PAD <= K - 1"),
            (dict(KH=3, KW=1), {}, "PAD <= K - 1"),
            # The kernel offsets columns by PAD: a different column padding
            # is refused.
            (dict(pad_w=0), {}, "one padding for both axes (got PAD=1x0)"),
            (dict(C=36), {}, "multiples of 8"),
            ({}, dict(waves_k=3, waves_c=1), "2, 4, 8 or 16 waves"),
            ({}, dict(tiles_k=3), "tiles_k / tiles_c"),
            ({}, dict(q_steps=3), "q_steps in (1, 2)"),
            ({}, dict(waves_k=2, waves_c=2, rows=3, tdm=True), "spread evenly"),
            ({}, dict(rows=0), "rows >= 1"),
            ({}, dict(rows=2, stages=2, prefetch=True), "stages >= 3"),
            ({}, dict(pad=4), "pad: a multiple of 8"),
            ({}, dict(stages=9), "stages in 2..8"),
            ({}, dict(target_blocks=0), "target_blocks >= 1"),
            ({}, dict(tiles_k=4, tiles_c=4), "VGPRs"),
            ({}, dict(waves_k=4, waves_c=4, rows=4, stages=8), "300 KiB"),
        )
        for pkw, knobs, needle in cases:
            with self.subTest(pkw=pkw, knobs=knobs):
                why = _reason(DirectWgradWmmaSpec(problem=_problem(**pkw), **knobs))
                self.assertIn(needle, why)
        # The arch gate reads ArchTarget capabilities, not the arch name.
        spec = DirectWgradWmmaSpec(problem=_problem())
        self.assertEqual(_reason(spec, "gfx950"), "wave32 kernel (got gfx950, wave64)")
        self.assertEqual(
            _reason(spec, "gfx1201"), "missing wmma 16x16x32 bf16 on gfx1201"
        )
        self.assertIn("gfx0", _reason(spec, "gfx0"))
        # An explicit column padding equal to PAD is the same problem.
        self.assertEqual(_reason(DirectWgradWmmaSpec(problem=_problem(pad_w=1))), "ok")

    def test_tdm_spreads_rows_over_waves(self):
        """2 * rows transfers go to waves 0 .. 2 * rows - 1 (mod the wave
        count): more transfers than waves must be a whole multiple."""
        for wk, wc, rows, ok in (
            (2, 1, 3, True),  # 6 transfers on 2 waves
            (2, 2, 3, False),  # 6 on 4
            (4, 4, 4, True),  # 8 on 16
            (2, 2, 4, True),  # 8 on 4
        ):
            spec = DirectWgradWmmaSpec(
                problem=_problem(),
                waves_k=wk,
                waves_c=wc,
                rows=rows,
                stages=3,
                tiles_k=1,
                tiles_c=1,
                tdm=True,
            )
            self.assertEqual(_reason(spec) == "ok", ok, (wk, wc, rows))

    def test_launch_geometry(self):
        # K 96 / block_k 128 -> 1 tile, C 64 / block_c 64 -> 1 tile; Wo 40
        # -> 2 column chunks; 2 images x (13 + 2) steps -> 60 steps.
        p = _problem()
        spec = DirectWgradWmmaSpec(problem=p, target_blocks=256)
        self.assertEqual((spec.n_k_tiles(), spec.n_c_tiles()), (1, 1))
        self.assertEqual(spec.total_steps(), 60)
        self.assertEqual(spec.n_split(), 60)  # capped by the step count
        self.assertEqual(spec.launch_geometry(), ((1, 1, 60), (256, 1, 1)))
        spec = DirectWgradWmmaSpec(problem=p, target_blocks=7)
        self.assertEqual(spec.launch_geometry(), ((1, 1, 7), (256, 1, 1)))
        # XCD grouping flattens the grid.
        big = _problem(N=4, H=56, W=56, C=256, K=256)
        spec = DirectWgradWmmaSpec(problem=big, **_AOT)
        planes = spec.n_k_tiles() * spec.n_c_tiles()
        self.assertEqual(planes, 4 * 2)
        # Nearest whole number of runs: (256 + 4) // 8 = 32.
        self.assertEqual(spec.n_split(), 32)
        self.assertEqual(spec.launch_geometry(), ((256, 1, 1), (256, 1, 1)))
        spec = DirectWgradWmmaSpec(problem=big, **{**_AOT, "target_blocks": 1})
        self.assertEqual(spec.n_split(), 1)

    def test_lds_bytes(self):
        spec = DirectWgradWmmaSpec(problem=_problem(), **_AOT)
        # X: (32 + 2 + PAD 1) cols x (64 + 16); dY: 32 x (128 + 16); 4 x 2 slots.
        per_stage = 35 * 80 + 32 * 144
        # TDM: only the dY row carries the 8-element dummy slot.
        self.assertEqual(spec.lds_bytes(), 4 * 2 * (per_stage + 8) * 2)

    def test_lds_bytes_match_the_allocations(self):
        """lds_bytes() is the sum of the builder's LDS allocations, with and
        without TDM (whose X row has no dummy slot but PAD slack rows)."""
        for knobs in (_AOT, dict(_AOT, tdm=False), {}):
            with self.subTest(knobs=knobs):
                spec = DirectWgradWmmaSpec(problem=_problem(), **knobs)
                kernel = build_direct_wgrad_wmma(spec, "gfx1250")
                allocated = 0
                for op in kernel.body.ops:
                    if op.name == "tile.smem_alloc":
                        shape = op.results[0].type.shape
                        allocated += shape[0] * shape[1] * 2
                self.assertEqual(spec.lds_bytes(), allocated)

    def test_kernel_name(self):
        spec = DirectWgradWmmaSpec(problem=_problem(), **_AOT)
        self.assertEqual(
            spec.kernel_name(),
            "direct_wgrad_wmma_f3x3p1s1_w4x2_t2x2_qs1_st4_tb256_pad16"
            "_bf16_pf_sched_xcd8_tdm_g2",
        )

    def test_ir_is_shape_independent(self):
        """One binary serves every extent and channel count."""
        from rocke.core.ir_serialize import serialize

        a = build_direct_wgrad_wmma(DirectWgradWmmaSpec(problem=_problem(), **_AOT))
        b = build_direct_wgrad_wmma(
            DirectWgradWmmaSpec(problem=_problem(N=7, H=40, W=9, C=256, K=32), **_AOT)
        )
        self.assertEqual(serialize(a), serialize(b))


class TestWgradWmmaEngines(unittest.TestCase):
    """The C++ engine against Python (skipped without ``rocke_library_engine``)."""

    def _aot_specs(self):
        from kernels.conv.direct_aot_jobs import DIRECT_CAPABILITIES, _knob_grid

        specs = []
        _, caps_list = DIRECT_CAPABILITIES["direct_wgrad_wmma"]
        for dtype in ("bf16", "fp16"):
            for caps in caps_list:
                for knobs in _knob_grid("direct_wgrad_wmma", "gfx1250", dtype, caps):
                    specs.append(
                        DirectWgradWmmaSpec(
                            problem=_problem(
                                N=3,
                                H=29,
                                W=37,
                                C=128,
                                K=192,
                                KH=caps.KH,
                                PAD=caps.PAD,
                                dtype=dtype,
                            ),
                            name="rocke_direct_wgrad_wmma",
                            **knobs,
                        )
                    )
        return specs

    def test_engines_agree_on_specs(self):
        """Validity, reason, kernel name, launch geometry and LDS size agree
        over every cell of the AOT grid and a set of rejected specs."""
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_wgrad_wmma_spec_to_dict

        specs = self._aot_specs()
        n_grid = len(specs)
        self.assertEqual(n_grid, 2 * 2 * 4)  # dtypes x PAD x (waves, blocks)
        for pkw, knobs in (
            ({}, {}),
            ({}, dict(tdm=True, rows=3, waves_k=2, waves_c=1, stages=3)),
            ({}, dict(q_steps=2, tiles_k=1, tiles_c=4)),
            (dict(KH=5, PAD=2), dict(tiles_k=1, tiles_c=1, tdm=True)),
            (dict(KH=1, KW=3, PAD=0), dict(xcd=3)),
            (dict(N=1, H=1, W=1, C=8, K=8), dict(target_blocks=1)),
            (dict(dil_h=2), {}),
            (dict(dil_w=0), {}),
            (dict(groups=2), {}),
            (dict(dtype="fp32"), {}),
            (dict(stride=2), {}),
            (dict(PAD=3), {}),
            (dict(KH=3, KW=1), {}),
            (dict(pad_w=0), {}),
            (dict(C=36), {}),
            (dict(K=12), {}),
            ({}, dict(waves_k=3, waves_c=1)),
            ({}, dict(waves_k=0)),
            ({}, dict(tiles_c=0)),
            ({}, dict(q_steps=0)),
            ({}, dict(waves_k=2, waves_c=2, rows=3, tdm=True)),
            ({}, dict(rows=0)),
            ({}, dict(rows=2, stages=2, prefetch=True)),
            ({}, dict(pad=-8)),
            ({}, dict(pad=12)),
            ({}, dict(stages=1)),
            ({}, dict(stages=9)),
            ({}, dict(target_blocks=0)),
            ({}, dict(tiles_k=4, tiles_c=4)),
            ({}, dict(tiles_k=4, tiles_c=2, q_steps=2, prefetch=True)),
            ({}, dict(waves_k=4, waves_c=4, rows=4, stages=8)),
        ):
            specs.append(DirectWgradWmmaSpec(problem=_problem(**pkw), **knobs))
        n_ok = 0
        for arch in ("gfx1250", "gfx950", "gfx1201"):
            for spec in specs:
                with self.subTest(arch=arch, spec=spec):
                    ok, why_c, name, grid, block, lds = (
                        eng.conv_direct_wgrad_wmma_describe(
                            conv_direct_wgrad_wmma_spec_to_dict(spec), arch
                        )
                    )
                    ok_py, why = is_valid_direct_wgrad_wmma_spec(spec, arch)
                    self.assertEqual(ok, ok_py)
                    self.assertEqual(why_c, why)
                    self.assertEqual(name, spec.kernel_name())
                    self.assertEqual(lds, spec.lds_bytes())
                    if ok:
                        n_ok += 1
                        self.assertEqual(
                            (tuple(grid), tuple(block)), spec.launch_geometry()
                        )
        # The whole AOT grid and the first six extra specs build on gfx1250.
        self.assertEqual(n_ok, n_grid + 6)

    def test_serialized_ir_matches(self):
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_wgrad_wmma_spec_to_dict
        from rocke.core.ir_serialize import serialize

        specs = self._aot_specs()[:2] + [
            DirectWgradWmmaSpec(problem=_problem(dtype="fp16")),
            DirectWgradWmmaSpec(
                problem=_problem(), rows=2, stages=3, prefetch=True, xcd=8
            ),
            DirectWgradWmmaSpec(
                problem=_problem(KH=5, PAD=2),
                tiles_k=1,
                tiles_c=1,
                prefetch=True,
                tdm=True,
            ),
        ]
        for spec in specs:
            with self.subTest(spec=spec):
                ir_c = eng.conv_direct_wgrad_wmma_serialize_ir(
                    conv_direct_wgrad_wmma_spec_to_dict(spec), "gfx1250"
                )
                self.assertEqual(ir_c, serialize(build_direct_wgrad_wmma(spec)))

    def test_binding_rejects_bad_input(self):
        """A rejected spec has no launch geometry and refuses to lower with
        Python's message; a missing, mistyped or partial problem raises."""
        eng = pytest.importorskip("rocke_library_engine")
        from kernels.conv.engine_bridge import conv_direct_wgrad_wmma_spec_to_dict

        spec = DirectWgradWmmaSpec(problem=_problem(), tiles_k=3)
        sd = conv_direct_wgrad_wmma_spec_to_dict(spec)
        ok, why, _, grid, block, _ = eng.conv_direct_wgrad_wmma_describe(sd, "gfx1250")
        self.assertFalse(ok)
        self.assertEqual((tuple(grid), tuple(block)), ((0, 0, 0), (0, 0, 0)))
        with self.assertRaises(RuntimeError) as cm:
            eng.conv_direct_wgrad_wmma_lower_llvm(sd, arch="gfx1250")
        self.assertIn(f"invalid DirectWgradWmmaSpec: {why}", str(cm.exception))
        good = conv_direct_wgrad_wmma_spec_to_dict(
            DirectWgradWmmaSpec(problem=_problem())
        )
        partial = dict(good["problem"])
        del partial["kpg"]
        # An explicit dtype=None is refused (Python rejects it), not read as fp16.
        no_dtype = dict(good["problem"], dtype=None)
        for problem in (None, "N2H13", partial, no_dtype):
            sd = dict(good)
            if problem is None:
                del sd["problem"]
            else:
                sd["problem"] = problem
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_wgrad_wmma_describe(sd, "gfx1250")
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                eng.conv_direct_wgrad_wmma_lower_llvm(sd, arch="gfx1250")


if __name__ == "__main__":
    unittest.main()
