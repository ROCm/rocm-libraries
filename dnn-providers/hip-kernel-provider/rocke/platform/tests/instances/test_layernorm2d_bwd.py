# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the AOT-generic LayerNorm2D backward instance.

The structure / validity / dispatch-admission tests need no GPU. The numeric
test launches all three parts (``dx``, ``dwdb``, ``dwdb_reduce``) against a
torch reference, for both the direct (``P == 1``) and the workspace
(``P > 1``) dgamma / dbeta paths and a row width with a masked ``dx`` tail; it
is skipped without torch or a GPU. Byte-identity with the C++ engine is
covered by ``parity/layernorm2d_bwd_emit.{py,c}``.
"""

from __future__ import annotations

import unittest

from rocke.core.verify import verify
from rocke.dispatch.families.norm import NormRequest, dynamic_norm_bwd_candidates
from rocke.instances.common.layernorm2d_bwd import (
    PARTS,
    LayerNorm2DBwdSpec,
    build_layernorm2d_bwd,
    default_row_splits,
    is_valid_spec,
    layernorm2d_bwd_grid,
    layernorm2d_bwd_signature,
    layernorm2d_bwd_workspace_bytes,
)


class SpecTest(unittest.TestCase):
    def test_kernel_names(self):
        self.assertEqual(
            LayerNorm2DBwdSpec(part="dx", dtype="bf16", block_size=64, vec=8,
                               fast_reduce=True).kernel_name(),
            "rocke_layernorm2d_bwd_dx_bf16_b64_v8_fr",
        )
        self.assertEqual(
            LayerNorm2DBwdSpec(part="dwdb", dtype="f16", unroll=2).kernel_name(),
            "rocke_layernorm2d_bwd_dwdb_f16_b256_v4_u2",
        )
        self.assertEqual(
            LayerNorm2DBwdSpec(part="dwdb_reduce", vec=2).kernel_name(),
            "rocke_layernorm2d_bwd_dwdb_reduce_f16_b256_v2_u4",
        )

    def test_validity(self):
        self.assertTrue(is_valid_spec(LayerNorm2DBwdSpec())[0])
        bad = [
            LayerNorm2DBwdSpec(part="dw"),
            LayerNorm2DBwdSpec(vec=3),
            LayerNorm2DBwdSpec(block_size=96),
            LayerNorm2DBwdSpec(part="dwdb", unroll=3),
            LayerNorm2DBwdSpec(part="dx", unroll=8),  # unroll is a dwdb knob
            LayerNorm2DBwdSpec(part="dwdb", fast_reduce=True),  # fr is a dx knob
            LayerNorm2DBwdSpec(part="dx", block_size=64, wave_size=128, fast_reduce=True),
        ]
        for spec in bad:
            with self.subTest(spec=spec):
                ok, why = is_valid_spec(spec)
                self.assertFalse(ok)
                self.assertTrue(why)
                with self.assertRaises(ValueError):
                    build_layernorm2d_bwd(spec)

    def test_every_part_builds_clean(self):
        for part in PARTS:
            for spec in (
                LayerNorm2DBwdSpec(part=part, dtype="bf16", block_size=64, vec=8),
                LayerNorm2DBwdSpec(part=part, dtype="f16", block_size=1024, vec=2,
                                   **({"fast_reduce": True} if part == "dx" else {"unroll": 1})),
            ):
                with self.subTest(spec=spec):
                    kernel = build_layernorm2d_bwd(spec)
                    self.assertEqual(kernel.name, spec.kernel_name())
                    self.assertEqual(list(verify(kernel)), [])
                    names = [p.name.lstrip("%") for p in kernel.params]
                    self.assertEqual(names, [e["name"] for e in layernorm2d_bwd_signature(spec)])

    def test_geometry(self):
        dx = LayerNorm2DBwdSpec(part="dx", block_size=64, vec=8)
        dwdb = LayerNorm2DBwdSpec(part="dwdb", block_size=64, vec=2)
        red = LayerNorm2DBwdSpec(part="dwdb_reduce", block_size=64, vec=2)
        self.assertEqual(layernorm2d_bwd_grid(dx, rows=1152, cols=1280), (1152, 1, 1))
        self.assertEqual(layernorm2d_bwd_grid(dwdb, rows=1152, cols=1280, splits=8), (10, 8, 1))
        self.assertEqual(layernorm2d_bwd_grid(red, rows=1152, cols=1280), (20, 1, 1))
        self.assertEqual(layernorm2d_bwd_workspace_bytes(cols=80, splits=1), 0)
        self.assertEqual(layernorm2d_bwd_workspace_bytes(cols=80, splits=4), 4 * 2 * 80 * 4)
        # Power of two, capped by sqrt(rows) and by ~4 CTAs per CU.
        self.assertEqual(default_row_splits(rows=16, cols=1024, block_size=64, vec=8,
                                            num_cus=256), 4)
        self.assertEqual(default_row_splits(rows=524288, cols=128, block_size=64, vec=2,
                                            num_cus=256), 512)  # sqrt(524288) ~ 724
        self.assertEqual(default_row_splits(rows=1, cols=128, block_size=64, vec=2,
                                            num_cus=256), 1)


class DispatchTest(unittest.TestCase):
    def test_parts_registered(self):
        for part in PARTS:
            cands = dynamic_norm_bwd_candidates(part)
            self.assertTrue(cands)
            self.assertTrue(all(c.opt_in for c in cands))
            self.assertTrue(all(c.select_spec.__name__ for c in cands))
        self.assertEqual(
            len(dynamic_norm_bwd_candidates()),
            sum(len(dynamic_norm_bwd_candidates(p)) for p in PARTS),
        )

    def test_vec_admission(self):
        req = NormRequest(rows=3200, cols=80, arch="gfx950", kind="layernorm", dtype="bf16")
        odd = NormRequest(rows=3200, cols=81, arch="gfx950", kind="layernorm", dtype="bf16")
        rms = NormRequest(rows=3200, cols=80, arch="gfx950", kind="rmsnorm", dtype="bf16")
        for c in dynamic_norm_bwd_candidates():
            spec_vec = int(c.spec_id.split("_v")[1].split("_")[0])
            with self.subTest(candidate=c.name):
                self.assertFalse(c.admits(rms)[0])
                self.assertTrue(c.admits(req)[0])  # 80 % {2, 4, 8} == 0
                if c.spec_id.startswith("layernorm_bwd_dx_"):
                    self.assertTrue(c.admits(odd)[0])  # masked tail
                else:
                    self.assertEqual(c.admits(odd)[0], 81 % spec_vec == 0)
        spec = dynamic_norm_bwd_candidates("dwdb")[0].select_spec(req)
        self.assertEqual((spec.part, spec.dtype), ("dwdb", "bf16"))


class NumericTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        if not torch.cuda.is_available():
            self.skipTest("no GPU")
        self.torch = torch

    def _run(self, rows, cols, dtype, splits, dx_kw, w_kw):
        torch = self.torch
        from rocke.helpers.compile import compile_kernel
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release

        tdt = torch.bfloat16 if dtype == "bf16" else torch.float16
        gen = torch.Generator(device="cuda").manual_seed(rows * 7919 + cols)
        x = torch.randn(rows, cols, device="cuda", generator=gen).to(tdt)
        dy = torch.randn(rows, cols, device="cuda", generator=gen).to(tdt)
        g = (torch.rand(cols, device="cuda", generator=gen) + 0.5).to(tdt)
        xf = x.float()
        mean = xf.mean(-1).to(tdt)
        rstd = torch.rsqrt(xf.var(-1, unbiased=False) + 1e-5).to(tdt)
        dx = torch.empty_like(x)
        dw = torch.empty_like(g)
        db = torch.empty_like(g)
        ws = torch.zeros(max(1, layernorm2d_bwd_workspace_bytes(cols=cols, splits=splits) // 4),
                         device="cuda")

        launches = [
            (LayerNorm2DBwdSpec(part="dx", dtype=dtype, **dx_kw),
             {"DY": dy.data_ptr(), "X": x.data_ptr(), "Gamma": g.data_ptr(),
              "Mean": mean.data_ptr(), "Rstd": rstd.data_ptr(), "DX": dx.data_ptr(),
              "M": rows, "N": cols}),
            (LayerNorm2DBwdSpec(part="dwdb", dtype=dtype, **w_kw),
             {"DY": dy.data_ptr(), "X": x.data_ptr(), "Mean": mean.data_ptr(),
              "Rstd": rstd.data_ptr(), "DW": dw.data_ptr(), "DB": db.data_ptr(),
              "WS": ws.data_ptr(), "M": rows, "N": cols, "P": splits}),
        ]
        if splits > 1:
            launches.append(
                (LayerNorm2DBwdSpec(part="dwdb_reduce", dtype=dtype, **w_kw),
                 {"WS": ws.data_ptr(), "DW": dw.data_ptr(), "DB": db.data_ptr(),
                  "N": cols, "P": splits}))
        for spec, values in launches:
            artifact = compile_kernel(build_layernorm2d_bwd(spec), arch="gfx950")
            grid = layernorm2d_bwd_grid(spec, rows=rows, cols=cols, splits=splits)
            KernelLauncher.from_artifact(artifact)(
                values, config=LaunchConfig(grid=grid, block=(spec.block_size, 1, 1)))
        synchronize_and_release()

        m, r = mean.float()[:, None], rstd.float()[:, None]
        xhat = (xf - m) * r
        dyf = dy.float()
        dyg = dyf * g.float()
        refs = {
            "dx": (dx, r * (dyg - dyg.mean(-1, keepdim=True)
                            - xhat * (dyg * xhat).mean(-1, keepdim=True))),
            "dgamma": (dw, (dyf * xhat).sum(0)),
            "dbeta": (db, dyf.sum(0)),
        }
        tol = 1.6e-2 if dtype == "bf16" else 2e-3
        for name, (got, ref) in refs.items():
            rel = float((got.float() - ref).abs().max() / ref.abs().max())
            self.assertLessEqual(rel, tol, f"{name} rows={rows} cols={cols} P={splits}")

    def test_numeric(self):
        from rocke.runtime.hip_module import get_device_arch

        if get_device_arch(0) != "gfx950":
            self.skipTest("numeric case compiled for gfx950")
        cases = [
            # rows, cols, dtype, P, dx knobs, dwdb/dwdb_reduce knobs
            (1152, 1280, "bf16", 32, {"block_size": 64, "vec": 8},
             {"block_size": 64, "vec": 2, "unroll": 4}),
            (3200, 80, "bf16", 1, {"block_size": 64, "vec": 2, "fast_reduce": True},
             {"block_size": 64, "vec": 8, "unroll": 8}),
            (777, 1000, "f16", 7, {"block_size": 128, "vec": 4},  # masked dx tail
             {"block_size": 128, "vec": 4, "unroll": 1}),
        ]
        for case in cases:
            with self.subTest(case=case[:4]):
                self._run(*case)


if __name__ == "__main__":
    unittest.main()
