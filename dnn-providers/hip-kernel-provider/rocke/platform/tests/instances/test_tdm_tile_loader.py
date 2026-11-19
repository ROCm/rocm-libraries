# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM window lowering and gfx1250 numerical coverage for GEMM fills."""

from __future__ import annotations

from contextlib import ExitStack
import json
import struct
import unittest
from unittest.mock import Mock

from rocke.core.ir import I64
from rocke.helpers.loads import TdmSourceWindow2D, TdmTileLoader2D


class TestTdmTileLoader(unittest.TestCase):
    def test_prepare_rebases_window_and_does_not_issue(self):
        b = Mock()
        b.zext.side_effect = lambda value, _: value
        b.mul.side_effect = lambda a, c: a * c
        b.add.side_effect = lambda a, c: a + c
        b.sub.side_effect = lambda a, c: a - c
        b.smax.side_effect = max
        b.global_addr_of.return_value = "rebased"
        groups = ("d0", "d1", "d2", "d3", "d4")
        b.tdm_descriptor_2d.return_value = groups
        source = TdmSourceWindow2D(
            ptr="ptr",
            base_offset=2**32,
            row_origin=2**16,
            col_origin=32,
            row_extent=2**16 + 9,
            cols_remaining=16,
            row_stride=2**16 + 16,
        )
        loader = TdmTileLoader2D(32, 32, 2, pad_interval=2, pad_amount=16)

        load = loader.prepare(b, source, "lds_slot", zero=0)

        self.assertEqual(
            [call.args for call in b.zext.call_args_list],
            [(source.row_origin, I64), (source.row_stride, I64), (32, I64)],
        )
        b.global_addr_of.assert_called_once_with(
            "ptr", 2**32 + (2**16) * (2**16 + 16) + 32
        )
        b.tdm_descriptor_2d.assert_called_once_with(
            "rebased",
            "lds_slot",
            elem_bytes=2,
            tensor_dim0=16,
            tensor_dim1=9,
            row_stride=2**16 + 16,
            tile_dim0=32,
            tile_dim1=32,
            pad_interval=2,
            pad_amount=16,
        )
        b.tensor_load_to_lds.assert_not_called()
        load.issue(b, cachepolicy=0)
        b.tensor_load_to_lds.assert_called_once_with(*groups, cachepolicy=0)
        b.s_wait_tensorcnt.assert_not_called()

    def test_gfx1250_gemm_numeric(self):
        import numpy as np

        from rocke.helpers.compile import compile_kernel
        from rocke.instances.common.gemm_universal import (
            TileSpec,
            TraitSpec,
            UniversalGemmSpec,
            build_universal_gemm,
        )
        from rocke.runtime.hip_module import Runtime, get_device_arch

        if get_device_arch() != "gfx1250":
            self.skipTest("requires a gfx1250 HIP GPU")

        cases = (
            ("single", 1, 0, False, (32, 80)),
            ("ping_pong", 2, 8, False, (32, 64, 96, 80)),
            ("ring", 3, 8, False, (32, 80, 128)),
            ("early_refill", 2, 8, True, (64, 80, 96)),
        )
        tile = TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 32)
        rt = Runtime()
        rng = np.random.default_rng(2302)
        for name, depth, pad, early, sizes in cases:
            with self.subTest(case=name):
                spec = UniversalGemmSpec(
                    name=f"tdm_tile_loader_{name}",
                    tile=tile,
                    trait=TraitSpec(
                        pipeline="mem",
                        epilogue="default",
                        tdm=True,
                        tdm_depth=depth,
                        lds_k_pad=pad,
                        split_barrier=early,
                        tdm_alternate_issuers=early,
                        tdm_early_refill=early,
                    ),
                    wave_size=32,
                )
                kernel = build_universal_gemm(spec, arch="gfx1250")
                artifact = compile_kernel(kernel, arch="gfx1250", backend="python")
                print(
                    json.dumps({"case": name, "phase": "compiled", "isa": artifact.isa})
                )
                self.assertIn(
                    "call void @llvm.amdgcn.tensor.load.to.lds", artifact.llvm_text
                )
                for k in sizes:
                    with self.subTest(k=k):
                        self._launch_and_compare(rt, spec, artifact, rng, name, k)

    @staticmethod
    def _launch_and_compare(rt, spec, artifact, rng, label, k):
        import numpy as np

        from rocke.instances.common.manifest_runner.utils import as_u8_buffer

        m, n = 256, 384
        a = rng.integers(-3, 4, (m, k), dtype=np.int16).astype(np.float16)
        b = rng.integers(-3, 4, (n, k), dtype=np.int16).astype(np.float16)
        c = np.full((m, n), np.nan, dtype=np.float16)
        with ExitStack() as stack:
            module = rt.load_module(artifact.hsaco)
            stack.callback(module.unload)
            ptrs = []
            for array in (a, b, c):
                ptr = rt.alloc(array.nbytes)
                stack.callback(rt.free, ptr)
                rt.memcpy_h2d(ptr, as_u8_buffer(array), array.nbytes)
                ptrs.append(ptr)
            rt.launch_blocking(
                module.get_function(artifact.kernel_name),
                (n // spec.tile.tile_n, m // spec.tile.tile_m, 1),
                (spec.block_size, 1, 1),
                struct.pack("<QQQiii", *ptrs, m, n, k),
            )
            rt.memcpy_d2h(as_u8_buffer(c), ptrs[2], c.nbytes)
        ref = (a.astype(np.float32) @ b.astype(np.float32).T).astype(np.float16)
        np.testing.assert_array_equal(c, ref)
        print(json.dumps({"case": label, "k": k, "phase": "numeric_pass"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
