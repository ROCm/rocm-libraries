# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the synchronous UniversalGemm coalesced tile copy."""

from __future__ import annotations

from contextlib import ExitStack
import json
import struct
import unittest
from unittest.mock import patch

from rocke.helpers.loads import CoalescedTileLoader
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def _build(*, pad_k: bool):
    spec = UniversalGemmSpec(
        name="universal_gemm_tile_copy",
        tile=TileSpec(
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="mem", epilogue="default", pad_k=pad_k),
    )
    copies = []
    load = CoalescedTileLoader.load

    def record(loader, builder, **kwargs):
        copies.append((loader.tile_rows, loader.tile_cols, loader.load_vec))
        return load(loader, builder, **kwargs)

    with patch.object(CoalescedTileLoader, "load", record):
        kernel = build_universal_gemm(spec, arch="gfx950")
    return spec, kernel, copies


class TestUniversalGemmTileCopy(unittest.TestCase):
    def test_copy_selection(self):
        for pad_k in (False, True):
            with self.subTest(pad_k=pad_k):
                _, _, copies = _build(pad_k=pad_k)
                self.assertEqual(copies, [] if pad_k else [(64, 32, 8)] * 2)

    def test_gfx950_numeric(self):
        import numpy as np

        from rocke.helpers.compile import compile_kernel
        from rocke.instances.common.manifest_runner.utils import as_u8_buffer
        from rocke.runtime.hip_module import Runtime, get_device_arch

        if get_device_arch() != "gfx950":
            self.skipTest("requires a gfx950 HIP GPU")

        rt = Runtime()
        rng = np.random.default_rng(2301)
        for label, K, pad_k in (("coalesced", 96, False), ("legacy_tail", 91, True)):
            with self.subTest(copy=label, k=K):
                spec, kernel, copies = _build(pad_k=pad_k)
                self.assertEqual(len(copies), 0 if pad_k else 2)
                artifact = compile_kernel(kernel, arch="gfx950", backend="python")
                print(json.dumps({"case": label, "phase": "compiled", "isa": artifact.isa}))

                M, N = 128, 192
                A = rng.integers(-3, 4, (M, K), dtype=np.int16).astype(np.float16)
                B = rng.integers(-3, 4, (N, K), dtype=np.int16).astype(np.float16)
                C = np.full((M, N), np.nan, dtype=np.float16)
                with ExitStack() as stack:
                    mod = rt.load_module(artifact.hsaco)
                    stack.callback(mod.unload)
                    ptrs = []
                    for array in (A, B, C):
                        ptr = rt.alloc(array.nbytes)
                        stack.callback(rt.free, ptr)
                        rt.memcpy_h2d(ptr, as_u8_buffer(array), array.nbytes)
                        ptrs.append(ptr)
                    rt.launch_blocking(
                        mod.get_function(artifact.kernel_name),
                        (N // spec.tile.tile_n, M // spec.tile.tile_m, 1),
                        (spec.block_size, 1, 1),
                        struct.pack("<QQQiii", *ptrs, M, N, K),
                    )
                    print(json.dumps({"case": label, "phase": "launched"}))
                    rt.memcpy_d2h(as_u8_buffer(C), ptrs[2], C.nbytes)

                ref = (A.astype(np.float32) @ B.astype(np.float32).T).astype(np.float16)
                np.testing.assert_array_equal(C, ref)
                print(json.dumps({"case": label, "phase": "numeric_pass", "elements": C.size}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
