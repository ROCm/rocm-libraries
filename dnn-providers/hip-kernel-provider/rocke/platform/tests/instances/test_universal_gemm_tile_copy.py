# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for UniversalGemm tile copies and runtime K schedules."""

from __future__ import annotations

from contextlib import ExitStack
import json
import struct
import unittest
from unittest.mock import patch

from rocke.helpers.loads import CoalescedTileLoader
from rocke.instances.common._gemm_kloop import GemmKLoop
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)


def _build(trait: TraitSpec):
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
        trait=trait,
    )
    copies = []
    schedules = []
    load = CoalescedTileLoader.load
    single = GemmKLoop.single_buffered
    ping_pong = GemmKLoop.ping_pong

    def record_load(loader, builder, **kwargs):
        copies.append((loader.tile_rows, loader.tile_cols, loader.load_vec))
        return load(loader, builder, **kwargs)

    def record_single(loop):
        schedules.append("single")
        return single(loop)

    def record_ping_pong(loop, *, direct_to_lds):
        schedules.append("async" if direct_to_lds else "sync")
        return ping_pong(loop, direct_to_lds=direct_to_lds)

    with (
        patch.object(CoalescedTileLoader, "load", record_load),
        patch.object(GemmKLoop, "single_buffered", record_single),
        patch.object(GemmKLoop, "ping_pong", record_ping_pong),
    ):
        kernel = build_universal_gemm(spec, arch="gfx950")
    return spec, kernel, copies, schedules


def _cases():
    return (
        ("coalesced", TraitSpec(pipeline="mem", epilogue="default"), "single", (96,)),
        (
            "legacy_tail",
            TraitSpec(pipeline="mem", epilogue="default", pad_k=True),
            "single",
            (91,),
        ),
        (
            "sync_ping_pong",
            TraitSpec(pipeline="compv4", epilogue="default"),
            "sync",
            (32, 64, 96),
        ),
        (
            "dtl_ping_pong",
            TraitSpec(
                pipeline="compv4",
                epilogue="default",
                direct_to_lds=True,
                dtl_prefetch=True,
            ),
            "async",
            (32, 64, 96),
        ),
    )


class TestUniversalGemmTileCopy(unittest.TestCase):
    def _build_case(self, label, trait, schedule):
        spec, kernel, copies, schedules = _build(trait)
        self.assertEqual(copies, [(64, 32, 8)] * 2 if label == "coalesced" else [])
        self.assertEqual(schedules, [schedule])
        return spec, kernel

    def test_copy_and_schedule_selection(self):
        for label, trait, schedule, _ in _cases():
            with self.subTest(case=label):
                self._build_case(label, trait, schedule)

    def test_gfx950_numeric(self):
        import numpy as np

        from rocke.helpers.compile import compile_kernel
        from rocke.runtime.hip_module import Runtime, get_device_arch

        if get_device_arch() != "gfx950":
            self.skipTest("requires a gfx950 HIP GPU")

        rt = Runtime()
        rng = np.random.default_rng(2301)
        for label, trait, schedule, k_sizes in _cases():
            with self.subTest(case=label):
                spec, kernel = self._build_case(label, trait, schedule)
                artifact = compile_kernel(kernel, arch="gfx950", backend="python")
                print(
                    json.dumps(
                        {"case": label, "phase": "compiled", "isa": artifact.isa}
                    )
                )
                for k in k_sizes:
                    with self.subTest(k=k):
                        self._launch_and_compare(rt, spec, artifact, rng, label, k)

    @staticmethod
    def _launch_and_compare(rt, spec, artifact, rng, label, k):
        import numpy as np

        from rocke.instances.common.manifest_runner.utils import as_u8_buffer

        m, n = 128, 192
        a = rng.integers(-3, 4, (m, k), dtype=np.int16).astype(np.float16)
        b = rng.integers(-3, 4, (n, k), dtype=np.int16).astype(np.float16)
        c = np.full((m, n), np.nan, dtype=np.float16)
        with ExitStack() as stack:
            mod = rt.load_module(artifact.hsaco)
            stack.callback(mod.unload)
            ptrs = []
            for array in (a, b, c):
                ptr = rt.alloc(array.nbytes)
                stack.callback(rt.free, ptr)
                rt.memcpy_h2d(ptr, as_u8_buffer(array), array.nbytes)
                ptrs.append(ptr)
            rt.launch_blocking(
                mod.get_function(artifact.kernel_name),
                (n // spec.tile.tile_n, m // spec.tile.tile_m, 1),
                (spec.block_size, 1, 1),
                struct.pack("<QQQiii", *ptrs, m, n, k),
            )
            print(json.dumps({"case": label, "k": k, "phase": "launched"}))
            rt.memcpy_d2h(as_u8_buffer(c), ptrs[2], c.nbytes)

        ref = (a.astype(np.float32) @ b.astype(np.float32).T).astype(np.float16)
        np.testing.assert_array_equal(c, ref)
        print(
            json.dumps(
                {"case": label, "k": k, "phase": "numeric_pass", "elements": c.size}
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
