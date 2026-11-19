# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for UniversalGemm tile copies and runtime K schedules."""

from __future__ import annotations

from contextlib import ExitStack
import json
import struct
import unittest
from unittest.mock import patch

from rocke.helpers.loads import AsyncTileLoaderSlot, CoalescedTileLoader
from rocke.instances.common import gemm_universal
from rocke.instances.common._gemm_kloop import GemmKLoop
from rocke.instances.common.gemm_universal import (
    DataSpec,
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
    visit_chunks = CoalescedTileLoader.visit_chunks
    single = GemmKLoop.single_buffered
    ping_pong = GemmKLoop.ping_pong

    def record_copy(loader, builder, **kwargs):
        copies.append((loader.tile_rows, loader.tile_cols, loader.load_vec))
        return visit_chunks(loader, builder, **kwargs)

    def record_single(loop):
        schedules.append("single")
        return single(loop)

    def record_ping_pong(loop, *, direct_to_lds):
        schedules.append("async" if direct_to_lds else "sync")
        return ping_pong(loop, direct_to_lds=direct_to_lds)

    with (
        patch.object(CoalescedTileLoader, "visit_chunks", record_copy),
        patch.object(GemmKLoop, "single_buffered", record_single),
        patch.object(GemmKLoop, "ping_pong", record_ping_pong),
        patch.object(
            AsyncTileLoaderSlot,
            "issue_bytes",
            autospec=True,
            side_effect=AsyncTileLoaderSlot.issue_bytes,
        ) as dtl_issues,
        patch.object(
            gemm_universal,
            "load_clipped_global_fragment",
            wraps=gemm_universal.load_clipped_global_fragment,
        ) as clipped_reads,
    ):
        kernel = build_universal_gemm(spec, arch="gfx950")
    return (
        spec,
        kernel,
        copies,
        schedules,
        clipped_reads.call_count,
        dtl_issues.call_count,
    )


def _cases():
    return (
        ("coalesced", TraitSpec(pipeline="mem", epilogue="default"), "single", (96,)),
        (
            "clipped_tail",
            TraitSpec(pipeline="mem", epilogue="default", pad_k=True),
            "single",
            (65, 91),
        ),
        (
            "preshuffled_ping_pong",
            TraitSpec(pipeline="compv4", epilogue="default", preshuffle_b=True),
            "sync",
            (96,),
        ),
        (
            "preshuffled_tail",
            TraitSpec(
                pipeline="mem", epilogue="default", pad_k=True, preshuffle_b=True
            ),
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
        (
            "dtl_swizzled",
            TraitSpec(
                pipeline="compv4",
                epilogue="default",
                direct_to_lds=True,
                dtl_prefetch=True,
                lds_swizzle=True,
            ),
            "async",
            (96,),
        ),
        (
            "split_k_edges",
            TraitSpec(
                pipeline="mem", epilogue="default", split_k=2, pad_m=True, pad_n=True
            ),
            "single",
            (128,),
        ),
    )


class TestUniversalGemmTileCopy(unittest.TestCase):
    def _build_case(self, label, trait, schedule):
        spec, kernel, copies, schedules, clipped_reads, dtl_issues = _build(trait)
        expected = {"single": 2, "sync": 4, "async": 0}[schedule]
        self.assertEqual(copies, [(64, 32, 8)] * expected)
        self.assertEqual(schedules, [schedule])
        self.assertEqual(bool(clipped_reads), trait.pad_k)
        self.assertEqual(dtl_issues, 4 if schedule == "async" else 0)
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

    def test_gfx942_fp64_dtl_numeric(self):
        import numpy as np

        from rocke.helpers.compile import compile_kernel
        from rocke.runtime.hip_module import Runtime, get_device_arch

        if get_device_arch() != "gfx942":
            self.skipTest("requires gfx942's sub-element FP64 direct-to-LDS path")

        rt = Runtime()
        rng = np.random.default_rng(2302)
        for swizzle in (False, True):
            with self.subTest(swizzle=swizzle):
                spec = UniversalGemmSpec(
                    name="fp64_dtl_sub_element",
                    tile=TileSpec(64, 64, 16, 2, 2, 1, 16, 16, 4),
                    trait=TraitSpec(
                        pipeline="compv4",
                        epilogue="default",
                        direct_to_lds=True,
                        dtl_prefetch=True,
                        lds_swizzle=swizzle,
                    ),
                    data=DataSpec("fp64", "fp64", "fp64", "fp64"),
                )
                artifact = compile_kernel(
                    build_universal_gemm(spec, arch="gfx942"),
                    arch="gfx942",
                    backend="python",
                )
                label = "fp64_dtl_swizzled" if swizzle else "fp64_dtl"
                for k in (16, 48):
                    self._launch_and_compare(rt, spec, artifact, rng, label, k)

    @staticmethod
    def _launch_and_compare(rt, spec, artifact, rng, label, k):
        import numpy as np

        from rocke.instances.common.manifest_runner.utils import as_u8_buffer

        split_k = spec.trait.split_k
        m, n = (67, 131) if split_k > 1 else (128, 192)
        tile_m, tile_n = spec.tile.tile_m, spec.tile.tile_n
        gm, gn = (m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n
        storage_dtype = np.float64 if spec.data.dtype_a == "fp64" else np.float16
        a = rng.integers(-3, 4, (m, k), dtype=np.int16).astype(storage_dtype)
        b = rng.integers(-3, 4, (n, k), dtype=np.int16).astype(storage_dtype)
        a_storage, b_storage = a, b
        if split_k > 1:
            # Pad physical input rows but keep the logical M/N bounds. Nonzero
            # padding makes an unguarded atomic write corrupt C or the canary.
            a_storage = np.full((gm * tile_m, k), 3, dtype=a.dtype)
            b_storage = np.full((gn * tile_n, k), 5, dtype=b.dtype)
            a_storage[:m], b_storage[:n] = a, b
            output = np.full(gm * tile_m * gn * tile_n, 137, dtype=np.float32)
            c = output[: m * n].reshape(m, n)
            c.fill(0)
        else:
            c = output = np.full((m, n), np.nan, dtype=storage_dtype)
        if spec.trait.preshuffle_b:
            tile_n, tile_k = spec.tile.tile_n, spec.tile.tile_k
            k_tiles = (k + tile_k - 1) // tile_k
            # Fill the physically allocated but logically clipped K tail with
            # nonzero data so a missed clip changes the result.
            padded = np.full((n, k_tiles * tile_k), 17, dtype=b.dtype)
            padded[:, :k] = b
            b_storage = np.ascontiguousarray(
                padded.reshape(n // tile_n, tile_n, k_tiles, tile_k).transpose(
                    2, 0, 1, 3
                )
            )
        with ExitStack() as stack:
            mod = rt.load_module(artifact.hsaco)
            stack.callback(mod.unload)
            ptrs = []
            for array in (a_storage, b_storage, output):
                ptr = rt.alloc(array.nbytes)
                stack.callback(rt.free, ptr)
                rt.memcpy_h2d(ptr, as_u8_buffer(array), array.nbytes)
                ptrs.append(ptr)
            rt.launch_blocking(
                mod.get_function(artifact.kernel_name),
                (gn, gm, split_k),
                (spec.block_size, 1, 1),
                struct.pack("<QQQiii", *ptrs, m, n, k),
            )
            print(json.dumps({"case": label, "k": k, "phase": "launched"}))
            rt.memcpy_d2h(as_u8_buffer(output), ptrs[2], output.nbytes)

        ref_dtype = np.float64 if c.dtype == np.float64 else np.float32
        ref = (a.astype(ref_dtype) @ b.astype(ref_dtype).T).astype(c.dtype)
        np.testing.assert_array_equal(c, ref)
        if split_k > 1:
            np.testing.assert_array_equal(output[m * n :], 137)
        print(
            json.dumps(
                {"case": label, "k": k, "phase": "numeric_pass", "elements": c.size}
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
