# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correctness of the fused-weight generic direct-MFMA dgrad as the AOT cache
ships it.

Two layers:

* ``TestDirectMfmaDgradValidation`` (CPU only): the spec validator's geometry
  rules that used to let silently-wrong kernels through (a transposed ``kpg``
  that is not a multiple of the 4-channel store vector, non-"same" padding),
  the LDS footprint check, ``direct_conv_lds_bytes`` against the LDS pool
  the lowering actually allocates, and the row-stream knobs' validation,
  names and register split.
* ``TestDirectMfmaDgradCachedNumeric`` (GPU): builds the
  ``direct_grouped_dgrad_fused`` binaries of a capability exactly as
  ``--compile-all`` does (on the probe problem), puts them in a private
  cache, takes the launch plans ``direct_plans`` offers for an adversarial
  problem (odd H/W, N = 1, partial W and H tiles, tiles taller than the image,
  cpg != kpg, partial K-atoms, 1x1, 5x5 and 7x7) and runs every one through
  the AOT ABI, comparing dX against a numpy reference with the
  manifest-runner conv rule (``|D - ref| <= 1e-2 + 1e-2*|ref|``, NaN is a
  failure). dX is pre-filled with 0xFF bytes -- NaN in fp16 and bf16 -- so an
  unwritten output fails instead of passing on a lucky zero.

numpy is the only hard dependency; no torch.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

if importlib.util.find_spec("torch") is not None:
    # This file does not use torch, but other test files in the same pytest
    # process do: torch's HIP runtime has to claim the process context before
    # rocke's does (see test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

from rocke.runtime.hip_module import get_device_arch

GPU_ARCH = get_device_arch(0)
_FUSED = "direct_grouped_dgrad_fused"


def _problem(N, C, K, H, W, G, KH=3, PAD=1, dtype="bf16"):
    from kernels.common.conv_direct_grouped import DirectConvProblem

    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=G,
        cpg=C // G,
        kpg=K // G,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=1,
        dtype=dtype,
    )


def _fprop(p, **kw):
    from kernels.common.conv_direct_grouped import make_dgrad_fprop_spec

    return make_dgrad_fprop_spec(p, **kw)


class TestDirectMfmaDgradValidation(unittest.TestCase):
    def test_transposed_kpg_must_be_a_store_vector_multiple(self):
        # Original cpg = 6 -> transposed kpg = 6: the 4-channel store vector
        # would spill into the next group's channels.
        from kernels.common.conv_direct_grouped import is_valid_spec

        p = _problem(2, 48, 32, 8, 8, 8)
        for kw in ({}, {"dgrad_fused_weights": True}):
            with self.subTest(kw=kw):
                ok, why = is_valid_spec(_fprop(p, block_groups=1, **kw))
                self.assertFalse(ok)
                self.assertIn("multiple of 4", why)

    def test_non_same_padding_is_rejected(self):
        # The row stream emits output rows 0..H-1 of its input height; with
        # pad 0 on a 3x3 the transposed problem pads by 2 and its last two
        # output rows would never be written.
        from kernels.common.conv_direct_grouped import is_valid_spec

        p = _problem(2, 32, 32, 9, 11, 4, KH=3, PAD=0)
        ok, why = is_valid_spec(_fprop(p, block_groups=1, dgrad_fused_weights=True))
        self.assertFalse(ok)
        self.assertIn("same", why)

    def test_lds_footprint_is_checked(self):
        from kernels.common.conv_direct_grouped import (
            direct_conv_lds_bytes,
            is_valid_spec,
        )

        p = _problem(2, 1024, 1024, 8, 8, 16, KH=7, PAD=3)  # cpg 64
        spec = _fprop(p, block_q=128, block_groups=16)
        self.assertGreater(direct_conv_lds_bytes(spec), 160 * 1024)
        ok, why = is_valid_spec(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("LDS", why)

    def test_lds_bytes_match_the_lowered_pool(self):
        # direct_conv_lds_bytes is the allocation the LDS pool packer makes:
        # the staged weight slice shares a slot with the first row buffer,
        # never with the second.
        from kernels.common.conv_direct_grouped import (
            build_direct_conv,
            direct_conv_lds_bytes,
        )
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        fwl = {"dgrad_fused_weights": True, "dgrad_weights_lds": True}
        cases = [
            (_problem(9, 256, 256, 10, 13, 16), {"block_groups": 2, **fwl}),
            (_problem(2, 32, 32, 9, 17, 4, dtype="fp16"), {"block_groups": 2, "double_buffer": False}),
            (_problem(2, 64, 128, 9, 17, 4, dtype="fp16"), {"block_q": 32, "block_groups": 1, "waves_q": 2, "waves_k": 2}),
            (_problem(8, 128, 128, 11, 19, 32), {"block_q": 32, "block_groups": 4, "block_h": 8, **fwl}),
            (_problem(3, 256, 256, 7, 9, 8), {"block_groups": 1, "fold_k32": True, **fwl}),
            (_problem(3, 32, 32, 7, 9, 2, KH=5, PAD=2), {"block_groups": 2, "block_h": 4, **fwl}),
            (_problem(9, 256, 256, 10, 13, 16), {"block_groups": 2, "lds_pad": 8, "stage_out": True, **fwl}),
            (_problem(8, 256, 256, 7, 15, 8), {"block_groups": 1, "fold_k32": True, "waves_m": 2, "lds_pad": 8, "stage_out": True, **fwl}),
            (_problem(2, 32, 32, 9, 17, 4, dtype="fp16"), {"block_groups": 2, "double_buffer": False, "lds_pad": 8}),
            (_problem(8, 128, 128, 11, 19, 32), {"block_q": 32, "block_groups": 4, "block_h": 8, "lds_pad": 8, **fwl}),
        ]  # fmt: skip
        for p, kw in cases:
            spec = _fprop(p, **kw)
            with self.subTest(spec=spec.kernel_name()):
                ll = lower_kernel_to_llvm(
                    build_direct_conv(spec, arch="gfx950"), arch="gfx950"
                )
                m = re.search(r"@smem_pool\.\S+ = .*global \[(\d+) x i8\]", ll)
                self.assertIsNotNone(m)
                self.assertEqual(direct_conv_lds_bytes(spec), int(m.group(1)))

    def test_row_stream_knobs_validate_and_name(self):
        from dataclasses import replace

        from kernels.common.conv_direct_grouped import (
            direct_conv_grid,
            direct_launch_geometry,
            is_valid_spec,
        )

        p = _problem(9, 256, 256, 10, 13, 16)
        base = _fprop(
            p, block_groups=2, dgrad_fused_weights=True, dgrad_weights_lds=True
        )
        tags = {
            "prefetch_rows": (3, "_pf3"),
            "lds_only_sync": (True, "_lso"),
            "lds_pad": (8, "_lp8"),
            "stage_out": (True, "_so"),
            "xcd_tiles": (True, "_xc"),
        }
        for field, (value, tag) in tags.items():
            with self.subTest(field=field):
                spec = replace(base, **{field: value})
                ok, why = is_valid_spec(spec)
                self.assertTrue(ok, why)
                self.assertIn(tag, spec.kernel_name())
                self.assertNotIn(tag, base.kernel_name())
        # prefetch_rows 0 and 1 are the same one-row-ahead kernel: one name.
        self.assertEqual(
            replace(base, prefetch_rows=1).kernel_name(), base.kernel_name()
        )
        # The launch grid is the same with or without the XCD order: the
        # kernel remaps its workgroup id in place.
        xc = replace(base, xcd_tiles=True)
        self.assertEqual(direct_launch_geometry(xc), direct_launch_geometry(base))
        self.assertEqual(direct_conv_grid(xc), (1, 8, 9))
        rejects = {
            "prefetch_rows must be in 0..3": replace(base, prefetch_rows=4),
            "stage_out needs kpg % 16": replace(
                base, stage_out=True, problem=replace(base.problem, kpg=8)
            ),
            "lds_pad must be a multiple of 8": replace(base, lds_pad=12),
            "waves_m > 1 needs": replace(base, waves_m=2),
            "need double_buffer=True": replace(
                base, lds_only_sync=True, double_buffer=False
            ),
            "xcd_tiles needs at least 8 image row bands": replace(
                base, xcd_tiles=True, problem=replace(base.problem, N=4)
            ),
        }  # fmt: skip
        for frag, spec in rejects.items():
            with self.subTest(reject=frag):
                try:
                    spec.validate()
                    ok, why = is_valid_spec(spec)
                except ValueError as e:
                    ok, why = False, str(e)
                self.assertFalse(ok)
                self.assertIn(frag, why)
        # The XCD order bakes the XCD count, so it is built for gfx950 only.
        gather = replace(xc, dgrad_weights_lds=False)
        self.assertTrue(is_valid_spec(gather, arch="gfx950")[0])
        ok, why = is_valid_spec(gather, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("xcd_tiles", why)

    def test_waves_m_splits_the_weight_register_budget(self):
        # Each of the waves_m waves keeps only its share of the output tiles'
        # weight fragments, so a 64-channel output (4 tiles) fits the preload
        # budget with waves_m = 4 and not with one wave.
        from dataclasses import replace

        from kernels.common.conv_direct_grouped import (
            PRELOAD_WEIGHT_VGPR_BUDGET,
            is_valid_spec,
            preload_weight_vgprs,
        )

        p = _problem(2, 256, 128, 9, 9, 4)  # transposed: cpg' 32, kpg' 64
        one = _fprop(p, block_groups=1, fold_k32=True, dgrad_fused_weights=True)
        four = replace(one, waves_m=4)
        self.assertGreater(preload_weight_vgprs(one), PRELOAD_WEIGHT_VGPR_BUDGET)
        self.assertEqual(preload_weight_vgprs(four), preload_weight_vgprs(one) // 4)
        self.assertFalse(is_valid_spec(one)[0])
        ok, why = is_valid_spec(four)
        self.assertTrue(ok, why)
        self.assertEqual(four.threads_per_block, 4 * 64)


# ---------------------------------------------------------------------------
# GPU numeric
# ---------------------------------------------------------------------------


def _rand_bits(shape, dtype, seed):
    f = np.random.default_rng(seed).uniform(-1, 1, size=shape).astype(np.float32)
    if dtype == "bf16":
        u = f.view(np.uint32)
        bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32), bits
    bits = f.astype(np.float16).view(np.uint16)
    return bits.view(np.float16).astype(np.float32), bits


def _dgrad_ref(dY, Wt, p):
    """dX[n,h,w,g*cpg+c] = sum_{r,s,k} dY[n,h+P-r,w+P-s,g*kpg+k] W[g*kpg+k,r,s,c]."""
    N = dY.shape[0]
    pad = p.KH
    dYp = np.pad(dY, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    dX = np.zeros((N, p.H, p.W, p.total_c), dtype=np.float32)
    for r in range(p.KH):
        for s in range(p.KW):
            h0 = p.PAD - r + pad
            w0 = p.PAD - s + pad
            win = dYp[:, h0 : h0 + p.H, w0 : w0 + p.W, :]
            for g in range(p.groups):
                ks = slice(g * p.kpg, (g + 1) * p.kpg)
                dX[..., g * p.cpg : (g + 1) * p.cpg] += np.einsum(
                    "nhwk,kc->nhwc", win[..., ks], Wt[ks, r, s, :], optimize=True
                )
    return dX


def _bad_count(out_bits, ref, dtype):
    tol = 1e-2
    if dtype == "bf16":
        D = (out_bits.astype(np.uint32) << 16).view(np.float32)
        u = ref.astype(np.float32).view(np.uint32)
        u = u + np.uint32(0x7FFF) + ((u >> 16) & 1).astype(np.uint32)
        R = (u & np.uint32(0xFFFF0000)).view(np.float32)
    else:
        D = out_bits.view(np.float16).astype(np.float32)
        R = ref.astype(np.float32)
    ok = np.abs(D - R) <= tol + tol * np.abs(R)
    return int((~ok).sum())


def _fill_cache(cache, caps_list, dtype, select=None):
    """Compile the fused variant's binaries of ``caps_list`` into ``cache``
    exactly as --compile-all does (``select(knobs)`` narrows the knobs)."""
    from benchmarks.common import direct_kernel_sweep as dks
    from benchmarks.common.kernel_sweep import compile_jobs
    from rocke.core.arch import ArchTarget

    wave = ArchTarget.from_gfx(GPU_ARCH).wave_size
    jobs = []
    for caps in caps_list:
        for _, job in dks._caps_jobs(GPU_ARCH, wave, _FUSED, "dgrad", caps, dtype)[1]:
            if select is None or select(job.spec_kwargs):
                jobs.append(job)
    compile_jobs(
        cache=cache,
        all_jobs=jobs,
        build=dks.direct_kernel_for_job,
        arch=GPU_ARCH,
        dtype=dtype,
        directions=["direct_dgrad"],
        jobs=os.cpu_count() or 1,
        log=lambda *_: None,
    )
    return len(jobs)


def _run_plan(plan, p):
    """Launch every step of a direct plan on ``p``; returns (dX bits, ref)."""
    from benchmarks.common.direct_kernel_sweep import launch_values
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    dY32, dYb = _rand_bits((p.N, p.Ho, p.Wo, p.total_k), p.dtype, 1)
    W32, Wb = _rand_bits((p.total_k, p.KH, p.KW, p.cpg), p.dtype, 2)
    sizes = {"dy": dYb.nbytes, "w": Wb.nbytes, "dx": p.N * p.H * p.W * p.total_c * 2}
    sizes.update(plan.workspaces)
    rt = Runtime()
    ptrs = {name: rt.alloc(nb) for name, nb in sizes.items()}
    try:
        for name, arr in (("dy", dYb), ("w", Wb)):
            rt.memcpy_h2d(
                ptrs[name],
                (ctypes.c_uint8 * arr.nbytes).from_address(arr.ctypes.data),
                arr.nbytes,
            )
        rt.memset(ptrs["dx"], 0xFF, sizes["dx"])  # NaN in fp16 and bf16
        for step in plan.steps:
            KernelLauncher(
                hsaco=step.hsaco_path.read_bytes(),
                kernel_name=step.kernel_name,
                signature=step.signature,
            )(
                launch_values(step, ptrs, sizes),
                config=LaunchConfig(grid=step.grid, block=step.block, fence=True),
            )
        synchronize_and_release(0)
        out = np.empty((p.N, p.H, p.W, p.total_c), dtype=np.uint16)
        rt.memcpy_d2h(
            (ctypes.c_uint8 * out.nbytes).from_address(out.ctypes.data),
            ptrs["dx"],
            out.nbytes,
        )
    finally:
        for d in ptrs.values():
            rt.free(d)
    return out, _dgrad_ref(dY32, W32, p)


# (label, (N, C, K, H, W, G, KH, dtype)): adversarial problems for the cached
# binaries -- odd H/W, N = 1, partial W tiles, images shorter and taller than
# the tiles (the row guard), cpg != kpg both ways, partial K-atoms, groups not
# a multiple of the wider block_groups, 1x1 / 5x5 / 7x7.
_SHAPES = [
    ("c4_g8_N1_odd", (1, 32, 32, 7, 13, 8, 3, "bf16")),
    ("c8_g4_partialW_fp16", (2, 32, 32, 13, 17, 4, 3, "fp16")),
    ("c16_g4_H17", (2, 64, 64, 17, 17, 4, 3, "bf16")),
    ("c32k32_g2", (3, 64, 64, 15, 9, 2, 3, "bf16")),
    ("c4_k8_asym_fp16", (2, 32, 64, 9, 9, 8, 3, "fp16")),
    ("c8_k4_asym", (2, 64, 32, 9, 9, 8, 3, "bf16")),
    ("c24_partial_atom_g3", (2, 72, 72, 7, 7, 3, 3, "bf16")),
    ("5x5_c16_g2_fp16", (2, 32, 32, 13, 13, 2, 5, "fp16")),
    ("7x7_c8_k12_g4", (1, 32, 48, 11, 10, 4, 7, "bf16")),
    ("tallH_c8", (1, 32, 32, 70, 20, 4, 3, "bf16")),
    # Enough row bands for the row-stream stack (XCD order) at every tile.
    ("c16_N8_stream", (8, 64, 64, 13, 15, 4, 3, "bf16")),
    ("c8_N9_stream_fp16", (9, 64, 64, 11, 21, 8, 3, "fp16")),
    ("1x1_c12_k24_H1", (2, 48, 96, 1, 11, 4, 1, "bf16")),
]


@unittest.skipUnless(
    GPU_ARCH in ("gfx942", "gfx950"), f"needs a gfx942/gfx950 GPU (got {GPU_ARCH!r})"
)
class TestDirectMfmaDgradCachedNumeric(unittest.TestCase):
    def _check_shape(self, label, shape, select=None):
        from benchmarks.common import direct_kernel_sweep as dks
        from benchmarks.common.kernel_cache import KernelCache

        N, C, K, H, W, G, KH, dtype = shape
        p = _problem(N, C, K, H, W, G, KH=KH, PAD=KH // 2, dtype=dtype)
        caps = dks.DirectCaps(KH=KH, PAD=KH // 2, stride=1, cpg=p.cpg, kpg=p.kpg)
        with tempfile.TemporaryDirectory() as tmp:
            cache = KernelCache(Path(tmp), GPU_ARCH)
            self.assertGreater(_fill_cache(cache, [caps], dtype, select), 0)
            plans, rejected = dks.direct_plans(cache, p, "dgrad", GPU_ARCH)
            fused = [pl for pl in plans if pl.identity.algorithm == _FUSED]
            self.assertTrue(fused, f"{label}: no fused plan ({rejected[:2]})")
            for plan in fused:
                with self.subTest(case=label, kernel=plan.steps[0].kernel_name):
                    out, ref = _run_plan(plan, p)
                    bad = _bad_count(out, ref, dtype)
                    self.assertEqual(
                        bad, 0, f"{label}: {bad}/{out.size} elements outside tolerance"
                    )

    def test_cached_binaries_on_adversarial_shapes(self):
        for label, shape in _SHAPES:
            self._check_shape(label, shape)


if __name__ == "__main__":
    unittest.main()
