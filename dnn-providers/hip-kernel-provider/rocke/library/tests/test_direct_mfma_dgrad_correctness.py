# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Correctness of the direct-MFMA grouped dgrad pipeline as dispatch ships it.

Two layers:

* ``TestDirectMfmaDgradPlanAndValidation`` (CPU only): the launch plan and the
  spec validator -- the pre-pass stage list per main-kernel path, the
  reorganize grid under ``fold_k32``, and the geometry rules that used to let
  silently-wrong kernels through (``fold_k32`` + ``waves_k`` atom slicing, a
  transposed ``kpg`` that is not a multiple of the 4-channel store vector).
* ``TestDirectMfmaDgradNumeric`` (GPU): builds every stage the plan lists,
  launches the pipeline and compares dX against a numpy reference with the
  manifest-runner conv rule (``|D - ref| <= 1e-2 + 1e-2*|ref|``, NaN is a
  failure). dX and every workspace are pre-filled with 0xFF bytes -- NaN in
  fp16 and bf16 -- so an unwritten output, or a workspace lane the pre-pass
  never wrote but the main kernel reads, fails instead of passing on a lucky
  zero. Covers the dispatcher's own picks on adversarial shapes above its size
  gate, the candidate's knob policy on small adversarial shapes below it (odd
  H/W, N=1, partial W tiles, partial H tiles, cpg != kpg, partial K-atoms, 5x5
  and 7x7) and regressions for the coalesced-weight path
  (``runtime_k_loop`` / ``waves_k > 1``), which read the reorganized workspace
  and were wrong for every grouped problem.

numpy is the only hard dependency; no torch.
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import replace

import numpy as np

if importlib.util.find_spec("torch") is not None:
    # This file does not use torch, but other test files in the same pytest
    # process do: torch's HIP runtime has to claim the process context before
    # rocke's does, or torch reports "No HIP GPUs are available" for the rest
    # of the session (see test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

from rocke.runtime.hip_module import get_device_arch

GPU_ARCH = get_device_arch(0)

_PREPASS_SIG = [
    {"name": "A", "type": "ptr<f16, global>", "size_bytes": 8},
    {"name": "D", "type": "ptr<f16, global>", "size_bytes": 8},
    {"name": "A_bytes", "type": "i32", "size_bytes": 4},
    {"name": "D_bytes", "type": "i32", "size_bytes": 4},
]


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

    k32 = kw.pop("fold_k32", False)
    spec = make_dgrad_fprop_spec(p, **kw)
    return replace(spec, fold_k32=True) if k32 else spec


class TestDirectMfmaDgradPlanAndValidation(unittest.TestCase):
    def test_default_path_reads_plain_transposed_weights(self):
        from kernels.common.conv_direct_grouped import plan_direct_mfma_dgrad

        p = _problem(2, 64, 64, 8, 8, 4)
        plan = plan_direct_mfma_dgrad(p, _fprop(p, block_groups=2))
        self.assertEqual([s.role for s in plan.stages], ["transpose", "main"])
        self.assertEqual(plan.main.b, "ws_wt")
        self.assertEqual(plan.workspace_bytes, p.total_c * 9 * p.kpg * 2)

    def test_coalesced_paths_add_the_reorganize_pass(self):
        from kernels.common.conv_direct_grouped import plan_direct_mfma_dgrad

        p = _problem(2, 256, 256, 8, 8, 8)  # cpg = kpg = 32
        for kw, n_k_atoms in (
            ({"runtime_k_loop": True}, 2),
            ({"runtime_k_loop": True, "fold_k32": True}, 1),
            ({"waves_k": 2}, 2),
        ):
            with self.subTest(kw=kw):
                plan = plan_direct_mfma_dgrad(p, _fprop(p, block_groups=1, **kw))
                self.assertEqual(
                    [s.role for s in plan.stages], ["transpose", "reorganize", "main"]
                )
                self.assertEqual(plan.main.b, "ws_coa")
                # One 64-lane block per (group, r, s, K-atom, M-tile); the
                # K-atom width follows the main kernel's atom (32 under fold).
                reorg = plan.stages[1]
                self.assertEqual(reorg.grid, (p.groups * 9 * n_k_atoms * 2, 1, 1))

    def test_transpose_grid_one_workgroup_per_staged_slice(self):
        # The LDS-staged transpose runs one 256-lane workgroup per (group,
        # filter row, chunk of source k rows): the largest chunk (within the
        # LDS budget) whose grid still has one workgroup per CU, else the
        # smallest. Small weight tensors and slices too large to stage use
        # the flat one-lane-per-element grid.
        from kernels.common.conv_direct_grouped import (
            _TRANSPOSE_STAGED_MIN_ELEMENTS,
            DIRECT_TRANSPOSE_WEIGHTS_BLOCK,
            _transpose_block_pad,
            _transpose_staging,
            plan_direct_mfma_dgrad,
        )

        def flat_grid(p):
            lanes = p.total_c * p.KH * p.KW * p.kpg
            return (-(-lanes // DIRECT_TRANSPOSE_WEIGHTS_BLOCK), 1, 1)

        for C, K, G, KH, chunks in (
            (6400, 3200, 200, 7, 1),
            (2048, 2048, 64, 7, 1),
            (1024, 1024, 16, 5, 4),
            (512, 512, 1, 3, 64),
        ):
            with self.subTest(C=C, K=K, G=G, KH=KH):
                p = _problem(2, C, K, 9, 9, G, KH=KH, PAD=KH // 2)
                self.assertGreaterEqual(
                    K * KH * KH * p.cpg, _TRANSPOSE_STAGED_MIN_ELEMENTS
                )
                plan = plan_direct_mfma_dgrad(p, _fprop(p, block_groups=1))
                st = plan.stages[0]
                self.assertEqual(st.role, "transpose")
                self.assertEqual(st.block, (DIRECT_TRANSPOSE_WEIGHTS_BLOCK, 1, 1))
                vin, vout, row, kc = _transpose_staging(p)
                self.assertEqual(_transpose_block_pad(p, vout, row, kc, vin) % vin, 0)
                self.assertEqual(p.kpg // kc, chunks)
                self.assertEqual(kc % vout, 0)
                self.assertLessEqual(kc * row * 2, 48 * 1024)
                self.assertTrue(G * KH * chunks >= 256 or kc == vout)
                self.assertEqual(st.grid, (G * KH * chunks, 1, 1))
        # Small weight tensors (latency-bound copy) and cpg 2048 at 7x7 (one
        # 4-row block of the slice is past the LDS cap) stay flat.
        for C, K, G, KH in (
            (1024, 128, 32, 5),
            (128, 1024, 32, 3),
            (72, 72, 3, 7),
            (256, 256, 1, 3),
            (2048, 4, 1, 7),
        ):
            with self.subTest(C=C, K=K, G=G, KH=KH, form="flat"):
                p = _problem(1, C, K, 4, 4, G, KH=KH, PAD=KH // 2)
                self.assertIsNone(_transpose_staging(p))
                plan = plan_direct_mfma_dgrad(p, _fprop(p, block_groups=1))
                self.assertEqual(plan.stages[0].grid, flat_grid(p))

    def test_transpose_block_pad_spreads_the_gather_over_banks(self):
        # cpg 32 / kpg 32 at 7x7: the 8-row block stride is a multiple of the
        # bank count, so without a shift most of a wave gathers from a few
        # banks; the chosen shift is one staging vector per block (16 bytes
        # for cpg 32, 8 bytes for cpg 28, whose runs take 4-element vectors).
        from kernels.common.conv_direct_grouped import (
            _transpose_block_pad,
            _transpose_staging,
        )

        for C, K, G, pad in ((9600, 9600, 300, 8), (8148, 9312, 291, 4)):
            with self.subTest(C=C, K=K, G=G):
                p = _problem(1, C, K, 10, 10, G, KH=7, PAD=3)
                vin, vout, row, kc = _transpose_staging(p)
                self.assertEqual(_transpose_block_pad(p, vout, row, kc, vin), pad)

    def test_plan_rejects_a_spec_for_another_problem(self):
        from kernels.common.conv_direct_grouped import plan_direct_mfma_dgrad

        p = _problem(2, 64, 128, 8, 8, 4)
        other = _problem(2, 128, 64, 8, 8, 4)
        with self.assertRaises(ValueError):
            plan_direct_mfma_dgrad(p, _fprop(other))

    def test_fold_k32_waves_k_counts_32_wide_atoms(self):
        # kpg = 32 -> one 32-wide atom; waves_k=2 would give each wave zero
        # atoms and the kernel would write zeros. Must be rejected.
        from kernels.common.conv_direct_grouped import is_valid_spec

        p = _problem(2, 256, 256, 8, 8, 8)
        ok, why = is_valid_spec(_fprop(p, block_groups=1, waves_k=2, fold_k32=True))
        self.assertFalse(ok)
        self.assertIn("waves_k", why)
        ok, _ = is_valid_spec(_fprop(p, block_groups=1, waves_k=2))
        self.assertTrue(ok)

    def test_transposed_kpg_must_be_a_store_vector_multiple(self):
        # Original cpg = 6 -> transposed kpg = 6: the 4-channel store vector
        # would spill into the next group's channels.
        from kernels.common.conv_direct_grouped import is_valid_spec

        p = _problem(2, 48, 32, 8, 8, 8)
        ok, why = is_valid_spec(_fprop(p, block_groups=1))
        self.assertFalse(ok)
        self.assertIn("multiple of 4", why)

    def test_non_same_padding_is_rejected(self):
        # The row stream emits output rows 0..H-1 of its input height; with
        # pad 0 on a 3x3 the transposed problem pads by 2 and its last two
        # output rows were silently never written.
        from kernels.common.conv_direct_grouped import is_valid_spec

        p = _problem(2, 32, 32, 9, 11, 4, KH=3, PAD=0)
        ok, why = is_valid_spec(_fprop(p, block_groups=1))
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


def _run_plan(plan, arch):
    """Compile + launch every stage of ``plan``; returns (dX bits, ref)."""
    from kernels.common.conv_direct_grouped import direct_mfma_dgrad_stage_kernel
    from rocke import compile_kernel
    from rocke.helpers.manifest import conv_args_signature
    from rocke.runtime import synchronize_and_release
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    p = plan.problem
    dY32, dYb = _rand_bits((p.N, p.Ho, p.Wo, p.total_k), p.dtype, 1)
    W32, Wb = _rand_bits((p.total_k, p.KH, p.KW, p.cpg), p.dtype, 2)
    nbytes = dict(plan.buffer_bytes)
    rt = Runtime()
    ptr = {role: rt.alloc(nb) for role, nb in nbytes.items()}
    for role, arr in (("dY", dYb), ("W", Wb)):
        rt.memcpy_h2d(
            ptr[role],
            (ctypes.c_uint8 * arr.nbytes).from_address(arr.ctypes.data),
            arr.nbytes,
        )
    for role, nb in nbytes.items():
        if role not in ("dY", "W"):
            rt.memset(ptr[role], 0xFF, nb)  # NaN in fp16 and bf16
    try:
        for st in plan.stages:
            art = compile_kernel(
                direct_mfma_dgrad_stage_kernel(st, arch=arch), arch=arch
            )
            if st.b is None:
                launcher = KernelLauncher(
                    hsaco=art.hsaco, kernel_name=art.kernel_name, signature=_PREPASS_SIG
                )
                values = {
                    "A": ptr[st.a],
                    "D": ptr[st.d],
                    "A_bytes": nbytes[st.a],
                    "D_bytes": nbytes[st.d],
                }
            else:
                launcher = KernelLauncher(
                    hsaco=art.hsaco,
                    kernel_name=art.kernel_name,
                    signature=conv_args_signature(p.dtype),
                )
                values = {
                    "A": ptr[st.a],
                    "B": ptr[st.b],
                    "D": ptr[st.d],
                    "A_bytes": nbytes[st.a],
                    "B_bytes": nbytes[st.b],
                    "D_bytes": nbytes[st.d],
                }
            launcher(
                values, config=LaunchConfig(grid=st.grid, block=st.block, fence=True)
            )
        synchronize_and_release(0)
        out = np.empty((p.N, p.H, p.W, p.total_c), dtype=np.uint16)
        rt.memcpy_d2h(
            (ctypes.c_uint8 * out.nbytes).from_address(out.ctypes.data),
            ptr["dX"],
            out.nbytes,
        )
    finally:
        for d in ptr.values():
            rt.free(d)
    return out, _dgrad_ref(dY32, W32, p)


# (N, C, K, H, W, G, KH, PAD, dtype): adversarial geometry run with the knobs
# the direct candidate's policy picks (``_select_direct_dgrad_spec``). Most are
# outside the dispatcher's measured win region -- dispatch sends them to
# igemm -- but the knob policy must still produce a correct pipeline for them:
# the region is a performance decision, not a correctness one.
_POLICY = [
    ("cpg4_g8_N1_odd", (1, 32, 32, 7, 13, 8, 3, 1, "bf16")),
    ("cpg8_g4_odd_partialW", (2, 32, 32, 13, 17, 4, 3, 1, "fp16")),
    ("cpg16_g4_H17", (2, 64, 64, 17, 17, 4, 3, 1, "bf16")),
    ("cpg32_k32_g2", (3, 64, 64, 15, 9, 2, 3, 1, "bf16")),
    ("cpg4_kpg8_asym", (2, 32, 64, 9, 9, 8, 3, 1, "fp16")),
    ("cpg8_kpg4_asym", (2, 64, 32, 9, 9, 8, 3, 1, "bf16")),
    ("cpg24_partial_atom_g3", (2, 72, 72, 7, 7, 3, 3, 1, "bf16")),
    ("5x5_cpg16_g2", (2, 32, 32, 13, 13, 2, 5, 2, "fp16")),
    ("7x7_cpg32_g2", (1, 64, 64, 17, 17, 2, 7, 3, "bf16")),
    ("tallH_partial_tile_cpg8", (1, 32, 32, 70, 20, 4, 3, 1, "fp16")),
]

# Shapes the dispatch policy admits, so dispatch itself routes them to the
# direct candidate; still adversarial (odd H/W, partial W and H tiles,
# cpg != kpg, partial K-atom, H > 64, 7x7 + fold_k32 on the pre-pass fallback,
# the 4c row on odd widths and 1x1).
_DISPATCHED = [
    ("N1_57x57_cpg4", (1, 128, 128, 57, 57, 32, 3, 1, "bf16")),
    ("cpg8_kpg16_29x31_fp16", (3, 256, 512, 29, 31, 32, 3, 1, "fp16")),
    ("tallH70_cpg8_g16", (3, 128, 128, 70, 20, 16, 3, 1, "bf16")),
    ("cpg24_partial_atom_g3", (32, 72, 72, 28, 28, 3, 3, 1, "bf16")),
    ("7x7_cpg32_k32_23x23", (8, 1024, 1024, 23, 23, 32, 7, 3, "bf16")),
    # cpg > kpg (narrow reduction) where dispatch admits it, with the wide
    # 32-column strip and 4-row H tiles: odd and partial strips / tiles.
    ("4c_cpg4_29x31_N24", (24, 128, 128, 29, 31, 32, 3, 1, "bf16")),
    ("4c_1x1_15x17_fp16", (64, 64, 64, 15, 17, 16, 1, 0, "fp16")),
    ("5x5_cpg16_kpg4_31x30_fp16", (8, 512, 128, 31, 30, 32, 5, 2, "fp16")),
    ("3x3_cpg32_kpg8_30x30_N4", (4, 1024, 256, 30, 30, 32, 3, 1, "bf16")),
]

# Explicit main-kernel knobs that read the coalesced workspace (the paths the
# reorganize pass feeds), plus the default path on the same shapes.
_COALESCED = [
    (
        "rk_wk1_cpg4_g8",
        (2, 32, 32, 8, 8, 8, 3, 1, "fp16"),
        {"block_groups": 1, "runtime_k_loop": True},
    ),
    (
        "rk_wk1_cpg4_bg2",
        (2, 32, 32, 8, 8, 8, 3, 1, "bf16"),
        {"block_groups": 2, "runtime_k_loop": True},
    ),
    (
        "rk_cpg32_k32",
        (2, 256, 256, 8, 8, 8, 3, 1, "fp16"),
        {"block_groups": 1, "runtime_k_loop": True, "fold_k32": True},
    ),
    (
        "wk2_cpg32",
        (2, 256, 256, 8, 8, 8, 3, 1, "fp16"),
        {"block_groups": 1, "waves_k": 2},
    ),
    (
        "rk_cpg24_partial_atom",
        (2, 48, 96, 7, 13, 4, 3, 1, "bf16"),
        {"block_groups": 1, "runtime_k_loop": True},
    ),
]


@unittest.skipUnless(GPU_ARCH == "gfx950", f"needs a gfx950 GPU (got {GPU_ARCH!r})")
class TestDirectMfmaDgradNumeric(unittest.TestCase):
    def _check(self, plan, label):
        out, ref = _run_plan(plan, GPU_ARCH)
        bad = _bad_count(out, ref, plan.problem.dtype)
        self.assertEqual(
            bad, 0, f"{label}: {bad}/{out.size} elements outside tolerance"
        )

    @staticmethod
    def _request(N, C, K, H, W, G, KH, PAD, dtype):
        from dispatch.grouped_convolution import ConvGroupedRequest

        return ConvGroupedRequest(
            N=N,
            C=C,
            K=K,
            Hi=H,
            Wi=W,
            Y=KH,
            X=KH,
            G=G,
            pad_h=PAD,
            pad_w=PAD,
            dtype=dtype,
            arch="gfx950",
            direction="dgrad",
        )

    def test_weight_transpose_is_bit_exact(self):
        # The pre-pass alone, against numpy's flip + per-group k<->c
        # transpose, workspace 0xFF-poisoned. Each case runs in the form the
        # size rule picks and with staging forced on every tensor that fits
        # (minimum size 0): covers 8- and 16-byte vectors, tail passes, k
        # chunking (G=1, cpg = kpg = 256), the per-block LDS shift (cpg 32
        # at 7x7, one and two k blocks), the flat form of small tensors and
        # the flat fallback past the LDS cap (cpg 1024 at 7x7).
        from unittest import mock

        import kernels.common.conv_direct_grouped as cdg
        from kernels.common.conv_direct_grouped import (
            DIRECT_TRANSPOSE_WEIGHTS_BLOCK,
            DirectTransposeWeightsDgradSpec,
            build_direct_transpose_weights_dgrad,
            direct_transpose_weights_dgrad_grid,
        )
        from rocke import compile_kernel
        from rocke.runtime import synchronize_and_release
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        rt = Runtime()
        for cpg, kpg, KH, G, dt in (
            (32, 16, 7, 200, "fp16"),
            (32, 32, 7, 30, "bf16"),
            (28, 32, 3, 7, "bf16"),
            (12, 20, 5, 3, "fp16"),
            (20, 12, 7, 5, "bf16"),
            (4, 4, 1, 16, "fp16"),
            (36, 44, 7, 3, "fp16"),
            (256, 256, 3, 1, "bf16"),
            (1024, 8, 7, 1, "fp16"),
        ):
            for staged_min in (None, 0):
                min_elems = (
                    cdg._TRANSPOSE_STAGED_MIN_ELEMENTS if staged_min is None else 0
                )
                with (
                    self.subTest(
                        cpg=cpg, kpg=kpg, KH=KH, G=G, dtype=dt, staged_min=staged_min
                    ),
                    mock.patch.object(cdg, "_TRANSPOSE_STAGED_MIN_ELEMENTS", min_elems),
                ):
                    p = _problem(
                        1, G * cpg, G * kpg, 8, 8, G, KH=KH, PAD=KH // 2, dtype=dt
                    )
                    w = np.random.default_rng(5).integers(
                        0, 0x7C00, size=(G * kpg, KH, KH, cpg), dtype=np.uint16
                    )
                    ref = np.ascontiguousarray(
                        w.reshape(G, kpg, KH, KH, cpg)[:, :, ::-1, ::-1, :]
                        .transpose(0, 4, 2, 3, 1)
                        .reshape(G * cpg, KH, KH, kpg)
                    )
                    art = compile_kernel(
                        build_direct_transpose_weights_dgrad(
                            DirectTransposeWeightsDgradSpec(problem=p), arch=GPU_ARCH
                        ),
                        arch=GPU_ARCH,
                    )
                    d_w, d_ws = rt.alloc(w.nbytes), rt.alloc(w.nbytes)
                    try:
                        rt.memcpy_h2d(
                            d_w,
                            (ctypes.c_uint8 * w.nbytes).from_address(w.ctypes.data),
                            w.nbytes,
                        )
                        rt.memset(d_ws, 0xFF, w.nbytes)
                        KernelLauncher(
                            hsaco=art.hsaco,
                            kernel_name=art.kernel_name,
                            signature=_PREPASS_SIG,
                        )(
                            {
                                "A": d_w,
                                "D": d_ws,
                                "A_bytes": w.nbytes,
                                "D_bytes": w.nbytes,
                            },
                            config=LaunchConfig(
                                grid=direct_transpose_weights_dgrad_grid(p),
                                block=(DIRECT_TRANSPOSE_WEIGHTS_BLOCK, 1, 1),
                                fence=True,
                            ),
                        )
                        synchronize_and_release(0)
                        out = np.empty(ref.shape, dtype=np.uint16)
                        rt.memcpy_d2h(
                            (ctypes.c_uint8 * out.nbytes).from_address(out.ctypes.data),
                            d_ws,
                            out.nbytes,
                        )
                    finally:
                        rt.free(d_w)
                        rt.free(d_ws)
                    self.assertEqual(int((out != ref).sum()), 0)

    def test_dispatched_configs(self):
        from dispatch.grouped_convolution import dispatch_conv_grouped

        for label, shape in _DISPATCHED:
            with self.subTest(case=label):
                req = self._request(*shape)
                r = dispatch_conv_grouped(req)
                self.assertEqual(r.candidate.name, "direct_mfma_conv_dgrad", label)
                self._check(r.spec.launch_plan(req), label)

    def test_pre_pass_fallback_on_dispatched_shapes(self):
        # The same shapes on the pre-pass pipeline (transpose + main reading
        # W_T), which the candidate falls back to past the fused form's
        # register budget.
        from dataclasses import replace

        from dispatch.grouped_convolution import dispatch_conv_grouped

        for label, shape in _DISPATCHED:
            req = self._request(*shape)
            spec = dispatch_conv_grouped(req).spec
            if spec.variant != "generic":
                continue
            with self.subTest(case=label):
                pre = replace(
                    spec, fused_weights=False, weights_lds=False, waves_per_eu=0
                )
                plan = pre.launch_plan(req)
                self.assertEqual(plan.stages[0].role, "transpose")
                self._check(plan, label)

    def test_policy_knobs_outside_the_win_region(self):
        from dispatch.grouped_convolution import _select_direct_dgrad_spec

        for label, shape in _POLICY:
            with self.subTest(case=label):
                req = self._request(*shape)
                self._check(_select_direct_dgrad_spec(req).launch_plan(req), label)

    def test_coalesced_weight_paths(self):
        from kernels.common.conv_direct_grouped import plan_direct_mfma_dgrad

        for label, (N, C, K, H, W, G, KH, PAD, dtype), kw in _COALESCED:
            with self.subTest(case=label):
                p = _problem(N, C, K, H, W, G, KH=KH, PAD=PAD, dtype=dtype)
                plan = plan_direct_mfma_dgrad(p, _fprop(p, **kw))
                self.assertIn("reorganize", [s.role for s in plan.stages])
                self._check(plan, label)


if __name__ == "__main__":
    unittest.main()
