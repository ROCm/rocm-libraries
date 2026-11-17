# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Direct-MFMA dgrad with the weight transform fused into the main kernel.

``DirectConvSpec`` knobs under test (spec from ``make_dgrad_fprop_spec``, the
forward row stream over the transposed problem):

* ``preload_weights``     -- waves_k == 1: load every W_T fragment once in the
  prologue (the pre-pass pipeline is unchanged otherwise);
* ``dgrad_fused_weights`` -- B is the ORIGINAL weight ``W[K, KH, KW, cpg]``;
  the prologue reads it with flipped taps and per-group k<->c transposed
  addressing, so the transpose pre-pass kernel and its workspace disappear;
* ``dgrad_weights_lds``   -- with fused weights: stage the workgroup's raw W
  slice in LDS (pool-overlaid with the row buffers) and build the fragments
  with ``ds_read_b64_tr_b16`` instead of per-element gathers (gfx950);
* ``waves_per_eu``        -- occupancy hint for the scheduler;
* ``row_guard``           -- the MFMA work of a row past the image is skipped
  under a block-uniform guard (a tile taller than the image);
* the row-stream knobs ``prefetch_rows`` / ``lds_only_sync`` / ``waves_m`` /
  ``lds_pad`` / ``stage_out`` / ``xcd_tiles`` and the early exit
  ``row_exit``, alone and stacked;

plus the kernel-level hooks ``direct_dgrad_spec_for_problem`` /
``direct_dgrad_launch`` / ``build_direct_dgrad``.

CPU tests cover spec plumbing, validation and emission. GPU tests (gfx942 /
gfx950 + torch) check numerics against ``torch.nn.grad.conv2d_input`` over
adversarial shapes: odd H / W, N = 1, W not a multiple of block_q, partial
K atoms (cpg 8 / 12), partial M tiles, cpg != kpg, 3 K atoms, fold_k32,
H tiles taller than the image and block_q = 32. Every kernel is launched
through the AOT forward direct ABI over the transposed problem
(``ConvArgs`` / ``conv_direct_args_signature``), and one binary per form is
launched on several unrelated shapes. Elementwise rule: an element is bad when
``|out - ref| > tol + tol * |ref|`` with ``tol = 1e-2``; a case passes iff no
element is bad (dX is NaN-filled first).
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import replace

from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConvProblem,
    DirectConvSpec,
    DirectTransposeWeightsDgradSpec,
    build_direct_conv,
    build_direct_dgrad,
    build_direct_transpose_weights_dgrad,
    dgrad_4c_transposed_problem,
    direct_dgrad_launch,
    direct_dgrad_spec_for_problem,
    direct_dgrad_workspace_bytes,
    direct_launch_geometry,
    is_valid_dgrad_fused_problem,
    is_valid_spec,
    make_dgrad_fprop_spec,
)
from rocke.runtime.hip_module import get_device_arch

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    # Claim the HIP context for torch before rocke's runtime touches it (see
    # test_direct_conv_correctness.py).
    import torch

    torch.cuda.is_available()

GPU_ARCH = get_device_arch(0)
_GPU_SKIP = (
    ""
    if (GPU_ARCH in ("gfx942", "gfx950") and _HAS_TORCH)
    else f"needs gfx942/gfx950 + torch (arch={GPU_ARCH!r}, torch={_HAS_TORCH})"
)
_TOL = 1e-2
_TR_READ = "call <4 x i16> @llvm.amdgcn.ds.read.tr16.b64("


def _problem(N, H, W, groups, cpg, kpg, dtype="bf16", K=3, stride=1):
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=cpg,
        kpg=kpg,
        KH=K,
        KW=K,
        PAD=(K - 1) // 2,
        stride=stride,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# CPU-only: spec plumbing, validation, emission, kernel-level hooks
# ---------------------------------------------------------------------------


class TestFusedWeightSpec(unittest.TestCase):
    def setUp(self):
        self.p = _problem(N=2, H=7, W=13, groups=32, cpg=16, kpg=16)

    def test_kernel_name_flags(self):
        base = make_dgrad_fprop_spec(self.p, block_groups=2)
        name = base.kernel_name()
        self.assertEqual(
            name, "direct_mfma_dgrad_N2H7W13_g32_c16k16_bq16_bg2_db_bh16_bf16"
        )
        self.assertEqual(
            replace(base, preload_weights=True).kernel_name(), name + "_pw"
        )
        fw = replace(base, dgrad_fused_weights=True)
        self.assertEqual(fw.kernel_name(), name + "_fw")
        fwl = replace(fw, dgrad_weights_lds=True)
        self.assertEqual(fwl.kernel_name(), name + "_fwl")
        self.assertEqual(replace(fwl, waves_per_eu=4).kernel_name(), name + "_fwl_we4")
        self.assertEqual(
            replace(fwl, waves_per_eu=4, row_guard=True).kernel_name(),
            name + "_fwl_we4_rg",
        )

    def test_invalid_combinations(self):
        base = make_dgrad_fprop_spec(self.p, block_groups=1)
        bad = {
            "rk": replace(base, dgrad_fused_weights=True, runtime_k_loop=True),
            "pg": replace(base, preload_weights=True, block_h=8, persistent_grid=True),
            "lds_only": replace(base, dgrad_weights_lds=True),
            "kpg_not4": replace(
                make_dgrad_fprop_spec(
                    _problem(N=1, H=8, W=8, groups=8, cpg=6, kpg=64), block_groups=1
                ),
                dgrad_fused_weights=True,
                dgrad_weights_lds=True,
            ),
            "vgpr_budget": make_dgrad_fprop_spec(
                _problem(N=1, H=8, W=8, groups=4, cpg=64, kpg=64),
                dgrad_fused_weights=True,
            ),
            "lds_budget": make_dgrad_fprop_spec(
                _problem(N=1, H=8, W=8, groups=16, cpg=32, kpg=32),
                block_groups=8,
                fold_k32=True,
                dgrad_fused_weights=True,
                dgrad_weights_lds=True,
            ),
            "row_guard_plain": replace(base, row_guard=True),
            "rtc": make_dgrad_fprop_spec(
                self.p,
                block_groups=1,
                runtime_channels=True,
                k_atoms=1,
                m_tiles=1,
                dgrad_fused_weights=True,
            ),
        }
        for tag, s in bad.items():
            with self.subTest(case=tag):
                with self.assertRaises(ValueError):
                    s.validate()
                self.assertFalse(is_valid_spec(s, arch="gfx950")[0])
        wk1 = make_dgrad_fprop_spec(self.p, block_groups=1, waves_k=1)
        with self.assertRaises(ValueError):
            replace(wk1, waves_k=2, dgrad_fused_weights=True).validate()

    def test_lds_needs_transpose_reads(self):
        s = make_dgrad_fprop_spec(
            self.p, block_groups=2, dgrad_fused_weights=True, dgrad_weights_lds=True
        )
        self.assertTrue(is_valid_spec(s, arch="gfx950")[0])
        ok, why = is_valid_spec(s, arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("ds_read_b64_tr_b16", why)
        self.assertTrue(
            is_valid_spec(replace(s, dgrad_weights_lds=False), arch="gfx942")[0]
        )

    def test_kpg_must_be_a_store_vector_multiple(self):
        # The epilogue stores 4 output channels per lane: a ragged kpg would
        # spill into the next group's channels, fused or not.
        p = DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=6)
        ok, why = is_valid_spec(DirectConvSpec(problem=p, block_groups=1))
        self.assertFalse(ok)
        self.assertIn("multiple of 4", why)

    def test_emission(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        base = make_dgrad_fprop_spec(self.p, block_groups=2, block_h=8)
        ll_b = lower_kernel_to_llvm(
            build_direct_conv(base, arch="gfx950"), arch="gfx950"
        )
        fw = replace(base, dgrad_fused_weights=True)
        ll_g = lower_kernel_to_llvm(build_direct_conv(fw, arch="gfx950"), arch="gfx950")
        ll_l = lower_kernel_to_llvm(
            build_direct_conv(replace(fw, dgrad_weights_lds=True), arch="gfx950"),
            arch="gfx950",
        )
        self.assertNotIn(_TR_READ, ll_b)
        self.assertNotIn(_TR_READ, ll_g)
        # KH * KW taps x 1 K atom x 1 M tile, one transpose read each.
        self.assertEqual(ll_l.count(_TR_READ), 9)
        # Two extra barriers: publish the staged slice, then retire it before
        # the overlaid row buffer is written.
        barrier = "call void @llvm.amdgcn.s.barrier()"
        self.assertEqual(ll_l.count(barrier), ll_b.count(barrier) + 2)
        k32 = make_dgrad_fprop_spec(
            _problem(N=1, H=7, W=9, groups=16, cpg=32, kpg=32),
            block_groups=2,
            block_h=8,
            fold_k32=True,
            dgrad_fused_weights=True,
            dgrad_weights_lds=True,
        )
        ll_k = lower_kernel_to_llvm(
            build_direct_conv(k32, arch="gfx950"), arch="gfx950"
        )
        # 9 taps x 1 K32 atom x 2 M tiles, two b64 transpose reads per fragment.
        self.assertEqual(ll_k.count(_TR_READ), 36)
        # waves_per_eu reaches the kernel attributes.
        ll_w = lower_kernel_to_llvm(
            build_direct_conv(replace(fw, waves_per_eu=4), arch="gfx950"),
            arch="gfx950",
        )
        self.assertIn('"amdgpu-waves-per-eu"="4,4"', ll_w)

    def test_row_guard_is_one_trip_loop_per_late_row(self):
        # Rows y_local > PAD of the BLOCK_H + KH - 1 unrolled rows can lie past
        # the image: each gets one guarded (runtime trip count) loop holding
        # its MFMAs; rows <= PAD never can.
        spec = make_dgrad_fprop_spec(
            self.p, block_groups=2, block_h=8, dgrad_fused_weights=True
        )
        plain = build_direct_conv(spec, arch="gfx950")
        guarded = build_direct_conv(replace(spec, row_guard=True), arch="gfx950")

        def loops(k):
            return [op for op in k.body.ops if op.name == "scf.for"]

        def mma(ops):
            return sum(1 for op in ops if op.name == "tile.mma")

        self.assertEqual(loops(plain), [])
        guard = loops(guarded)
        n_rows = 8 + 3 - 1
        self.assertEqual(len(guard), n_rows - 1 - 1)  # y_local in (PAD, n_rows)
        total = mma(guarded.body.ops) + sum(mma(g.regions[0].ops) for g in guard)
        self.assertEqual(total, mma(plain.body.ops))

    def test_fused_problem_rule(self):
        P = DirectConvProblem
        self.assertTrue(is_valid_dgrad_fused_problem(self.p)[0])
        bad = {
            "stride2": P(N=1, H=8, W=8, groups=8, cpg=16, kpg=16, stride=2),
            "pad0_3x3": P(N=2, H=10, W=12, groups=8, cpg=16, kpg=16, PAD=0),
            "1x3_pad1": P(N=2, H=10, W=12, groups=8, cpg=16, kpg=16, KH=1, KW=3),
            "cpg6": P(N=2, H=9, W=11, groups=8, cpg=6, kpg=16),
            "kpg10": P(N=2, H=9, W=11, groups=8, cpg=8, kpg=10),
        }
        for tag, p in bad.items():
            with self.subTest(case=tag):
                self.assertFalse(is_valid_dgrad_fused_problem(p)[0])

    def test_dgrad_spec_hook(self):
        p4 = _problem(N=2, H=13, W=13, groups=32, cpg=4, kpg=4)
        s4 = direct_dgrad_spec_for_problem(p4)
        self.assertIsInstance(s4, DirectConv4cSpec)
        self.assertTrue(s4.dgrad_fused_weights and s4.dgrad_weights_lds)

        s16 = direct_dgrad_spec_for_problem(self.p)
        self.assertIsInstance(s16, DirectConvSpec)
        self.assertTrue(s16.dgrad_fused_weights and s16.dgrad_weights_lds)
        # A 7-row image is one 8-row tile, the row past it guarded.
        self.assertEqual((s16.block_q, s16.block_groups, s16.block_h), (16, 2, 8))
        self.assertTrue(s16.row_guard)
        self.assertEqual(s16.waves_per_eu, 4)
        self.assertFalse(s16.fold_k32)

        p32 = _problem(N=2, H=8, W=17, groups=16, cpg=32, kpg=32)
        s32 = direct_dgrad_spec_for_problem(p32)
        self.assertTrue(s32.fold_k32 and s32.dgrad_weights_lds)
        self.assertEqual(s32.block_h, 8)
        self.assertFalse(s32.row_guard)  # 8 rows fill the tile
        self.assertEqual(s32.waves_per_eu, 0)
        s32_942 = direct_dgrad_spec_for_problem(p32, arch="gfx942")
        self.assertTrue(s32_942.dgrad_fused_weights)
        self.assertFalse(s32_942.dgrad_weights_lds or s32_942.fold_k32)

        for p in (
            _problem(N=1, H=8, W=8, groups=4, cpg=64, kpg=64),  # VGPR budget
            _problem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16, stride=2),
            _problem(N=1, H=8, W=8, groups=8, cpg=16, kpg=6),
        ):
            with self.subTest(p=p.short(), stride=p.stride):
                self.assertIsNone(direct_dgrad_spec_for_problem(p))
        self.assertIsNone(direct_dgrad_spec_for_problem(self.p, arch="gfx000"))

    def test_dgrad_spec_hook_rejects_unsupported_shapes(self):
        """Shapes the row-streaming kernel would compute wrongly get no spec.

        The kernel needs "same" padding (Ho == H, Wo == W) and both channel
        counts in whole 4-channel slices. The same rule rejects the transposed
        spec in ``validate`` / ``is_valid_spec`` (pre-pass included) and the
        generic forward spec.
        """
        P = DirectConvProblem
        bad = {
            "pad0_3x3": P(N=2, H=10, W=12, groups=8, cpg=16, kpg=16, PAD=0),
            "pad2_3x3": P(N=2, H=10, W=12, groups=8, cpg=16, kpg=16, PAD=2),
            "1x3_pad1": P(N=2, H=10, W=12, groups=8, cpg=16, kpg=16, KH=1, KW=3),
            "3x1_pad1": P(N=2, H=10, W=12, groups=8, cpg=16, kpg=16, KH=3, KW=1),
            "4c_pad0": P(N=2, H=10, W=12, groups=32, cpg=4, kpg=4, PAD=0),
            "cpg6_kpg64": P(N=1, H=8, W=8, groups=8, cpg=6, kpg=64, dtype="fp16"),
            "cpg6_kpg16": P(N=2, H=9, W=11, groups=8, cpg=6, kpg=16),
            "cpg2_kpg8": P(N=2, H=9, W=11, groups=16, cpg=2, kpg=8),
            "cpg10_kpg12": P(N=2, H=9, W=11, groups=8, cpg=10, kpg=12),
        }
        for tag, p in bad.items():
            for arch in ("gfx950", "gfx942"):
                with self.subTest(case=tag, arch=arch):
                    self.assertIsNone(direct_dgrad_spec_for_problem(p, arch=arch))
            if p.kpg % 4 == 0 and p.kpg >= 4 and p.KH == p.KW:
                for knobs in ({}, {"dgrad_fused_weights": True}):
                    s = make_dgrad_fprop_spec(p, block_groups=1, **knobs)
                    with self.subTest(case=tag, knobs=knobs, path="spec"):
                        self.assertFalse(is_valid_spec(s, arch="gfx950")[0])
            # The forward kernel on the original problem is rejected by the
            # same rule whenever its output side (kpg / padding) is unsupported.
            if p.cpg % 4 == 0:
                fp = DirectConvSpec(problem=p, block_groups=1)
                ok_shape = 2 * p.PAD == p.KH - 1 == p.KW - 1 and p.kpg % 4 == 0
                with self.subTest(case=tag, path="fprop"):
                    self.assertEqual(is_valid_spec(fp, arch="gfx950")[0], ok_shape)

    def test_launch_and_build(self):
        s = direct_dgrad_spec_for_problem(self.p)
        L = direct_dgrad_launch(s)
        self.assertEqual(
            L, {"grid": (1, 16, 2), "block": (128, 1, 1), "workspace_bytes": 0}
        )
        self.assertEqual(build_direct_dgrad(s, arch="gfx950").name, s.kernel_name())
        bh = replace(s, block_h=4)
        self.assertEqual(direct_dgrad_launch(bh)["grid"], (1, 16, 4))
        with self.assertRaises(ValueError):
            direct_dgrad_launch(
                replace(s, dgrad_fused_weights=False, dgrad_weights_lds=False)
            )
        s4 = direct_dgrad_spec_for_problem(
            _problem(N=2, H=13, W=13, groups=32, cpg=4, kpg=4)
        )
        self.assertEqual(direct_dgrad_launch(s4)["workspace_bytes"], 0)
        self.assertEqual(build_direct_dgrad(s4, arch="gfx950").name, s4.kernel_name())


# ---------------------------------------------------------------------------
# GPU numerics
# ---------------------------------------------------------------------------


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _compile(kernel):
    from rocke import compile_kernel

    return compile_kernel(kernel, arch=GPU_ARCH)


def _run(p: DirectConvProblem, spec: DirectConvSpec, art=None) -> tuple[bool, str]:
    """dgrad of ``p`` with ``spec`` (built on any problem of its shape class,
    or taken from ``art``): fused, one kernel; else the transpose pre-pass and
    the kernel. Every launch takes the AOT ABI over the runtime problem."""
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_grouped import DirectConvSpec as _Spec
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    tp = dgrad_4c_transposed_problem(p)
    run_spec: _Spec = replace(spec, problem=tp)
    ok, why = is_valid_spec(run_spec, arch=GPU_ARCH)
    if not ok:
        return False, f"invalid spec: {why}"
    if art is None:
        art = _compile(build_direct_conv(spec, arch=GPU_ARCH))
    torch.manual_seed(3)
    td = torch.bfloat16 if p.dtype == "bf16" else torch.float16
    dy = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=td).uniform_(-1.0, 1.0)
    w = torch.empty(p.total_k, p.KH, p.KW, p.cpg, dtype=td).uniform_(-1.0, 1.0)
    dx = torch.empty(p.N, p.H, p.W, p.total_c, dtype=td)
    ref = torch.nn.grad.conv2d_input(
        (p.N, p.total_c, p.H, p.W),
        w.float().permute(0, 3, 1, 2),
        dy.float().permute(0, 3, 1, 2),
        stride=1,
        padding=p.PAD,
        groups=p.groups,
    ).permute(0, 2, 3, 1)

    rt = Runtime()
    d_dy, d_w, d_dx = (rt.alloc(t.nbytes) for t in (dy, w, dx))
    rt.memcpy_h2d(d_dy, _u8(dy), dy.nbytes)
    rt.memcpy_h2d(d_w, _u8(w), w.nbytes)
    rt.memset(d_dx, 0xFF, dx.nbytes)  # NaN-fill: unwritten outputs fail
    extra = []
    if spec.dgrad_fused_weights:
        b_buf, b_bytes = d_w, w.nbytes
    else:
        ws = direct_dgrad_workspace_bytes(p)
        d_ws = rt.alloc(ws)
        extra.append(d_ws)
        tspec = DirectTransposeWeightsDgradSpec(problem=p)
        kt = _compile(build_direct_transpose_weights_dgrad(tspec, arch=GPU_ARCH))
        elem = "bf16" if p.dtype == "bf16" else "f16"
        wsig = [
            {"name": "A", "type": f"ptr<{elem}, global>", "size_bytes": 8},
            {"name": "D", "type": f"ptr<{elem}, global>", "size_bytes": 8},
            {"name": "A_bytes", "type": "i32", "size_bytes": 4},
            {"name": "D_bytes", "type": "i32", "size_bytes": 4},
        ]
        t_grid, t_block = direct_launch_geometry(tspec)
        KernelLauncher(hsaco=kt.hsaco, kernel_name=kt.kernel_name, signature=wsig)(
            {"A": d_w, "D": d_ws, "A_bytes": w.nbytes, "D_bytes": ws},
            config=LaunchConfig(grid=t_grid, block=t_block, fence=True),
        )
        b_buf, b_bytes = d_ws, ws
    grid, block = direct_launch_geometry(run_spec)
    values = ConvArgs.from_problem(tp, direction="fwd").to_launch_values(
        int(d_dy), int(b_buf), int(d_dx), dy.nbytes, b_bytes, dx.nbytes
    )
    KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_direct_args_signature(p.dtype, direction="fwd"),
    )(values, config=LaunchConfig(grid=grid, block=block, fence=True))
    rt.memcpy_d2h(_u8(dx), d_dx, dx.nbytes)
    for b in (d_dy, d_w, d_dx, *extra):
        rt.free(b)
    diff = (dx.float() - ref).abs()
    n_bad = int((~(diff <= _TOL + _TOL * ref.abs())).sum())
    return n_bad == 0, f"bad={n_bad} max_abs={float(diff.nan_to_num(1e9).max()):.3e}"


#: (problem, extra make_dgrad_fprop_spec knobs) -- adversarial coverage.
_CASES = [
    (_problem(N=1, H=13, W=15, groups=16, cpg=16, kpg=16, dtype="fp16"), {}),
    (_problem(N=2, H=7, W=17, groups=16, cpg=32, kpg=32), {}),
    (_problem(N=2, H=7, W=17, groups=16, cpg=32, kpg=32), {"fold_k32": True}),
    (_problem(N=4, H=9, W=11, groups=8, cpg=12, kpg=12, dtype="fp16"), {}),
    (_problem(N=2, H=15, W=13, groups=8, cpg=16, kpg=8), {}),
    (_problem(N=2, H=13, W=7, groups=4, cpg=16, kpg=48), {}),
    (_problem(N=2, H=11, W=9, groups=32, cpg=8, kpg=8, dtype="fp16"), {}),
    (_problem(N=1, H=17, W=19, groups=16, cpg=16, kpg=16), {"block_h": 8}),
    (
        _problem(N=2, H=9, W=37, groups=16, cpg=16, kpg=16),
        {"block_q": 32, "block_groups": 4, "block_h": 4},
    ),
    (_problem(N=1, H=5, W=7, groups=8, cpg=16, kpg=16, K=1), {"block_h": 8}),
    (_problem(N=3, H=11, W=10, groups=6, cpg=20, kpg=8, K=5), {"block_h": 4}),
    (_problem(N=2, H=6, W=9, groups=4, cpg=8, kpg=12, K=7), {"block_h": 8}),
    (_problem(N=1, H=3, W=5, groups=8, cpg=12, kpg=28), {"block_h": 32}),
]

_MODES = {
    "preload": {"preload_weights": True},
    "gather": {"dgrad_fused_weights": True},
    "gather_we4_rg": {
        "dgrad_fused_weights": True,
        "waves_per_eu": 4,
        "row_guard": True,
    },
}
if GPU_ARCH == "gfx950":
    _MODES["lds"] = {"dgrad_fused_weights": True, "dgrad_weights_lds": True}
    _MODES["lds_we4_rg"] = {
        "dgrad_fused_weights": True,
        "dgrad_weights_lds": True,
        "waves_per_eu": 4,
        "row_guard": True,
    }


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu")
class TestFusedWeightNumerics(unittest.TestCase):
    def test_modes(self):
        for p, knobs in _CASES:
            kw = {"block_groups": 2 if p.groups % 2 == 0 else 1, **knobs}
            for mode, mk in _MODES.items():
                spec = make_dgrad_fprop_spec(p, **kw, **mk)
                if not is_valid_spec(spec, arch=GPU_ARCH)[0]:
                    continue  # e.g. a 7x7 preload past the register budget
                with self.subTest(p=p.short(), K=p.KH, knobs=knobs, mode=mode):
                    ok, msg = _run(p, spec)
                    self.assertTrue(ok, msg)

    def test_kernel_level_default(self):
        for p, _ in _CASES:
            spec = direct_dgrad_spec_for_problem(p, arch=GPU_ARCH)
            with self.subTest(p=p.short(), K=p.KH):
                self.assertIsNotNone(spec)
                if isinstance(spec, DirectConv4cSpec):
                    continue
                ok, msg = _run(p, spec)
                self.assertTrue(ok, msg)

    def test_one_binary_serves_every_shape(self):
        # One build per form on a probe problem, launched on unrelated
        # extents (N, H, W and groups are kernargs): images shorter than the
        # tile, taller than it, and not a multiple of it.
        probe = _problem(N=1, H=64, W=64, groups=64, cpg=16, kpg=8)
        shapes = (
            _problem(N=2, H=5, W=11, groups=16, cpg=16, kpg=8),
            _problem(N=1, H=17, W=3, groups=6, cpg=16, kpg=8),
            _problem(N=3, H=8, W=24, groups=2, cpg=16, kpg=8),
            _problem(N=1, H=1, W=6, groups=4, cpg=16, kpg=8),
        )
        for mode, mk in _MODES.items():
            if mode == "preload":
                continue
            spec = make_dgrad_fprop_spec(
                probe, block_groups=2, block_h=8, **{**mk, "row_guard": True}
            )
            art = _compile(build_direct_conv(spec, arch=GPU_ARCH))
            for p in shapes:
                with self.subTest(mode=mode, p=p.short()):
                    ok, msg = _run(p, spec, art=art)
                    self.assertTrue(ok, msg)


#: Row-stream knobs, alone and stacked (the policy's stack with and without
#: the early exit, and the split-M stack on outputs with two 16-wide tiles).
_STREAM_KNOBS = [
    {"prefetch_rows": 2},
    {"prefetch_rows": 3, "lds_only_sync": True},
    {"lds_pad": 8},
    {"stage_out": True},
    {"xcd_tiles": True},
    {"row_exit": True},
    {"prefetch_rows": 2, "lds_only_sync": True, "lds_pad": 8, "stage_out": True, "xcd_tiles": True},
    {"prefetch_rows": 2, "lds_only_sync": True, "lds_pad": 8, "stage_out": True, "xcd_tiles": True, "row_exit": True},
    {"waves_m": 2, "prefetch_rows": 2, "lds_only_sync": True, "stage_out": True, "xcd_tiles": True},
]  # fmt: skip

#: (problem, base knobs): odd H / W, partial q strips and H tiles, a tile
#: taller than the image, partial K atoms, 5x5, LDS-staged and gathered
#: weights, the pre-pass weights, fp16 / bf16 -- each with at least eight
#: image row bands so the XCD order is valid.
_STREAM_CASES = [
    (_problem(N=8, H=9, W=13, groups=8, cpg=16, kpg=16), {"block_groups": 2, "block_h": 16, "dgrad_fused_weights": True, "dgrad_weights_lds": True, "row_guard": True}),
    (_problem(N=8, H=11, W=37, groups=8, cpg=16, kpg=16, dtype="fp16"), {"block_q": 32, "block_groups": 2, "block_h": 4, "dgrad_fused_weights": True, "dgrad_weights_lds": True, "row_guard": True}),
    (_problem(N=9, H=7, W=9, groups=8, cpg=32, kpg=32), {"block_groups": 1, "block_h": 8, "fold_k32": True, "dgrad_fused_weights": True, "dgrad_weights_lds": True, "row_guard": True}),
    (_problem(N=8, H=9, W=11, groups=8, cpg=32, kpg=8, K=5, dtype="fp16"), {"block_groups": 1, "block_h": 4, "dgrad_fused_weights": True, "row_guard": True}),
    (_problem(N=8, H=10, W=13, groups=4, cpg=12, kpg=12), {"block_groups": 1, "block_h": 4, "dgrad_fused_weights": True, "waves_per_eu": 4}),
    (_problem(N=8, H=10, W=13, groups=4, cpg=16, kpg=16), {"block_groups": 1, "block_h": 4}),
]  # fmt: skip


@unittest.skipIf(bool(_GPU_SKIP), _GPU_SKIP or "gpu")
class TestRowStreamKnobNumerics(unittest.TestCase):
    def test_knob_grid(self):
        for p, base in _STREAM_CASES:
            for kw in _STREAM_KNOBS:
                spec = make_dgrad_fprop_spec(p, **base, **kw)
                try:
                    spec.validate()
                except ValueError:
                    continue  # e.g. waves_m on a single 16-wide output tile
                if not is_valid_spec(spec, arch=GPU_ARCH)[0]:
                    continue  # e.g. the XCD order or LDS weights on gfx942
                with self.subTest(p=p.short(), K=p.KH, spec=spec.kernel_name()):
                    ok, msg = _run(p, spec)
                    self.assertTrue(ok, msg)

    def test_one_stream_binary_serves_every_shape(self):
        # The policy's stack on a whole-image tile with the early exit, built
        # once on a probe and launched on unrelated extents that keep at
        # least eight image row bands (images shorter than the tile, and one
        # whose second tile is mostly past the image).
        probe = _problem(N=8, H=64, W=64, groups=64, cpg=16, kpg=16)
        knobs = {"dgrad_fused_weights": True, "row_guard": True, "row_exit": True}
        if GPU_ARCH == "gfx950":
            knobs.update(
                dgrad_weights_lds=True,
                prefetch_rows=2,
                lds_only_sync=True,
                xcd_tiles=True,
                lds_pad=8,
                stage_out=True,
            )
        else:
            knobs.update(prefetch_rows=2, lds_only_sync=True, stage_out=True)
        spec = make_dgrad_fprop_spec(probe, block_groups=2, block_h=16, **knobs)
        art = _compile(build_direct_conv(spec, arch=GPU_ARCH))
        for p in (
            _problem(N=8, H=5, W=11, groups=16, cpg=16, kpg=16),
            _problem(N=4, H=17, W=3, groups=6, cpg=16, kpg=16),
            _problem(N=9, H=8, W=24, groups=2, cpg=16, kpg=16),
        ):
            with self.subTest(p=p.short()):
                ok, msg = _run(p, spec, art=art)
                self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
