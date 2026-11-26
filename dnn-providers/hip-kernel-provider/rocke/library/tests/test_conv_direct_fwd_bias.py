# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Conv + per-output-channel bias on the direct forward kernels.

A direct forward spec built with ``bias=True`` takes the ``Bias`` /
``Bias_bytes`` kernarg tail after the direct-conv block and computes
``round(acc_fp32 + bias[g*kpg + k])``: the bias is added to the fp32
accumulators before the one rounding to the output dtype. A split-C conv pass
declares the tail but leaves its partial sums raw; its fold
(``DirectSplitReduceSpec(bias=True)``) adds the bias once.

CPU: the kernarg tail and launch values, the name tags, the identity round
trip through the direct AOT grid, the refusals (the variants with no bias form
and the specs whose validators refuse it). GPU (gfx942 / gfx950): every
family that runs on the device, fp16 / bf16 (and fp32 storage where the family
has it), against the fp32 torch convolution plus the bias rounded once to the
output dtype; each case also checks that the bias moved the output.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import unittest
from dataclasses import replace

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _p(N=2, H=12, W=20, C=64, K=64, *, groups=1, KH=3, PAD=1, stride=1, **kw):
    from kernels.common.conv_direct_grouped import DirectConvProblem

    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=kw.pop("KW", KH),
        PAD=PAD,
        stride=stride,
        **kw,
    )


def _ng(problem, **kw):
    from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

    base = dict(tile_h=8, tile_w=32, tile_k=64, ck=32, waves_m=2, waves_n=2)
    base.update(kw)
    return DirectNongroupedConvSpec(problem=problem, bias=True, **base)


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


class TestDirectBiasAbi(unittest.TestCase):
    def test_tail_follows_the_direct_block(self):
        from kernels.common.conv_abi import (
            conv_bias_arg_names,
            conv_direct_arg_names,
        )

        plain = conv_direct_arg_names(direction="fwd")
        self.assertEqual(
            conv_direct_arg_names(direction="fwd", bias=True),
            plain + conv_bias_arg_names(),
        )
        for d in ("dgrad", "wgrad"):
            with self.assertRaisesRegex(ValueError, "fwd ABI only"):
                conv_direct_arg_names(direction=d, bias=True)

    def test_signature(self):
        from kernels.common.conv_abi import conv_direct_args_signature

        sig = conv_direct_args_signature("bf16", bias=True)
        self.assertEqual(sig[-2]["name"], "Bias")
        self.assertEqual(sig[-2]["type"], sig[0]["type"].replace("A", "A"))
        self.assertEqual(sig[-1]["name"], "Bias_bytes")
        self.assertEqual(
            conv_direct_args_signature("bf16"), sig[:-2], "no-bias ABI unchanged"
        )

    def test_launch_values(self):
        from kernels.common.conv_args import ConvArgs

        args = ConvArgs.from_problem(_p())
        plain = args.to_launch_values(1, 2, 3, 10, 20, 30)
        vals = args.to_launch_values(
            1, 2, 3, 10, 20, 30, bias_ptr=0xB00, bias_bytes=128
        )
        self.assertEqual(list(vals)[:-2], list(plain))
        self.assertEqual(
            list(vals.items())[-2:], [("Bias", 0xB00), ("Bias_bytes", 128)]
        )
        # Bias_bytes bounds the kernel's bias reads: K values, no fewer.
        for short in (None, 127):
            with self.subTest(bias_bytes=short):
                with self.assertRaisesRegex(ValueError, "K=64 channels needs"):
                    args.to_launch_values(
                        1, 2, 3, 10, 20, 30, bias_ptr=0xB00, bias_bytes=short
                    )

    def test_kernel_params_match_the_abi(self):
        from kernels.common.conv_abi import conv_direct_arg_names
        from kernels.common.conv_direct_nongrouped import (
            DirectSplitReduceSpec,
            build_direct_conv_nongrouped,
            build_direct_conv_split_reduce,
        )

        want = [n for n, _ in conv_direct_arg_names(direction="fwd", bias=True)]
        for k in (
            build_direct_conv_nongrouped(_ng(_p(dtype="bf16"))),
            # The split-C conv pass declares the tail too: one shared block.
            build_direct_conv_nongrouped(_ng(_p(C=128, dtype="bf16"), split_c=2)),
            build_direct_conv_split_reduce(DirectSplitReduceSpec(bias=True)),
        ):
            self.assertEqual([p.name for p in k.params], want, k.name)


class TestDirectBiasSpecs(unittest.TestCase):
    def test_name_tags(self):
        from kernels.common.conv_direct_nongrouped import DirectSplitReduceSpec

        spec = _ng(_p(dtype="bf16"))
        self.assertTrue(spec.kernel_name().endswith("_cbias_bf16"))
        self.assertNotIn("cbias", replace(spec, bias=False).kernel_name())
        self.assertEqual(
            DirectSplitReduceSpec(dtype="bf16", bias=True).kernel_name(),
            "direct_conv_split_reduce_sc2_b256_cbias_bf16",
        )

    def test_make_direct_spec(self):
        from kernels.conv.direct_variants import (
            DIRECT_FWD_BIAS_VARIANTS,
            make_direct_spec,
        )

        spec = make_direct_spec(
            "direct_nongrouped", _p(), dict(tile_h=8, ck=32), bias=True
        )
        self.assertTrue(spec.bias)
        self.assertIn("direct_split_reduce", DIRECT_FWD_BIAS_VARIANTS)
        for variant in ("direct_grouped_dgrad_fused", "direct_wgrad"):
            with self.assertRaisesRegex(ValueError, "has no bias form"):
                make_direct_spec(variant, _p(), {}, bias=True)

    def test_identity_round_trip(self):
        from kernels.conv.direct_aot_jobs import (
            DirectCaps,
            _identity,
            _split_reduce_identity,
            direct_job_for_identity,
        )

        caps = DirectCaps(KH=3, PAD=1, stride=1, cpg=0, kpg=0)
        knobs = dict(tile_h=8, tile_w=32, tile_k=64, ck=32, waves_m=2, waves_n=2)
        plain = _identity("gfx950", 64, "direct_nongrouped", "direct_fwd", caps, knobs)
        ident = replace(plain, bias=True)
        self.assertNotEqual(ident.stable_hash(), plain.stable_hash())
        self.assertIn("cbias", ident.label())
        self.assertEqual(json.loads(ident.knobs), knobs, "bias is not a knob")
        job = direct_job_for_identity(ident)
        self.assertTrue(job.identity.bias)
        red = _split_reduce_identity("gfx950", 64, 4, "bf16", bias=True)
        self.assertTrue(direct_job_for_identity(red).identity.bias)
        # A dgrad identity has no bias form.
        dgrad = _identity(
            "gfx950", 64, "direct_grouped_dgrad_fused", "direct_dgrad", caps, {}
        )
        with self.assertRaises(LookupError):
            direct_job_for_identity(replace(dgrad, bias=True))
        # Nor does a forward variant a dgrad identity runs over the
        # transposed problem: its launch has no bias tail.
        with self.assertRaisesRegex(LookupError, "only a forward kernel"):
            direct_job_for_identity(replace(ident, direction="direct_dgrad"))

    def test_bias_dtypes(self):
        """The bias carries a stored output dtype: f16 / bf16 (2 bytes) or
        f32 (4); any other is refused rather than read at a wrong stride."""
        from rocke.core.ir import BF16, F16, F32, I8, I32

        from kernels.conv.conv_bias import _bias_elem_bytes

        self.assertEqual([_bias_elem_bytes(t) for t in (F16, BF16, F32)], [2, 2, 4])
        for t in (I8, I32):
            with self.assertRaisesRegex(ValueError, "f16, bf16 or f32 only"):
                _bias_elem_bytes(t)

    def test_engine_bridge_forwards_bias(self):
        from kernels.conv.engine_bridge import conv_direct_nongrouped_spec_to_dict

        self.assertIs(conv_direct_nongrouped_spec_to_dict(_ng(_p()))["bias"], True)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _gpu_arch() -> str:
    if not _HAS_TORCH:
        return ""
    try:
        from rocke.runtime.hip_module import get_device_arch

        return (get_device_arch() or "").split(":")[0]
    except Exception:  # pragma: no cover - no device
        return ""


_ARCH = _gpu_arch()
_GPU_SKIP = (
    "" if _ARCH in ("gfx942", "gfx950") else f"needs gfx942 / gfx950 (got {_ARCH!r})"
)
# Peak-relative error bound of the forward correctness tests.
_TOL = {"fp16": 2e-2, "bf16": 5e-2, "fp32": 1e-4, "tf32": 1e-2}


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _geometry(spec):
    from kernels.common.conv_direct_grouped import direct_launch_geometry
    from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

    from kernels.common.conv_direct_grouped import (
        DirectConv16cSpec,
        DirectConv32cSpec,
        DirectConv8cSpec,
    )

    if isinstance(spec, DirectNongroupedConvSpec):
        return spec.grid(), (spec.threads_per_block, 1, 1)
    if isinstance(spec, (DirectConv16cSpec, DirectConv8cSpec, DirectConv32cSpec)):
        # bx = output-column tile, by = group tile, bz = image.
        p = spec.problem
        grid = (-(-p.Wo // spec.block_q), p.groups // spec.block_groups, p.N)
        return grid, (spec.threads_per_block, 1, 1)
    if hasattr(spec, "launch_geometry"):
        return spec.launch_geometry()
    return direct_launch_geometry(spec)


def _valid_and_builder(variant, spec, arch):
    """``(ok, why)`` and the kernel builder of ``variant``: a direct AOT
    variant, or one of the specialised grouped kernels outside the AOT grid
    ("16c", "8c", "32c", "4c")."""
    from kernels.common import conv_direct_grouped as dc
    from kernels.conv.direct_variants import (
        select_direct_kernel_builder,
        validate_direct_spec,
    )

    local = {
        "16c": (dc.is_valid_spec_16c, dc.build_direct_conv_16c),
        "8c": (dc.is_valid_spec_8c, dc.build_direct_conv_8c),
        "32c": (dc.is_valid_spec_32c, dc.build_direct_conv_32c),
        "4c": (dc.is_valid_spec_4c, dc.build_direct_conv_4c),
    }
    if variant in local:
        valid, build = local[variant]
        return valid(spec, arch=arch), lambda: build(spec, arch=arch)
    return validate_direct_spec(variant, spec, arch), lambda: (
        select_direct_kernel_builder(variant, spec, arch)
    )


def _run(variant, spec, problem=None):
    """Build, launch and verify ``spec`` (of ``variant``) with a bias; returns
    the peak-relative error against ``round(conv_fp32 + bias)`` and the same
    error against the convolution alone. ``problem`` is the convolution the
    kernel computes when the spec's own problem is a transformed one."""
    import torch
    import torch.nn.functional as F
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_nongrouped import DirectSplitReduceSpec
    from kernels.conv.direct_variants import (
        select_direct_kernel_builder,
        validate_direct_spec,
    )
    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    arch = _ARCH
    p = problem or spec.problem
    (ok, why), build = _valid_and_builder(variant, spec, arch)
    assert ok, why
    dt = p.dtype
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dt, torch.float32)
    sig = conv_direct_args_signature(
        "fp32" if dt == "tf32" else dt, direction="fwd", bias=True
    )
    art = compile_kernel(build(), arch=arch)
    torch.manual_seed(0)
    A = torch.empty(p.N, p.H, p.W, p.total_c).uniform_(-1, 1).to(tdt)
    B = torch.empty(p.total_k, p.KH, p.KW, p.cpg).uniform_(-1, 1).to(tdt)
    bias = torch.empty(p.total_k).uniform_(-8, 8).to(tdt)
    D = torch.empty(p.N, p.Ho, p.Wo, p.total_k, dtype=tdt)
    conv = (
        F.conv2d(
            A.permute(0, 3, 1, 2).float(),
            B.permute(0, 3, 1, 2).float(),
            padding=(p.PAD, getattr(p, "PAD_W", p.PAD)),
            stride=p.stride,
            dilation=(p.dil_h, p.dil_w),
            groups=p.groups,
        )
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    ref = (conv + bias.float()).to(tdt).float()
    split = getattr(spec, "split_c", 1) > 1
    ws_bytes = spec.workspace_bytes if split else 0
    rt = Runtime()
    ptrs = [rt.alloc(t.nbytes) for t in (A, B, D, bias)]
    ws = rt.alloc(ws_bytes) if split else None
    try:
        rt.memcpy_h2d(ptrs[0], _u8(A), A.nbytes)
        rt.memcpy_h2d(ptrs[1], _u8(B), B.nbytes)
        rt.memcpy_h2d(ptrs[3], _u8(bias), bias.nbytes)
        # NaN in every float dtype: an output never written fails.
        rt.memset(ptrs[2], 0xFF, D.nbytes)
        if split:
            rt.memset(ws, 0xFF, ws_bytes)
        args = ConvArgs.from_problem(p)
        out_ptr, out_bytes = (ws, ws_bytes) if split else (ptrs[2], D.nbytes)
        grid, block = _geometry(spec)
        KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)(
            args.to_launch_values(
                ptrs[0],
                ptrs[1],
                out_ptr,
                A.nbytes,
                B.nbytes,
                out_bytes,
                bias_ptr=ptrs[3],
                bias_bytes=bias.nbytes,
            ),
            config=LaunchConfig(grid=grid, block=block, fence=True),
        )
        if split:
            red = DirectSplitReduceSpec(dtype=dt, split_c=spec.split_c, bias=True)
            rart = compile_kernel(
                select_direct_kernel_builder("direct_split_reduce", red, arch),
                arch=arch,
            )
            KernelLauncher(
                hsaco=rart.hsaco, kernel_name=rart.kernel_name, signature=sig
            )(
                args.to_launch_values(
                    ws,
                    ptrs[1],
                    ptrs[2],
                    ws_bytes,
                    B.nbytes,
                    D.nbytes,
                    bias_ptr=ptrs[3],
                    bias_bytes=bias.nbytes,
                ),
                config=LaunchConfig(
                    grid=red.grid(p), block=(red.threads, 1, 1), fence=True
                ),
            )
        out = torch.empty_like(D)
        rt.memcpy_d2h(_u8(out), ptrs[2], D.nbytes)
    finally:
        for q in ptrs + ([ws] if split else []):
            rt.free(q)
    out = out.float()
    scale = ref.abs().max().clamp(min=1.0)
    err = float((out - ref).abs().max() / scale)
    no_bias = float((out - conv.to(tdt).float()).abs().max() / scale)
    return err, no_bias


class _GpuCase(unittest.TestCase):
    def _check(self, variant, spec, problem=None, *, arches=None):
        if arches is not None and _ARCH not in arches:
            self.skipTest(f"{spec.kernel_name()}: {'/'.join(arches)} only")
        err, no_bias = _run(variant, spec, problem)
        tol = _TOL[(problem or spec.problem).dtype]
        name = spec.kernel_name()
        self.assertLess(err, tol, f"{name}: rel err {err:.3e}")
        print(f"  {name}: rel err {err:.2e} (without bias {no_bias:.2e})")
        self.assertGreater(no_bias, 2 * tol, f"{name}: the bias did not move it")


@pytest.mark.gpu
@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestDirectNongroupedBiasGpu(_GpuCase):
    def _atom(self, wide: str) -> str:
        # gfx942 has the 32x32x8 / 16x16x16 atoms only.
        narrow = {"32x32x16": "32x32x8", "16x16x32": "16x16x16"}
        return narrow.get(wide, wide) if _ARCH == "gfx942" else wide

    def test_dense(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                spec = _ng(_p(dtype=dtype), atom=self._atom("32x32x16"), ck=16)
                self._check("direct_nongrouped", spec)
        # 16-wide atom on a partial width, K past one tile (96 of 128).
        spec = _ng(
            _p(H=9, W=20, K=96, dtype="bf16"),
            tile_w=16,
            tile_k=32,
            waves_m=1,
            atom=self._atom("16x16x32"),
            ck=16,
        )
        self._check("direct_nongrouped", spec)

    def test_grouped_merge(self):
        spec = _ng(
            _p(C=4, K=4, groups=16, dtype="bf16"),
            tile_k=32,
            atom=self._atom("16x16x32"),
            ck=16 if _ARCH == "gfx942" else 32,
            grouped=True,
            group_merge=8,
        )
        self._check("direct_grouped_halo", spec)

    def test_split_c_fold_adds_the_bias_once(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                spec = _ng(
                    _p(H=7, W=7, C=256, K=64, dtype=dtype),
                    atom=self._atom("32x32x16"),
                    ck=16,
                    split_c=4,
                )
                self._check("direct_nongrouped", spec)

    def test_fp32_storage(self):
        spec = _ng(
            _p(H=6, W=32, C=64, K=64, dtype="fp32"),
            tile_h=4,
            tile_k=32,
            ck=16,
            waves_m=1,
            lds_pad=4,
            atom="32x32x8" if _ARCH == "gfx942" else "32x32x16",
        )
        self._check("direct_nongrouped", spec)


def _gen(problem, **kw):
    from kernels.common.conv_direct_grouped import DirectConvSpec

    kw.setdefault("block_groups", 1)
    kw.setdefault("block_h", 4)
    return DirectConvSpec(problem=problem, bias=True, **kw)


@pytest.mark.gpu
@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestDirectGenericBiasGpu(_GpuCase):
    """DirectConvSpec (build_direct_conv): the per-row flush, the waves_k
    LDS reduction (bias added once, on the reduced sum), the staged-output
    tile, the persistent grid and runtime channels."""

    def test_flush(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                # kpg 20: a partial second M-tile.
                p = _p(N=2, H=9, W=20, C=16, K=20, groups=4, dtype=dtype)
                self._check("direct_grouped", _gen(p, block_groups=2))

    def test_knobs(self):
        # (waves_k > 1 and the runtime K loop read the reorganized weights of
        # the MFMA dgrad pipeline; the parity configs pin their bias add.)
        p = _p(N=1, H=9, W=20, C=64, K=32, groups=2, dtype="bf16")
        for kw in (
            dict(stage_out=True, lds_pad=8),
            dict(preload_weights=True, waves_m=2),
            dict(persistent_grid=True, block_h=8),
            dict(runtime_channels=True, k_atoms=4, m_tiles=2),
        ):
            with self.subTest(**kw):
                self._check("direct_grouped", _gen(p, **kw))

    def test_fp32(self):
        p = _p(N=1, H=9, W=20, C=16, K=16, groups=4, dtype="fp32")
        self._check("direct_grouped", _gen(p))


# ---------------------------------------------------------------------------
# gfx950 MFMA families in their own modules: flex_mfma, pp_mfma (forward
# only; the dgrad spec of the shared pipeline has no bias) and the small-C /
# small-K MFMA stems. Their kernels need gfx950 (16-byte LDS DMA / the
# 16x16x32 atoms): the GPU halves run there only.
# ---------------------------------------------------------------------------

_GFX950_SKIP = "" if _ARCH == "gfx950" else f"needs gfx950 (got {_ARCH!r})"


def _flex(problem, **kw):
    from kernels.common.conv_direct_fwd_flex_mfma import FwdFlexMfmaSpec

    return FwdFlexMfmaSpec(problem=problem, bias=True, **kw)


def _pp(problem, **kw):
    from kernels.common.conv_direct_fwd_pp_mfma import FwdPpMfmaSpec

    return FwdPpMfmaSpec(problem=problem, bias=True, **kw)


def _smallc(problem, **kw):
    from kernels.common.conv_fwd_small_mfma import FwdSmallcMfmaSpec

    return FwdSmallcMfmaSpec(problem=problem, bias=True, **kw)


def _smallk(problem, **kw):
    from kernels.common.conv_fwd_small_mfma import FwdSmallkMfmaSpec

    return FwdSmallkMfmaSpec(problem=problem, bias=True, **kw)


class _MfmaBiasCpu(unittest.TestCase):
    """Name tag, the bias tail at the end of the kernel's params, and the
    gfx950 validator accepting the bias form."""

    variant = ""

    def _cpu(self, spec, dtype_tag: str):
        from kernels.common.conv_abi import conv_direct_arg_names
        from kernels.conv.direct_variants import (
            select_direct_kernel_builder,
            validate_direct_spec,
        )

        name = spec.kernel_name()
        self.assertTrue(name.endswith("_cbias" + dtype_tag), name)
        self.assertNotIn("cbias", replace(spec, bias=False).kernel_name())
        ok, why = validate_direct_spec(self.variant, spec, "gfx950")
        self.assertTrue(ok, why)
        k = select_direct_kernel_builder(self.variant, spec, "gfx950")
        want = [n for n, _ in conv_direct_arg_names(direction="fwd", bias=True)]
        self.assertEqual([p.name for p in k.params], want)


class TestDirectFlexMfmaBias(_MfmaBiasCpu):
    variant = "direct_fwd_flex_mfma"

    def test_cpu(self):
        self._cpu(_flex(_p(C=128, K=96, dtype="bf16")), "_bf16")
        # split_c > 1: the conv pass declares the tail, the fold adds the bias.
        self._cpu(_flex(_p(C=128, K=96), split_c=2, bk=128), "")


class TestDirectPpMfmaBias(_MfmaBiasCpu):
    variant = "direct_fwd_pp_mfma"

    def test_cpu(self):
        self._cpu(_pp(_p(C=72, K=96, dtype="bf16"), block_k=64), "_bf16")

    def test_dgrad_spec_has_no_bias(self):
        from dataclasses import fields

        from kernels.common.conv_direct_dgrad_pp_mfma import DgradPpMfmaSpec

        self.assertNotIn("bias", {f.name for f in fields(DgradPpMfmaSpec)})


class TestDirectSmallMfmaBias(_MfmaBiasCpu):
    def test_cpu(self):
        self.variant = "direct_fwd_smallc_mfma"
        self._cpu(_smallc(_p(H=32, W=40, C=3, K=64, dtype="bf16")), "_bf16")
        self.variant = "direct_fwd_smallk_mfma"
        self._cpu(_smallk(_p(H=32, W=40, C=64, K=3), out_h=8, out_w=8), "")


@pytest.mark.gpu
@unittest.skipIf(_GFX950_SKIP, _GFX950_SKIP)
class TestDirectMfmaBiasGpu(_GpuCase):
    def test_flex(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = _p(N=2, H=14, W=18, C=128, K=96, dtype=dtype)
                self._check("direct_fwd_flex_mfma", _flex(p, bk=128))
        # split_c 2: raw slabs, then the fold adds the bias once.
        p = _p(N=1, H=7, W=7, C=256, K=64, dtype="bf16")
        self._check("direct_fwd_flex_mfma", _flex(p, split_c=2))

    def test_pp(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = _p(N=1, H=20, W=24, C=72, K=96, dtype=dtype)
                self._check("direct_fwd_pp_mfma", _pp(p, block_k=64))

    def test_small(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = _p(N=2, H=32, W=40, C=3, K=36, dtype=dtype)
                self._check("direct_fwd_smallc_mfma", _smallc(p))
                for k in (3, 4):
                    p = _p(N=2, H=32, W=40, C=64, K=k, dtype=dtype)
                    spec = _smallk(p, out_h=8, out_w=8)
                    self._check("direct_fwd_smallk_mfma", spec)


# ---------------------------------------------------------------------------
# gfx1250 wave32 WMMA families: fwd_wmma (block and per-wave epilogues),
# tapouter, gstream (padded M tiles), small-C (register and TDM stores) and
# small-K (the overlap-add). The kernels need gfx1250 (TDM, wave32 WMMA):
# the GPU halves run there only.
# ---------------------------------------------------------------------------

_GFX1250_SKIP = "" if _ARCH == "gfx1250" else f"needs gfx1250 (got {_ARCH!r})"


def _wmma_specs():
    """``(variant, spec)`` of every WMMA family, built with bias."""
    from kernels.common.conv_direct_dw_fwd_wmma import DwFwdWmmaSpec
    from kernels.common.conv_direct_fwd_gstream_wmma import FwdGstreamWmmaSpec
    from kernels.common.conv_direct_fwd_tapouter_wmma import DirectFwdTapOuterSpec
    from kernels.common.conv_direct_fwd_wmma import DirectFwdWmmaSpec
    from kernels.common.conv_fwd_small_wmma import (
        FwdSmallcWmmaSpec,
        FwdSmallkWmmaSpec,
    )

    dense = _p(N=2, H=20, W=24, C=64, K=96, dtype="bf16")
    return (
        ("direct_fwd_wmma", DirectFwdWmmaSpec(problem=dense, bias=True)),
        (
            "direct_fwd_wmma",
            DirectFwdWmmaSpec(
                problem=replace(dense, dtype="fp16"),
                waves_k=1,
                waves_p=4,
                tiles_k=4,
                tiles_p=6,
                tile_n=2,
                split_barrier=True,
                wave_epilogue=True,
                bias=True,
            ),
        ),
        (
            "direct_fwd_tapouter_wmma",
            DirectFwdTapOuterSpec(
                problem=_p(H=33, W=33, C=64, K=96, PAD=2, dil_h=2, dil_w=2),
                waves_k=1,
                waves_p=4,
                tiles_k=8,
                tiles_p=2,
                ck=64,
                stages=2,
                bias=True,
            ),
        ),
        (
            "direct_fwd_gstream_wmma",
            FwdGstreamWmmaSpec(
                problem=_p(N=2, H=28, W=28, C=56, K=56, groups=4, dtype="bf16"),
                waves_k=8,
                waves_p=1,
                tiles_k=2,
                tiles_p=4,
                tile_w=32,
                walk=4,
                bias=True,
            ),
        ),
        (
            "direct_fwd_smallc_wmma",
            FwdSmallcWmmaSpec(problem=_p(H=32, W=40, C=3, K=64), bias=True),
        ),
        (
            "direct_fwd_smallc_wmma",
            FwdSmallcWmmaSpec(
                problem=_p(H=32, W=40, C=4, K=64, dtype="fp16"),
                waves_k=2,
                waves_p=2,
                tiles_k=2,
                tiles_p=4,
                out_h=8,
                out_w=16,
                tdm_store=True,
                bias=True,
            ),
        ),
        (
            "direct_fwd_smallk_wmma",
            FwdSmallkWmmaSpec(
                problem=_p(H=32, W=40, C=128, K=4),
                waves_c=1,
                waves_p=4,
                tiles_c=3,
                tiles_p=3,
                out_h=6,
                out_w=16,
                ck=64,
                prefetch=True,
                bias=True,
            ),
        ),
        (
            "direct_fwd_smallk_wmma",
            FwdSmallkWmmaSpec(
                problem=_p(H=32, W=40, C=32, K=3),
                waves_c=1,
                waves_p=3,
                tiles_c=2,
                tiles_p=3,
                out_h=6,
                out_w=16,
                ck=32,
                prefetch=True,
                bias=True,
            ),
        ),
        # The Toeplitz depthwise forward (the windowed dgrad builder's wave32
        # fwd form): 8 channels per wave, a partial channel block.
        (
            "direct_dw_fwd_wmma",
            DwFwdWmmaSpec(
                problem=_p(H=13, W=19, C=1, K=1, groups=24, dtype="fp16"),
                block_waves=8,
                block_h=2,
                w_fold=4,
                prefetch_rows=3,
                bias=True,
            ),
        ),
    )


class TestDirectWmmaBias(unittest.TestCase):
    """Name tag, the bias tail at the end of the kernel's params, and the
    gfx1250 validator accepting the bias form, per WMMA family."""

    def test_cpu(self):
        from kernels.common.conv_abi import conv_direct_arg_names
        from kernels.conv.direct_variants import (
            select_direct_kernel_builder,
            validate_direct_spec,
        )

        want = [n for n, _ in conv_direct_arg_names(direction="fwd", bias=True)]
        for variant, spec in _wmma_specs():
            with self.subTest(name=spec.kernel_name()):
                name = spec.kernel_name()
                tag = "_cbias_bf16" if spec.problem.dtype == "bf16" else "_cbias"
                self.assertTrue(name.endswith(tag), name)
                self.assertNotIn("cbias", replace(spec, bias=False).kernel_name())
                ok, why = validate_direct_spec(variant, spec, "gfx1250")
                self.assertTrue(ok, why)
                k = select_direct_kernel_builder(variant, spec, "gfx1250")
                self.assertEqual([p.name for p in k.params], want)

    def test_engine_bridge_forwards_bias(self):
        from kernels.conv import engine_bridge as eb

        to_dict = {
            "direct_fwd_wmma": eb.conv_direct_fwd_wmma_spec_to_dict,
            "direct_fwd_tapouter_wmma": eb.conv_direct_fwd_tapouter_wmma_spec_to_dict,
            "direct_fwd_gstream_wmma": eb.conv_direct_fwd_gstream_wmma_spec_to_dict,
            "direct_fwd_smallc_wmma": eb.conv_direct_fwd_smallc_wmma_spec_to_dict,
            "direct_fwd_smallk_wmma": eb.conv_direct_fwd_smallk_wmma_spec_to_dict,
            "direct_dw_fwd_wmma": eb.conv_direct_dw_fwd_wmma_spec_to_dict,
        }
        for variant, spec in _wmma_specs():
            self.assertIs(to_dict[variant](spec)["bias"], True, variant)


@pytest.mark.gpu
@unittest.skipIf(_GFX1250_SKIP, _GFX1250_SKIP)
class TestDirectWmmaBiasGpu(_GpuCase):
    def test_families(self):
        for variant, spec in _wmma_specs():
            with self.subTest(name=spec.kernel_name()):
                self._check(variant, spec)


def _dwp(C=64, *, N=2, H=11, W=13, KH=3, stride=1, dtype="fp16"):
    return _p(N, H, W, 1, 1, groups=C, KH=KH, PAD=KH // 2, stride=stride, dtype=dtype)


@pytest.mark.gpu
@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestDirectDepthwiseBiasGpu(_GpuCase):
    """The depthwise forward kernels: bias[ch] per lane, added to every
    flushed output (scalar and channel-packed)."""

    def test_streaming(self):
        from kernels.common.conv_direct_grouped import DirectDepthwiseSpec

        for dtype in ("fp16", "bf16", "fp32"):
            for KH, stride in ((3, 1), (5, 2)):
                with self.subTest(dtype=dtype, KH=KH, stride=stride):
                    p = _dwp(C=96, KH=KH, stride=stride, dtype=dtype)
                    spec = DirectDepthwiseSpec(problem=p, block_w=8, bias=True)
                    self._check("direct_depthwise", spec)
        spec = DirectDepthwiseSpec(problem=_dwp(C=128), cpl=2, bias=True)
        self._check("direct_depthwise", spec, arches=("gfx950",))

    def test_spatial(self):
        from kernels.common.conv_direct_grouped import DirectDepthwiseSpatialSpec

        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                spec = DirectDepthwiseSpatialSpec(
                    problem=_dwp(C=16, dtype=dtype), block_waves=2, bias=True
                )
                self._check("direct_depthwise_spatial", spec)

    def test_tiled(self):
        from kernels.common.conv_direct_grouped import DirectDepthwiseTiledSpec

        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                spec = DirectDepthwiseTiledSpec(
                    problem=_dwp(C=80, KH=5, dtype=dtype),
                    block_w=8,
                    block_h=4,
                    bias=True,
                )
                self._check("direct_depthwise_tiled", spec)

    def test_col(self):
        from kernels.common.conv_direct_grouped import DirectDepthwiseColSpec

        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = _dwp(C=80, KH=7, stride=2, dtype=dtype)
                spec = DirectDepthwiseColSpec(
                    problem=p, dtype=dtype, block_w=2, block_h=8, bias=True
                )
                self._check("direct_depthwise_col", spec)

    def test_windowed_toeplitz_fwd(self):
        from kernels.common.conv_direct_grouped import (
            DirectDepthwiseDgradWindowedSpec,
            dw_win_fwd_problem,
        )

        for C, cpg, stride in ((64, 1, 1), (64, 1, 2), (8, 16, 1)):
            with self.subTest(C=C, cpg=cpg, stride=stride):
                p = _p(2, 14, 14, cpg, cpg, groups=C, KH=7, PAD=3, stride=stride)
                spec = DirectDepthwiseDgradWindowedSpec(
                    problem=dw_win_fwd_problem(p),
                    mfma=True,
                    fwd=True,
                    out_stride=stride,
                    block_waves=8,
                    w_fold=2,
                    prefetch_rows=2,
                    block_h=14,
                    bias=True,
                )
                self._check("direct_depthwise_win", spec, p, arches=("gfx950",))


@pytest.mark.gpu
@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestDirectSpecialisedBiasGpu(_GpuCase):
    """The specialised grouped kernels (16 / 8 / 32 / 4 channels per group)."""

    def test_16c(self):
        from kernels.common.conv_direct_grouped import DirectConv16cSpec

        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                p = _p(2, 9, 20, 16, 16, groups=8, dtype=dtype)
                spec = DirectConv16cSpec(
                    problem=p, block_groups=4, fold_k32=_ARCH == "gfx950", bias=True
                )
                self._check("16c", spec)

    def test_8c_32c(self):
        from kernels.common.conv_direct_grouped import (
            DirectConv32cSpec,
            DirectConv8cSpec,
        )

        p8 = _p(2, 9, 20, 8, 8, groups=16, dtype="bf16")
        self._check("8c", DirectConv8cSpec(problem=p8, bias=True), arches=("gfx950",))
        p32 = _p(2, 9, 40, 32, 32, groups=8, dtype="fp16")
        self._check(
            "32c", DirectConv32cSpec(problem=p32, bias=True), arches=("gfx950",)
        )

    def test_4c(self):
        from kernels.common.conv_direct_grouped import DirectConv4cSpec

        p = _p(2, 9, 20, 4, 4, groups=32, dtype="bf16")
        spec = DirectConv4cSpec(problem=p, block_q=4, block_groups=16, bias=True)
        self._check("4c", spec)


if __name__ == "__main__":
    unittest.main()
