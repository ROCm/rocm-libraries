# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Guard: the AOT conv kernarg ABI agrees across all three places that use it.

Kernargs are packed *positionally* from the launch signature
(``rocke.runtime.packing.pack_args`` only uses names to look values up), so if
the kernel declares its params in a different order than the signature lists
them, every argument past the first divergence silently shifts — pointers get
read out of i32 slots and the kernel faults or corrupts memory. Nothing else
in the test suite catches that, because both sides individually look fine.

This pins together:

1. ``kernels.common.conv_abi``    — the ordered name list (the ABI) and the
                                       launch signature built from it
2. the kernel builders                — ``[p.name for p in kernel.params]``
3. ``kernels.common.conv_args``       — the host-side ``values`` dict keys
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import pytest

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    build_implicit_gemm_conv_dgrad,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
)
from kernels.common.conv_args import ConvArgs
from kernels.common.conv_abi import conv_arg_names, conv_direct_arg_names
from kernels.common.conv_abi import conv_args_signature, conv_direct_args_signature

_ARCH = "gfx950"
_DATA = ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16")

_TILE = dict(
    tile_m=64,
    tile_n=64,
    tile_k=32,
    warp_m=2,
    warp_n=2,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
    wave_size=64,
    vector_size_c=1,
)

_P2D = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1)
_P2D_GROUPED = ConvProblem(
    N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, groups=2
)
_P3D = ConvProblem(
    N=2,
    Di=8,
    Hi=14,
    Wi=14,
    C=64,
    K=64,
    Z=3,
    Y=3,
    X=3,
    sH=1,
    sW=1,
    pH=1,
    pW=1,
    sD=1,
    pD=1,
    dD=1,
)


def _names(sig):
    return [a["name"] for a in sig]


def _assert_abi(kernel, abi_names, signature, label):
    """The kernel, the ABI list and the signature must be the same sequence."""
    kernel_order = [p.name for p in kernel.params]
    expected = [n for n, _ in abi_names]
    assert kernel_order == expected, (
        f"{label}: kernel param order diverges from conv_abi.\n"
        f"  kernel: {kernel_order}\n  abi:    {expected}"
    )
    assert _names(signature) == expected, (
        f"{label}: launch signature diverges from conv_abi.\n"
        f"  signature: {_names(signature)}\n  abi:       {expected}"
    )


# ---------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,problem,is_3d",
    [
        ("fwd-2d", _P2D, False),
        ("fwd-2d-grouped", _P2D_GROUPED, False),
        ("fwd-3d", _P3D, True),
    ],
)
def test_fwd_abi(label, problem, is_3d):
    kernel = build_implicit_gemm_conv(
        ImplicitGemmConvSpec(problem=problem, data=_DATA, **_TILE), arch=_ARCH
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="fwd", is_3d=is_3d),
        conv_args_signature("fp16", is_3d=is_3d),
        label,
    )


@pytest.mark.parametrize(
    "label,pipeline,extra",
    [
        ("mem", "mem", {}),
        ("compv3", "compv3", {}),
        ("compv4", "compv4", {}),
        ("basic", "basic", {}),
        ("unroll_k", "mem", {"unroll_k": True}),
        ("async_dma", "mem", {"async_dma": True}),
        ("chiplet", "mem", {"chiplet_swizzle": True}),
    ],
)
def test_fwd_abi_is_pipeline_independent(label, pipeline, extra):
    """The kernarg block must not depend on the K-loop driver."""
    kernel = build_implicit_gemm_conv(
        ImplicitGemmConvSpec(
            problem=_P2D, data=_DATA, pipeline=pipeline, **_TILE, **extra
        ),
        arch=_ARCH,
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="fwd"),
        conv_args_signature("fp16"),
        f"fwd-{label}",
    )


# ---------------------------------------------------------------------
# wgrad — one case per optional trailing-arg combination
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,problem,spec_kw,abi_kw",
    [
        ("wgrad-2d", _P2D, {"split_k": 1}, {}),
        ("wgrad-3d", _P3D, {"split_k": 1}, {"is_3d": True}),
        (
            "wgrad-split-k",
            _P2D,
            {"split_k": 2, "epilogue": "cshuffle", "vector_size_c": 2},
            {},
        ),
        (
            "wgrad-split-k-grouped",
            _P2D_GROUPED,
            {"split_k": 2, "epilogue": "cshuffle", "vector_size_c": 2},
            {},
        ),
    ],
)
def test_wgrad_abi(label, problem, spec_kw, abi_kw):
    tile = dict(_TILE)
    tile.update({k: v for k, v in spec_kw.items() if k in tile})
    spec_kw = {k: v for k, v in spec_kw.items() if k not in tile}
    kernel = build_implicit_gemm_conv_wgrad(
        WgradConvSpec(problem=problem, data=_DATA, **tile, **spec_kw), arch=_ARCH
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="wgrad", **abi_kw),
        conv_args_signature("fp16", direction="wgrad", **abi_kw),
        label,
    )


# ---------------------------------------------------------------------
# dgrad
# ---------------------------------------------------------------------


def test_dgrad_abi():
    kernel = build_implicit_gemm_conv_dgrad(
        DgradConvSpec(problem=_P2D, data=_DATA, **_TILE), arch=_ARCH
    )
    _assert_abi(
        kernel,
        conv_arg_names(direction="dgrad"),
        conv_args_signature("fp16", direction="dgrad"),
        "dgrad",
    )


# ---------------------------------------------------------------------
# host-side values dicts
# ---------------------------------------------------------------------


@pytest.mark.parametrize("problem", [_P2D, _P2D_GROUPED, _P3D])
def test_fwd_launch_values_cover_the_abi(problem):
    values = ConvArgs.from_problem(problem, tile_m=64, tile_n=64).to_launch_values(
        1, 2, 3, 4, 5, 6
    )
    expected = [n for n, _ in conv_arg_names(direction="fwd", is_3d=problem.is_3d)]
    assert sorted(values) == sorted(expected)


@pytest.mark.parametrize("problem", [_P2D, _P3D])
def test_wgrad_launch_values_cover_the_abi(problem):
    values = ConvArgs.from_problem(
        problem, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
    ).to_launch_values(1, 2, 3, 4, 5, 6)
    expected = [n for n, _ in conv_arg_names(direction="wgrad", is_3d=problem.is_3d)]
    assert sorted(values) == sorted(expected)


def test_wgrad_launch_values_split_k_grouped():
    """The split degree is a launch parameter, so it changes values not shape.

    Splitting must not select a different ABI: the same kernel serves any
    degree, and ``ks``/``ks_count`` simply carry different numbers.
    """
    args = ConvArgs.from_problem(
        _P2D_GROUPED, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
    )
    expected = sorted(n for n, _ in conv_arg_names(direction="wgrad"))
    unsplit = args.to_launch_values(1, 2, 3, 4, 5, 6, split_k=1)
    split = args.to_launch_values(1, 2, 3, 4, 5, 6, split_k=4)
    assert sorted(unsplit) == expected
    assert sorted(split) == expected
    assert unsplit["ks_count"] == 1 and split["ks_count"] == 4
    # Every slice is tile-aligned and together they cover the whole reduction.
    assert split["ks"] % 32 == 0
    assert split["ks"] * 4 >= args.gemm_k


def test_wgrad_launch_values_two_stage():
    values = ConvArgs.from_problem(
        _P2D, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
    ).to_launch_values(1, 2, 3, 4, 5, 6, ws_ptr=99, ws_bytes=128)
    expected = [n for n, _ in conv_arg_names(direction="wgrad", two_stage=True)]
    assert sorted(values) == sorted(expected)


def test_dgrad_launch_values_cover_the_abi():
    values = ConvArgs.from_problem(
        _P2D, direction="dgrad", tile_m=64, tile_n=64
    ).to_launch_values(1, 2, 3, 4, 5, 6, sub_gemm_buf=7, num_sub_gemms=1)
    expected = [n for n, _ in conv_arg_names(direction="dgrad")]
    assert sorted(values) == sorted(expected)


def test_fwd_launch_values_reject_a_reduction_past_the_mul24_bound():
    """The fwd K-loop decodes k with 24-bit multiplies; the host enforces it."""
    # Y = X = 3, cpg = 2**20: Z*Y*X*cpg = 9 * 2**20 >= 2**23.
    big = ConvProblem(N=1, Hi=4, Wi=4, C=1 << 20, K=8, Y=3, X=3, pH=1, pW=1)
    with pytest.raises(ValueError, match="2\\*\\*23"):
        ConvArgs.from_problem(big, tile_m=64, tile_n=64).to_launch_values(
            1, 2, 3, 4, 5, 6
        )
    # wgrad decodes no filter-channel index in its K loop: no bound there.
    ConvArgs.from_problem(
        big, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
    ).to_launch_values(1, 2, 3, 4, 5, 6)


def test_dgrad_3d_abi_exists_but_the_spec_refuses_to_build_it():
    """The 3-D dgrad ABI is declared; only the spec validator says no.

    Keeping the refusal in exactly one place matters: if the argument builder
    also refused, a future 3-D dgrad kernel would land and silently have no
    way to be launched, and the second refusal would be found by debugging
    rather than by reading the validator.
    """
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    # The host can fill the 3-D block, and it matches the declared ABI.
    values = ConvArgs.from_problem(
        _P3D, direction="dgrad", tile_m=64, tile_n=64
    ).to_launch_values(1, 2, 3, 4, 5, 6, sub_gemm_buf=7, num_sub_gemms=1)
    assert sorted(values) == sorted(
        n for n, _ in conv_arg_names(direction="dgrad", is_3d=True)
    )

    # Building a kernel for it is what fails, and it says why.
    with pytest.raises(ValueError, match="3-D"):
        DgradConvSpec(problem=_P3D, data=_DATA, **_TILE).validate()


# ---------------------------------------------------------------------
# magic-number semantics
# ---------------------------------------------------------------------


def _magic_div(x: int, mult: int, shift: int) -> int:
    """Mirror of ``do_magic_division`` on unsigned 32-bit values."""
    if mult < 0:
        mult += 1 << 32
    return (((x * mult) >> 32) + x) >> shift


@pytest.mark.parametrize("problem", [_P2D, _P2D_GROUPED])
def test_m_decode_matches_the_reference_split(problem):
    """``m -> (n, ho, wo)`` must divide by Wo then Ho, not by Ho*Wo.

    Dividing by the product instead collapses every batch index onto n == 0,
    which is correct only for N == 1 — exactly the shape a smoke test uses.
    """
    args = ConvArgs.from_problem(problem, tile_m=64, tile_n=64)
    v = args.to_launch_values(0, 0, 0, 0, 0, 0)
    Ho, Wo, N = v["p_Ho"], v["p_Wo"], v["p_N"]
    for m in range(N * Ho * Wo):
        tmp = m
        wo = tmp - _magic_div(tmp, v["p_magic_m_Wo_mult"], v["p_magic_m_Wo_shift"]) * Wo
        tmp = _magic_div(tmp, v["p_magic_m_Wo_mult"], v["p_magic_m_Wo_shift"])
        ho = tmp - _magic_div(tmp, v["p_magic_m_Ho_mult"], v["p_magic_m_Ho_shift"]) * Ho
        n = _magic_div(tmp, v["p_magic_m_Ho_mult"], v["p_magic_m_Ho_shift"])
        assert (n, ho, wo) == (m // (Ho * Wo), (m // Wo) % Ho, m % Wo), f"m={m}"


def test_k_decode_matches_the_reference_split():
    """``k -> (y, x, c)`` must divide by cpg then X, not by X*cpg."""
    args = ConvArgs.from_problem(_P2D, tile_m=64, tile_n=64)
    v = args.to_launch_values(0, 0, 0, 0, 0, 0)
    X, cpg, Y = v["p_X"], v["p_cpg"], v["p_Y"]
    for k in range(Y * X * cpg):
        q = _magic_div(k, v["p_magic_k_cpg_mult"], v["p_magic_k_cpg_shift"])
        c = k - q * cpg
        y = _magic_div(q, v["p_magic_k_X_mult"], v["p_magic_k_X_shift"])
        x = q - y * X
        assert (y, x, c) == (k // (X * cpg), (k // cpg) % X, k % cpg), f"k={k}"


# ---------------------------------------------------------------------
# direct grouped convolution
# ---------------------------------------------------------------------


def _op_signature(kernel):
    """Structural fingerprint of the emitted IR: param names + op sequence.

    Two kernels with the same fingerprint emit the same instructions, which is
    the property that makes a kernel shape-generic.

    Constant *values* are part of the fingerprint, not just the op names. A
    baked ``const_i32(Ho)`` emits the same ``arith.constant`` op whatever the
    shape, so comparing op names alone would call such a kernel shape-generic
    while it silently only computes the right answer for the height it was
    built for -- exactly the bug this test exists to catch.
    """
    out = [p.name for p in kernel.params]

    def walk(op):
        if op.name == "arith.constant":
            out.append(f"{op.name}={op.attrs.get('value')!r}")
        else:
            out.append(op.name)
        for region in op.regions or []:
            for inner in region.ops:
                walk(inner)

    for op in kernel.body.ops:
        walk(op)
    return tuple(out)


def _direct_builders():
    """``(label, channel kwargs, direction, build fn)`` for every direct variant."""
    from kernels.common import conv_direct_grouped as dc

    return [
        (
            "16c",
            dict(cpg=16, kpg=16, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv_16c(
                dc.DirectConv16cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "4c",
            dict(cpg=4, kpg=4, groups=32),
            "fwd",
            lambda pr: dc.build_direct_conv_4c(
                dc.DirectConv4cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "8c",
            dict(cpg=8, kpg=8, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv_8c(
                dc.DirectConv8cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "32c",
            dict(cpg=32, kpg=32, groups=4),
            "fwd",
            lambda pr: dc.build_direct_conv_32c(
                dc.DirectConv32cSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "generic",
            dict(cpg=16, kpg=16, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv(dc.DirectConvSpec(problem=pr), arch=_ARCH),
        ),
        (
            "depthwise",
            dict(cpg=1, kpg=1, groups=32),
            "fwd",
            lambda pr: dc.build_direct_depthwise(
                dc.DirectDepthwiseSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "depthwise_col",
            dict(cpg=1, kpg=1, groups=32),
            "fwd",
            lambda pr: dc.build_direct_depthwise_col(
                dc.DirectDepthwiseColSpec(problem=pr, block_h=4, block_w=2), arch=_ARCH
            ),
        ),
        (
            "depthwise_spatial",
            dict(cpg=1, kpg=1, groups=32),
            "fwd",
            lambda pr: dc.build_direct_depthwise_spatial(
                dc.DirectDepthwiseSpatialSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "dgrad",
            dict(cpg=16, kpg=16, groups=8),
            "dgrad",
            lambda pr: dc.build_direct_conv_dgrad(
                dc.DirectConvDgradSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "depthwise_dgrad",
            dict(cpg=1, kpg=1, groups=32),
            "dgrad",
            lambda pr: dc.build_direct_depthwise_dgrad(
                dc.DirectDepthwiseDgradSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "depthwise_dgrad_streaming",
            dict(cpg=1, kpg=1, groups=32),
            "dgrad",
            lambda pr: dc.build_direct_depthwise_dgrad_streaming(
                dc.DirectDepthwiseDgradStreamSpec(problem=pr), arch=_ARCH
            ),
        ),
        (
            "wgrad",
            dict(cpg=16, kpg=16, groups=8),
            "wgrad",
            lambda pr: dc.build_direct_conv_wgrad(
                dc.DirectConvWgradSpec(problem=pr), arch=_ARCH
            ),
        ),
        # waves_q > 1 derives n_q_blocks from the runtime wo-tile count with a
        # second ceiling; bf16 at the narrow atom covers the other loader width.
        (
            "wgrad_bf16_mk16_wq2",
            dict(cpg=32, kpg=32, groups=4, dtype="bf16"),
            "wgrad",
            lambda pr: dc.build_direct_conv_wgrad(
                dc.DirectConvWgradSpec(problem=pr, mfma_k=16, waves_q=2), arch=_ARCH
            ),
        ),
        # The MFMA dgrad's compute pass: a forward kernel on the transposed
        # problem, so it takes the forward ABI.
        (
            "mfma_dgrad_fprop",
            dict(cpg=16, kpg=64, groups=8),
            "fwd",
            lambda pr: dc.build_direct_conv(
                dc.make_dgrad_fprop_spec(pr, block_h=8, waves_k=4), arch=_ARCH
            ),
        ),
    ]


# The spatial depthwise kernel packs every channel of an image into one
# wavefront, so its lane layout -- and with it the emitted IR -- follows the
# group count. It is the one direct kernel whose groups are not a kernarg.
_GROUPS_BAKED = {"depthwise_spatial"}


@pytest.mark.parametrize(
    "label,channels,direction,build",
    _direct_builders(),
    ids=[c[0] for c in _direct_builders()],
)
def test_direct_conv_abi_and_shape_invariance(label, channels, direction, build):
    """Direct conv must be shape-generic and match the direct kernarg ABI.

    The shape-invariance half is the load-bearing one: these builders used to
    unroll the H-row loop at build time, so a kernel silently only worked for
    the height it was compiled for. Comparing the emitted op sequence across
    three unrelated shapes is what catches a regression back to that.
    """
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from kernels.common.conv_abi import conv_direct_arg_names
    from kernels.common.conv_abi import conv_direct_args_signature

    fingerprints = []
    kernel = None
    # The group count varies too: it is a kernarg, so a binary built for one
    # count has to serve every multiple of the block's group tile.
    for (N, H, W), group_mult in zip(
        [(2, 14, 14), (3, 29, 37), (1, 64, 64)], (1, 2, 3)
    ):
        shape = dict(channels)
        if label not in _GROUPS_BAKED:
            shape["groups"] *= group_mult
        problem = DirectConvProblem(N=N, H=H, W=W, KH=3, KW=3, PAD=1, stride=1, **shape)
        kernel = build(problem)
        fingerprints.append(_op_signature(kernel))

    assert len(set(fingerprints)) == 1, (
        f"{label}: emitted IR depends on the problem shape, so the kernel is "
        f"not shape-generic"
    )

    expected = [n for n, _ in conv_direct_arg_names(direction=direction)]
    assert [p.name for p in kernel.params] == expected, f"{label}: kernarg order"
    signature = conv_direct_args_signature("fp16", direction=direction)
    assert _names(signature) == expected, f"{label}: launch signature"


@pytest.mark.parametrize("fold_k32", [False, True])
def test_direct_dgrad_weight_transforms_are_shape_invariant(fold_k32):
    """The MFMA dgrad's weight transforms are cached once per filter/channels.

    They bake KH/KW/cpg/kpg and nothing else, so the same binary has to come
    out for any batch, image size or group count.
    """
    from kernels.common import conv_direct_grouped as dc

    fingerprints = set()
    for N, H, W, groups in [(2, 14, 14, 8), (3, 29, 37, 16), (1, 64, 64, 24)]:
        problem = dc.DirectConvProblem(
            N=N, H=H, W=W, groups=groups, cpg=32, kpg=32, KH=3, KW=3, PAD=1, stride=1
        )
        transpose = dc.build_direct_transpose_weights_dgrad(
            dc.DirectTransposeWeightsDgradSpec(problem=problem), arch=_ARCH
        )
        reorganize = dc.build_direct_reorganize_weights(
            dc.DirectReorganizeWeightsSpec(problem=problem, fold_k32=fold_k32),
            arch=_ARCH,
        )
        fingerprints.add((_op_signature(transpose), _op_signature(reorganize)))
    assert len(fingerprints) == 1


@pytest.mark.parametrize("direction", ["fwd", "dgrad", "wgrad"])
def test_direct_launch_values_cover_the_abi(direction):
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_abi import conv_direct_arg_names

    problem = DirectConvProblem(N=2, H=14, W=14, groups=8, cpg=16, kpg=16)
    values = ConvArgs.from_problem(problem, direction=direction).to_launch_values(
        1, 2, 3, 4, 5, 6
    )
    expected = [n for n, _ in conv_direct_arg_names(direction=direction)]
    assert sorted(values) == sorted(expected)


# ---------------------------------------------------------------------
# implicit GEMM: shape invariance
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,build",
    [
        (
            "fwd",
            lambda pr: build_implicit_gemm_conv(
                ImplicitGemmConvSpec(problem=pr, data=_DATA, **_TILE), arch=_ARCH
            ),
        ),
        (
            "wgrad",
            lambda pr: build_implicit_gemm_conv_wgrad(
                WgradConvSpec(problem=pr, data=_DATA, split_k=1, **_TILE), arch=_ARCH
            ),
        ),
        (
            "dgrad",
            lambda pr: build_implicit_gemm_conv_dgrad(
                DgradConvSpec(problem=pr, data=_DATA, **_TILE), arch=_ARCH
            ),
        ),
    ],
)
def test_implicit_gemm_is_shape_invariant(label, build):
    """One compiled kernel per tile config must serve every compatible shape."""
    shapes = [
        ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1),
        ConvProblem(N=5, Hi=31, Wi=27, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1),
        ConvProblem(N=1, Hi=64, Wi=64, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1),
    ]
    fingerprints = {_op_signature(build(p)) for p in shapes}
    assert len(fingerprints) == 1, f"{label}: emitted IR depends on the shape"


# ---------------------------------------------------------------------
# capability coverage: what the IR *does* depend on must be in the identity
# ---------------------------------------------------------------------


_CAPABILITY_CASES = [
    # (label, build fn, attribute kwargs, KernelIdentity fields that record it)
    ("fwd grouped", "fwd", dict(groups=2), ("grouped",)),
    ("wgrad grouped", "wgrad", dict(groups=2), ("grouped",)),
    ("dgrad grouped", "dgrad", dict(groups=2), ("grouped",)),
    ("dgrad stride", "dgrad", dict(sH=2, sW=2), ("stride_h", "stride_w")),
    ("dgrad dilation", "dgrad", dict(dH=2, dW=2), ("dilation_h", "dilation_w")),
]


def _build_direction(direction, problem):
    if direction == "fwd":
        return build_implicit_gemm_conv(
            ImplicitGemmConvSpec(problem=problem, data=_DATA, **_TILE), arch=_ARCH
        )
    if direction == "wgrad":
        return build_implicit_gemm_conv_wgrad(
            WgradConvSpec(problem=problem, data=_DATA, split_k=1, **_TILE), arch=_ARCH
        )
    return build_implicit_gemm_conv_dgrad(
        DgradConvSpec(problem=problem, data=_DATA, **_TILE), arch=_ARCH
    )


@pytest.mark.parametrize(
    "label,direction,attrs,identity_fields",
    _CAPABILITY_CASES,
    ids=[c[0] for c in _CAPABILITY_CASES],
)
def test_baked_attributes_are_recorded_as_capabilities(
    label, direction, attrs, identity_fields
):
    """Anything the IR bakes must be a capability the AOT cache can filter on.

    A problem attribute that changes the emitted IR but is *not* in
    :class:`~benchmarks.common.kernel_cache.KernelIdentity` is the worst kind
    of AOT bug: the
    cache hands out a kernel built for a different value, the launch succeeds,
    and the numbers are quietly wrong. Two halves are asserted here --
    the attribute really does change the IR (so it is not over-declared), and
    the identity has a field for it (so the cache can reject the mismatch).
    """
    from dataclasses import fields as dc_fields

    from benchmarks.common.kernel_cache import KernelIdentity

    base = ConvProblem(N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1)
    variant = dc_replace(base, **attrs)

    assert _op_signature(_build_direction(direction, base)) != _op_signature(
        _build_direction(direction, variant)
    ), (
        f"{label}: this attribute no longer changes the emitted IR. If that is "
        f"intentional, drop it from the identity so the cache stops splitting "
        f"binaries on it."
    )

    known = {f.name for f in dc_fields(KernelIdentity)}
    missing = [f for f in identity_fields if f not in known]
    assert not missing, f"{label}: KernelIdentity has no field(s) {missing}"


def test_cache_rejects_capability_mismatches():
    """``supports_problem`` must refuse a kernel built for other capabilities.

    The identity carrying the field is only half the guard; the cache has to
    actually check it. Each case below produced a silently wrong result before
    the check existed.
    """
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    cache = KernelCache("/nonexistent-cache-dir-for-a-pure-predicate-test", _ARCH)
    common = dict(
        arch=_ARCH,
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=2,
        vector_size_b=2,
        vector_size_c=2,
    )
    plain = ConvProblem(N=1, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, pH=1, pW=1)
    strided = dc_replace(plain, sH=2, sW=2)
    dilated = dc_replace(plain, dH=2, dW=2)
    grouped = dc_replace(plain, groups=2)

    ungrouped_fwd = KernelIdentity(direction="fwd", grouped=False, **common)
    grouped_fwd = KernelIdentity(direction="fwd", grouped=True, **common)
    dgrad_s1 = KernelIdentity(
        direction="dgrad",
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        max_sub_gemms=64,
        **common,
    )

    assert cache.supports_problem(ungrouped_fwd, plain)[0]
    assert cache.supports_problem(grouped_fwd, grouped)[0]
    assert not cache.supports_problem(ungrouped_fwd, grouped)[0]
    # The grouped path computes groups == 1 too, so it serves both.
    assert cache.supports_problem(grouped_fwd, plain)[0]

    assert cache.supports_problem(dgrad_s1, plain)[0]
    assert not cache.supports_problem(dgrad_s1, strided)[0]
    assert not cache.supports_problem(dgrad_s1, dilated)[0]


# ---------------------------------------------------------------------
# Manifests, the two-stage signature and the dgrad cache bound
# ---------------------------------------------------------------------


def test_two_stage_stage1_signature_matches_builder_abi():
    """The two-stage stage-1 signature is the wgrad ABI with the workspace pair."""
    from kernels.common.conv_implicit_gemm_wgrad_two_stage import (
        _wgrad_stage1_signature,
    )

    spec = WgradConvSpec(problem=_P2D, data=_DATA, split_k=2, two_stage=True, **_TILE)
    assert _wgrad_stage1_signature(spec) == conv_args_signature(
        "fp16", direction="wgrad", two_stage=True
    )


class _Artifact:
    kernel_name = "k"
    timings: dict = {}
    hsaco_bytes = 0


class _FakeRuntime:
    """Just enough of ``Runtime`` for the runner to pack its kernargs."""

    def __init__(self):
        self._next = 0x1000

    def alloc(self, _n):
        self._next += 0x1000
        return self._next

    def memcpy_h2d(self, *_a):
        pass

    def memset(self, *_a):
        pass


def _conv_manifest(conv, conv_layout, **kw):
    from kernels.common.conv_abi import conv_manifest_args_signature
    from rocke.helpers.manifest import make_conv_manifest

    base = dict(
        artifact=_Artifact(),
        block_m=64,
        block_n=64,
        block_k=32,
        threads_per_block=256,
        groups=1,
        cpg=conv[3] if len(conv) == 13 else conv[4],
        kpg=conv[4] if len(conv) == 13 else conv[5],
    )
    base.update(kw)
    base.setdefault(
        "args_signature",
        conv_manifest_args_signature(
            conv_layout=conv_layout, direction=base.get("direction", "fwd")
        ),
    )
    return make_conv_manifest(conv=conv, conv_layout=conv_layout, **base)


@pytest.mark.parametrize("layout", ["direct_grouped", "direct_grouped_4c"])
def test_direct_manifest_advertises_and_packs_direct_abi(layout):
    """Every direct layout variant gets the direct ABI, and the runner packs it."""
    from kernels.common.manifest_runner.conv import run_conv_manifest_problem

    m = _conv_manifest(
        [1, 8, 8, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1],
        layout,
        groups=4,
        cpg=16,
        kpg=16,
        grid_explicit=[1, 1, 1],
    )
    assert m["args_signature"] == conv_direct_args_signature("fp16")
    make_args, *_ = run_conv_manifest_problem(m, None, False)
    make_args(_FakeRuntime())


def test_3d_manifest_accepts_18_fields_and_packs_3d_abi():
    from kernels.common.manifest_runner.conv import run_conv_manifest_problem

    conv3d = [1, 4, 8, 8, 16, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    m = _conv_manifest(conv3d, "implicit_gemm_3d")
    assert m["args_signature"] == conv_args_signature("fp16", is_3d=True)
    make_args, *_ = run_conv_manifest_problem(m, None, False)
    make_args(_FakeRuntime())
    # The field count follows the layout: a 2-D manifest still needs 13.
    with pytest.raises(ValueError):
        _conv_manifest(conv3d, "implicit_gemm")


def test_manifest_runner_rejects_backward_directions():
    """The runner's buffers and reference are forward-only."""
    from kernels.common.manifest_runner.conv import run_conv_manifest_problem

    m = _conv_manifest(
        [1, 8, 8, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1], "implicit_gemm", direction="wgrad"
    )
    with pytest.raises(ValueError, match="direction"):
        run_conv_manifest_problem(m, None, False)


def test_cache_rejects_dgrad_beyond_max_sub_gemms():
    """A dgrad binary cannot serve a problem with more tilde sub-GEMMs than
    its CTA dispatch search was unrolled for."""
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    cache = KernelCache("/nonexistent-cache-dir-for-a-pure-predicate-test", _ARCH)
    ident = dict(
        arch=_ARCH,
        direction="dgrad",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=2,
        vector_size_b=2,
        vector_size_c=2,
    )
    # Stride 2 with a 3x3 filter decomposes into 2x2 = 4 tilde sub-GEMMs.
    strided = ConvProblem(N=1, Hi=16, Wi=16, C=32, K=32, Y=3, X=3, sH=2, sW=2)
    assert cache.supports_problem(KernelIdentity(max_sub_gemms=4, **ident), strided)[0]
    ok, reason = cache.supports_problem(
        KernelIdentity(max_sub_gemms=2, **ident), strided
    )
    assert not ok and "sub-GEMMs" in reason


def test_deep_fused_conv_pool_signature_matches_kernel_params():
    """deep_fused_conv_pool builds conv0 through the AOT implicit-GEMM builder,
    so its launch signature has to carry the runtime problem block too -- and
    the launch values have to fill every entry of it."""
    from kernels.common.deep_fused_conv_pool import (
        build_deep_fused_conv_pool,
        deep_fused_conv_pool_problem_values,
        deep_fused_conv_pool_signature,
        make_deep_fused_conv_pool_spec,
    )
    from rocke.runtime.packing import pack_args

    spec = make_deep_fused_conv_pool_spec(
        n=1,
        h=64,
        w=128,
        c=8,
        k0=16,
        k1=16,
        r=3,
        s=3,
        pool_tile_h=4,
        pool_tile_w=8,
        tile_n=16,
        tile_k=16,
        warp_m=2,
        warp_n=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        wave_size=64,
    )
    kernel = build_deep_fused_conv_pool(spec, arch="gfx950")
    sig = deep_fused_conv_pool_signature(spec)
    # The kernel names its output D; the manifest/launcher calls it Y.
    kernel_order = [
        {"D": "Y", "D_bytes": "Y_bytes"}.get(p.name, p.name) for p in kernel.params
    ]
    assert _names(sig) == kernel_order

    values = dict.fromkeys(("A", "B", "Y", "W1"), 0x1000)
    values.update(dict.fromkeys(("W1_bytes", "A_bytes", "B_bytes", "Y_bytes"), 1))
    values.update(
        deep_fused_conv_pool_problem_values(
            spec.problem.conv, tile_m=spec.tile_m, tile_n=spec.tile_n
        )
    )
    assert set(values) == set(_names(sig))
    pack_args(sig, values)


def _cache_identity(**kw):
    from benchmarks.common.kernel_cache import KernelIdentity

    base = dict(
        arch=_ARCH,
        direction="fwd",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=2,
        vector_size_b=2,
        vector_size_c=2,
    )
    base.update(kw)
    return KernelIdentity(**base)


def test_cache_rejects_stale_provenance():
    """A binary lowered for another LLVM flavor never matches, and an entry
    that predates the flavor field reads as stale."""
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    cache = KernelCache("/nonexistent-cache-dir-for-a-pure-predicate-test", _ARCH)
    current = _cache_identity()
    assert cache.supports_problem(current, _P2D)[0]
    stale = dc_replace(current, llvm_flavor="llvm-other")
    assert not cache.supports_problem(stale, _P2D)[0]
    assert stale.stable_hash() != current.stable_hash()

    legacy = current.to_dict()
    del legacy["llvm_flavor"]
    assert not cache.supports_problem(KernelIdentity.from_dict(legacy), _P2D)[0]


def test_identity_survives_emitter_changes(tmp_path):
    """An identity names a configuration, not an emitter version: the same
    configuration keeps its entry, and an entry built from other sources is
    still offered but counted as stale for --compile-all to refresh."""
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    current = _cache_identity()
    legacy = dict(current.to_dict(), emitter_digest="0" * 40)
    assert KernelIdentity.from_dict(legacy).stable_hash() == current.stable_hash()

    cache = KernelCache(tmp_path, _ARCH)
    cache.put(current, b"bin", {"kernel_name": "k", "emitter_digest": "0" * 40})
    assert cache.supports_problem(current, _P2D)[0]
    assert cache.stale_entries() == 1


def test_cache_async_chunk_must_divide_cpg():
    """The async loaders' chunk width is baked from the build-time cpg; a
    problem whose cpg it does not divide would read across filter positions."""
    from benchmarks.common.kernel_cache import KernelCache
    from benchmarks.common.kernel_sweep import _async_chunks

    cache = KernelCache("/nonexistent-cache-dir-for-a-pure-predicate-test", _ARCH)
    cfg = dict(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=2,
        vector_size_b=2,
        vector_size_c=2,
        async_dma=True,
    )
    chunks = _async_chunks(cfg, ("fp16", "fp16", "fp16"), {})
    assert chunks["async_chunk_a"] > 2 and chunks["async_chunk_b"] > 2
    ident = _cache_identity(async_dma=True, **chunks)

    wide = ConvProblem(N=1, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    narrow = dc_replace(wide, C=2, K=2)
    assert cache.supports_problem(ident, wide)[0]
    ok, reason = cache.supports_problem(ident, narrow)
    assert not ok and "chunk" in reason
    # An async identity without a recorded width is not trusted.
    assert not cache.supports_problem(_cache_identity(async_dma=True), wide)[0]


@pytest.mark.parametrize(
    "directions", [("fwd",), ("wgrad",), ("dgrad",), ("fwd", "wgrad", "dgrad")]
)
def test_count_jobs_matches_the_generators(monkeypatch, directions):
    """``count_jobs`` is the enumeration progress denominator.

    It multiplies the inner axes out instead of walking the generators, so a
    new axis added to a generator and not to the count would make the
    percentage wrong. Shrink the grid so the generators can be walked here.
    """
    from rocke.core.arch import ArchTarget

    ks = _shrink_sweep_grid(monkeypatch)
    target = ArchTarget.from_gfx(_ARCH)
    family = "wmma" if target.wave_size == 32 else "mma"
    split_ks = (1, 4)
    gens = {
        "fwd": lambda: ks._fwd_jobs(_ARCH, "fp16", target.wave_size, family, target),
        "wgrad": lambda: ks._wgrad_jobs(
            _ARCH, "fp16", target.wave_size, family, target, split_ks
        ),
        "dgrad": lambda: ks._dgrad_jobs(
            _ARCH, "fp16", target.wave_size, family, target, 64
        ),
    }
    walked = sum(sum(1 for _ in gens[d]()) for d in directions)
    counted = ks.count_jobs(
        dtype="fp16", target=target, directions=directions, split_ks=split_ks
    )
    assert walked > 0
    assert counted == walked


def _shrink_sweep_grid(monkeypatch):
    from benchmarks.common import kernel_sweep as ks

    monkeypatch.setattr(ks, "_TILE_MN", (32, 64))
    monkeypatch.setattr(ks, "_TILE_K", (32,))
    monkeypatch.setattr(ks, "_WARP_MN", (1, 2))
    monkeypatch.setattr(ks, "_VECS", (2, 8))
    # basic/wavelet are in so the AOT alias filter is exercised.
    monkeypatch.setattr(ks, "_PIPELINES", ("mem", "compv3", "basic", "wavelet"))
    return ks


@pytest.mark.parametrize("jobs", [1, 3])
def test_enumerate_jobs_matches_a_serial_round_robin(monkeypatch, jobs):
    """The chunked/parallel enumeration must return exactly the serial result.

    Order matters, not just membership: ``--limit`` truncates this list, and
    the round-robin across directions is what keeps a truncated cache from
    holding forward kernels only. The reference below is the original
    single-stream walk.
    """
    import multiprocessing

    from rocke.core.arch import ArchTarget

    if jobs > 1 and multiprocessing.get_start_method() != "fork":
        pytest.skip("workers only inherit the shrunken grid under fork")
    ks = _shrink_sweep_grid(monkeypatch)
    target = ArchTarget.from_gfx(_ARCH)
    family = "wmma" if target.wave_size == 32 else "mma"
    directions = ("fwd", "wgrad", "dgrad")
    split_ks = (1, 4)

    streams = [
        iter(ks._fwd_jobs(_ARCH, "fp16", target.wave_size, family, target)),
        iter(ks._wgrad_jobs(_ARCH, "fp16", target.wave_size, family, target, split_ks)),
        iter(ks._dgrad_jobs(_ARCH, "fp16", target.wave_size, family, target, 64)),
    ]
    seen = {}
    while streams:
        still_running = []
        for stream in streams:
            for job in stream:
                key = job.identity.stable_hash()
                if key in seen or not ks._spec_is_valid(job, _ARCH, "fp16"):
                    continue
                seen[key] = job
                still_running.append(stream)
                break
        streams = still_running
    expected = list(seen)

    got = ks.enumerate_jobs(
        arch=_ARCH,
        dtype="fp16",
        target=target,
        directions=directions,
        split_ks=split_ks,
        jobs=jobs,
    )
    assert expected
    assert [j.identity.stable_hash() for j in got] == expected


def test_cache_rejects_packed_atomic_wgrad_on_odd_dw_row(tmp_path):
    """A split-K wgrad binary with a 16-bit dW uses the packed atomic.

    That epilogue needs an even dW row (``wg_N = Y*X*cpg``); on an odd one it
    misaddresses without an error, so the cache must not offer the binary.
    """
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    ident = KernelIdentity(
        arch=_ARCH,
        direction="wgrad",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=1,
        vector_size_b=1,
        vector_size_c=2,  # the packed atomic needs a partner element
        split_k=2,
    )
    cache = KernelCache(tmp_path, _ARCH)
    shape = dict(N=2, Hi=16, Wi=16, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1)
    shape.update(dH=1, dW=1)
    odd = ConvProblem(C=3, **shape)  # wg_N = 3*3*3 = 27
    even = ConvProblem(C=64, **shape)
    assert not cache.supports_problem(ident, odd)[0]
    assert cache.supports_problem(ident, even)[0]


def test_aot_grid_skips_alias_pipelines(monkeypatch):
    """basic is mem everywhere and wavelet is mem in wgrad: never cached twice."""
    from rocke.core.arch import ArchTarget

    ks = _shrink_sweep_grid(monkeypatch)
    target = ArchTarget.from_gfx(_ARCH)
    jobs = ks.enumerate_jobs(
        arch=_ARCH,
        dtype="fp16",
        target=target,
        directions=("fwd", "wgrad", "dgrad"),
        validate=False,
    )
    pipes = {(j.direction, j.identity.pipeline) for j in jobs}
    assert not any(p == "basic" for _, p in pipes)
    assert ("wgrad", "wavelet") not in pipes
    assert ("wgrad", "mem") in pipes


def test_wgrad_rejects_wavelet():
    """The wgrad builder has no wavelet kernel; the predicate must say so."""
    from kernels.common.conv_implicit_gemm_wgrad import (
        WgradConvSpec,
        is_valid_wgrad_spec,
    )

    problem = ConvProblem(N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    spec = WgradConvSpec(problem=problem, data=_DATA, pipeline="wavelet", **_TILE)
    ok, why = is_valid_wgrad_spec(spec, arch=_ARCH)
    assert not ok and "wavelet" in why


def test_compile_jobs_is_incremental(monkeypatch, tmp_path):
    """compile_jobs compiles each distinct binary once and only what changed.

    Real kernels and a real COMGR compile (no GPU): a rebuild with nothing
    changed does nothing, an emitter change that leaves the code alone only
    refreshes entries, and a change to one kernel's code recompiles that one.
    """
    import re

    from rocke.core.arch import ArchTarget
    from benchmarks.common.kernel_cache import KernelCache

    ks = _shrink_sweep_grid(monkeypatch)
    target = ArchTarget.from_gfx(_ARCH)
    family = "wmma" if target.wave_size == 32 else "mma"
    pool = [
        j
        for j in ks._fwd_jobs(_ARCH, "fp16", target.wave_size, family, target)
        if not j.identity.async_dma and ks._spec_is_valid(j, _ARCH, "fp16")
    ]
    jobs, spare = pool[:3], pool[-1]
    cache = KernelCache(tmp_path, _ARCH)
    swap = {}

    def build(job, arch, dtype):
        # A swapped job stands in for "the emitter now emits other code for
        # this identity".
        return ks.build_kernel(swap.get(job.identity, job), arch, dtype)

    def run():
        lines = []
        ks.compile_jobs(
            cache=cache,
            all_jobs=jobs,
            build=build,
            arch=_ARCH,
            dtype="fp16",
            directions=("fwd",),
            log=lines.append,
        )
        done = lines[-1]
        if "nothing to do" in done:
            return 0, 0
        m = re.search(r"(\d+) compiled, (\d+) entries updated", done)
        return int(m.group(1)), int(m.group(2))

    compiled, updated = run()
    assert updated == 3 and 1 <= compiled <= 3
    assert all(cache.has(j.identity) for j in jobs)

    assert run() == (0, 0)  # nothing changed

    monkeypatch.setattr(ks, "current_emitter_digest", lambda: "new-sources")
    assert run() == (0, 3)  # sources changed, code did not

    before = cache.meta(jobs[0].identity)["blob"]
    swap[jobs[0].identity] = spare
    monkeypatch.setattr(ks, "current_emitter_digest", lambda: "newer-sources")
    assert run() == (1, 3)  # one kernel's code changed
    assert cache.meta(jobs[0].identity)["blob"] != before


# Set by test_compile_jobs_survives_a_dead_worker before the pool forks.
_CRASH_IDENTITY: list = []
_REAL_COMPILE_WORKER: list = []


def _crashing_compile_worker(payload):
    if payload[1].identity.stable_hash() in _CRASH_IDENTITY:
        import os

        os._exit(1)  # what an OOM kill or a segfault in COMGR looks like
    return _REAL_COMPILE_WORKER[0](payload)


def test_compile_jobs_survives_a_dead_worker(monkeypatch, tmp_path):
    """A worker dying mid-compile fails that binary, not the whole run."""
    import multiprocessing
    import re

    from rocke.core.arch import ArchTarget
    from benchmarks.common.kernel_cache import KernelCache

    if multiprocessing.get_start_method() != "fork":
        pytest.skip("the crashing worker is injected by monkeypatch + fork")
    ks = _shrink_sweep_grid(monkeypatch)
    target = ArchTarget.from_gfx(_ARCH)
    family = "wmma" if target.wave_size == 32 else "mma"
    jobs = [
        j
        for j in ks._fwd_jobs(_ARCH, "fp16", target.wave_size, family, target)
        if not j.identity.async_dma and ks._spec_is_valid(j, _ARCH, "fp16")
    ][:3]
    _REAL_COMPILE_WORKER[:] = [ks._compile_worker]
    _CRASH_IDENTITY[:] = [jobs[0].identity.stable_hash()]
    monkeypatch.setattr(ks, "_compile_worker", _crashing_compile_worker)
    cache = KernelCache(tmp_path, _ARCH)
    lines = []
    try:
        ks.compile_jobs(
            cache=cache,
            all_jobs=jobs,
            build=ks.build_kernel,
            arch=_ARCH,
            dtype="fp16",
            directions=("fwd",),
            jobs=2,
            log=lines.append,
        )
    finally:
        _CRASH_IDENTITY.clear()
        _REAL_COMPILE_WORKER.clear()
    log = "\n".join(lines)
    assert "a worker process died" in log
    assert "[fail]" in log and "worker process died" in log
    assert not cache.has(jobs[0].identity)
    assert re.search(r"[1-9]\d* failed", lines[-1])


def _fake_compile_worker(payload):
    if payload[-1] == "boom":
        import os

        os._exit(1)
    return payload[-1], b"hsaco", "k", None


def test_isolated_rerun_fits_a_low_open_file_limit(monkeypatch):
    """Hundreds of suspects rerun within a 256-fd limit; only the culprit fails."""
    import multiprocessing

    resource = pytest.importorskip("resource")
    if multiprocessing.get_start_method() != "fork":
        pytest.skip("the fake worker is injected by monkeypatch + fork")
    from benchmarks.common import kernel_sweep as ks

    monkeypatch.setattr(ks, "_compile_worker", _fake_compile_worker)
    keys = [f"k{i}" for i in range(300)]
    keys[137] = "boom"
    suspects = [("compile", (None, None, None, None, k)) for k in keys]
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, hard), hard))
    try:
        results = {p[-1]: r for _, p, r in ks._run_isolated(suspects, jobs=128)}
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert len(results) == 300
    assert results["boom"][3] == ks._WORKER_DIED
    assert all(r[3] is None for k, r in results.items() if k != "boom")


@pytest.mark.parametrize("two_stage", [False, True])
def test_wgrad_split_degree_is_not_compiled_in(two_stage):
    """Every split_k > 1 builds the same wgrad kernel, name included.

    The degree is a launch parameter (ks / ks_count kernargs), so a degree in
    the IR or the symbol would split one binary into one per degree.
    """
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import (
        WgradConvSpec,
        build_implicit_gemm_conv_wgrad,
    )
    from rocke.helpers.compile import lower_kernel_for_comgr

    problem = ConvProblem(
        N=2, Hi=16, Wi=16, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
    )

    def ir(split_k):
        spec = WgradConvSpec(
            problem=problem,
            data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="default" if two_stage else "cshuffle",
            wave_size=64,
            split_k=split_k,
            two_stage=two_stage,
        )
        kernel = build_implicit_gemm_conv_wgrad(spec, arch=_ARCH)
        return lower_kernel_for_comgr(kernel, arch=_ARCH, backend="python").llvm_text

    base = ir(2)
    assert "_spk" in base
    for degree in (4, 8, 64):
        assert ir(degree) == base, f"split_k={degree} emitted a different kernel"


@pytest.mark.parametrize(
    "direction, ok_vecs, bad_vecs",
    [
        # fwd: A=X (cpg), B=W (cpg), D=Y (kpg)
        ("fwd", (8, 8, 4), (4, 4, 8)),
        # wgrad: A=dY (kpg), B=X (cpg), D=dW (cpg)
        ("wgrad", (4, 8, 8), (8, 8, 8)),
        # dgrad: A=dY (kpg), B=W (cpg), D=dX (cpg)
        ("dgrad", (4, 8, 8), (8, 8, 8)),
    ],
)
def test_cache_vector_extents_follow_direction(tmp_path, direction, ok_vecs, bad_vecs):
    """Each operand's width is checked against its own contiguous extent.

    cpg=64, kpg=4: a backward kernel loading dY 8 wide needs kpg % 8 == 0,
    which the fwd layout (A along cpg) would have waved through.
    """
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    cache = KernelCache(tmp_path, _ARCH)
    problem = ConvProblem(
        N=2, Hi=16, Wi=16, C=64, K=4, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
    )

    def ident(vecs):
        a, b, c = vecs
        return KernelIdentity(
            arch=_ARCH,
            direction=direction,
            algorithm="implicit_gemm",
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_d="fp16",
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            vector_size_a=a,
            vector_size_b=b,
            vector_size_c=c,
        )

    ok, why = cache.supports_problem(ident(ok_vecs), problem)
    assert ok, why
    ok, why = cache.supports_problem(ident(bad_vecs), problem)
    assert not ok and "divisible" in why


def test_aot_grid_capability_axes(monkeypatch):
    """wgrad group-merged (depthwise, two-stage) kernels are in the AOT grid,
    built from probes with the matching capabilities; pointwise is not."""
    from rocke.core.arch import ArchTarget

    ks = _shrink_sweep_grid(monkeypatch)
    target = ArchTarget.from_gfx(_ARCH)
    jobs = ks.enumerate_jobs(
        arch=_ARCH, dtype="fp16", target=target, directions=("fwd", "wgrad", "dgrad")
    )
    assert not any(j.identity.is_pointwise for j in jobs)
    # Only grouped binaries: they serve groups == 1 as well.
    assert all(j.identity.grouped for j in jobs)
    merged = [j for j in jobs if j.identity.group_merge > 1]
    assert merged and all(j.direction == "wgrad" for j in merged)
    for j in merged:
        assert j.identity.grouped and j.identity.two_stage
        p = ks._probe_problem("wgrad", j.caps)
        assert p.cpg == p.kpg == 1 and p.groups % j.identity.group_merge == 0


def test_cache_group_merge_needs_depthwise_that_fits(tmp_path):
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    ident = KernelIdentity(
        arch=_ARCH,
        direction="wgrad",
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_m=32,
        tile_n=64,
        tile_k=32,
        warp_m=1,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="default",
        wave_size=64,
        vector_size_a=0,
        vector_size_b=0,
        vector_size_c=0,
        split_k=2,
        two_stage=True,
        grouped=True,
        group_merge=4,
    )
    cache = KernelCache(tmp_path, _ARCH)
    shape = dict(N=2, Hi=16, Wi=16, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1)
    dw3 = ConvProblem(C=32, K=32, Y=3, X=3, groups=32, **shape)  # 9*4 <= 64
    assert cache.supports_problem(ident, dw3)[0]
    dw5 = ConvProblem(C=32, K=32, Y=5, X=5, groups=32, **shape)  # 25*4 > 64
    assert not cache.supports_problem(ident, dw5)[0]
    odd = ConvProblem(C=30, K=30, Y=3, X=3, groups=30, **shape)  # 4 does not divide 30
    assert not cache.supports_problem(ident, odd)[0]
    wide = ConvProblem(C=64, K=64, Y=3, X=3, groups=32, **shape)  # cpg = 2
    assert not cache.supports_problem(ident, wide)[0]


def test_describe_jobs_breaks_the_grid_down(monkeypatch):
    from rocke.core.arch import ArchTarget

    ks = _shrink_sweep_grid(monkeypatch)
    jobs = ks.enumerate_jobs(
        arch=_ARCH,
        dtype="fp16",
        target=ArchTarget.from_gfx(_ARCH),
        directions=("fwd", "wgrad", "dgrad"),
    )
    lines = []
    ks.describe_jobs(jobs, log=lines.append)
    text = "\n".join(lines)
    for direction in ("fwd", "wgrad", "dgrad"):
        n = sum(j.direction == direction for j in jobs)
        assert f"  {direction}: {n} kernels" in lines
    assert "k-loop:" in text and "split-K:" in text and "stride x dilation:" in text


def test_aot_grid_caps_accumulator_registers(monkeypatch):
    """No cached kernel holds more f32 accumulators per lane than the cap."""
    from rocke.core.arch import ArchTarget

    ks = _shrink_sweep_grid(monkeypatch)
    monkeypatch.setattr(ks, "_TILE_MN", (64, 128, 256))
    target = ArchTarget.from_gfx(_ARCH)
    jobs = ks.enumerate_jobs(
        arch=_ARCH, dtype="fp16", target=target, directions=("fwd", "wgrad", "dgrad")
    )

    def acc(i):
        return (i.tile_m // i.warp_m) * (i.tile_n // i.warp_n) // target.wave_size

    assert jobs
    assert max(acc(j.identity) for j in jobs) <= ks.CACHE_MAX_ACC_REGS
    # The cap is what removed them: the shrunk grid does reach past it.
    monkeypatch.setattr(ks, "_MAX_ACC_REGS", 10**6)
    uncapped = ks.enumerate_jobs(
        arch=_ARCH, dtype="fp16", target=target, directions=("fwd",)
    )
    assert max(acc(j.identity) for j in uncapped) > ks.CACHE_MAX_ACC_REGS


def test_cache_applies_runtime_launch_limits(tmp_path):
    """Rules that depend on the launch-time problem, not on the binary:
    the per-axis grid cap and implicit dgrad's missing depthwise path."""
    from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

    cache = KernelCache(tmp_path, _ARCH)
    common = dict(
        arch=_ARCH,
        algorithm="implicit_gemm",
        dtype_a="fp16",
        dtype_b="fp16",
        dtype_d="fp16",
        tile_k=32,
        warp_m=1,
        warp_n=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        pipeline="mem",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=1,
        vector_size_b=1,
        vector_size_c=1,
        grouped=True,
    )
    shape = dict(C=128, K=128, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1)
    small = ConvProblem(N=2, Hi=16, Wi=16, **shape)
    huge = ConvProblem(N=128, Hi=120, Wi=160, **shape)  # M / 16 > 65535
    fwd16 = KernelIdentity(direction="fwd", tile_m=16, tile_n=64, **common)
    fwd256 = KernelIdentity(direction="fwd", tile_m=256, tile_n=64, **common)
    assert cache.supports_problem(fwd16, small)[0]
    ok, why = cache.supports_problem(fwd16, huge)
    assert not ok and "65535" in why
    assert cache.supports_problem(fwd256, huge)[0]

    dgrad = KernelIdentity(
        direction="dgrad",
        tile_m=64,
        tile_n=64,
        stride_h=1,
        stride_w=1,
        dilation_h=1,
        dilation_w=1,
        max_sub_gemms=64,
        **common,
    )
    grouped = ConvProblem(N=2, Hi=16, Wi=16, **dict(shape, C=64, K=64), groups=4)
    depthwise = ConvProblem(N=2, Hi=16, Wi=16, **dict(shape, C=64, K=64), groups=64)
    assert cache.supports_problem(dgrad, grouped)[0]
    ok, why = cache.supports_problem(dgrad, depthwise)
    assert not ok and "depthwise" in why
