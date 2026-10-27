# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side conv helpers that decide *which* problem runs, and how it is cut.

* ``parse_miopen_cmd`` / ``parse_miopen_cmd_direct`` -- the MIOpenDriver
  keyword is the first non-flag token. A flag value (the ``conv`` of
  ``-m conv``) must never be taken for it: the flags before it would be
  dropped, and the problem silently replaced by defaults.
* ``_batch_chunks`` -- the batch slices one AOT launch each covers, so no
  operand exceeds the kernels' i32 byte addressing.
* ``_batch_slices`` -- the batch slices the torch reference runs, so no fp32
  operand crosses torch's 2^32-byte break point.

CPU only; no kernel is built or launched.
"""

from __future__ import annotations

import pytest

from kernels.common.conv_implicit_gemm import ConvProblem

_INT32_MAX = 2**31 - 1

# fwd-style command with every geometry flag *before* ``-m conv``.
_FLAGS = (
    "-n 8 -c 64 -H 56 -W 28 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 "
    "-l 1 -j 1 -m conv -g {g} -F {f} -t 1 -in_layout NHWC"
)


# ---------------------------------------------------------------------------
# parse_miopen_cmd (implicit GEMM)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, dtype",
    [
        ("conv", "fp32"),
        ("convfp32", "fp32"),
        ("convfp16", "fp16"),
        ("convbfp16", "bf16"),
    ],
)
def test_implicit_parse_keeps_flags_before_mode(kw, dtype):
    from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd

    problem, got_dtype, forw = parse_miopen_cmd(
        f"./bin/MIOpenDriver {kw} " + _FLAGS.format(g=1, f=2)
    )
    assert got_dtype == dtype
    assert forw == 2
    # Taking the "conv" of "-m conv" for the keyword parses only what follows
    # it, and every field below falls back to its default of 1.
    assert (problem.N, problem.C, problem.K) == (8, 64, 64)
    assert (problem.Hi, problem.Wi, problem.Y, problem.X) == (56, 28, 3, 3)
    assert (problem.pH, problem.pW) == (1, 1)


def test_implicit_parse_rejects_mode_value_as_keyword():
    from benchmarks.common.benchmark_implicit_gemm_conv import parse_miopen_cmd

    with pytest.raises(ValueError, match="No MIOpenDriver keyword"):
        parse_miopen_cmd("./bin/MIOpenDriver " + _FLAGS.format(g=1, f=1))


# ---------------------------------------------------------------------------
# parse_miopen_cmd_direct
# ---------------------------------------------------------------------------


def test_direct_parse_keeps_flags_before_mode():
    from benchmarks.common.benchmark_direct_conv import parse_miopen_cmd_direct

    problem, dtype, forw = parse_miopen_cmd_direct(
        "MIOpenDriver convbfp16 " + _FLAGS.format(g=64, f=1)
    )
    assert (dtype, forw) == ("bf16", 1)
    assert (problem.N, problem.H, problem.W) == (8, 56, 28)
    assert (problem.groups, problem.cpg, problem.kpg) == (64, 1, 1)
    assert (problem.KH, problem.KW, problem.PAD, problem.stride) == (3, 3, 1, 1)


@pytest.mark.parametrize("kw", ["conv", "convfp32"])
def test_direct_parse_rejects_fp32_by_its_own_keyword(kw):
    from benchmarks.common.benchmark_direct_conv import parse_miopen_cmd_direct

    # The error names the keyword the command gave, not the "-m conv" value.
    with pytest.raises(ValueError, match=f"fp32 \\('{kw}'\\)"):
        parse_miopen_cmd_direct(f"MIOpenDriver {kw} " + _FLAGS.format(g=64, f=1))


def test_direct_parse_rejects_mode_value_as_keyword():
    from benchmarks.common.benchmark_direct_conv import parse_miopen_cmd_direct

    with pytest.raises(ValueError, match="No MIOpenDriver keyword"):
        parse_miopen_cmd_direct("MIOpenDriver " + _FLAGS.format(g=64, f=1))


@pytest.mark.parametrize(
    "geom, what",
    [("-u 2 -v 1 -p 1 -q 1", "stride"), ("-u 1 -v 1 -p 1 -q 0", "padding")],
)
def test_direct_parse_rejects_per_axis_stride_or_padding(geom, what):
    from benchmarks.common.benchmark_direct_conv import parse_miopen_cmd_direct

    with pytest.raises(ValueError, match=f"one {what} for both axes"):
        parse_miopen_cmd_direct(
            "MIOpenDriver convfp16 -n 2 -c 64 -H 16 -W 16 -k 64 -y 3 -x 3 "
            f"{geom} -g 64 -F 1"
        )


# ---------------------------------------------------------------------------
# _batch_chunks (AOT launches)
# ---------------------------------------------------------------------------


def _chunks(direction, problem, dtype="fp16"):
    from benchmarks.common.benchmark_implicit_gemm_conv import _batch_chunks

    return _batch_chunks(direction, problem, dtype)


def _image(N, C=64, K=64, **kw):
    """1024x1024 1x1 problem: C=K=64 fp16 is exactly 2^27 bytes per image."""
    return ConvProblem(N=N, Hi=1024, Wi=1024, C=C, K=K, Y=1, X=1, **kw)


def _assert_partition(chunks, N, max_images):
    assert chunks[0][0] == 0
    for (n0, n), (n1, _) in zip(chunks, chunks[1:]):
        assert n0 + n == n1
    assert chunks[-1][0] + chunks[-1][1] == N
    assert all(0 < n <= max_images for _, n in chunks)


@pytest.mark.parametrize("direction", ["fwd", "dgrad", "wgrad"])
def test_batch_chunks_single_launch_up_to_the_limit(direction):
    # 15 * 2^27 bytes is the largest whole-image count under 2^31.
    assert _chunks(direction, _image(15)) == [(0, 15)]


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_batch_chunks_splits_one_past_the_limit(direction):
    # 16 images are exactly 2^31 bytes: one byte past the i32 maximum.
    assert _chunks(direction, _image(16)) == [(0, 8), (8, 8)]


def test_batch_chunks_balances_slices():
    chunks = _chunks("fwd", _image(31))
    assert chunks == [(0, 11), (11, 11), (22, 9)]
    _assert_partition(chunks, 31, 15)


def test_batch_chunks_fp32_doubles_the_bytes():
    assert _chunks("fwd", _image(7), "fp32") == [(0, 7)]
    assert _chunks("fwd", _image(8), "fp32") == [(0, 4), (4, 4)]


def test_batch_chunks_sized_by_the_larger_activation():
    # The input is 2^24 bytes per image, the output 2^27: the output decides.
    assert _chunks("dgrad", _image(16, C=8, K=64)) == [(0, 8), (8, 8)]
    assert _chunks("fwd", _image(16, C=64, K=8)) == [(0, 8), (8, 8)]


def test_batch_chunks_output_extent_from_stride():
    # Stride 2 quarters the output; the input (2^27 per image) still decides.
    p = ConvProblem(N=16, Hi=1024, Wi=1024, C=64, K=64, Y=1, X=1, sH=2, sW=2)
    assert _chunks("fwd", p) == [(0, 8), (8, 8)]
    # With a 4x wider K the 512x512 output equals the input again.
    p = ConvProblem(N=16, Hi=1024, Wi=1024, C=16, K=64, Y=1, X=1, sH=2, sW=2)
    assert _chunks("fwd", p) == [(0, 16)]


def test_batch_chunks_3d_counts_depth():
    p = ConvProblem(
        N=16, Hi=256, Wi=256, C=64, K=64, Y=1, X=1, Di=16, Z=1, sD=1, pD=0, dD=1
    )
    assert _chunks("fwd", p) == [(0, 8), (8, 8)]


def test_batch_chunks_wgrad_cannot_split():
    with pytest.raises(ValueError, match="wgrad reduces over"):
        _chunks("wgrad", _image(16))


def test_batch_chunks_rejects_one_oversized_image():
    p = ConvProblem(N=1, Hi=8192, Wi=8192, C=32, K=32, Y=1, X=1)
    with pytest.raises(ValueError, match="one image's activations"):
        _chunks("fwd", p)


def test_batch_chunks_rejects_oversized_weights():
    # K * cpg * 2 bytes == 2^31; grouping it 2 ways halves cpg and fits.
    p = ConvProblem(N=1, Hi=1, Wi=1, C=32768, K=32768, Y=1, X=1)
    with pytest.raises(ValueError, match="weight tensor"):
        _chunks("fwd", p)
    p = ConvProblem(N=1, Hi=1, Wi=1, C=32768, K=32768, Y=1, X=1, groups=2)
    assert _chunks("fwd", p) == [(0, 1)]


# ---------------------------------------------------------------------------
# _batch_slices (torch reference)
# ---------------------------------------------------------------------------


def _slices(p, n):
    pytest.importorskip("torch")
    from builders.common.conv_reference import _batch_slices

    return _batch_slices(p, n)


def test_batch_slices_small_batch_is_one_slice():
    p = ConvProblem(N=4, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1)
    assert _slices(p, 4) == [(0, 4)]


def test_batch_slices_fp32_cap():
    # 2^26 elements -> 2^28 fp32 bytes per image: 7 fit under 2^31.
    p = _image(16)
    assert _slices(p, 7) == [(0, 7)]
    slices = _slices(p, 16)
    assert slices == [(0, 7), (7, 7), (14, 2)]
    _assert_partition(slices, 16, 7)


def test_batch_slices_sized_by_the_larger_activation():
    # Output 2^26 elements, input 2^23: the output decides.
    assert _slices(_image(16, C=8, K=64), 16) == [(0, 7), (7, 7), (14, 2)]
    # Padding grows the output: a 1022x1022 image is just under 2^28 fp32
    # bytes (8 per slice); padded by 1 its output is 1024x1024, exactly 2^28.
    p = ConvProblem(N=8, Hi=1022, Wi=1022, C=64, K=64, Y=1, X=1)
    assert _slices(p, 8) == [(0, 8)]
    p = ConvProblem(N=8, Hi=1022, Wi=1022, C=64, K=64, Y=1, X=1, pH=1, pW=1)
    assert _slices(p, 8) == [(0, 7), (7, 1)]


def test_batch_slices_3d_counts_depth():
    p = ConvProblem(
        N=8, Hi=256, Wi=256, C=64, K=64, Y=1, X=1, Di=16, Z=1, sD=1, pD=0, dD=1
    )
    assert _slices(p, 8) == [(0, 7), (7, 1)]


def test_batch_slices_oversized_image_runs_alone():
    p = ConvProblem(N=3, Hi=8192, Wi=8192, C=32, K=32, Y=1, X=1)
    assert _slices(p, 3) == [(0, 1), (1, 1), (2, 1)]
