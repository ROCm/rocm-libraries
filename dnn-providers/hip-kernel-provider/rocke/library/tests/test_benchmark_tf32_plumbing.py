# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TF32 through the conv benchmark sweep.

The benchmark/cache layer spells TF32 as the dtype triple
``("tf32", "tf32", "fp32")``; kernels only ever see fp32 storage plus
``mma_dtype="tf32"``, built by the one translation helper
``kernel_sweep.conv_data_spec``. These tests pin that split, the MIOpenDriver
parsing (``-M``, 3-D, layouts), the verify tolerance table, and the ranking rule
that keeps verify failures out of ``Best:``.

CPU only: nothing is compiled or launched.
"""

from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import pytest
from rocke.core.arch import ArchTarget

from benchmarks.common import benchmark_implicit_gemm_conv as big
from benchmarks.common import kernel_sweep as ks
from kernels.conv import aot_jobs
from benchmarks.common.kernel_cache import KernelIdentity

_ARCH = "gfx950"
_FLAVOR = "llvm22"


# ---------------------------------------------------------------------------
# The triple and its one translation
# ---------------------------------------------------------------------------


def test_dtype_triple_and_storage():
    assert ks.dtype_triple("tf32") == ("tf32", "tf32", "fp32") == ks.TF32_TRIPLE
    for dt in ("fp16", "bf16", "fp32"):
        assert ks.dtype_triple(dt) == (dt, dt, dt)
        assert ks.storage_dtype(dt) == dt
    assert ks.storage_dtype("tf32") == "fp32"
    # The private spelling the enumerators use stays the same function.
    assert ks._dtype_triple is ks.dtype_triple


def test_conv_data_spec_translation():
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    tf = ks.conv_data_spec("tf32")
    assert (tf.dtype_a, tf.dtype_b, tf.dtype_d, tf.mma_dtype) == (
        "fp32",
        "fp32",
        "fp32",
        "tf32",
    )
    assert tf == ks.conv_data_spec(*ks.TF32_TRIPLE)
    assert tf.mma_dtype_reason() is None
    # Every existing dtype passes through untouched: same spec as before.
    for dt in ("fp16", "bf16", "fp32"):
        assert ks.conv_data_spec(dt) == ConvDataSpec(dtype_a=dt, dtype_b=dt, dtype_d=dt)
        assert ks.conv_data_spec(dt).mma_dtype is None
    with pytest.raises(ValueError):
        ks.conv_data_spec("tf32", "fp32", "fp32")


def test_tf32_signature_uses_storage_pointers():
    from kernels.common.conv_abi import conv_args_signature

    for direction in ("fwd", "wgrad", "dgrad"):
        sig = conv_args_signature(ks.storage_dtype("tf32"), direction=direction)
        ptrs = {a["name"]: a["type"] for a in sig if a["type"].startswith("ptr")}
        tensors = {k: t for k, t in ptrs.items() if not t.startswith("ptr<i32")}
        assert len(tensors) == 3, ptrs
        assert all(t.startswith("ptr<f32") for t in tensors.values()), ptrs


# ---------------------------------------------------------------------------
# Identity: hash, label, job_for_identity, kernel name
# ---------------------------------------------------------------------------

_BASE = dict(
    arch="gfx950",
    direction="fwd",
    algorithm="implicit_gemm",
    tile_m=128,
    tile_n=128,
    tile_k=32,
    warp_m=2,
    warp_n=2,
    warp_tile_m=32,
    warp_tile_n=32,
    warp_tile_k=16,
    pipeline="compv3",
    epilogue="cshuffle",
    wave_size=64,
    vector_size_a=8,
    vector_size_b=8,
    vector_size_c=8,
    llvm_flavor="x",
)

# stable_hash() of existing identities, recorded before TF32 was added: a
# change here orphans every cache entry on disk.
_PINNED_HASHES = {
    ("fp16", "fwd"): "2132d668aeede8a4dfb1ea56e94b969b7a3b4201",
    ("bf16", "wgrad"): "afdd0d212eedd2dbe1633df6d6df490736d81b63",
    ("fp32", "fwd"): "17c6a05ac13fc3644647ee002fbdd5f5b564a53a",
    ("fp32", "dgrad"): "dab4728d2f9df1b0d2d3017626146417bdf2b374",
}


def _ident(dtype, direction="fwd", **kw):
    da, db, dd = ks.dtype_triple(dtype)
    extra = {"wgrad": dict(split_k=4), "dgrad": dict(max_sub_gemms=4)}.get(
        direction, {}
    )
    return KernelIdentity(
        **dict(
            _BASE,
            direction=direction,
            dtype_a=da,
            dtype_b=db,
            dtype_d=dd,
            **extra,
            **kw,
        )
    )


@pytest.mark.parametrize("key", sorted(_PINNED_HASHES))
def test_existing_identity_hash_unchanged(key):
    dtype, direction = key
    assert _ident(dtype, direction).stable_hash() == _PINNED_HASHES[key]


def test_tf32_identity_is_distinct_with_no_new_field():
    tf, fp = _ident("tf32"), _ident("fp32")
    assert tf.stable_hash() != fp.stable_hash()
    assert tf.label() != fp.label()
    assert "tf32.tf32.fp32" in tf.label()
    # No field beyond the dtype triple separates them.
    diff = {
        f.name
        for f in dataclasses.fields(KernelIdentity)
        if getattr(tf, f.name) != getattr(fp, f.name)
    }
    assert diff == {"dtype_a", "dtype_b"}


@pytest.mark.parametrize("direction", ["fwd", "wgrad", "dgrad"])
def test_tf32_label_round_trips(direction):
    ident = _ident("tf32", direction)
    assert KernelIdentity.from_label(ident.label()) == ident


def _tf32_jobs(direction, n=4):
    target = ArchTarget.from_gfx(_ARCH)
    geos = ks._geometry_list(target, "tf32", direction)
    return list(
        itertools.islice(
            ks._direction_jobs(
                direction,
                _ARCH,
                "tf32",
                target,
                (1, 2),
                64,
                geometries=geos[:: max(1, len(geos) // n)],
                llvm_flavor=_FLAVOR,
            ),
            0,
            None,
            11,
        )
    )


@pytest.mark.parametrize("direction", ["fwd", "wgrad", "dgrad"])
def test_tf32_grid_uses_tf32_atoms_and_round_trips(direction):
    jobs = _tf32_jobs(direction)
    assert jobs
    target = ArchTarget.from_gfx(_ARCH)
    for job in jobs:
        ident = job.identity
        assert (ident.dtype_a, ident.dtype_b, ident.dtype_d) == ks.TF32_TRIPLE
        atom = target.mma.select_largest_k(
            a_dtype="tf32",
            b_dtype="tf32",
            c_dtype="fp32",
            m=ident.warp_tile_m,
            n=ident.warp_tile_n,
        )
        assert atom is not None and atom.k == ident.warp_tile_k
        assert ks.job_for_identity(ident) == job
        assert KernelIdentity.from_label(ident.label()) == ident


def test_job_for_identity_rejects_mixed_dtypes():
    ident = _tf32_jobs("fwd")[0].identity
    for bad in (
        dict(dtype_d="fp16"),
        dict(dtype_d="tf32"),
        dict(dtype_b="fp32"),
    ):
        with pytest.raises(LookupError):
            ks.job_for_identity(dataclasses.replace(ident, **bad))


def test_tf32_wgrad_layout_helpers_see_storage_dtype():
    """The K-outer predicate is a 16-bit path: storage fp32 must turn it off,
    exactly as for an fp32 grid (passing "tf32" would read as 16-bit)."""
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    for wt in (16, 32):
        kw = dict(arch=_ARCH, warp_tile_m=wt, warp_tile_n=wt, wave_size=64)
        assert WgradConvSpec.default_lds_k_outer(
            dtype_a=ks.storage_dtype("tf32"), dtype_b=ks.storage_dtype("tf32"), **kw
        ) == WgradConvSpec.default_lds_k_outer(dtype_a="fp32", dtype_b="fp32", **kw)
    for job in _tf32_jobs("wgrad"):
        assert not job.spec_kwargs.get("lds_k_outer", False)


def test_tf32_streamk_load_width_is_fp32():
    builds = aot_jobs._wgrad_streamk_builds(
        "default", 128, "tf32", ("dp_sk",), ("atomic",)
    )
    fp32 = aot_jobs._wgrad_streamk_builds(
        "default", 128, "fp32", ("dp_sk",), ("atomic",)
    )
    assert builds == fp32


def test_tf32_kernel_name_carries_flag():
    from kernels.common.conv_implicit_gemm import ConvProblem, ImplicitGemmConvSpec

    p = ConvProblem(
        N=1, Hi=8, Wi=8, C=32, K=32, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
    )
    kw = dict(
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
    )
    fp32 = ImplicitGemmConvSpec(
        problem=p, data=ks.conv_data_spec("fp32"), warp_tile_k=4, **kw
    )
    tf32 = ImplicitGemmConvSpec(
        problem=p, data=ks.conv_data_spec("tf32"), warp_tile_k=32, **kw
    )
    assert tf32.kernel_name() != fp32.kernel_name()
    assert tf32.kernel_name().endswith("tf32")
    assert "tf32" not in fp32.kernel_name()


# ---------------------------------------------------------------------------
# MIOpenDriver command parsing
# ---------------------------------------------------------------------------

_CMD2D = "./MIOpenDriver {kw} -n 2 -c 32 -H 16 -W 16 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -F 1 -t 1 -i 10 {extra}"


@pytest.mark.parametrize(
    "kw,extra,want",
    [
        ("conv", "-M 0 -I NHWC", "tf32"),
        ("conv", "--math_type 0 --in_layout NHWC", "tf32"),
        ("convfp32", "-M 0 -I NHWC", "tf32"),
        ("conv", "-I NHWC", "fp32"),
        ("conv", "-M 1 -in_layout=NHWC", "fp32"),
        ("convfp16", "-M 0 -I NHWC", "fp16"),
        ("convbfp16", "-M 0 -I NHWC -m conv", "bf16"),
    ],
)
def test_parse_miopen_cmd_math_type(kw, extra, want):
    problem, dtype, forw = big.parse_miopen_cmd(_CMD2D.format(kw=kw, extra=extra))
    assert dtype == want
    assert forw == 1
    assert not problem.is_3d
    assert (problem.N, problem.C, problem.K, problem.Hi, problem.Y) == (
        2,
        32,
        64,
        16,
        3,
    )


def test_parse_miopen_cmd_3d():
    cmd = (
        "MIOpenDriver conv -n 2 -c 32 -H 16 -W 12 -k 64 -y 3 -x 3 -p 1 -q 1 "
        "-! 8 -@ 5 -# 2 -$ 2 -^ 1 -_ 3 --in_layout NDHWC -F 4 -M 0"
    )
    problem, dtype, forw = big.parse_miopen_cmd(cmd)
    assert dtype == "tf32" and forw == 4
    assert problem.is_3d
    assert (problem.Di, problem.Z, problem.sD, problem.pD, problem.dD) == (
        8,
        5,
        2,
        2,
        1,
    )
    assert (problem.Hi, problem.Wi) == (16, 12)


def test_parse_miopen_cmd_3d_long_flags_and_defaults():
    cmd = (
        "MIOpenDriver conv --batchsize 1 --in_channels 16 --in_h 8 --in_w 8 "
        "--out_channels 16 --fil_h 3 --fil_w 3 --spatial_dim 3 -I NDHWC"
    )
    problem, _, _ = big.parse_miopen_cmd(cmd)
    # MIOpenDriver's own 3-D defaults: depth 32, filter depth 3.
    assert (problem.Di, problem.Z, problem.sD, problem.pD, problem.dD) == (
        32,
        3,
        1,
        0,
        1,
    )


def test_parse_miopen_cmd_layouts(capsys):
    with pytest.raises(ValueError, match="no -I"):
        big.parse_miopen_cmd(_CMD2D.format(kw="conv", extra=""), require_layout=True)
    # Without require_layout: a warning, and the shape read as NHWC.
    big.parse_miopen_cmd(_CMD2D.format(kw="conv", extra=""))
    assert "NCHW" in capsys.readouterr().err
    with pytest.raises(ValueError):
        big.parse_miopen_cmd(_CMD2D.format(kw="conv", extra="-I NCHW"))
    with pytest.raises(ValueError):
        big.parse_miopen_cmd(_CMD2D.format(kw="conv", extra="-I NHWC -f NCHW"))
    with pytest.raises(ValueError):  # 3-D needs NDHWC
        big.parse_miopen_cmd(_CMD2D.format(kw="conv", extra="-_ 3 -I NHWC"))
    with pytest.raises(ValueError):
        big.parse_miopen_cmd(_CMD2D.format(kw="conv", extra="-_ 1 -I NHWC"))


def test_parse_miopen_cmd_iterations_are_not_a_layout():
    # "-i" must not be read as an abbreviation of "-in_layout".
    _, dtype, _ = big.parse_miopen_cmd(_CMD2D.format(kw="conv", extra="-i 5 -I NHWC"))
    assert dtype == "fp32"


def test_parse_json_case_tf32():
    base = dict(op="conv2d", layout="NHWC", N=1, Cin=8, Cout=8, H=8, W=8, Kh=3, Kw=3)
    assert big.parse_json_case(dict(base, dtype="tf32"))[1] == "tf32"
    assert big.parse_json_case(dict(base, dtype="fp32", math="tf32"))[1] == "tf32"
    assert big.parse_json_case(dict(base, dtype="fp32"))[1] == "fp32"
    with pytest.raises(ValueError):
        big.parse_json_case(dict(base, dtype="fp8"))


def test_json_file_cases_carry_a_direction(tmp_path, monkeypatch):
    """--json-file without the cache used to append 2-tuples the sweep loop
    could not unpack."""
    import json

    path = tmp_path / "cases.json"
    entry = dict(op="conv2d", layout="NHWC", N=1, Cin=8, Cout=8, H=8, W=8, Kh=3, Kw=3)
    path.write_text(json.dumps([dict(entry, dtype="tf32")]))
    seen = []

    def fake_sweep(**kw):
        seen.append((kw["problem"].short(), kw["dtype"]))
        return 0, []

    monkeypatch.setattr(big, "_run_wgrad_sweep", fake_sweep)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_implicit_gemm_conv.py",
            "--json-file",
            str(path),
            "--direction",
            "wgrad",
        ],
    )
    try:
        rc = big.main()
    except Exception as e:  # noqa: BLE001 - runtime imports without a GPU stack
        pytest.skip(f"main() needs the runtime: {type(e).__name__}: {e}")
    assert rc == 0
    assert seen == [("N1H8W8C8_K8Y3X3", "tf32")]


# ---------------------------------------------------------------------------
# Verify tolerance and ranking
# ---------------------------------------------------------------------------


def test_verify_tol_table():
    assert big._verify_tol("fp16", "gfx950") == 5e-2
    assert big._verify_tol("bf16", "gfx942") == 5e-2
    assert big._verify_tol("fp32", "gfx950") == 1e-3
    # gfx950 emulates TF32 (bf16x3), gfx942 encodes XF32 natively.
    assert big._verify_tol("tf32", "gfx950") == 1e-4
    assert big._verify_tol("tf32", "gfx942") == 1e-3
    # The atom decides when it is known.
    assert big._verify_tol("tf32", "gfx942", "mfma_f32_32x32x16_xf32") == 1e-4
    assert big._verify_tol("tf32", "gfx950", "mfma_f32_16x16x8_xf32") == 1e-3


def test_rank_results_never_ranks_a_failed_kernel():
    def mk(name, tflops, passed):
        return big.Result(
            kernel_name=name,
            tile_m=0,
            tile_n=0,
            tile_k=0,
            warp_m=0,
            warp_n=0,
            warp_tile_mn=0,
            warp_tile_k=0,
            pipeline="",
            epilogue="",
            split_k=1,
            ms=1.0,
            tflops=tflops,
            gbps=0.0,
            passed=passed,
        )

    ranked, failed = big._rank_results(
        [mk("slow_ok", 1.0, True), mk("fast_bad", 9.0, False), mk("mid_ok", 5.0, True)]
    )
    assert [r.kernel_name for r in ranked] == ["mid_ok", "slow_ok"]
    assert [r.kernel_name for r in failed] == ["fast_bad"]
    # Verify not run (passed=None) ranks as before.
    ranked, failed = big._rank_results([mk("a", 1.0, None), mk("b", 2.0, None)])
    assert [r.kernel_name for r in ranked] == ["b", "a"] and not failed


def test_conv_reference_pins_ieee_and_restores():
    torch = pytest.importorskip("torch")
    from builders.common.conv_reference import ieee_fp32

    conv = getattr(torch.backends.cudnn, "conv", None)
    if conv is not None and hasattr(conv, "fp32_precision"):
        before = conv.fp32_precision
        with ieee_fp32():
            assert conv.fp32_precision == "ieee"
        assert conv.fp32_precision == before
    else:
        before = torch.backends.cudnn.allow_tf32
        with ieee_fp32():
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 == before


def _problem(is_3d=False):
    from kernels.common.conv_implicit_gemm import ConvProblem

    depth = dict(Di=8, Z=3, sD=2, pD=1, dD=1) if is_3d else {}
    return ConvProblem(
        N=2, Hi=16, Wi=12, C=32, K=64, Y=3, X=3, sH=1, sW=2, pH=1, pW=0, dH=1, dW=1,
        groups=2, **depth,
    )  # fmt: skip


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_every_sweep_result_carries_its_verify_verdict():
    """``_rank_results`` drops only kernels whose ``passed`` is False, so a leg
    that builds its ``Result`` without the verdict lets a kernel that failed
    verify rank -- and win. The legs are GPU loops no CPU test reaches, so pin
    the contract on the source: every ``Result(...)`` call passes ``passed=``.
    """
    import ast

    tree = ast.parse(Path(big.__file__).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Result"
    ]
    assert calls, "no Result(...) construction found"
    missing = [
        node.lineno
        for node in calls
        if not any(kw.arg == "passed" for kw in node.keywords)
    ]
    assert not missing, f"Result(...) without passed= at lines {missing}"
