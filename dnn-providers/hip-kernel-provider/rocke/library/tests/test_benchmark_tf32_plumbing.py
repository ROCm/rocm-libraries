# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TF32 through the conv benchmark sweep and the MIOpen timing harness.

The benchmark/cache layer spells TF32 as the dtype triple
``("tf32", "tf32", "fp32")``; kernels only ever see fp32 storage plus
``mma_dtype="tf32"``, built by the one translation helper
``kernel_sweep.conv_data_spec``. These tests pin that split, the MIOpenDriver
parsing (``-M``, 3-D, layouts), the verify tolerance table, the ranking rule
that keeps verify failures out of ``Best:``, and the MIOpen harness's command
builder and output parser on canned text.

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
from benchmarks.common import benchmark_miopen_conv as bmc
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


def test_miopen_cmd_builder_2d():
    argv = bmc.build_miopen_cmd(_problem(), "fp32", "wgrad", bmc.MATH_TF32, iters=7)
    assert argv[:2] == ["MIOpenDriver", "conv"]
    want = {
        "-n": "2", "-c": "32", "-H": "16", "-W": "12", "-k": "64", "-y": "3",
        "-x": "3", "-p": "1", "-q": "0", "-u": "1", "-v": "2", "-g": "2",
        "-_": "2", "-F": "4", "-M": "0", "-t": "1", "-V": "1", "-i": "7",
        "--in_layout": "NHWC", "--fil_layout": "NHWC", "--out_layout": "NHWC",
    }  # fmt: skip
    for flag, value in want.items():
        assert _flag(argv, flag) == value, flag
    assert "-!" not in argv and "-S" not in argv and "-w" not in argv
    # It parses back to the same problem.
    problem, dtype, forw = big.parse_miopen_cmd(" ".join(argv), require_layout=True)
    assert problem == _problem() and dtype == "tf32" and forw == 4


def test_miopen_cmd_builder_3d_and_16bit():
    argv = bmc.build_miopen_cmd(
        _problem(True), "bf16", "dgrad", bmc.MATH_IEEE, solution="138"
    )
    assert argv[1] == "convbfp16"
    assert [_flag(argv, f) for f in ("-_", "-!", "-@", "-#", "-$", "-^")] == [
        "3", "8", "3", "2", "1", "1",
    ]  # fmt: skip
    assert _flag(argv, "--in_layout") == "NDHWC" and _flag(argv, "-F") == "2"
    assert _flag(argv, "-S") == "138"
    problem, dtype, _ = big.parse_miopen_cmd(" ".join(argv), require_layout=True)
    assert problem == _problem(True) and dtype == "bf16"


def test_miopen_env_unsets_tf32_overrides(tmp_path):
    base = {
        "MIOPEN_TF32_OVERRIDE": "1",
        "NVIDIA_TF32_OVERRIDE": "1",
        "LD_LIBRARY_PATH": "/x",
    }
    env = bmc.miopen_env(tmp_path, str(tmp_path / "db"), base=base)
    assert "MIOPEN_TF32_OVERRIDE" not in env and "NVIDIA_TF32_OVERRIDE" not in env
    assert env["LD_LIBRARY_PATH"] == str(tmp_path / "lib")
    assert env["MIOPEN_FIND_MODE"] == "NORMAL"
    assert env["MIOPEN_USER_DB_PATH"] == str(tmp_path / "db")


def test_miopen_root_needs_flag_or_env(tmp_path, monkeypatch):
    monkeypatch.delenv(bmc.ENV_ROOT, raising=False)
    with pytest.raises(SystemExit):
        bmc.miopen_root(None)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "MIOpenDriver").write_text("")
    monkeypatch.setenv(bmc.ENV_ROOT, str(tmp_path))
    assert bmc.miopen_root(None) == tmp_path.resolve()


def test_preflight_refuses_driver_without_math_type(monkeypatch, tmp_path):
    import subprocess

    def fake_run(text):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=text, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run("      --verify -V  Verify\n"))
    with pytest.raises(SystemExit):
        bmc.preflight(tmp_path, {})
    monkeypatch.setattr(subprocess, "run", fake_run("  --math_type  -M  math type\n"))
    bmc.preflight(tmp_path, {})


# Synthetic driver output for the parser: the times are made-up round values
# chosen to exercise the number formats, not measurements.
_CANNED = """\
MIOpenDriver conv -n 4 -c 64 -H 28 -W 28 -k 64 -y 3 -x 3 -F 0 -M 0 -t 1 -V 1
PRNG seed: 12345678
MIOpen Forward Conv. Algorithm: 1, Solution: 220/ConvHipConv
GPU Kernel Time Forward Conv. Elapsed: 0.250000 ms (average)
stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs
Forward Convolution Verifies OK on GPU reference (1.1e-06 < 0.082)
MIOpen Backward Data Conv. Algorithm: 5, Solution: 108/ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC
GPU Kernel Time Backward Data Conv. Elapsed: 0.5 ms (average)
Backward Convolution Data FAILED: 0.2 > 0.082
MIOpen Backward Weights Conv. Algorithm: 5, Solution: 140/ConvHipImplicitGemm3DGroupWrwXdlops
GPU Kernel Time Backward Weights Conv. Elapsed: 1.5e-01 ms (average)
Backward Convolution Weights Verifies OK on GPU reference (8.3e-08 < 0.01)
"""


def test_parse_driver_output():
    fwd = bmc.parse_driver_output(_CANNED, "fwd")
    assert (fwd.ms, fwd.algorithm, fwd.solution_id, fwd.solver) == (
        0.25,
        1,
        "220",
        "ConvHipConv",
    )
    assert (fwd.verify, fwd.verify_err, fwd.verify_tol) == ("OK", 1.1e-06, 0.082)
    dgrad = bmc.parse_driver_output(_CANNED, "dgrad")
    assert dgrad.solver == "ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC"
    assert (dgrad.ms, dgrad.verify, dgrad.verify_err) == (0.5, "FAILED", 0.2)
    wgrad = bmc.parse_driver_output(_CANNED, "wgrad")
    assert (wgrad.ms, wgrad.solution_id, wgrad.verify) == (0.15, "140", "OK")
    empty = bmc.parse_driver_output("garbage", "fwd")
    assert empty.ms is None and empty.solver is None and empty.verify == ""


@pytest.mark.parametrize(
    "name,tf32",
    [
        # hipconv: DataType 5 is tf32 (fp16, bf16, fp32, fp8, bf8, tf32).
        (
            (
                "void hipconv::cdna4::direct_l1::conv2d_direct_l1_cdna4<hipconv::cdna4::"
                "direct_l1::Config{4, 2, 1}, (hipconv::DataType)5>(float const*, int)"
            ),
            True,
        ),
        (
            (
                "void hipconv::cdna4::direct_l1::conv2d_direct_l1_cdna4<hipconv::cdna4::"
                "direct_l1::Config{4, 2, 1}, (hipconv::DataType)2>(float const*, int)"
            ),
            False,
        ),
        ("_ZN7hipconv5cdna49direct_l1IXadL_ZN7hipconv8DataTypeE5EEEvv", True),
        # CK: tf32_t operand type vs plain float.
        (
            (
                "void ck::tensor_operation::device::(anonymous namespace)::kernel_grouped_conv"
                "_fwd_xdl_cshuffle_v3<ck::GridwiseGemm<float, float, ck::tf32_t, ck::tf32_t>>()"
            ),
            True,
        ),
        ("_ZN2ck16tensor_operation6device6kernelINS_6tf32_tEEEvv", True),
        (
            (
                "void ck::tensor_operation::device::(anonymous namespace)::kernel_grouped_conv"
                "_fwd_xdl_cshuffle_v3<ck::GridwiseGemm<float, float, float, float>>()"
            ),
            False,
        ),
        ("igemm_fwd_gtcx35_nhwc_fp32_bx0_ex1_bt128x64x32_wt32x32x2", False),
        ("Cijk_Alik_Bljk_S_MT64x96x32_MI16x16x4x1_XF32", True),
        # CK's tf32_t is _BitInt(19): demangled and mangled forms.
        (
            "void ck::tensor_operation::device::(anonymous namespace)::kernel_grouped_conv"
            "_fwd_multiple_abd_xdl_cshuffle<ck::GridwiseGemmMultipleD_xdl_cshuffle<float, "
            "float, _BitInt(19), float, float>>()",
            True,
        ),
        ("_ZN2ck6kernelIffDB19_ffEEvv", True),
        # An fp32 kernel whose name merely ends a word in "xF32" is not TF32.
        ("reduce_MaxF32", False),
        ("SoftmaxF32_kernel", False),
    ],
)
def test_is_tf32_kernel(name, tf32):
    assert bmc.is_tf32_kernel(name) is tf32


def test_trace_kernels_without_rocprofv3_reports_a_note(tmp_path, monkeypatch):
    """No rocprofv3 anywhere: a trace note, not an unpacking crash."""
    monkeypatch.setenv("PATH", str(tmp_path))
    kernels, note, text = bmc.trace_kernels(tmp_path, ["true"], {}, 5)
    assert kernels is None and note == "no rocprofv3" and text == ""


def test_short_kernel_name():
    name = (
        "void ck::tensor_operation::device::(anonymous namespace)::kernel_x"
        "<float, float>(float const*)"
    )
    assert bmc.short_kernel_name(name) == "ck::tensor_operation::device::kernel_x"
    assert bmc.short_kernel_name("igemm_fwd_plain") == "igemm_fwd_plain"


def test_read_kernel_trace_drops_utility_kernels(tmp_path):
    trace = tmp_path / "run_kernel_trace.csv"
    trace.write_text(
        '"Kind","Kernel_Name","Start_Timestamp"\n'
        '"K","__amd_rocclr_fillBufferUnAligned","1"\n'
        '"K","kernel_b","2"\n'
        '"K","kernel_a","3"\n'
        '"K","kernel_b","4"\n'
    )
    assert bmc.read_kernel_trace(trace) == ["kernel_b", "kernel_a"]


def test_expand_runs_math_modes():
    p = _problem()
    cases = [
        bmc.Case(p, "fp32", "fwd"),
        bmc.Case(p, "tf32", "dgrad"),
        bmc.Case(p, "bf16", "fwd"),
    ]
    both = bmc.expand_runs(cases, "both")
    assert [(r.dtype, r.storage, r.math_type) for r in both] == [
        ("fp32", "fp32", 1),
        ("tf32", "fp32", 0),
        ("fp32", "fp32", 1),
        ("tf32", "fp32", 0),
        ("bf16", "bf16", 1),
    ]
    only = bmc.expand_runs(cases, "tf32")
    assert [(r.dtype, r.math_type) for r in only] == [
        ("tf32", 0),
        ("tf32", 0),
        ("bf16", 1),
    ]
    as32 = bmc.expand_runs(cases[2:], "tf32", as_fp32=True)
    assert [(r.dtype, r.storage) for r in as32] == [("tf32", "fp32")]


def test_csv_row_joins_with_the_sweep():
    p = _problem()
    run = bmc.Run(bmc.Case(p, "tf32", "fwd"), "fp32", 0, "tf32")
    run.result = bmc.DriverResult(
        ms=2.0, solution_id="220", solver="ConvHipConv", verify="OK"
    )
    run.kernels = ["void hipconv::k<(hipconv::DataType)5>()"]
    row = bmc.csv_row(run, "conv ...")
    assert set(row) == set(bmc._CSV_FIELDS)
    # The sweep's CSV join key.
    assert (row["shape"], row["dtype"], row["direction"]) == (p.short(), "tf32", "fwd")
    assert float(row["miopen_tflops"]) == pytest.approx(p.flops / 2.0 * 1e-9, rel=1e-3)
    assert row["tf32_kernel"] == "yes"
    run.kernels = None
    assert bmc.csv_row(run, "")["tf32_kernel"] == "unknown"


def test_dry_run_needs_no_install(capsys, monkeypatch):
    monkeypatch.delenv(bmc.ENV_ROOT, raising=False)
    rc = bmc.main(
        [
            "--dry-run",
            "--math",
            "both",
            "--miopen-cmd",
            "MIOpenDriver conv -n 1 -c 8 -H 8 -W 8 -k 8 -y 3 -x 3 -F 3 -I NHWC",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("=== ") == 4  # fwd + dgrad, each at -M 1 and -M 0
    assert "-M 0" in out and "-M 1" in out


def test_harness_file_has_no_absolute_install_path():
    text = Path(bmc.__file__).read_text()
    assert "/home/" not in text and "/opt/" not in text


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
