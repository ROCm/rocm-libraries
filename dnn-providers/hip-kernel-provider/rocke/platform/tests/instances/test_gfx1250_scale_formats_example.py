# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scale example preserves independent selectors through real lowering."""

import argparse
import pytest
from rocke.examples.gfx1250.gemm.scale_formats_gemm import make_spec
from rocke.instances.gfx1250.block_scaled_gemm import (
    build_block_scaled_gemm,
    is_valid_spec,
)
from rocke.core.arch.target import normalize_dtype
from rocke.core.lower_llvm import lower_kernel_to_llvm


@pytest.mark.parametrize(
    "a,b,sa,sb", [("fp4", "fp4", "e4m3", "e4m3"), ("fp6", "fp4", "e8m0", "e5m3")]
)
@pytest.mark.parametrize("matrix_path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize("output_dtype", ["bf16", "fp16"])
def test_scale_example_contract(a, b, sa, sb, matrix_path, output_dtype):
    spec = make_spec(
        argparse.Namespace(
            m=32,
            n=48,
            k=256,
            matrix_path=matrix_path,
            output_dtype=output_dtype,
            dtype_a=a,
            dtype_b=b,
            scale_dtype_a=sa,
            scale_dtype_b=sb,
        )
    )
    assert spec.resolved_scale_dtypes() == (sa, sb)
    assert spec.dtype_c == output_dtype
    assert is_valid_spec(spec)[0]
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    intrinsic = "scale16" if matrix_path == "wmma_scale16" else "scale"
    assert f"@llvm.amdgcn.wmma.{intrinsic}." in llvm
    storage_type = "half" if output_dtype == "fp16" else "bfloat"
    assert f"store {storage_type} " in llvm


@pytest.mark.parametrize("matrix_path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize("output_dtype", [None, "bf16", "fp16"])
def test_scale_formats_gemm_cli_selects_output(monkeypatch, matrix_path, output_dtype):
    from rocke.examples.gfx1250.gemm import _scaled_gemm_example
    from rocke.examples.gfx1250.gemm import scale_formats_gemm as example

    seen = []

    def run(spec, cases, **kwargs):
        seen.append(spec.dtype_c)
        return len(cases)

    monkeypatch.setattr(_scaled_gemm_example, "run_cases", run)
    argv = ["--matrix-path", matrix_path]
    if output_dtype is not None:
        argv += ["--output-dtype", output_dtype]
    assert example.main(argv) == 0
    assert seen == [output_dtype or "bf16"]


def test_scale_formats_gemm_cli_rejects_unsupported_output(monkeypatch, capsys):
    from rocke.examples.gfx1250.gemm import scale_formats_gemm as example

    monkeypatch.setattr(example, "verify", lambda *args: pytest.fail("must not launch"))
    with pytest.raises(SystemExit) as exc:
        example.main(["--output-dtype", "fp32"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "dtype",
    [
        "fp8",
        "bf8",
        "fp8e4m3",
        "bf8e5m2",
        "fp6",
        "bf6",
        "fp6e2m3",
        "fp6e3m2",
        "fp4",
        "fp4e2m1",
    ],
)
@pytest.mark.parametrize("operand", ["a", "b"])
def test_scale_formats_gemm_cli_normalizes_matrix_names(monkeypatch, dtype, operand):
    from rocke.examples.gfx1250.gemm import scale_formats_gemm as example

    calls = []

    def verify(spec, args):
        calls.append(spec)
        llvm = lower_kernel_to_llvm(
            build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
        )
        assert "@llvm.amdgcn.wmma.scale" in llvm
        return 0

    monkeypatch.setattr(example, "verify", verify)
    argv = [
        "--dtype-a",
        "fp4",
        "--dtype-b",
        "fp4",
        "--scale-dtype-a",
        "e8m0",
        "--scale-dtype-b",
        "e8m0",
    ]
    argv += [f"--dtype-{operand}", dtype]
    assert example.main(argv) == 0
    assert len(calls) == 1
    assert getattr(calls[0], f"dtype_{operand}") == normalize_dtype(dtype)


@pytest.mark.parametrize("axis", ["m", "n"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_scale_formats_gemm_cli_rejects_invalid_spec_before_verify(
    monkeypatch, capsys, axis, value
):
    from rocke.examples.gfx1250.gemm import scale_formats_gemm as example

    monkeypatch.setattr(example, "verify", lambda *args: pytest.fail("must not verify"))
    with pytest.raises(SystemExit) as exc:
        example.main([f"--{axis}", value])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "M/N/K must be positive" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp6", "bf6", "fp4"])
@pytest.mark.parametrize("operand", ["a", "b"])
def test_scale_formats_gemm_cli_filters_alternate_scales(
    monkeypatch, capsys, dtype, operand
):
    from rocke.examples.gfx1250.gemm import scale_formats_gemm as example

    calls = []

    def verify(spec, args):
        assert is_valid_spec(spec)[0]
        calls.append(spec)
        return 0

    monkeypatch.setattr(example, "verify", verify)
    if dtype == "fp4":
        assert example.main([f"--dtype-{operand}", dtype]) == 0
        assert len(calls) == 1
        assert calls[0].resolved_scale_dtypes() == ("e4m3", "e4m3")
    else:
        with pytest.raises(SystemExit) as exc:
            example.main([f"--dtype-{operand}", dtype])
        assert exc.value.code == 2
        assert not calls
        assert "no gfx1250 scaled WMMA atom" in capsys.readouterr().err


@pytest.mark.parametrize("operand", ["a", "b"])
def test_scale_formats_gemm_cli_rejects_unequal_fp4_scales(
    monkeypatch, capsys, operand
):
    from rocke.examples.gfx1250.gemm import scale_formats_gemm as example

    monkeypatch.setattr(example, "verify", lambda *args: pytest.fail("must not verify"))
    with pytest.raises(SystemExit) as exc:
        example.main([f"--scale-dtype-{operand}", "e5m3"])
    assert exc.value.code == 2
    assert "no gfx1250 scaled WMMA atom" in capsys.readouterr().err
