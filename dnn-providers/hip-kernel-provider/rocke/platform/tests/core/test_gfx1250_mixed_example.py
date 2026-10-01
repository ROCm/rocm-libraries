# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mixed example selects independent normalized operands and the shared verifier."""

import pytest

from rocke.examples.gfx1250.gemm import mixed_scaled_gemm


def test_default_pair_and_verification_options(monkeypatch):
    seen = []
    monkeypatch.setattr(
        mixed_scaled_gemm, "verify", lambda spec, args: seen.append((spec, args)) or 0
    )
    assert (
        mixed_scaled_gemm.main(
            ["--matrix-path", "wmma_scale16", "--compile-route", "hip", "--case", "all"]
        )
        == 0
    )
    spec, args = seen[0]
    assert (spec.dtype_a, spec.dtype_b) == ("fp8e4m3", "fp4e2m1")
    assert spec.block_k == 16 and spec.scale_dtype == "e8m0"
    assert args.compile_route == "hip" and args.case == "all"


@pytest.mark.parametrize(
    "a,b,expected",
    [("fp6", "bf6", ("fp6e2m3", "fp6e3m2")), ("fp4", "bf8", ("fp4e2m1", "bf8e5m2"))],
)
def test_aliases_preserve_operand_order(monkeypatch, a, b, expected):
    seen = []
    monkeypatch.setattr(
        mixed_scaled_gemm, "verify", lambda spec, args: seen.append(spec) or 0
    )
    assert mixed_scaled_gemm.main(["--dtype-a", a, "--dtype-b", b]) == 0
    assert (seen[0].dtype_a, seen[0].dtype_b) == expected


@pytest.mark.parametrize(
    "a,b", [("fp8", "fp8e4m3"), ("bf6", "fp6e3m2"), ("fp4", "fp4e2m1")]
)
def test_equal_normalized_formats_rejected(a, b, capsys):
    with pytest.raises(SystemExit) as exc:
        mixed_scaled_gemm.main(["--dtype-a", a, "--dtype-b", b])
    assert exc.value.code == 2
    assert "choose different A/B formats" in capsys.readouterr().err
