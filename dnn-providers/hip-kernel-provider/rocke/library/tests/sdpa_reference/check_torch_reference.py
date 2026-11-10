# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit optional Torch oracle checks, excluded from installed test discovery."""

from __future__ import annotations

import numpy as np
import pytest

from sdpa_reference.contract import Case, independent_reference, make_inputs
from sdpa_reference.torch_reference import cross_check, validate_device

pytest.importorskip("torch")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 4])
def test_torch_reference_agrees_on_mha_gqa_and_mask(dtype, causal, kv_heads):
    case = Case(dtype, 64, 4, kv_heads, False, causal, batch=2, sequence_length=16)
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    report = cross_check(case, inputs, reference, reference + 1)
    assert report["device"] == "cpu"
    assert set(report["results"]) == {"float64", "float32"}
    for result in report["results"].values():
        assert result["numpy_distance_upper"] < 1e-5
        assert result["baseline_error_bound"] > case.tolerance


def test_torch_reference_reports_disagreement_without_arbitrary_reserve():
    case = Case("fp16", 64, 4, 1, False, True, sequence_length=8)
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    report = cross_check(case, inputs, reference + 1, reference)
    for result in report["results"].values():
        assert result["numpy_distance_upper"] > 0.9
        assert result["baseline_error_bound"] < 1e-5


@pytest.mark.parametrize("bad", ["nan", "shape"])
def test_torch_reference_rejects_invalid_output(monkeypatch, bad):
    import torch
    import torch.nn.functional as functional

    case = Case("fp16", 64, 4, 1, False, True, sequence_length=8)
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    output = torch.full((1, 4, 8, 64), float("nan")) if bad == "nan" else torch.zeros(1)
    monkeypatch.setattr(
        functional, "scaled_dot_product_attention", lambda *a, **k: output
    )
    with pytest.raises((ValueError, RuntimeError)):
        cross_check(case, inputs, reference)


def test_gpu_reference_rejects_missing_rocm(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(ValueError, match="ROCm GPU"):
        validate_device("cuda", "gfx942")


def test_gpu_reference_rejects_wrong_processor(monkeypatch):
    import torch
    from types import SimpleNamespace

    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-"),
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_device("cuda", "gfx942")
