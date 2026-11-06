# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Optional offline oracle checks; outside the required installed host suite."""

import numpy as np
import pytest

from conv_reference.architectures import ARCHITECTURES, get_adapter, variants
from conv_reference.contract import independent_reference, make_inputs, reference_scale
from conv_reference.torch_reference import cross_check

pytest.importorskip("torch")

# The CPU Torch oracles depend only on the case, not on the processor or the
# kernel variant, so check each distinct enrolled case once.
CASES = tuple(
    dict.fromkeys(
        case
        for name in ARCHITECTURES
        for variant in variants(name)
        for case in get_adapter(name, variant).CASES
    )
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_torch_oracles_agree_with_numpy_and_detect_bad_baseline(case):
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    scale = reference_scale(reference)
    report = cross_check(case, inputs, reference, reference + scale, scale)
    assert report["device"] == "cpu"
    assert set(report["results"]) == {"float32", "float64"}
    for result in report["results"].values():
        assert result["numpy_distance_upper"] < case.tolerance
        assert result["baseline_error_bound"] > case.tolerance


def test_torch_cross_check_reports_disagreement_without_reserve():
    case = CASES[0]
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    scale = reference_scale(reference)
    report = cross_check(case, inputs, reference + scale, reference, scale)
    for result in report["results"].values():
        assert result["numpy_distance_upper"] > 0.9
        assert result["baseline_error_bound"] < case.tolerance


def test_torch_cross_check_rejects_nonfinite_answer(monkeypatch):
    import torch
    import torch.nn.functional as functional

    case = CASES[0]
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    monkeypatch.setattr(
        functional,
        "conv2d",
        lambda *args, **kwargs: torch.full(
            (case.N, case.K, *case.output_shape[1:3]), np.nan
        ),
    )
    with pytest.raises(ValueError, match="invalid Torch"):
        cross_check(case, inputs, reference, reference, reference_scale(reference))
