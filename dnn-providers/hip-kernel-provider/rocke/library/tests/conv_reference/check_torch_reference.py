# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Optional offline oracle checks; outside the required installed host suite."""

import numpy as np
import pytest

from conv_reference.architectures import get_architecture
from conv_reference.contract import independent_reference, make_inputs, reference_scale
from conv_reference.torch_reference import cross_check

pytest.importorskip("torch")


@pytest.mark.parametrize("case", get_architecture("gfx942").CASES, ids=lambda c: c.id)
def test_torch_oracles_agree_with_numpy_and_detect_bad_baseline(case):
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    scale = reference_scale(reference)
    report = cross_check(case, inputs, reference, reference + scale, scale)
    assert report["device"] == "cpu"
    assert set(report["results"]) == {"float32", "float64"}
    for result in report["results"].values():
        assert result["numpy_distance_upper"] <= case.margin
        assert result["baseline_error_bound"] > case.tolerance


def test_torch_cross_check_rejects_wrong_numpy_answer():
    case = get_architecture("gfx942").CASES[0]
    inputs = make_inputs(case)
    reference = independent_reference(case, inputs)
    scale = reference_scale(reference)
    with pytest.raises(ValueError, match="disagreement"):
        cross_check(case, inputs, reference + scale, reference, scale)


def test_torch_cross_check_rejects_nonfinite_answer(monkeypatch):
    import torch
    import torch.nn.functional as functional

    case = get_architecture("gfx942").CASES[0]
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
