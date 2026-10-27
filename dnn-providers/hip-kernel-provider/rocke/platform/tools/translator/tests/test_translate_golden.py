# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The fixture modules translate to their goldens, the same way every time."""

from __future__ import annotations

import pytest
from trtest import CPP_MODULES, GOLDEN, normalize, outputs


@pytest.mark.parametrize("name", sorted(outputs(GOLDEN)))
def test_golden(name, generated, request):
    out, _runs = generated
    text = outputs(out)[name].read_text()
    golden = GOLDEN / name
    if request.config.getoption("--update-goldens"):
        golden.write_text(text)
        pytest.skip(f"updated {golden.name}")
    assert golden.exists(), f"no golden {golden.name}: run with --update-goldens"
    assert normalize(text) == normalize(
        golden.read_text()
    ), f"{name} differs from its golden; if the change is intended, rerun with --update-goldens and review the diff"


@pytest.mark.parametrize("module", CPP_MODULES)
def test_fully_mechanical(module, generated):
    """Every function translates by rule: no hole, no hand fill."""
    _out, runs = generated
    last = runs[module].stdout.strip().splitlines()[-1]
    n, total = last.split()[0].split("/")
    assert n == total and last.endswith(", 0 holes"), last
    assert "facts stale" not in runs[module].stderr


def test_deterministic(generated, fixture_tr, tmp_path):
    out, _runs = generated
    fixture_tr.generate(tmp_path)
    first, second = outputs(out), outputs(tmp_path)
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes(), name
