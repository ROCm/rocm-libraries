################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.printWarning``.

Pins the EXACT warning line emitted to stdout so that mutations to the format
string (e.g. wrapping it in ``XX...XX`` sentinels) are detected. The existing
``in``-based assertion survives such mutations because the original prefix is
still a substring of the mutated output."""

from Tensile.Common.Utilities import printWarning

import pytest

pytestmark = pytest.mark.unit


def test_print_warning_exact_format(capsys):
    # Kills x_printWarning__mutmut_3: the original emits exactly
    # "Tensile::WARNING: <message>" with no surrounding sentinels.
    printWarning("careful")
    out = capsys.readouterr().out
    assert out == "Tensile::WARNING: careful\n"
