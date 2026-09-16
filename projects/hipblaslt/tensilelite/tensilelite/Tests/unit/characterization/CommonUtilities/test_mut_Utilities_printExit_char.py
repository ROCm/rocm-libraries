################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.printExit``.

``printExit`` prints a ``Tensile::FATAL: <message>`` line, flushes stdout, and
terminates the process via ``sys.exit(-1)``. These tests pin the exact message
prefix and the exact exit code so that mutations of either are caught.
"""

import pytest

from Tensile.Common.Utilities import printExit

pytestmark = pytest.mark.unit


def test_print_exit_exact_message_prefix(capsys):
    """Kills mutmut_3: the literal must be exactly ``Tensile::FATAL: %s``.

    A mutant that changes the format literal to ``XXTensile::FATAL: %sXX``
    produces ``XXTensile::FATAL: fatalXX``; asserting the exact line content
    (prefix + no surrounding ``XX`` markers) distinguishes it.
    """
    with pytest.raises(SystemExit):
        printExit("fatal")
    out = capsys.readouterr().out
    # Exact line, not just a containment check.
    assert out == "Tensile::FATAL: fatal\n"
    assert out.startswith("Tensile::FATAL: ")
    assert not out.startswith("XX")


def test_print_exit_code_is_negative_one(capsys):
    """Kills mutmut_6/_7/_8: the exit code must be exactly ``-1``.

    Distinguishes the original ``sys.exit(-1)`` from ``sys.exit(None)`` (code
    ``None``), ``sys.exit(+1)`` (code ``1``), and ``sys.exit(-2)`` (code ``-2``).
    """
    with pytest.raises(SystemExit) as excinfo:
        printExit("boom")
    assert excinfo.value.code == -1
    # Guard against None / positive codes explicitly.
    assert excinfo.value.code is not None
    assert excinfo.value.code < 0
