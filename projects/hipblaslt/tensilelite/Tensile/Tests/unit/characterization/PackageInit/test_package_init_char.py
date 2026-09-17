################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the ``Tensile`` package ``__init__``: the version
constant, ``ROOT_PATH``, and ``PrintTensileRoot``."""

import os

import pytest

import Tensile

pytestmark = pytest.mark.unit


def test_version(snapshot):
    assert Tensile.__version__ == snapshot


def test_root_path():
    # The absolute path is env-specific, so only pin that it is absolute.
    assert os.path.isabs(Tensile.ROOT_PATH)


def test_print_tensile_root(capsys):
    Tensile.PrintTensileRoot()
    out = capsys.readouterr().out
    assert out == Tensile.ROOT_PATH  # printed with end='' (no newline)
