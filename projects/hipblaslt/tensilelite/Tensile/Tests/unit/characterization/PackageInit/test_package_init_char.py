################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the ``Tensile`` package ``__init__``: the version
constant, the derived ROOT/SOURCE/CUSTOM_KERNEL paths, and ``PrintTensileRoot``."""

import os

import pytest

import Tensile

pytestmark = pytest.mark.unit


def test_version(snapshot):
    assert Tensile.__version__ == snapshot


def test_derived_paths():
    # Absolute paths are env-specific; pin their structure, not the prefix.
    assert os.path.isabs(Tensile.ROOT_PATH)
    assert Tensile.SOURCE_PATH == os.path.join(Tensile.ROOT_PATH, "Source")
    assert Tensile.CUSTOM_KERNEL_PATH == os.path.join(Tensile.ROOT_PATH, "CustomKernels")


def test_print_tensile_root(capsys):
    Tensile.PrintTensileRoot()
    out = capsys.readouterr().out
    assert out == Tensile.ROOT_PATH  # printed with end='' (no newline)
