################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the ``tensilelite`` package ``__init__``: the
version constant, ``ROOT_PATH``, and ``PrintTensileRoot``."""

import os

import pytest

import tensilelite

pytestmark = pytest.mark.unit


def test_version(snapshot):
    assert tensilelite.__version__ == snapshot


def test_root_path():
    # The absolute path is env-specific, so only pin that it is absolute.
    assert os.path.isabs(tensilelite.ROOT_PATH)


def test_print_tensile_root(capsys):
    tensilelite.PrintTensileRoot()
    out = capsys.readouterr().out
    assert out == tensilelite.ROOT_PATH  # printed with end='' (no newline)
