################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Unit tests for the ``--global-parameters`` value parser (``splitExtraParameters``).

These pin the security fix that replaced ``eval()`` with ``ast.literal_eval()`` in
``Tensile.addCommonArguments``: literal values still parse exactly as before, non-literal
values fall back to their raw string, and an attacker-controlled value can no longer
execute arbitrary code.
"""

import argparse

import pytest

pytestmark = pytest.mark.unit

from Tensile import Tensile


def _parse(*pairs):
    """Parse ``--global-parameters`` pairs through the real shared parser.

    Returns the resulting ``{key: value}`` mapping built from ``args.global_parameters``.
    """
    parser = argparse.ArgumentParser()
    Tensile.addCommonArguments(parser)
    args = parser.parse_args(["--global-parameters", *pairs])
    return dict(args.global_parameters)


def test_literal_bool_is_parsed():
    assert _parse("CheckASMCodeSize=True") == {"CheckASMCodeSize": True}


def test_literal_int_and_list_are_parsed():
    assert _parse("A=5", "B=[1, 2]") == {"A": 5, "B": [1, 2]}


def test_non_literal_bareword_falls_back_to_string():
    assert _parse("CompilerKind=yaml") == {"CompilerKind": "yaml"}


def test_arithmetic_expression_is_not_evaluated():
    # Old eval() would yield 2; ast.literal_eval rejects operators and keeps the raw string.
    assert _parse("X=1+1") == {"X": "1+1"}


def test_malicious_payload_is_not_executed(tmp_path):
    sentinel = tmp_path / "pwned"
    payload = f"__import__('os').system('touch {sentinel}')"
    result = _parse(f"Evil={payload}")
    assert result == {"Evil": payload}  # kept verbatim, not executed
    assert not sentinel.exists()
