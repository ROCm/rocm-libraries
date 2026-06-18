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
"""Sibling InternalSupportParams validator tests."""

import pytest

from Tensile.Common.ValidParameters import validateInternalSupportParams
from Tensile.Common.TypeValidationErrors import ConfigTypeError
from Tensile.Common.GlobalParameters import defaultInternalSupportParams

pytestmark = pytest.mark.usefixtures("_force_strict_gate_on")


class TestCleanDict:
    def test_empty_dict_passes(self):
        validateInternalSupportParams({})

    def test_full_defaults_dict_passes(self):
        validateInternalSupportParams(dict(defaultInternalSupportParams))

    def test_kern_args_version_int_passes(self):
        validateInternalSupportParams({"KernArgsVersion": 2})


class TestBadType:
    def test_kern_args_version_string_raises(self):
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({"KernArgsVersion": "two"})
        assert "KernArgsVersion" in str(exc.value)
        assert "expected int" in str(exc.value)

    def test_support_user_gsu_int_raises(self):
        """SupportUserGSU default is bool True; int must be rejected."""
        # Use a value that's an actual int (not bool).
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({"SupportUserGSU": 1})
        assert "SupportUserGSU" in str(exc.value)
        assert "expected bool" in str(exc.value)


class TestUnknownKey:
    def test_unknown_key_raises(self):
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({"NoSuchKey": 0})
        assert "NoSuchKey" in str(exc.value)
        assert "unknown key" in str(exc.value).lower()


class TestKeyPathPrefix:
    def test_default_prefix_used(self):
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({"KernArgsVersion": "two"})
        assert "InternalSupportParams.KernArgsVersion" in str(exc.value)

    def test_custom_prefix_used(self):
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams(
                {"KernArgsVersion": "two"},
                keyPathPrefix="BenchmarkProblems[0][1].InternalSupportParams",
            )
        assert "BenchmarkProblems[0][1].InternalSupportParams.KernArgsVersion" in str(exc.value)


class TestRaisesOnFirstError:
    def test_raises_on_first_bad_key(self):
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({
                "KernArgsVersion": "two",   # bad: str for int — first key, raises here
                "SupportUserGSU": 1,        # bad: int for bool — not reached
            })
        assert "KernArgsVersion" in str(exc.value)
