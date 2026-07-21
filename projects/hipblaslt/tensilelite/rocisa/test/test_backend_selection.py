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

"""Tests for rocisa's backend selection (_resolve_backend).

These exercise the decision logic directly with injected fake probes, so both
the "successful switch" and the "failed fallback" scenarios are covered without
needing a real stinkytofu build. The key contract is that an *explicitly
requested* stinkytofu backend that cannot be used falls back to native with a
visible warning (never silently), while any other request stays quiet.
"""

import warnings

import pytest

from rocisa import _resolve_backend


def _never_called(*args, **kwargs):
    raise AssertionError("probe should not be called")


class TestResolveBackend:
    def test_unset_selects_native_without_probing_or_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            assert _resolve_backend("", _never_called, _never_called) is False

    def test_other_backend_selects_native_without_probing_or_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve_backend("native", _never_called, _never_called) is False

    def test_successful_switch_returns_true_without_warning(self):
        calls = {"load": 0}

        def load_fn():
            calls["load"] += 1
            return True, ""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _resolve_backend("stinkytofu", lambda: True, load_fn)

        assert result is True
        assert calls["load"] == 1

    def test_fallback_when_unavailable_warns_not_built(self):
        with pytest.warns(UserWarning, match="not built/available") as record:
            result = _resolve_backend("stinkytofu", lambda: False, _never_called)

        assert result is False
        assert len(record) == 1
        assert "native rocisa backend" in str(record[0].message)

    def test_fallback_when_load_fails_warns_with_reason(self):
        reason = "import failed: ImportError('boom')"

        with pytest.warns(UserWarning, match="failed to load") as record:
            result = _resolve_backend(
                "stinkytofu", lambda: True, lambda: (False, reason)
            )

        assert result is False
        assert reason in str(record[0].message)

    def test_warn_callable_is_injectable(self):
        captured = []

        def fake_warn(msg, **kwargs):
            captured.append(msg)

        result = _resolve_backend(
            "stinkytofu", lambda: False, _never_called, warn=fake_warn
        )

        assert result is False
        assert len(captured) == 1
        assert "not built/available" in captured[0]
