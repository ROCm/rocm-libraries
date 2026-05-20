################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Forward-compat coverage for deprecated `Assert*` keys.

PR #7443 ("manual revert KRingShift") removed the parser handlers for
several `Assert*` keys; PR #7513 ran the matching YAML cleanup in the
hipBLASLt / hipSPARSELt logic trees. Library-logic YAML files added to
develop *after* either revert can still carry the stale keys, and
parsing such a YAML used to bomb the whole `TensileCreateLibrary` run
with::

    RuntimeError: Unknown assertion key: AssertFree1DivByMT1LowbitGT1

These tests pin the silent-ignore behavior we added in
`ProblemPredicate.FromOriginalKeyPair` so a future refactor that
re-tightens the strict-Assert path doesn't silently re-introduce the
build-time regression.
"""
import pytest

from Tensile.Contractions import ProblemPredicate


class TestDeprecatedAssertKeys:
    """`ProblemPredicate.FromOriginalKeyPair` must silently drop the
    deprecated `Assert*` keys instead of raising.
    """

    @pytest.mark.parametrize("key,value", [
        ("AssertFree1DivByMT1LowbitGT1", 0),
        ("AssertFree1DivByMT1LowbitGT1", 1),
        ("AssertKRingShiftTailWrapOnly", 0),
        ("AssertKRingShiftTailWrapOnly", 1),
    ])
    def test_returns_none_no_raise(self, key, value):
        """Deprecated key -> `None` (i.e. no predicate emitted); must
        not raise `RuntimeError`.
        """
        assert ProblemPredicate.FromOriginalKeyPair((key, value)) is None

    def test_unknown_non_deprecated_assert_still_raises(self):
        """The deprecated-key fallback must NOT swallow genuinely
        unknown `Assert*` keys -- those still need to fail loudly so a
        new assertion isn't silently dropped on the floor.
        """
        with pytest.raises(RuntimeError, match=r"Unknown assertion key: AssertSomeFutureKey"):
            ProblemPredicate.FromOriginalKeyPair(("AssertSomeFutureKey", 1))

    def test_known_assert_keys_still_work(self):
        """Spot check: a recognized `Assert*Multiple` key must still
        round-trip to a real predicate (the fallback didn't accidentally
        short-circuit normal keys).
        """
        pred = ProblemPredicate.FromOriginalKeyPair(
            ("AssertFree0ElementMultiple", 2)
        )
        assert pred is not None
        pred = ProblemPredicate.FromOriginalKeyPair(
            ("AssertSummationElementMultiple", 32)
        )
        assert pred is not None

    def test_known_assert_multiple_with_value_1_returns_none(self):
        """Sanity: the existing `Multiple == 1 -> None` short-circuit
        is unchanged by the deprecated-key fallback (it fires *after*
        the deprecated-key check but *before* the strict-raise).
        """
        assert ProblemPredicate.FromOriginalKeyPair(
            ("AssertFree0ElementMultiple", 1)
        ) is None
