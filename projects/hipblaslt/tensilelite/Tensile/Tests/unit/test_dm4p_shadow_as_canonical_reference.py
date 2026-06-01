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
"""rocm-libraries-dm4p Phase 2: lock-in for SHADOW-as-canonical-reference.

The Phase 2 swap replaces the Approach-A `build_non_cms_reference(...)` call
at the `_captureNonCmsBuild` CMS-callsite branch with consumption of
`self._last_default_capture` (the SHADOW capture already populated by the
`_captureDefaultSchedule` machinery earlier in `kernelBody`).

This test pins the swap at the source level so that a future change which
accidentally restores the old `build_non_cms_reference` call (e.g. during
Phase 4 Approach-A retirement, or during a merge conflict) fails loud here
rather than silently bypassing the nmsx/g9fi capture-quality work and
making compare_graphs depend on Approach A again.

Why source-level (instead of a build-driven assertion):

  - The end-to-end build (`Tensile/bin/Tensile ... --build-only`) already
    exercises the path; failure there is the primary signal.
  - This file is a fast, deterministic regression-pin that runs in the unit
    test suite alongside the cwd-trap guard.
  - It does NOT depend on rocisa/assembler infrastructure to assert the
    invariant, so it stays green in environments where the full build
    cannot run.
"""

import inspect
import os
import re

import pytest


# rocm-libraries-g9fi (cwd-trap guard, replicated from
# test_capture_pipeline_checks.py:54-85). When pytest is invoked from a
# directory containing a sibling `Tensile/` package, `import
# Tensile.KernelWriter` resolves to THAT tree's KernelWriter.py rather
# than the worktree's. The test would then assert against the wrong
# source code.
def _assert_tensile_tree_matches_test_tree():
    import Tensile.KernelWriter as _kw
    test_tree = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    kw_tree = os.path.abspath(
        os.path.join(os.path.dirname(_kw.__file__), "..")
    )
    if test_tree != kw_tree:
        raise RuntimeError(
            f"Tensile package loaded from a different tree than this test "
            f"file. test_tree={test_tree!r}, kw_tree={kw_tree!r}. This "
            f"usually means pytest was invoked from a directory containing "
            f"a sibling `Tensile/` package that shadows the intended one — "
            f"`import Tensile.*` resolves to the cwd's tree, not the one "
            f"PYTHONPATH points at. Fix: `cd {test_tree}` before invoking "
            f"pytest. See the round-3 review note on the g9fi commit."
        )


_assert_tensile_tree_matches_test_tree()


def _kernelbody_source():
    """Return the source text of `KernelWriter.kernelBody`.

    `kernelBody` is the method that hosts both the SHADOW capture block
    (`_captureDefaultSchedule`) and the xj16 real-vs-real assertion block
    (`_captureNonCmsBuild`). The Phase 2 swap lives inside the latter.
    """
    import Tensile.KernelWriter as kw
    src, _ = inspect.getsourcelines(kw.KernelWriter.kernelBody)
    return "".join(src)


def test_xj16_cms_callsite_consumes_shadow_capture():
    """Phase 2 lock-in: the CMS-callsite branch of `_captureNonCmsBuild`
    MUST assign `ctx.default` from `self._last_default_capture` (the
    SHADOW capture), NOT call `build_non_cms_reference(...)`.

    If this test fails after a future change, you have re-introduced
    Approach A's parallel Build #2 as the validator's reference. That is
    the exact regression Phase 2 was designed to retire — go read
    `Tensile/Components/DEFAULT_SCHEDULER_REFERENCE_DESIGN.md` §4 Phase
    2 before "fixing" the test.
    """
    src = _kernelbody_source()

    # The `_captureNonCmsBuild` block in kernelBody contains a
    # `if is_cms_callsite:` branch. That branch must consume the SHADOW
    # capture.
    cms_branch_match = re.search(
        r"if is_cms_callsite:\s*\n(?P<body>(?:^[ \t]+.*\n)+?)\s*else:",
        src,
        re.MULTILINE,
    )
    assert cms_branch_match is not None, (
        "Could not locate `if is_cms_callsite: ... else:` block in "
        "KernelWriter.kernelBody. Has the structure of the "
        "_captureNonCmsBuild block changed? Update this test to match."
    )
    cms_branch_body = cms_branch_match.group("body")

    # Strip line-comments so the assertions check executable Python, not
    # narrative prose that legitimately mentions `build_non_cms_reference`
    # while explaining why it's no longer called.
    cms_branch_code = "\n".join(
        re.sub(r"#.*$", "", line) for line in cms_branch_body.splitlines()
    )

    assert "self._last_default_capture" in cms_branch_code, (
        "Phase 2 invariant violated: the CMS-callsite branch of the "
        "_captureNonCmsBuild block in KernelWriter.kernelBody no longer "
        "assigns from `self._last_default_capture`. The SHADOW capture "
        "MUST be the validator's reference per the dm4p Phase 2 swap. "
        "Branch body:\n"
        + cms_branch_body
    )

    assert "build_non_cms_reference" not in cms_branch_code, (
        "Phase 2 regression: the CMS-callsite branch of the "
        "_captureNonCmsBuild block in KernelWriter.kernelBody has "
        "re-introduced a `build_non_cms_reference(...)` call. That was "
        "Approach A's parallel Build #2; Phase 2 (dm4p) replaced it "
        "with consumption of the SHADOW capture "
        "(`self._last_default_capture`). If you genuinely need to "
        "restore Approach A here, update the design doc "
        "(DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §4) and delete this "
        "test with a clear rationale. Otherwise: revert your change. "
        "Branch body:\n"
        + cms_branch_body
    )


def test_capture_context_default_survives_reset():
    """Supporting invariant: the SHADOW path's `_capture_context.reset()`
    in `kernelBody`'s finally block MUST preserve `ctx.default`.

    The Phase 2 swap relies on this — the SHADOW block runs first and
    populates `ctx.default`, then `reset()` clears scratch state, then
    the `_captureNonCmsBuild` block reads `self._last_default_capture`
    (which is `ctx.default`). If `reset()` ever clears `default`, the
    Phase 2 swap silently propagates None into the xj16 assertion.
    """
    from Tensile.Components.ScheduleCapture import CaptureContext

    ctx = CaptureContext()
    sentinel = object()
    ctx.default = sentinel
    ctx.cms = sentinel
    ctx.default_main = "scratch"
    ctx.builder = "scratch"

    ctx.reset()

    assert ctx.default is sentinel, (
        "CaptureContext.reset() cleared `default` — Phase 2 (dm4p) "
        "requires `default` to survive reset so the xj16 inline "
        "assertion can consume the SHADOW capture."
    )
    assert ctx.cms is sentinel, (
        "CaptureContext.reset() cleared `cms` — both `default` and "
        "`cms` must survive reset (they are the consumer-facing "
        "artifacts; only scratch state is cleared)."
    )
    assert ctx.default_main is None, (
        "CaptureContext.reset() failed to clear scratch state "
        "`default_main`."
    )
    assert ctx.builder is None, (
        "CaptureContext.reset() failed to clear scratch state "
        "`builder`."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
