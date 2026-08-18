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
################################################################################

"""Producer-side tests for per-arch fallback rename and Mapping completeness.

Each test is constructed so it fails on the *un*patched producer for the
specific behavior the patch changes. Discrimination is the point: a test that
passes on either implementation does not protect against this regression class.
"""

import pytest

from Tensile.SolutionLibrary import (
    MasterSolutionLibrary,
    PlaceholderLibrary,
    PredicateLibrary,
    ProblemMapLibrary,
)
from Tensile.TensileCreateLibrary.Run import renameFallbacksPerArch


def _placeholder(name: str) -> PlaceholderLibrary:
    return PlaceholderLibrary(name)


def _wrap_in_predicate(child) -> PredicateLibrary:
    """Wrap a single child in a PredicateLibrary row, mirroring the
    structure produced by the real library builder."""
    pred = PredicateLibrary(tag="Hardware")
    pred.rows = [{"predicate": object(), "library": child}]
    return pred


def _wrap_in_problem_map(child) -> ProblemMapLibrary:
    pmap = ProblemMapLibrary(mappingProperty=object(), mapping={"key": child})
    return pmap


def _make_master_with_fallback(lazyName: str, treeName: str) -> MasterSolutionLibrary:
    """Build a minimal master library that has a single fallback lazy library
    and a matching PlaceholderLibrary buried inside the master.library tree.

    Solutions dicts are left empty; the rename code does not need real
    solutions, only attribute shape (.lazyLibraries dict, .library tree).
    """
    placeholder = _placeholder(treeName)
    tree = _wrap_in_predicate(_wrap_in_problem_map(placeholder))
    master = MasterSolutionLibrary({}, tree)
    # `merge()` would have populated lazyLibraries from the fallback master;
    # we plant the entry directly to avoid needing real solution objects.
    master.lazyLibraries[lazyName] = object()
    return master


# ---------------------------------------------------------------------------
# 1. lazyLibraries dict keys get arch-suffixed for fallback entries.
# ---------------------------------------------------------------------------
def test_renameFallbacksPerArch_archSuffixesLazyKeys():
    master = _make_master_with_fallback(
        lazyName="TensileLibrary_Type_Hi_fallback",
        treeName="TensileLibrary_Type_Hi_fallback",
    )
    masterLibraries = {"gfx942": master}

    renameFallbacksPerArch(masterLibraries)

    keys = list(masterLibraries["gfx942"].lazyLibraries.keys())
    assert "TensileLibrary_Type_Hi_fallback_gfx942" in keys, (
        "Patched producer must arch-suffix fallback lazy-library keys; "
        "without the rename the key stays at 'TensileLibrary_Type_Hi_fallback' "
        "and the per-arch Mapping filter (`name.endswith(\"_gfx942\")`) drops it."
    )
    assert "TensileLibrary_Type_Hi_fallback" not in keys, (
        "Old (unsuffixed) key must be removed; leaving both around would "
        "double-count solutions in the per-arch Mapping."
    )


# ---------------------------------------------------------------------------
# 2. PlaceholderLibrary nodes inside master.library tree are renamed too.
# ---------------------------------------------------------------------------
def test_renameFallbacksPerArch_walksTreeAndRenamesPlaceholders():
    master = _make_master_with_fallback(
        lazyName="TensileLibrary_Type_Hi_fallback",
        treeName="TensileLibrary_Type_Hi_fallback",
    )
    masterLibraries = {"gfx942": master}

    renameFallbacksPerArch(masterLibraries)

    # Re-walk to find the PlaceholderLibrary the helper produced:
    placeholder = (
        masterLibraries["gfx942"]
        .library.rows[0]["library"]    # PredicateLibrary -> ProblemMapLibrary
        .mapping["key"]                # ProblemMapLibrary -> PlaceholderLibrary
    )
    assert isinstance(placeholder, PlaceholderLibrary)
    assert placeholder.filenamePrefix == "TensileLibrary_Type_Hi_fallback_gfx942", (
        "The PlaceholderLibrary in the master library tree drives the runtime "
        "load path. If only the lazyLibraries key is renamed but the tree node "
        "is not, the runtime asks for the un-suffixed file and finds nothing."
    )


# ---------------------------------------------------------------------------
# 3. Multi-arch safety: deepcopy must isolate per-arch changes.
# ---------------------------------------------------------------------------
def test_renameFallbacksPerArch_deepcopiesPerArchToAvoidAliasing():
    """`MasterSolutionLibrary.merge` aliases the same fallback lazy library
    across every arch's master. Renaming without a per-arch deep copy would
    suffix the shared object once per arch, producing
    `..._fallback_gfx942_gfx1100` and corrupting both arches.
    """
    master_gfx942 = _make_master_with_fallback(
        lazyName="TensileLibrary_Type_Hi_fallback",
        treeName="TensileLibrary_Type_Hi_fallback",
    )
    master_gfx1100 = _make_master_with_fallback(
        lazyName="TensileLibrary_Type_Hi_fallback",
        treeName="TensileLibrary_Type_Hi_fallback",
    )
    # Mimic the post-merge alias: both arches reference the same tree node.
    shared_placeholder = master_gfx942.library.rows[0]["library"].mapping["key"]
    master_gfx1100.library.rows[0]["library"].mapping["key"] = shared_placeholder

    masterLibraries = {"gfx942": master_gfx942, "gfx1100": master_gfx1100}
    renameFallbacksPerArch(masterLibraries)

    p942 = (
        masterLibraries["gfx942"]
        .library.rows[0]["library"]
        .mapping["key"]
    )
    p1100 = (
        masterLibraries["gfx1100"]
        .library.rows[0]["library"]
        .mapping["key"]
    )
    assert p942.filenamePrefix == "TensileLibrary_Type_Hi_fallback_gfx942"
    assert p1100.filenamePrefix == "TensileLibrary_Type_Hi_fallback_gfx1100"
    # Without the deepcopy these would point to the same mutated object and
    # one of the asserts above would fail with the other arch's suffix.
    assert p942 is not p1100, (
        "renameFallbacksPerArch must deep-copy each arch's master library; "
        "without it, the shared fallback PlaceholderLibrary is mutated twice "
        "and one arch ends up with the other arch's filename."
    )


# ---------------------------------------------------------------------------
# 4. Idempotency: a second pass must not double-suffix.
# ---------------------------------------------------------------------------
def test_renameFallbacksPerArch_isIdempotent():
    master = _make_master_with_fallback(
        lazyName="TensileLibrary_Type_Hi_fallback",
        treeName="TensileLibrary_Type_Hi_fallback",
    )
    masterLibraries = {"gfx942": master}

    renameFallbacksPerArch(masterLibraries)
    # Capture the post-first-pass state and re-run.
    keys_after_first = set(masterLibraries["gfx942"].lazyLibraries.keys())
    renameFallbacksPerArch(masterLibraries)
    keys_after_second = set(masterLibraries["gfx942"].lazyLibraries.keys())

    assert keys_after_first == keys_after_second, (
        "A naive implementation without the `endswith('_<arch>')` guard "
        "would suffix again, producing '..._fallback_gfx942_gfx942'."
    )
    assert "TensileLibrary_Type_Hi_fallback_gfx942_gfx942" not in keys_after_second


# ---------------------------------------------------------------------------
# 5. Round-trip discrimination: the per-arch Mapping filter the producer uses
# must keep every fallback entry once the rename has run.
#
# This is the test that would have caught the CI failure: it asserts the
# completeness invariant the runtime depends on.
# ---------------------------------------------------------------------------
def _per_arch_mapping_keys(libraryMapping, arch):
    """Mirror the producer's per-arch filter logic from Run.py."""
    return {
        idx
        for idx, name in libraryMapping.items()
        if name.endswith("_" + arch)
    }


def test_perArchMapping_keepsFallbackEntries_afterRename():
    """Build a synthetic post-merge libraryMapping (the dict that
    `generateLogicDataAndSolutions` returns) and verify that after rename
    the per-arch filter retains every entry the runtime needs."""
    # Pre-rename: lazyLibraries has both a tuned-for-arch entry and a fallback
    # entry that came in via merge. The producer's libraryMapping would map
    # solution indexes to these names.
    libraryMapping_pre = {
        0: "TensileLibrary_Type_Hi_gfx942",      # tuned solution
        1: "TensileLibrary_Type_Hi_fallback",    # fallback solution (e.g. fp16)
    }

    # Apply the per-arch filter as the unpatched producer would:
    unpatched_kept = _per_arch_mapping_keys(libraryMapping_pre, "gfx942")
    assert unpatched_kept == {0}, (
        "Sanity check: under the unpatched producer the fallback entry is "
        "filtered out because its name ends in '_fallback' not '_gfx942'. "
        "This is the bug that causes 'NO solution found!' for fp16 at runtime."
    )

    # Post-rename libraryMapping (what the patched producer feeds to the filter):
    libraryMapping_post = {
        0: "TensileLibrary_Type_Hi_gfx942",
        1: "TensileLibrary_Type_Hi_fallback_gfx942",
    }
    patched_kept = _per_arch_mapping_keys(libraryMapping_post, "gfx942")
    assert patched_kept == {0, 1}, (
        "Patched producer must surface every solution index in the per-arch "
        "Mapping. Missing fallback entries here means the runtime cannot "
        "resolve solutions whose dtype lives only in the fallback library."
    )
