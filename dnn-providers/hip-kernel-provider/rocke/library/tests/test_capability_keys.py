# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The capability keys say what ``identity_supports_problem`` reads.

``KernelCache.compatible`` evaluates the predicate once per (identity key,
problem key) pair instead of once per entry and shape, which is only sound if
the keys cover everything the predicate reads. A field the keys leave out but
the predicate reads would make one cached answer stand in for two different
ones -- a binary offered to a shape it cannot run, reported as a launch
failure or, worse, as a wrong result.

So the keys are checked the way they are used: perturb what a key leaves out
and assert the predicate does not move. Nothing is compiled and no GPU is
touched.
"""

from __future__ import annotations

import dataclasses

import pytest

from benchmarks.common.kernel_cache import KernelCache
from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_direct_grouped import DirectConvProblem
from kernels.conv.kernel_identity import (
    DIRECT_PROBLEM_CAPABILITY_ATTRS,
    IDENTITY_CAPABILITY_FIELDS,
    KernelIdentity,
    identity_capability_key,
    identity_supports_problem,
    problem_capability_key,
)

_ARCH = "gfx942"


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------


def _identity(**kw) -> KernelIdentity:
    base = dict(
        arch=_ARCH,
        direction="fwd",
        algorithm="implicit_gemm",
        dtype_a="bf16",
        dtype_b="bf16",
        dtype_d="bf16",
        tile_m=128,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="compv3",
        epilogue="cshuffle",
        wave_size=64,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=8,
    )
    base.update(kw)
    return KernelIdentity(**base)


def _identities() -> list:
    """Identities spanning every branch of the predicate.

    One per capability class that reads a problem differently: the three
    directions, the grouped and group-merged forms, and each of the
    implicit-GEMM dgrad specialisations (patch, tilde table, tap-outer-K,
    dY halo), plus the direct kernels, whose geometry is baked.
    """
    return [
        _identity(),
        _identity(grouped=True),
        _identity(is_3d=True),
        _identity(is_pointwise=True, filter_h=1, filter_w=1),
        _identity(async_dma=True, async_chunk_a=8, async_chunk_b=8),
        _identity(direction="wgrad", split_k=4),
        _identity(direction="wgrad", split_k=4, two_stage=True),
        _identity(direction="wgrad", group_merge=4, grouped=True),
        _identity(direction="fwd", group_merge=4, grouped=True),
        _identity(direction="dgrad", grouped=True),
        _identity(direction="dgrad", patch=True),
        _identity(direction="dgrad", max_sub_gemms=4),
        _identity(direction="dgrad", tap_outer_k=True),
        _identity(direction="dgrad", dy_halo=1, halo_max_wo=64, filter_h=3, filter_w=3),
        _identity(
            direction="direct_fwd",
            algorithm="direct_nongrouped",
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
            stride_h=1,
            stride_w=1,
            dilation_h=1,
            dilation_w=1,
            vector_size_a=0,
            vector_size_b=0,
            vector_size_c=0,
        ),
        _identity(
            direction="direct_wgrad",
            algorithm="direct_dw_wgrad",
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
            stride_h=1,
            stride_w=1,
            dilation_h=1,
            dilation_w=1,
            cpg=1,
            kpg=1,
            vector_size_a=0,
            vector_size_b=0,
            vector_size_c=0,
        ),
        _identity(
            direction="direct_dgrad",
            algorithm="direct_dgrad_mfma_rtc",
            filter_h=3,
            filter_w=3,
            pad_h=1,
            pad_w=1,
            stride_h=1,
            stride_w=1,
            dilation_h=1,
            dilation_w=1,
            vector_size_a=0,
            vector_size_b=0,
            vector_size_c=0,
        ),
    ]


def _conv_problems() -> list:
    return [
        ConvProblem(N=2, Hi=28, Wi=28, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        ConvProblem(N=1, Hi=56, Wi=56, C=32, K=64, Y=1, X=1),
        ConvProblem(N=4, Hi=14, Wi=14, C=128, K=128, Y=3, X=3, pH=1, pW=1, groups=128),
        ConvProblem(N=2, Hi=32, Wi=32, C=64, K=64, Y=3, X=3, sH=2, sW=2, pH=1, pW=1),
        ConvProblem(N=2, Hi=32, Wi=32, C=64, K=64, Y=3, X=3, dH=2, dW=2, pH=2, pW=2),
        ConvProblem(N=2, Hi=224, Wi=224, C=3, K=768, Y=16, X=16, sH=16, sW=16),
        ConvProblem(
            N=1,
            Hi=16,
            Wi=16,
            C=32,
            K=32,
            Y=3,
            X=3,
            pH=1,
            pW=1,
            Di=16,
            Z=3,
            sD=1,
            pD=1,
            dD=1,
        ),
    ]


def _direct_problems() -> list:
    return [
        DirectConvProblem(N=2, H=28, W=28, groups=1, cpg=64, kpg=64),
        DirectConvProblem(N=1, H=56, W=56, groups=16, cpg=16, kpg=16),
        DirectConvProblem(N=4, H=14, W=14, groups=128, cpg=1, kpg=1, KH=3, KW=3, PAD=1),
        DirectConvProblem(
            N=2, H=32, W=32, groups=1, cpg=32, kpg=64, KH=1, KW=1, PAD=0, stride=2
        ),
        DirectConvProblem(
            N=2, H=32, W=32, groups=8, cpg=8, kpg=8, dil_h=2, dil_w=2, PAD=2
        ),
        # Small groups at stride 2 (the packed direct WMMA dgrad rows).
        DirectConvProblem(N=2, H=28, W=28, groups=32, cpg=8, kpg=8, stride=2),
    ]


def _answers(identity, problems=None) -> list:
    """The predicate's verdicts (not its reasons) over a problem list.

    Each family is asked about its own problem type: the predicate reads a
    ``DirectConvProblem`` for a direct identity and a ``ConvProblem`` for an
    implicit-GEMM one, and the two do not share an attribute set.
    """
    if problems is None:
        problems = _direct_problems() if identity.is_direct else _conv_problems()
    return [identity_supports_problem(identity, problem)[0] for problem in problems]


def _perturbed(value):
    """A different value of the same kind, or ``None`` when there is none."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "_x"
    if value is None:
        return 3
    return None


# ---------------------------------------------------------------------------
# The identity side
# ---------------------------------------------------------------------------

# ``is_direct`` is a property over ``algorithm``; the key carries the property,
# so the field behind it is not free to move.
_IDENTITY_KEY_FIELDS = set(IDENTITY_CAPABILITY_FIELDS) | {"algorithm"}


def test_identity_key_covers_every_field_the_predicate_reads():
    """Moving a field outside the key must not move the predicate.

    If this fails, the named field belongs in ``IDENTITY_CAPABILITY_FIELDS``:
    the cache groups entries by that key and answers once per group, so two
    identities differing only in it would be told the same thing.
    """
    free = [
        f.name
        for f in dataclasses.fields(KernelIdentity)
        if f.name not in _IDENTITY_KEY_FIELDS
    ]
    assert free, "every field is in the key -- the grouping would be pointless"
    for identity in _identities():
        expected = _answers(identity)
        for name in free:
            other = _perturbed(getattr(identity, name))
            if other is None:
                continue
            moved = dataclasses.replace(identity, **{name: other})
            assert _answers(moved) == expected, (
                f"{identity.direction}/{identity.algorithm}: the predicate "
                f"reads {name!r}, which the identity capability key leaves out"
            )


def test_identity_key_separates_what_the_predicate_separates():
    """Two identities with one key is the claim; different keys for different
    capabilities is what makes the key useful rather than merely sound."""
    keys = {identity_capability_key(i) for i in _identities()}
    assert len(keys) == len(_identities())

    # ... and tuning that the predicate never reads must not split a group.
    base = _identity()
    assert identity_capability_key(base) == identity_capability_key(
        dataclasses.replace(base, pipeline="compv4", lds_k_pad=8, waves_per_eu=2)
    )


# ---------------------------------------------------------------------------
# The problem side
# ---------------------------------------------------------------------------


def test_direct_problem_key_covers_every_extent_the_predicate_reads():
    """A direct binary bakes its geometry and takes extents as kernargs, so
    the predicate must not read one. That is what lets thousands of shapes
    share a few hundred keys -- and it is a claim, so it is checked."""
    direct = [i for i in _identities() if i.is_direct]
    assert direct
    free = [
        f.name
        for f in dataclasses.fields(DirectConvProblem)
        if f.name not in DIRECT_PROBLEM_CAPABILITY_ATTRS
    ]
    assert set(free) == {"N", "H", "W", "dtype"}, free
    for problem in _direct_problems():
        expected = [_answers(i, [problem]) for i in direct]
        for name in free:
            other = _perturbed(getattr(problem, name))
            moved = dataclasses.replace(problem, **{name: other})
            assert [_answers(i, [moved]) for i in direct] == expected, (
                f"the direct predicate reads {name!r}, which "
                f"DIRECT_PROBLEM_CAPABILITY_ATTRS leaves out"
            )


def test_direct_problem_key_separates_column_paddings():
    # The predicate reads the column padding as PAD_W, which pad_w overrides,
    # so a key that drops pad_w would answer a per-axis padding with the
    # verdict for an equal one.
    p = DirectConvProblem(N=2, H=28, W=28, groups=16, cpg=16, kpg=16, PAD=1)
    assert problem_capability_key(p, is_direct=True) != problem_capability_key(
        dataclasses.replace(p, pad_w=0), is_direct=True
    )


def test_direct_problem_key_collapses_shapes_onto_geometries():
    p = DirectConvProblem(N=2, H=28, W=28, groups=16, cpg=16, kpg=16)
    same = dataclasses.replace(p, N=64, H=7, W=7, dtype="bf16")
    assert problem_capability_key(p, is_direct=True) == problem_capability_key(
        same, is_direct=True
    )
    other = dataclasses.replace(p, PAD=0)
    assert problem_capability_key(p, is_direct=True) != problem_capability_key(
        other, is_direct=True
    )


def test_implicit_problem_key_is_the_whole_shape():
    """An implicit-GEMM binary is shape-generic, so its admission does read
    extents (the grid, the dgrad sub-GEMM table). The key is exact there."""
    p = _conv_problems()[0]
    assert problem_capability_key(p, is_direct=False) == problem_capability_key(
        dataclasses.replace(p), is_direct=False
    )
    assert problem_capability_key(p, is_direct=False) != problem_capability_key(
        dataclasses.replace(p, N=p.N + 1), is_direct=False
    )


def test_unkeyable_problem_is_reported_as_such():
    """A caller must be able to tell "no key" from "this key", so it can fall
    back to evaluating the predicate rather than memoise everything as one."""

    class NotADataclass:
        cpg = kpg = 1

    assert problem_capability_key(NotADataclass(), is_direct=False) is None


# ---------------------------------------------------------------------------
# What the cache does with them
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path):
    """A cache holding every identity of the corpus with a placeholder blob."""
    c = KernelCache(tmp_path, _ARCH)
    for i, identity in enumerate(_identities()):
        c.put(identity, b"placeholder", {"kernel_name": f"k{i}"})
        # A second entry per capability class, differing only in tuning: the
        # grouping must return both, in the order a plain scan would.
        c.put(
            dataclasses.replace(identity, pipeline="compv4"),
            b"placeholder",
            {"kernel_name": f"k{i}_v4"},
        )
    return c


def _scan(cache, problem, direction):
    """``compatible`` as it read before grouping: one predicate per entry."""
    return [
        entry
        for entry in cache._entries(direction)
        if identity_supports_problem(entry[0], problem)[0]
    ]


@pytest.mark.parametrize(
    "direction", ["fwd", "wgrad", "dgrad", "direct_fwd", "direct_wgrad", "direct_dgrad"]
)
def test_compatible_matches_a_plain_scan(cache, direction):
    """Grouping and memoising must not change which kernels are offered, nor
    the order -- ``--sample``'s fixed-seed subset and the probe order are
    taken off this list."""
    direct = direction.startswith("direct_")
    problems = _direct_problems() if direct else _conv_problems()
    offered = 0
    for problem in problems:
        expected = _scan(cache, problem, direction)
        offered += len(expected)
        assert cache.compatible(problem, direction=direction) == expected
        # Again, from the memo this time.
        assert cache.compatible(problem, direction=direction) == expected
    # An agreement on "nothing matches" everywhere would prove nothing.
    assert offered


def test_compatible_returns_the_callers_own_list(cache):
    problem = _direct_problems()[0]
    first = cache.compatible(problem, direction="direct_fwd")
    first.clear()
    assert cache.compatible(problem, direction="direct_fwd")


def test_a_write_drops_the_memo(cache):
    """An entry added after a shape was answered has to be offered to it."""
    # The depthwise shape: the corpus's wgrad kernel bakes cpg = kpg = 1.
    problem = _direct_problems()[2]
    before = cache.compatible(problem, direction="direct_wgrad")
    assert before
    added = dataclasses.replace(
        [i for i in _identities() if i.direction == "direct_wgrad"][0],
        tile_k=32,
    )
    cache.put(added, b"placeholder", {"kernel_name": "added"})
    after = cache.compatible(problem, direction="direct_wgrad")
    assert len(after) == len(before) + 1


def test_entry_meta_reads_no_binary(cache, monkeypatch):
    """Plan building asks for a helper's entry point, not its code."""
    identity = _identities()[0]
    monkeypatch.setattr(
        type(cache.blob_path("x")),
        "read_bytes",
        lambda self: pytest.fail("entry_meta read an HSACO"),
    )
    assert cache.entry_meta(identity)["kernel_name"] == "k0"
    assert cache.entry_meta(dataclasses.replace(identity, tile_k=13)) is None
