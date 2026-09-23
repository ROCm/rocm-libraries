# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS budget-vs-allocation equality for the gfx942 dense prefill kernel.

``_lds_bytes`` (the budget consulted by ``supports_attention_dense``) and the
builder's ``smem_alloc`` calls are two INDEPENDENT re-derivations of the same
footprint. Nothing previously compared them: there was no gfx942 test that read the
emitted LDS pool size at all. That is a live drift vector, and the
:class:`Gfx942AttentionDenseSpec` promotion of ``lds_row_pad`` / ``v_row_pad`` to
sweepable knobs widens it -- ``v_row_pad`` in particular is read in exactly two
places, the budget in ``_lds_bytes`` and the ``V_lds`` allocation in the builder, and
if those two ever disagree the gate passes a config whose real allocation overflows
the 64 KB LDS. An over-budget kernel does not fail loudly: it reaches comgr and dies
with an opaque ``CODEGEN_BC_TO_RELOCATABLE`` abort, or (worse, when the model
UNDER-counts by less than the slack) compiles and stomps memory.

The assertion is therefore the strong form: for every legal spec in the cohort,
``_lds_bytes(spec)`` must EQUAL the ``addrspace(3) global [N x i8]`` pool actually
emitted -- not merely "the two sites read the same field".

``_lds_pool_bytes`` is PORTED from the gfx950 ``test_attention_dense_d64_lds.py``
rather than imported: that file is a gfx950 test with an 18-case x 3-flavor golden
behind it and is not ours to touch, and a cross-test import would couple the two
arches' suites. The regex is four lines; the duplication is cheaper than the coupling.

Pure text lowering -- no GPU and no comgr required. Like its sibling this file lives
under ``library/tests/``, which ``platform/tests/run_all.py`` does not collect; run it
with the library lane:

    cd rocke/library
    PYTHONPATH=../platform/python:. python -m pytest \
        tests/test_attention_dense_gfx942_lds.py
"""

import dataclasses
import itertools
import re

import pytest

from kernels.common.attention_arch import attention_lds_capacity_bytes
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
    _DEFAULT_LDS_ROW_PAD,
    _lds_bytes,
)

_POOL_RE = re.compile(r"addrspace\(3\)\s+global\s+\[(\d+) x i8\]")
_CAPACITY = attention_lds_capacity_bytes("gfx942")


def _spec(**kw) -> Gfx942AttentionDenseSpec:
    base = dict(
        batch=1,
        seqlen_q=2048,
        seqlen_kv=2048,
        num_query_heads=16,
        num_kv_heads=4,
        head_size=128,
        causal=True,
        dtype="fp16",
        block_n=64,
    )
    base.update(kw)
    return Gfx942AttentionDenseSpec(**base)


def _lower(spec) -> str:
    from rocke.core.lower_llvm import (
        _lower_kernel_to_llvm_python,
        _resolve_llvm_flavor,
    )

    return _lower_kernel_to_llvm_python(
        build_attention_dense(spec, arch="gfx942"),
        arch="gfx942",
        llvm_flavor=_resolve_llvm_flavor(),
    )


def _lds_pool_bytes(spec) -> int:
    """Size of the unified ``addrspace(3)`` smem pool this spec emits.

    Ported from the gfx950 ``test_attention_dense_d64_lds.py`` helper. Asserts there
    is EXACTLY one pool global: the whole point of comparing against ``_lds_bytes`` is
    that the model accounts for the entire footprint, and a second allocation the
    model knows nothing about would otherwise be silently ignored by a ``search``.
    """
    pools = _POOL_RE.findall(_lower(spec))
    assert len(pools) == 1, (
        f"expected exactly one addrspace(3) smem pool in the lowered IR, found "
        f"{len(pools)}: {pools} -- _lds_bytes models a single unified pool"
    )
    return int(pools[0])


# --------------------------------------------------------------------------- #
# the cohort
# --------------------------------------------------------------------------- #
# Tuning variants -- now plain field overrides on the spec itself. Every knob that
# can move the LDS footprint appears at its default AND at a non-default value, plus
# both cfvst states (cfvst switches V between the transposed [dim, token+v_row_pad]
# layout and the natural [token, dim] one, i.e. it selects WHICH of the two pads is
# live) and a block_m variant (the footprint is block_m-invariant; that invariance is
# itself worth pinning, since a builder that started sizing LDS from block_m would
# silently escape the budget check).
_TUNING_VARIANTS = (
    dict(),
    dict(lds_row_pad=0),
    dict(lds_row_pad=16),
    dict(v_row_pad=0),
    dict(v_row_pad=32),
    dict(lds_row_pad=0, v_row_pad=0),
    dict(lds_row_pad=16, v_row_pad=16),
    dict(use_cfvst=False),
    dict(use_cfvst=True),
    dict(use_cfvst=False, v_row_pad=32),
    dict(block_m=128),
)


def _cohort():
    """Legal specs across D64/D128 x bf16/fp16 x block_n x tuning override.

    Filtered through ``supports_attention_dense`` at COLLECTION time so the suite has
    no skipped cases: an out-of-scope combination (cfvst forced on at D64, an
    over-budget pad) simply is not a case. The ``spec.lds_k_group_pad`` axis is in
    here too -- it is the D64 K pad, the mutually-exclusive sibling of ``lds_row_pad``,
    and it feeds the same ``_lds_bytes`` branch.
    """
    cases = []
    for d, dtype, block_n, kpad in itertools.product(
        (64, 128), ("bf16", "fp16"), (32, 64, 128), (0, 8)
    ):
        if kpad and d != 64:
            continue  # inert at D128; keeps the cohort from tripling for nothing
        for kw in _TUNING_VARIANTS:
            spec = _spec(
                head_size=d,
                dtype=dtype,
                block_n=block_n,
                lds_k_group_pad=kpad,
                **kw,
            )
            ok, _ = supports_attention_dense(spec, arch="gfx942")
            if not ok:
                continue
            ident = f"d{d}-{dtype}-bn{block_n}-kpad{kpad}-" + (
                "default" if not kw else "-".join(f"{k}{v}" for k, v in kw.items())
            )
            cases.append(pytest.param(spec, id=ident))
    return cases


_COHORT = _cohort()


@pytest.mark.parametrize("spec", _COHORT)
def test_lds_budget_equals_the_emitted_allocation(spec):
    """``_lds_bytes`` must equal the pool the builder actually allocates, and fit.

    The equality is the drift guard: the budget and the allocation are separate
    re-derivations and the gate in ``supports_attention_dense`` trusts the budget, so
    an under-count admits a config whose real allocation overflows LDS.

    The capacity check that follows closes the loop -- gate reads the model, model
    equals the allocation, allocation is within capacity -- so that supports()
    accepting really does imply the emitted kernel fits. Both live in one case (one
    lowering serves both) with distinct messages, so the two failure modes stay
    distinguishable.
    """
    expected = _lds_bytes(spec)
    actual = _lds_pool_bytes(spec)
    assert actual == expected, (
        f"LDS budget/allocation drift: _lds_bytes says {expected} B, the emitted "
        f"addrspace(3) pool is {actual} B (delta {actual - expected}). The budget is "
        f"what supports_attention_dense gates on, so an under-count admits a config "
        f"that overflows LDS at compile time"
    )
    assert actual <= _CAPACITY, (
        f"supports_attention_dense accepted a config whose emitted LDS pool is "
        f"{actual} B, past the gfx942 capacity of {_CAPACITY} B"
    )


def test_cohort_is_not_vacuous():
    """Anti-vacuity for the two tests above.

    Both are trivially true over an empty or degenerate cohort, so pin that the cohort
    really spans the axes it claims: both head sizes, both dtypes, several block_n,
    both cfvst states, non-default values of BOTH pads, and enough distinct footprints
    that the equality is not being satisfied by one constant.
    """
    assert len(_COHORT) >= 60, f"cohort collapsed to {len(_COHORT)} cases"
    specs = [p.values[0] for p in _COHORT]
    assert {s.head_size for s in specs} == {64, 128}
    assert {s.dtype for s in specs} == {"bf16", "fp16"}
    assert {s.block_n for s in specs} >= {32, 64, 128}
    assert {s.resolved_use_cfvst() for s in specs} == {False, True}
    assert {s.lds_row_pad for s in specs} > {_DEFAULT_LDS_ROW_PAD}
    # v_row_pad is tri-state: None ("ask the policy") is its shipped default.
    assert {s.v_row_pad for s in specs} > {None}
    sizes = {_lds_bytes(s) for s in specs}
    assert len(sizes) >= 10, f"cohort produces only {len(sizes)} distinct footprints"


# --------------------------------------------------------------------------- #
# the specific drift vector: v_row_pad is read in TWO places
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pad", [0, 4, 16, 32])
def test_v_row_pad_moves_budget_and_allocation_by_the_same_amount(pad):
    """``v_row_pad`` is read at exactly two sites -- the ``_lds_bytes`` budget and the
    ``V_lds`` allocation in the builder -- and this pins that they move TOGETHER.

    Delta form rather than absolute, so it stays honest if the rest of the footprint
    changes: on the cfvst path V is ``[D, block_n + v_row_pad]``, so a pad delta of
    ``dp`` must move both numbers by exactly ``D * dp * 2`` bytes and nothing else.
    """
    # The cfvst path: v_row_pad is live. Hold the swizzle off so the pad varies
    # independently: an explicit non-pow2 pad with the swizzle on is rejected (the mask
    # would address out of bounds) -- exactly the pad/swizzle decoupling
    # use_v_swizzle exists for.
    base = _spec(head_size=128, dtype="fp16", v_row_pad=0, use_v_swizzle=False)
    alt = dataclasses.replace(base, v_row_pad=pad)
    assert base.resolved_use_cfvst(), "premise: this config takes the cfvst path"
    expect = base.head_size * pad * 2
    assert _lds_bytes(alt) - _lds_bytes(base) == expect
    assert _lds_pool_bytes(alt) - _lds_pool_bytes(base) == expect


@pytest.mark.parametrize("pad", [0, 16, 32])
def test_v_row_pad_is_inert_on_the_naive_v_path(pad):
    """Counterpart: with cfvst off, V keeps the natural [token, dim] async-DMA layout
    and ``v_row_pad`` must not reach the allocation at all. Without this the pair of
    tests could both pass on a builder that applied the pad unconditionally while the
    budget did the same -- consistent, but wrong for the D64 layout the DMA needs."""
    # naive V (cfvst is policy-off at D64)
    base = _spec(head_size=64, dtype="bf16", v_row_pad=0)
    alt = dataclasses.replace(base, v_row_pad=pad)
    assert not base.resolved_use_cfvst(), "premise: this config is naive-V"
    assert _lds_pool_bytes(alt) == _lds_pool_bytes(base)
    assert _lds_bytes(alt) == _lds_bytes(base)


@pytest.mark.parametrize("pad", [0, 16, 32])
def test_lds_row_pad_is_inert_on_the_packed_d64_path(pad):
    """``lds_row_pad`` is the per-ROW K pad and only D128 (one row per DMA
    instruction) can carry it; D64 packs two rows per instruction and needs a
    contiguous stride, so it pads between row-GROUPS via ``spec.lds_k_group_pad``
    instead. The two are mutually exclusive and merely happen to share the value 8 --
    which is exactly the kind of coincidence a sweep breaks. Pinned on the ALLOCATION,
    not just the model."""
    base = _lds_pool_bytes(_spec(head_size=64, dtype="fp16", lds_row_pad=0))
    alt = _spec(head_size=64, dtype="fp16", lds_row_pad=pad)
    assert _lds_pool_bytes(alt) == base
    assert _lds_bytes(alt) == base


@pytest.mark.parametrize("pad", [0, 8, 16])
def test_k_group_pad_grows_the_allocation_by_one_pad_per_row_group(pad):
    """The D64 K row-group pad reaches the real allocation, one pad per DMA group.

    Mirrors the gfx950 sibling's assertion (``NBUF * (block_n // 2) * pad * 2``) for
    the gfx942 single-buffer body: NBUF=1 here, so the growth over the unpadded layout
    is ``(block_n // rows_per_instr) * pad * 2`` bytes on the K tile only -- V is
    untouched. Asserted against the emitted pool so a builder that computed the group
    stride differently from ``_k_group_stride`` would be caught.
    """
    from kernels.gfx942.attention_dense import _rows_per_instr

    block_n = 64
    base = _lds_pool_bytes(
        _spec(head_size=64, dtype="bf16", block_n=block_n, lds_k_group_pad=0)
    )
    grown = _lds_pool_bytes(
        _spec(head_size=64, dtype="bf16", block_n=block_n, lds_k_group_pad=pad)
    )
    expect = (block_n // _rows_per_instr(64)) * pad * 2
    assert grown - base == expect, (
        f"D64 block_n={block_n} lds_k_group_pad={pad}: LDS grew {grown - base} B, "
        f"expected {expect} (one pad per K row-group, V untouched)"
    )


def test_capacity_boundary_allocation_is_exactly_capacity():
    """The gate is ``> capacity``, so a config landing on EXACTLY 65536 B is accepted.

    bn128 / D128 / bf16 at ``lds_row_pad=0`` is that config: K and V are each
    128*128*2 = 32768 B. Its emitted pool must be exactly the capacity -- if the
    allocation were even one byte larger than the model, this is the case where the
    gate's boundary would admit an overflow. The complementary supports()-side
    assertion lives in ``test_attention_dense_gfx942.py``.
    """
    spec = _spec(block_n=128, head_size=128, dtype="bf16", lds_row_pad=0)
    ok, why = supports_attention_dense(spec, arch="gfx942")
    assert ok, why
    assert _lds_bytes(spec) == _CAPACITY
    assert _lds_pool_bytes(spec) == _CAPACITY
