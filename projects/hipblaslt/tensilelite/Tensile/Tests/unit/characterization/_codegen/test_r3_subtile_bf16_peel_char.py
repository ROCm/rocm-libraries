################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Characterization for the bf16 UseSubtileImpl epilogue PEEL (gfx950, wave64).

Guards the 16bit subtile store fast-path in ``Tensile/Components/GlobalWriteBatch.py``
(gated by ``is16bitSubtile`` = UseSubtileImpl + non-edge + bf16/fp16 dest + HPA + wave64):

  Interior store peel: a guard-free/mask-free interior body plus a guarded boundary body
            (``label_subtile_peel_boundary`` / ``_peel_end``).
  Routing — the N-group skip label (``label_subtile_skip_store_N<n>_end``) must be emitted
            into the SAME module as the stores, so the per-element M-guard
            ``s_cbranch_scc0`` targeting it is a FORWARD branch. If it is routed to the
            outer module instead it is emitted *before* the stores, the branch becomes
            BACKWARD, and ragged-M tiles hang / scramble control flow.

Target: Tensile/Components/GlobalWriteBatch.py, the ranges this change adds:
  2304-2355  peel emit — interior test, boundary/end labels, and the selection between
             the interior body and the guarded body
  2465-2476  ``_flushNGroupSkipLabel`` — the routing helper shared by the three emit sites
  2683-2750  ``_buildSubtileInteriorStores`` — the guard-free/mask-free interior body

The routing check is deliberately an *ordering* assertion, not a presence assertion: under
the routing bug the label is still defined, just in the wrong place, so "is it defined"
cannot detect the regression.

Assertions are on instruction/label structure (opcodes, label names, SGPR names) rather
than on emitted comment prose, so rewording a comment does not break CI.

CPU-only — no GPU, no compile, no hardware access.
"""

import os
import re

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

# 3 MI shapes x StreamK{0,5} x PGR{1,2} = 12 fork permutations.
_LIMIT = 12

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "subtile_bf16_peel.yaml",
)

# Structural signature of the 16bit paired subtile store (the path both levers hang off).
# These opcodes are emitted by the ds_bpermute partner-lane packed store; they exist on the
# pre-peel base too, so they only SELECT the relevant kernels -- the peel-specific
# assertions below are what actually guard this change.
_PAIRED_STORE_OPS = (
    "v_permlane32_swap_b32",
    "v_permlane16_swap_b32",
    "ds_bpermute_b32",
)

_LABEL_DEF_RE = re.compile(r"\s*(label_\w+):")
_BRANCH_RE = re.compile(r"\b(s_cbranch_scc[01]|s_branch)\s+(label_\w+)\b")

_NGROUP_LABEL_RE = re.compile(r"label_subtile_skip_store_N\d+_end(?:_\d+)?")
_PEEL_BOUNDARY_RE = re.compile(r"label_subtile_peel_boundary(?:_\d+)?")
_PEEL_END_RE = re.compile(r"label_subtile_peel_end(?:_\d+)?")

# --- fixtures (emit once per module; the emit pipeline is the expensive part) ----------


@pytest.fixture(scope="module")
def emitted():
    """[(basename, src, err), ...] for the whole designed fork sweep."""
    return emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)


@pytest.fixture(scope="module")
def peel_kernels(emitted):
    """Every emitted kernel that took the 16bit subtile paired-store path."""
    hits = [(b, s) for (b, s, _e) in emitted if all(op in s for op in _PAIRED_STORE_OPS)]
    assert hits, (
        "no kernel took the 16bit subtile paired-store path -- is16bitSubtile never "
        f"engaged (config: {_CONFIG}); emitted: {[b for b, _s, _e in emitted]}"
    )
    return hits


# --- helpers --------------------------------------------------------------------------


def _label_defs(src):
    """label name -> [line indices where it is defined]."""
    out = {}
    for i, line in enumerate(src.splitlines()):
        m = _LABEL_DEF_RE.match(line)
        if m:
            out.setdefault(m.group(1), []).append(i)
    return out


def _branch_sites(src):
    """label name -> [(line index, branch opcode)] for every branch targeting it."""
    out = {}
    for i, line in enumerate(src.splitlines()):
        m = _BRANCH_RE.search(line)
        if m:
            out.setdefault(m.group(2), []).append((i, m.group(1)))
    return out


def _matching(labels, pattern):
    return sorted(l for l in labels if pattern.fullmatch(l))


def _assert_forward_branches(base, src, pattern, what, require=True):
    """Every label matching `pattern` is defined exactly once, is branched to, and all
    branches to it appear BEFORE its definition (i.e. forward branches only)."""
    defs, branches = _label_defs(src), _branch_sites(src)
    targets = _matching(defs, pattern)
    if require:
        assert targets, f"{base}: {what}: no label matching {pattern.pattern} was emitted"
    for tgt in targets:
        assert len(defs[tgt]) == 1, (
            f"{base}: {what}: label {tgt} defined {len(defs[tgt])} times (lines {defs[tgt]})"
        )
        def_line = defs[tgt][0]
        sites = branches.get(tgt, [])
        assert sites, f"{base}: {what}: label {tgt} is defined but never branched to (dead label)"
        backward = [(i, op) for (i, op) in sites if i >= def_line]
        assert not backward, (
            f"{base}: {what}: BACKWARD branch to {tgt} -- label defined at line {def_line} but "
            f"branched from {backward}. The label is not co-located with the stores, so the "
            f"M-guard branch jumps backwards (ragged-M hang; Stage-14 routing regression)."
        )
    return targets


# --- tests ----------------------------------------------------------------------------


def test_peel_config_emits_assembly(emitted):
    """The designed fork sweep emits real gfx950 assembly with err==0."""
    assert len(emitted) >= 1, f"expected >=1 kernel, got 0 (config: {_CONFIG})"
    failed = [(b, e) for (b, _s, e) in emitted if e != 0]
    assert not failed, f"some kernels failed to emit: {failed}"
    for base, src, _err in emitted:
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"


def test_paired_store_path_engaged(emitted, peel_kernels):
    """is16bitSubtile engages across the sweep (not just one pinned geometry)."""
    assert len(peel_kernels) >= 2, (
        "expected the 16bit subtile store on >=2 kernels of the sweep (varying "
        f"MIWaveTile / StreamK), got {len(peel_kernels)} of {len(emitted)}"
    )


def test_interior_store_peel(peel_kernels):
    """Guarded boundary body + interior body that branches over it."""
    for base, src in peel_kernels:
        boundary = _assert_forward_branches(
            base, src, _PEEL_BOUNDARY_RE, "interior store peel"
        )
        end = _assert_forward_branches(
            base, src, _PEEL_END_RE, "peel end"
        )
        # The interior body must reach the boundary body via a conditional guard branch,
        # and skip it via an unconditional s_branch to the peel-end label.
        branches = _branch_sites(src)
        for tgt in boundary:
            ops = {op for _i, op in branches[tgt]}
            assert "s_cbranch_scc0" in ops, (
                f"{base}: expected s_cbranch_scc0 -> {tgt} (interior guard failed -> "
                f"guarded boundary body), got {sorted(ops)}"
            )
        for tgt in end:
            ops = {op for _i, op in branches[tgt]}
            assert "s_branch" in ops, (
                f"{base}: expected unconditional s_branch -> {tgt} (interior stores done -> "
                f"skip guarded boundary body), got {sorted(ops)}"
            )


def test_ngroup_skip_label_is_colocated_with_stores(peel_kernels):
    """Routing fix: the N-group skip label must be a FORWARD branch target.

    This is the assertion that actually guards commit "co-locate bf16 subtile N-group skip
    label with stores". Under the bug the label is still defined, but emitted into the outer
    module ahead of the stores, making every per-element M-guard branch backward.
    """
    checked = 0
    for base, src in peel_kernels:
        targets = _assert_forward_branches(
            base, src, _NGROUP_LABEL_RE, "N-group skip label routing", require=False
        )
        for tgt in targets:
            ops = {op for _i, op in _branch_sites(src)[tgt]}
            assert "s_cbranch_scc0" in ops, (
                f"{base}: expected the per-element M-guard s_cbranch_scc0 -> {tgt}, "
                f"got {sorted(ops)}"
            )
        checked += len(targets)
    assert checked >= 1, (
        "no N-group skip label was emitted anywhere in the sweep, so the routing invariant "
        "was never exercised -- the guarded (N-guard) store path did not engage"
    )


def test_r3_subtile_bf16_peel_gfx950_golden(emitted, snapshot):
    """Order-invariant golden: pin {basename, err} for every kernel of the fork sweep.

    Catches a fork permutation silently dropping out of the sweep, which would leave the
    structural assertions above passing over a smaller set. The assembly TEXT is
    deliberately not hashed: it is order-coupled through process-global rocisa scheduler
    state (see characterization/README.md).
    """
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in emitted),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
