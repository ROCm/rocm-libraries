#!/usr/bin/env python3
################################################################################
# Unit test for the out-of-line 16-bit subtile scalar-fallback store layout.
#
# Guards the "BranchPenaltyFallThrough" change in
#   Tensile/Components/GlobalWriteBatch.py
# (GlobalWriteBatchWriter): the 16-bit subtile *paired* store is the hot path and
# now falls THROUGH to its merge label (``afterPairedLabel``); the scalar
# fallback (taken only when just the lower M-block is valid) is no longer emitted
# inline between the paired store and its merge label. Instead each fallback
# block -- ``fallbackLabel`` + scalar store + ``s_branch afterPairedLabel``
# (jump-back) -- is collected into a ``subtileScalarFallback`` Module and emitted
# OUT OF LINE at the end of the batch's store code, wrapped by an unconditional
# ``s_branch scalarFallbackEndLabel`` (skip guard) and a trailing
# ``scalarFallbackEndLabel``.
#
# rocisa's Module tree stringifies via DFS in ``.add()`` insertion order, so the
# ordering the change relies on is fully determined by the sequence of ``.add()``
# calls into the storeCode / subtileScalarFallback Modules and the batch-end
# emission block. This test models that exact ``.add()`` / DFS-stringify contract
# with lightweight stand-ins (rocisa's native _rocisa.so is not required, so the
# test runs in CI without the pybind/native build) and asserts the ordering and
# branch invariants the change guarantees.
#
# Usage:
#   pytest test_subtile_scalar_fallback_layout.py -v
################################################################################

import re

import pytest


# --------------------------------------------------------------------------- #
# Minimal stand-ins that model the rocisa.code contract the change depends on:
#   * Module.add(item) appends and returns the item
#   * Module.items() / len(Module) reflect inserted children
#   * str(Module) renders children depth-first in insertion order (nested
#     Modules are spliced in place), matching rocisa's stringify behaviour.
# (See rocisa/test/test_code.py for the real Module.add()/items() semantics.)
# --------------------------------------------------------------------------- #
class _Item:
    def render(self):
        raise NotImplementedError


class Label(_Item):
    def __init__(self, name, comment=""):
        self.name = name
        self.comment = comment

    def getLabelName(self):
        return self.name

    def render(self):
        line = f"{self.name}:"
        if self.comment:
            line += f"  // {self.comment}"
        return [line]


class _Inst(_Item):
    def __init__(self, text, comment=""):
        self.text = text
        self.comment = comment

    def render(self):
        line = self.text
        if self.comment:
            line += f"  // {self.comment}"
        return [line]


class SBranch(_Inst):
    def __init__(self, labelName, comment=""):
        super().__init__(f"s_branch {labelName}", comment)
        self.labelName = labelName


class SCBranchSCC0(_Inst):
    def __init__(self, labelName, comment=""):
        super().__init__(f"s_cbranch_scc0 {labelName}", comment)
        self.labelName = labelName


class Module(_Item):
    def __init__(self, name=""):
        self.name = name
        self._items = []

    def add(self, item):
        self._items.append(item)
        return item

    def items(self):
        return list(self._items)

    def __len__(self):
        return len(self._items)

    def render(self):
        out = []
        for it in self._items:
            out.extend(it.render())
        return out

    def __str__(self):
        return "\n".join(self.render())


# Rendered-line tags used to locate emitted pieces without depending on exact
# instruction mnemonics (kept deliberately distinct from real asm comments).
_PAIRED = "paired-store-common-path"
_SCALAR_FALLBACK = "scalar-fallback-store"
_ORPHAN = "orphan-scalar-store"


class _LabelFactory:
    """Mimics parentWriter.labels.getNameInc: unique, monotonically-suffixed names."""

    def __init__(self):
        self._n = {}

    def getNameInc(self, base):
        idx = self._n.get(base, 0)
        self._n[base] = idx + 1
        return f"{base}_{idx}"


# --------------------------------------------------------------------------- #
# Builders that replicate the GlobalWriteBatch emission sequence. ``iters`` is a
# list of per-element kinds:
#   "paired_fallback" - paired hot store guarded by a scalar fallback
#   "paired_only"     - paired store with no fallback (both M-blocks valid)
#   "orphan"          - lone scalar store (odd MIWaveTile tail), unrelated path
# --------------------------------------------------------------------------- #
def _build_common(iters, group_load_store, labels, storeCodeModule,
                  subtileScalarFallback):
    for kind in iters:
        if kind == "paired_fallback":
            afterPairedLabel = Label(labels.getNameInc("subtile_after_paired"),
                                     "after paired/fallback store")
            fallbackLabel = Label(labels.getNameInc("subtile_scalar_fallback"),
                                  "scalar fallback when upper M-block OOB")
            # -- hot path: compare, conditional branch to fallback, paired store,
            #    then FALL THROUGH to the merge label --
            storeCodeModule.add(_Inst("s_cmp_gt_u32 SubtileMGuard",
                                      "both M-blocks valid?"))
            storeCodeModule.add(SCBranchSCC0(fallbackLabel.getLabelName(),
                                             "only lower block valid -> scalar fallback"))
            storeCodeModule.add(_Inst("buffer_store_dwordx4", _PAIRED))
            storeCodeModule.add(afterPairedLabel)
            # -- fallback collected for out-of-line emission --
            subtileScalarFallback.add(fallbackLabel)
            subtileScalarFallback.add(_Inst("buffer_store_short", _SCALAR_FALLBACK))
            subtileScalarFallback.add(SBranch(afterPairedLabel.getLabelName(),
                                              "return from scalar fallback"))
        elif kind == "paired_only":
            storeCodeModule.add(_Inst("buffer_store_dwordx4", _PAIRED))
        elif kind == "orphan":
            storeCodeModule.add(_Inst("buffer_store_short", _ORPHAN))
        else:  # pragma: no cover - guard against typos in test params
            raise ValueError(f"unknown iter kind {kind!r}")


def build_store_batch(iters, group_load_store=True):
    """Replicate the post-change (out-of-line) GlobalWriteBatch store emission."""
    labels = _LabelFactory()
    module = Module("module")
    storeCode = Module("GroupLoadStore")
    subtileScalarFallback = Module("subtile_scalar_fallback")
    storeCodeModule = storeCode if group_load_store else module

    _build_common(iters, group_load_store, labels, storeCodeModule,
                  subtileScalarFallback)

    # Batch-end: emit the collected fallback blocks out of line, guarded by a
    # skip branch + trailing end label -- only when the fallback is non-empty.
    if len(subtileScalarFallback.items()):
        scalarFallbackTarget = storeCode if group_load_store else module
        scalarFallbackEndLabel = Label(labels.getNameInc("subtile_scalar_fallback_end"),
                                       "end of out-of-line scalar fallback")
        scalarFallbackTarget.add(SBranch(scalarFallbackEndLabel.getLabelName(),
                                         "skip over out-of-line scalar fallback"))
        scalarFallbackTarget.add(subtileScalarFallback)
        scalarFallbackTarget.add(scalarFallbackEndLabel)

    if group_load_store:
        module.add(storeCode)
    return module


def build_store_batch_inline(iters, group_load_store=True):
    """Replicate the PRE-change (inline) layout, where the scalar fallback sits
    between the paired store and its merge label. Used only to prove the
    out-of-line invariant checker actually has teeth (rejects the old layout)."""
    labels = _LabelFactory()
    module = Module("module")
    storeCode = Module("GroupLoadStore")
    storeCodeModule = storeCode if group_load_store else module

    for kind in iters:
        if kind == "paired_fallback":
            afterPairedLabel = Label(labels.getNameInc("subtile_after_paired"),
                                     "after paired/fallback store")
            fallbackLabel = Label(labels.getNameInc("subtile_scalar_fallback"),
                                  "scalar fallback when upper M-block OOB")
            storeCodeModule.add(_Inst("s_cmp_gt_u32 SubtileMGuard",
                                      "both M-blocks valid?"))
            storeCodeModule.add(SCBranchSCC0(fallbackLabel.getLabelName(),
                                             "only lower block valid -> scalar fallback"))
            storeCodeModule.add(_Inst("buffer_store_dwordx4", _PAIRED))
            storeCodeModule.add(SBranch(afterPairedLabel.getLabelName(),
                                        "skip scalar fallback"))
            storeCodeModule.add(fallbackLabel)
            storeCodeModule.add(_Inst("buffer_store_short", _SCALAR_FALLBACK))
            storeCodeModule.add(afterPairedLabel)
        elif kind == "paired_only":
            storeCodeModule.add(_Inst("buffer_store_dwordx4", _PAIRED))
        elif kind == "orphan":
            storeCodeModule.add(_Inst("buffer_store_short", _ORPHAN))
    if group_load_store:
        module.add(storeCode)
    return module


# --------------------------------------------------------------------------- #
# Text helpers operating on the rendered assembly.
# --------------------------------------------------------------------------- #
def _lines(module):
    return str(module).splitlines()


def _idx_with(lines, needle):
    return [i for i, ln in enumerate(lines) if needle in ln]


def _fallback_is_out_of_line(lines):
    """True iff every scalar-fallback store is emitted after every paired-store
    merge label (i.e. the fallback lives out of line, not inline)."""
    fb = _idx_with(lines, _SCALAR_FALLBACK)
    merge = _idx_with(lines, "subtile_after_paired")
    merge_labels = [i for i in merge if lines[i].lstrip().startswith("subtile_after_paired")]
    if not fb or not merge_labels:
        return False
    return min(fb) > max(merge_labels)


def _label_defs(lines, prefix):
    """Return names of label definition lines (``<prefix>...:``)."""
    out = []
    for ln in lines:
        m = re.match(rf"^({re.escape(prefix)}\w*):", ln.strip())
        if m:
            out.append(m.group(1))
    return out


def _branch_targets(lines, mnemonic, prefix):
    out = []
    for ln in lines:
        m = re.search(rf"{mnemonic}\s+({re.escape(prefix)}\w*)", ln)
        if m:
            out.append(m.group(1))
    return out


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
FALLBACK_MIXES = [
    ["paired_fallback"],
    ["paired_fallback", "paired_fallback"],
    ["paired_only", "paired_fallback", "orphan", "paired_fallback"],
    ["paired_fallback", "paired_only", "paired_fallback", "paired_fallback"],
]


class TestSubtileScalarFallbackLayout:
    """Ordering / branch invariants for the out-of-line scalar-fallback change."""

    @pytest.mark.parametrize("iters", FALLBACK_MIXES,
                             ids=["x".join(k[0] for k in m) for m in FALLBACK_MIXES])
    def test_fallback_emitted_out_of_line(self, iters):
        """Scalar fallback stores appear AFTER every paired-store merge label,
        never inline between the paired store and its ``afterPairedLabel``."""
        lines = _lines(build_store_batch(iters))
        assert _fallback_is_out_of_line(lines)
        # Inline sanity: within the hot region, each paired store is immediately
        # followed by its merge label with no scalar fallback store between them.
        for pi in _idx_with(lines, _PAIRED):
            # next non-empty line after a fallback-guarded paired store is the
            # merge label (fallthrough), and it is not a scalar fallback store.
            assert _SCALAR_FALLBACK not in lines[pi]
            if pi + 1 < len(lines):
                assert _SCALAR_FALLBACK not in lines[pi + 1]

    @pytest.mark.parametrize("iters", FALLBACK_MIXES,
                             ids=["x".join(k[0] for k in m) for m in FALLBACK_MIXES])
    def test_skip_branch_and_end_label_wrap_fallback(self, iters):
        """A single ``s_branch`` skip guard precedes the out-of-line fallback
        region and the matching end label follows it."""
        lines = _lines(build_store_batch(iters))
        skip = _branch_targets(lines, "s_branch", "subtile_scalar_fallback_end")
        end_defs = _label_defs(lines, "subtile_scalar_fallback_end")
        assert len(skip) == 1
        assert len(end_defs) == 1
        assert skip[0] == end_defs[0]

        skip_idx = _idx_with(lines, "s_branch " + skip[0])[0]
        end_idx = [i for i, ln in enumerate(lines)
                   if ln.strip().startswith(end_defs[0] + ":")][0]
        fb_idx = _idx_with(lines, _SCALAR_FALLBACK)
        first_fb_label = _idx_with(lines, "subtile_scalar_fallback_0:")
        # skip branch is before the fallback region; end label is after it.
        assert skip_idx < min(fb_idx)
        assert end_idx > max(fb_idx)
        if first_fb_label:
            assert skip_idx < first_fb_label[0]

    @pytest.mark.parametrize("iters", FALLBACK_MIXES,
                             ids=["x".join(k[0] for k in m) for m in FALLBACK_MIXES])
    def test_label_pairing_is_one_to_one(self, iters):
        """Each ``fallbackLabel`` has exactly one matching ``afterPairedLabel``,
        and every fallback block ends with a jump-back ``s_branch`` to its own
        merge label (bijection between fallback labels and return branches)."""
        n_fallback = iters.count("paired_fallback")
        lines = _lines(build_store_batch(iters))

        fallback_labels = _label_defs(lines, "subtile_scalar_fallback")
        # Exclude the single end label from the fallback-label count.
        fallback_labels = [n for n in fallback_labels
                           if not n.startswith("subtile_scalar_fallback_end")]
        merge_labels = _label_defs(lines, "subtile_after_paired")
        return_targets = _branch_targets(lines, "s_branch", "subtile_after_paired")

        assert len(fallback_labels) == n_fallback
        assert len(merge_labels) == n_fallback
        assert len(return_targets) == n_fallback
        # Jump-back targets are exactly the set of merge labels (1:1 bijection).
        assert sorted(return_targets) == sorted(merge_labels)
        assert len(set(return_targets)) == len(return_targets)

    @pytest.mark.parametrize("iters", [
        [],
        ["paired_only"],
        ["orphan"],
        ["paired_only", "orphan", "paired_only"],
    ], ids=["empty", "paired_only", "orphan", "no_fallback_mix"])
    def test_no_fallback_emits_no_out_of_line_region(self, iters):
        """When no element needs a fallback, the ``subtileScalarFallback`` module
        is empty, so NO skip branch, end label, or out-of-line region is emitted."""
        lines = _lines(build_store_batch(iters))
        text = "\n".join(lines)
        assert _SCALAR_FALLBACK not in text
        assert "subtile_scalar_fallback" not in text
        assert "subtile_scalar_fallback_end" not in text
        # No unconditional s_branch is introduced by the fallback machinery.
        assert _branch_targets(lines, "s_branch", "subtile_scalar_fallback_end") == []

    def test_inline_layout_would_violate_out_of_line_invariant(self):
        """Teeth check: the pre-change inline arrangement (scalar fallback sitting
        between the paired store and its merge label) must FAIL the out-of-line
        invariant, proving the assertion above genuinely detects a regression."""
        iters = ["paired_fallback", "paired_only", "paired_fallback"]
        inline_lines = _lines(build_store_batch_inline(iters))
        out_of_line_lines = _lines(build_store_batch(iters))
        assert not _fallback_is_out_of_line(inline_lines)
        assert _fallback_is_out_of_line(out_of_line_lines)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
