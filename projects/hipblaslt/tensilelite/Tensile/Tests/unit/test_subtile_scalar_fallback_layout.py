#!/usr/bin/env python3
################################################################################
# Guards the out-of-line 16-bit subtile scalar-fallback store layout
# ("BranchPenaltyFallThrough") in Components/GlobalWriteBatch.py: the paired
# dwordx store falls through to its merge label, and each scalar fallback block
# is emitted out of line at the end of the batch, bracketed by a skip s_branch
# and an end label. Drives the real KernelWriterAssembly.notLocalSplitUGlobalWrite
# emission (CPU-only, no GPU) and asserts the ordering/branch invariants.
#
# Usage:
#   pytest test_subtile_scalar_fallback_layout.py -v
################################################################################

import os
import re
import shutil
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, TENSILE_ROOT)

# Reuse the proven real-writer scaffolding from the store-D roundtrip suite
# (CPU-only builders; only its assemble/run paths need a GPU, which we skip).
import test_storeD_roundtrip as _storeD
from gpu_test_helpers import create_writer, TileConfig, WAVESIZE
from Tensile.KernelWriterModules import mapAcctoArchRegs

_VERSION = (9, 5, 0)


def _init_rocisa():
    from rocisa import rocIsa
    ri = rocIsa.getInstance()
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(_VERSION, asmpath)
    ri.setKernel(_VERSION, WAVESIZE)


def _emit_store_asm(cfg, mi_wave_group, use_bf16):
    """Emit the real notLocalSplitUGlobalWrite store module for a subtile kernel
    and return its rendered assembly text."""
    _init_rocisa()
    kernel = _storeD._build_store_kernel(cfg, mi_wave_group=mi_wave_group, use_bf16=use_bf16)
    kernel["UseSubtileImpl"] = True

    writer, _, _, _ = create_writer(cfg, mi_wave_group=mi_wave_group)
    sgprs = _storeD._build_sgprs_for_beta_test(writer)
    tileInfoD, _ = _storeD._allocate_d_tile(kernel, writer)
    kw = _storeD._build_kwa(kernel, writer, use_bf16=use_bf16)
    kw.states.d.tileInfo = tileInfoD

    # Wire the subtile M/N guard SGPRs so the 16-bit paired-store scalar-fallback
    # path is exercised (mirrors KernelWriter's real allocation).
    kw.states.subtileM32ValidBlocksSgpr = sgprs["subtileMValidBlocks"]
    kw.states.subtileN16ValidBlocksSgpr = sgprs["subtileNValidBlocks"]
    kw.sgprs["SubtileMGuard"] = sgprs["subtileMValidBlocks"]
    kw.sgprs["SubtileNGuard"] = sgprs["subtileNValidBlocks"]
    kw.states.subtileMBlockSize = 16

    kw.codes.accVgprRead = mapAcctoArchRegs(kernel, kw.states.maxLimitAgprs, write=False)
    kw.notLocalSplitUGlobalWriteIndices(kernel)
    kw.states.c.startVgprValu = 0
    store_write_mod, _ = kw.notLocalSplitUGlobalWrite(kernel, tPA=None, tPB=None)
    return str(store_write_mod)


# --- Emitted-assembly parsing helpers ------------------------------------- #
_FALLBACK = "label_subtile_scalar_fallback"
_END = "label_subtile_scalar_fallback_end"
_MERGE = "label_subtile_after_paired"


def _label_defs(lines):
    defs = {}
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(label_[A-Za-z0-9_]+):", ln)
        if m:
            defs.setdefault(m.group(1), i)
    return defs


def _branches(lines, mnemonic):
    out = []
    for i, ln in enumerate(lines):
        m = re.search(rf"\b{mnemonic}\s+(label_[A-Za-z0-9_]+)\b", ln)
        if m:
            out.append((i, m.group(1)))
    return out


def _is_fallback(name):
    return name.startswith(_FALLBACK) and not name.startswith(_END)


# Fallback-producing kernels: 16-bit dest, UseSubtileImpl, even MIWaveTile[0]
# (so paired even/odd tt0 stores exist), guard wired -> scalar fallback emitted.
FALLBACK_CONFIGS = [
    (TileConfig(mt_a=128, mt_b=128, depth_u=64), [2, 2]),
    (TileConfig(mt_a=256, mt_b=128, depth_u=64), [2, 2]),
    (TileConfig(mt_a=128, mt_b=256, depth_u=64), [2, 2]),
]
_FB_IDS = [c.label for c, _ in FALLBACK_CONFIGS]


class TestSubtileScalarFallbackLayout:
    """Out-of-line layout / branch invariants for the real GlobalWriteBatch emission."""

    @pytest.mark.parametrize("cfg,wg", FALLBACK_CONFIGS, ids=_FB_IDS)
    def test_fallback_blocks_emitted_out_of_line(self, cfg, wg):
        """Every scalar-fallback block is emitted after (out of line from) the
        merge label it jumps back to, never inline before it."""
        lines = _emit_store_asm(cfg, wg, use_bf16=True).splitlines()
        defs = _label_defs(lines)
        jumpbacks = {i: t for i, t in _branches(lines, "s_branch") if t.startswith(_MERGE)}
        fb_defs = {n: i for n, i in defs.items() if _is_fallback(n)}
        assert fb_defs, "expected at least one scalar-fallback block"
        assert jumpbacks, "each out-of-line block must jump back to its merge label"
        for name, fb_idx in fb_defs.items():
            # The jump-back branch that closes this block names its merge label;
            # the merge label must already be defined (fell through) before it.
            merge = next(t for i, t in jumpbacks.items() if i > fb_idx)
            assert defs[merge] < fb_idx, f"{name} not out of line vs {merge}"

    @pytest.mark.parametrize("cfg,wg", FALLBACK_CONFIGS, ids=_FB_IDS)
    def test_skip_branch_and_end_label_bracket_region(self, cfg, wg):
        """Each out-of-line region is bracketed by one skip s_branch and its end
        label, contains only fallback blocks (no hot-path merge labels)."""
        lines = _emit_store_asm(cfg, wg, use_bf16=True).splitlines()
        defs = _label_defs(lines)
        end_defs = {n: i for n, i in defs.items() if n.startswith(_END)}
        skips = [(i, t) for i, t in _branches(lines, "s_branch") if t.startswith(_END)]
        fb_idx = [i for n, i in defs.items() if _is_fallback(n)]
        merge_idx = [i for n, i in defs.items() if n.startswith(_MERGE)]

        assert end_defs, "expected at least one out-of-line region (end label)"
        assert len(skips) == len(end_defs)
        for end, e_idx in end_defs.items():
            matching = [i for i, t in skips if t == end]
            assert len(matching) == 1
            s_idx = matching[0]
            assert s_idx < e_idx
            assert any(s_idx < i < e_idx for i in fb_idx)
            assert not any(s_idx < i < e_idx for i in merge_idx)

    @pytest.mark.parametrize("cfg,wg", FALLBACK_CONFIGS, ids=_FB_IDS)
    def test_label_pairing_is_one_to_one(self, cfg, wg):
        """One conditional entry, one fallback block, and one jump-back branch per
        merge label (bijection between fallback labels and their merge labels)."""
        lines = _emit_store_asm(cfg, wg, use_bf16=True).splitlines()
        defs = _label_defs(lines)
        fb_defs = {n for n in defs if _is_fallback(n)}
        merge_defs = {n for n in defs if n.startswith(_MERGE)}
        entries = [t for _, t in _branches(lines, "s_cbranch_scc0") if _is_fallback(t)]
        jumpbacks = [t for _, t in _branches(lines, "s_branch") if t.startswith(_MERGE)]

        assert len(entries) == len(fb_defs) == len(jumpbacks) == len(merge_defs)
        assert set(entries) == fb_defs
        assert len(set(jumpbacks)) == len(jumpbacks)
        assert set(jumpbacks) == merge_defs

    @pytest.mark.parametrize("cfg,wg,use_bf16", [
        (TileConfig(mt_a=128, mt_b=128, depth_u=64), [2, 2], False),  # f32: no 16-bit fallback
        (TileConfig(mt_a=32, mt_b=128, depth_u=64), [2, 2], True),    # bf16, MIWaveTile[0]=1: no pairs
    ], ids=["f32", "bf16_no_pairs"])
    def test_no_out_of_line_region_without_fallback(self, cfg, wg, use_bf16):
        """No paired fallback -> no scalar-fallback labels, no end label, no skip branch."""
        asm = _emit_store_asm(cfg, wg, use_bf16=use_bf16)
        assert _FALLBACK not in asm
        assert _END not in asm


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
