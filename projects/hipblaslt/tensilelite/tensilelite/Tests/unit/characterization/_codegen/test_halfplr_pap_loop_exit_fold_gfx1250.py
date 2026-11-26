# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HalfPLR PAP shares one per-trip test between the loop exit and the PAP entry."""

import re

import pytest
from config_harness import assert_assembles, assert_real_gfx1250_kernels, emit_kernels_from_config
from test_pap_pipeline_overlap_gfx1250 import _config
from tensilelite.Components.PersistentLoop import PersistentKernelState

pytestmark = pytest.mark.unit

_LC = "s[sgprLoopCounterL]"


def _code(line):
    return line.split("//", 1)[0].strip()


def assert_folded_loop_exit(src):
    lines = [_code(line) for line in src.splitlines()]

    def label(name):
        return lines.index("label_" + name + ":")

    lo, hi = label("openLoopL"), label("LoopEndL")
    loop = lines[lo:hi]
    entries = [(i, int(m[1])) for i, line in enumerate(loop)
               if (m := re.match(r"s_cbranch_scc1 label_HalfPlrPAPEntry_(\d+)$", line))]
    # The packed carrier tests counters two and one separately at the unfolded sites.
    copies = sorted({copy for _, copy in entries})
    assert copies == list(range(len(copies))) and len(copies) >= 3, copies
    final = copies[-1]
    bound = None
    for i, copy in entries:
        if copy in (0, final):
            continue
        # The test the branch reads is the last compare above it; nothing else writes SCC between.
        test = next(line for line in reversed(loop[:i]) if re.match(r"s_(cmp|bitcmp)", line))
        m = re.fullmatch(r"s_cmp_le_u32 %s, (\d+)" % re.escape(_LC), test)
        assert m, test
        bound = bound or int(m[1])
        assert int(m[1]) == bound
        assert loop[i + 1] == "label_ReturnFromHalfPlrPAP_%u:" % copy
        assert not loop[i + 2].startswith("s_cmp_eq_i32 %s" % _LC), "exit test left on the per-trip path"
        stub = lines[label("HalfPlrPAPEntry_%u" % copy) + 1:][:2]
        assert stub == ["s_cmp_eq_u32 %s, 0" % _LC, "s_cbranch_scc1 label_LoopEndL"], stub

    # Only the final copy keeps an exit test, after its hot back edge.
    exits = [i for i, line in enumerate(loop) if line.startswith("s_cmp_eq_i32 %s, 0x0" % _LC)]
    assert len(exits) == 1
    back = next(i for i, line in enumerate(loop) if line == "s_cbranch_scc1 label_LoopBeginL")
    test = next(line for line in reversed(loop[:back]) if re.match(r"s_(cmp|bitcmp)", line))
    assert test == "s_cmp_gt_u32 %s, %u" % (_LC, bound), test
    finalEntries = [i for i, copy in entries if copy == final]
    assert back < min(finalEntries) and max(finalEntries) < exits[0]
    assert loop[exits[0] + 1] == "s_cbranch_scc0 label_LoopBeginL"
    finalStub = lines[label("HalfPlrPAPEntry_%u" % final) + 1:label("HalfPlrPAPEntry_%u" % final) + 4]
    assert not any("sgprLoopCounterL" in line for line in finalStub), finalStub

    # The last-trip increment mask runs once, on the issue path of the shared block.
    raw = src.splitlines()
    assert not any("sgprPersistentPrefetchState" in line for line in loop)
    assert not any(re.match(r"s_cmp_le_i32 \S+, 1$", line) for line in loop)
    masks = [i for i, line in enumerate(raw) if "no increment on the last iters unless primed" in line]
    assert masks
    issue = label("HalfPlrPAPIssue") if "label_HalfPlrPAPIssue:" in lines else label("HalfPlrPrefetchAcrossPersistent")
    end = label("HalfPlrPAPPreparedReturn") if "label_HalfPlrPAPPreparedReturn:" in lines else len(lines)
    assert all(issue < i < end for i in masks), (issue, masks, end)
    test = next(line for line in reversed(lines[:masks[0]]) if re.match(r"s_(cmp|bitcmp)", line))
    assert test == "s_bitcmp1_b32 s[sgprPersistentPrefetchState], 0"
    assert not any(re.match(r"s_(cmp|bitcmp)", line) for line in lines[masks[0]:masks[-1]])
    return bound


@pytest.mark.parametrize("strategy", ["DataParallel", "StreamK"])
@pytest.mark.parametrize("carrier", [True, False])
def test_halfplr_pap_folds_loop_exit_into_entry_test(tmp_path, monkeypatch, strategy, carrier):
    overrides = dict(HalfPLR=1, SuppressNoLoadLoop=True, TileProcessingStrategy=strategy,
                     AssertSummationElementMultiple=256)
    if not carrier:
        monkeypatch.setattr(PersistentKernelState, "papAddressCarrierPlan", lambda *args, **kwargs: None)
    results = emit_kernels_from_config(_config(tmp_path, **overrides), limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        prepared = "HalfPLR PAP shared dispatch prepares only at counter two" in src
        assert assert_folded_loop_exit(src) == (2 if prepared else 1)
        if not carrier:
            assert not prepared


@pytest.mark.parametrize("halfplr,pap", [(1, 0), (0, 1)])
def test_halfplr_loop_exit_unchanged_without_folding(tmp_path, halfplr, pap):
    """Without HalfPLR PAP every copy keeps its own exit test."""
    config = _config(tmp_path, HalfPLR=halfplr, SuppressNoLoadLoop=bool(halfplr), PrefetchAcrossPersistent=pap)
    results = emit_kernels_from_config(config, limit=1, arch="gfx1250")
    assert_real_gfx1250_kernels(results)
    for name, src, _ in results:
        assert_assembles(src, name)
        assert "folded loop-exit test" not in src
        assert "last trips leave the loop body" not in src
        assert re.search(r"s_cmp_eq_i32 s\[sgprLoopCounterL\], 0x\d+\s+// counterL==", src)
