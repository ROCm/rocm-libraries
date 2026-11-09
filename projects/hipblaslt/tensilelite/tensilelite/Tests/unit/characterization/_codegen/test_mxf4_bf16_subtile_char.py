# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXF4->bf16 subtile characterization: PostLoopStoreInNll + block scheduling.

Drives ``data/_designed/gfx950/mxf4_bf16_subtile.yaml`` through the config-driven
emit harness. The config has two problem groups that differ only in
``DestDataType``, which is what makes them a scoping pair:

  group 0  MXF4 -> bf16, in scope for ``isMxf4SubtilePath``
  group 1  MXF4 -> fp32, out of scope, otherwise identical
  group 2  MXF4 -> bf16 carrying the shipped epilogue (bias, scaleAlphaVec,
           scalar scaleAB, activation, user args), in scope

Target: the ``isMxf4SubtilePath`` gate in ``Tensile/Common/Utilities.py`` and
everything hanging off it -- the fused post-loop store (``PostLoopStoreInNll``),
its guard hoisting, and the partitioned block schedule. None of it had unit-level
coverage: the only other gfx950 MXF4 config in the tree is ``DestDataType: S``,
so ``plsinDataTypes`` is False for it and the gate never opened.

The ``{basename, err}`` goldens hold no assembly, so they pin the solution-name
hash and nothing about the store body. The tests here carry the rest, in two
groups.

Scope -- that each optimization is built for MXF4->bf16 and for nothing else,
which is what the rest of the library depends on:

  - PLSIN markers appear in scope and are wholly absent from the fp32 control,
    so the fused store cannot silently widen to other destination types.
  - The partitioned schedule appears only at MT256x256, pinning the
    ``MacroTile0 == 256 and MacroTile1 == 256`` equality in ``plsinBlockSchedTile``
    that keeps every other tile on its shipped schedule.

Content -- that what gets built is correct, asserted on instructions rather than
on the ``/* PLSIN: ... */`` comments the emit path writes. A comment assertion
only says the generator announced it ran, and rewording one silently removes the
coverage:

  - the pre-loop waits on half its global reads, and on the right half;
  - the pre-loop SRD advance is conditional here and immediate outside the gate;
  - the eight partitions are a 2x4 M/N split covering the tile exactly once;
  - no tile's store is issued before its K reduction finishes;
  - the fused-store flag is settled before the mainloop and only ever narrowed;
  - the owner's NGLL arm carries the store's hoisted coordinate math.

CPU-only. No GPU, no compile, no hardware access.
"""

import functools
import os
import re

import pytest

from config_harness import emit_kernels_from_config, golden_digest

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "mxf4_bf16_subtile.yaml",
)

# Problem-group indices within the config. They are a matched pair: same solution
# parameters, different destination type.
_IN_SCOPE = 0      # DestDataType: B  -> isMxf4SubtilePath() True
_CONTROL = 1       # DestDataType: S  -> isMxf4SubtilePath() False
# Same as group 0 but carrying the shipped epilogue, which is what gives the
# fused store kernargs to load. Groups 0 and 1 have none, so numStoreSgprToLoad
# is zero for them and the loader returns at its first guard.
_EPILOGUE = 2      # DestDataType: B + bias/scaleAlphaVec/scaleAB

# Emitted markers of the fused post-loop store. These are the comments and label
# the PLSIN emit path writes; matching them keeps the test tied to the feature
# rather than to the full, compiler-dependent assembly text.
_PLSIN_MARKERS = (
    ("the fused-store eligibility precompute",
     r"/\* PLSIN: precompute fused full-tile eligibility -> PostLoopFusedStore \*/"),
    ("the alpha==1 guard hoist",
     r"/\* PLSIN guard-hoist: fold alpha==1 into PostLoopFusedStore \*/"),
    ("the beta==0 guard hoist",
     r"/\* PLSIN guard-hoist: fold beta==0 into PostLoopFusedStore \(pre-loop shadow\) \*/"),
    ("the hoisted-coordinate rejoin label",
     r"^label_PLSIN_coordsValid(_\d+)?:"),
)


@functools.lru_cache(maxsize=None)
def _emit(problem_index):
    """Emit one problem group. Cached: each call is a full CPU codegen run.

    The result is read-only in every test here, and both groups are emitted by
    most of them, so without the cache the file pays for the same two codegen
    runs a dozen times over.
    """
    return tuple(emit_kernels_from_config(
        _CONFIG, limit=8, arch=_ARCH,
        problem_index=problem_index, expected_fork_count=2,
    ))


def _by_macro_tile(results):
    """Map ``"MT<m>x<n>"`` -> assembly, read from the kernel symbol in the source.

    The basename is a hash, so the tile is not recoverable from it; the descriptive
    name in the emitted ``.amdhsa_kernel`` directive carries it.
    """
    tiles = {}
    for base, src, _err in results:
        found = re.search(r"\.amdhsa_kernel \S+?_(MT\d+x\d+)x", src)
        assert found, f"kernel {base!r}: no MacroTile in the emitted kernel symbol"
        tiles[found.group(1)] = src
    return tiles


# ── Assembly readers ──────────────────────────────────────────────────────
#
# The tests below assert on instructions rather than on the ``/* PLSIN: ... */``
# comments the emit path writes. A comment assertion only says the generator
# announced it ran, and rewording one silently removes the coverage.

_PRELOOP_START = "/* PRELOOP start */"
_MAINLOOP_LABEL = "label_LoopBeginL:"
_FUSED_STORE_FLAG = "s[sgprPostLoopFusedStore]"

_WAIT_GR = re.compile(r"s_waitcnt vmcnt\((\d+)\)\s*//\s*Wait GR \(([^)]*)\)")
_WAIT_GR_COUNTS = re.compile(r"A=(\d+) B=(\d+) SA=(\d+) SB=(\d+)")
_PARTITION = re.compile(r"^/\* partition=(\d+) subIterK=\d+ \*/$")
_MFMA_TILE = re.compile(r"MFMA C\[(\d+),(\d+)\]")
_FUSED_GUARD = re.compile(r"/\* Fused-store guard:.*else -> (label_\w+) \*/")
# A kernarg scalar load, with its request width in dwords (bare s_load_dword is 1).
_EPILOGUE_LOAD = re.compile(r"\bs_load_dword(?:x(\d+))?\s+s\[?\d+.*sgprKernArgAddress")
_PLAIN_NGLL = re.compile(r"^label_NGLL\w*_PlainNGLL:")
_DONE_NGLL = re.compile(r"^label_NGLL\w*_PostInitNGLL:")


def _pairs_by_macro_tile():
    """``(tile, in-scope assembly, control assembly)`` for each MacroTile.

    Pairing by tile rather than concatenating the groups keeps a property that
    holds on one tile and not the other from passing.
    """
    in_scope = _by_macro_tile(_emit(_IN_SCOPE))
    control = _by_macro_tile(_emit(_CONTROL))
    assert set(in_scope) == set(control), "the two problem groups emitted different tiles"
    return [(tile, in_scope[tile], control[tile]) for tile in sorted(in_scope)]


def _preloop_wait(src):
    """The pre-loop's global-read wait: ``(loads issued before it, vmcnt, counts)``.

    ``counts`` is the per-tensor breakdown the wait carries, or None on the full
    drain, which has no breakdown to report.
    """
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == _PRELOOP_START)
    loads = 0
    for line in lines[start:]:
        wait = _WAIT_GR.search(line)
        if wait:
            counts = _WAIT_GR_COUNTS.search(wait.group(2))
            return loads, int(wait.group(1)), tuple(map(int, counts.groups())) if counts else None
        if line.strip().startswith("buffer_load"):
            loads += 1
    raise AssertionError("no Wait GR after the pre-loop start")


def _fused_arms(src):
    """The spans each fused-store guard protects, as line lists.

    The guard names the label it branches to when the tile is not eligible, so
    everything between the two is the arm that runs the fused store.
    """
    lines = src.splitlines()
    arms = []
    for i, line in enumerate(lines):
        guard = _FUSED_GUARD.search(line)
        if not guard:
            continue
        target = guard.group(1) + ":"
        end = next((j for j in range(i, len(lines)) if lines[j].startswith(target)), None)
        if end is not None:
            arms.append(lines[i:end])
    return arms


def _partitions(arm):
    """Per-partition ``(MFMA C tiles, D store count)``, in emission order."""
    tiles, stores, current = {}, {}, None
    for line in arm:
        marker = _PARTITION.match(line.strip())
        if marker:
            current = int(marker.group(1))
            tiles.setdefault(current, set())
            stores.setdefault(current, 0)
            continue
        if current is None:
            continue
        mfma = _MFMA_TILE.search(line)
        if mfma:
            tiles[current].add((int(mfma.group(1)), int(mfma.group(2))))
        if line.strip().startswith("buffer_store_"):
            stores[current] += 1
    return tiles, stores


def _block_scheduled_arm(src):
    """The one fused arm whose K reduction is split across partitions."""
    scheduled = [arm for arm in _fused_arms(src) if max(_partitions(arm)[0], default=0) > 0]
    assert len(scheduled) == 1, f"expected one block-scheduled arm, found {len(scheduled)}"
    return scheduled[0]


def _ngll_arms(src):
    """``(owner arm, plain arm)`` instruction lists for each dual NGLL in ``src``.

    The two arms are emitted in that order and closed by the rejoin label.
    """
    lines = src.splitlines()
    arms = []
    for i, line in enumerate(lines):
        if "_PostLoopInitInNGLL" not in line:
            continue
        plain = next(j for j in range(i, len(lines)) if _PLAIN_NGLL.match(lines[j]))
        done = next(j for j in range(plain, len(lines)) if _DONE_NGLL.match(lines[j]))
        arms.append((_instructions(lines[i + 1:plain]), _instructions(lines[plain + 1:done])))
    assert arms, "no dual NGLL in this kernel"
    return arms


def _instructions(segment):
    body = [l.split("//")[0].strip() for l in segment]
    return [l for l in body if l and not l.startswith("/*") and not l.endswith(":")]


def _valu(instructions):
    return [i for i in instructions if i.startswith("v_") and not i.startswith("v_mfma")]


def _flag_writes_and_reads(src):
    """Instructions that write and that test ``PostLoopFusedStore``."""
    body = [l.split("//")[0].strip() for l in src.splitlines()]
    flag = re.escape(_FUSED_STORE_FLAG)
    writes = [(i, l) for i, l in enumerate(body)
              if re.match(r"^s_(?!cmp_)\w+ " + flag + ",", l)]
    reads = [(i, l) for i, l in enumerate(body) if re.match(r"^s_cmp_\w+ " + flag + ",", l)]
    return writes, reads


def test_mxf4_bf16_subtile_emits_assembly():
    """The in-scope MXF4->bf16 config emits real gfx950 assembly, all err==0."""
    results = _emit(_IN_SCOPE)
    assert len(results) == 2, f"expected 2 kernels, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert len(src.splitlines()) > 100, f"kernel {base!r}: suspiciously short assembly"
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"


def test_mxf4_fp32_control_emits_assembly():
    """The out-of-scope MXF4->fp32 control emits real gfx950 assembly, all err==0."""
    results = _emit(_CONTROL)
    assert len(results) == 2, f"expected 2 kernels, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )


def test_plsin_is_scoped_to_mxf4_bf16():
    """The fused post-loop store is built for bf16 out and for nothing else.

    Both halves matter. Losing the markers on the left means the optimization
    stopped being built; gaining any on the right means it escaped its gate and is
    now reshaping the epilogue of kernels that were never measured with it.
    """
    in_scope = "\n".join(src for _b, src, _e in _emit(_IN_SCOPE))
    control = "\n".join(src for _b, src, _e in _emit(_CONTROL))

    for description, pattern in _PLSIN_MARKERS:
        assert re.search(pattern, in_scope, re.MULTILINE), (
            f"MXF4->bf16 assembly is missing {description}: /{pattern}/"
        )
        assert not re.search(pattern, control, re.MULTILINE), (
            f"MXF4->fp32 is outside isMxf4SubtilePath but its assembly contains "
            f"{description}: /{pattern}/"
        )


def test_block_scheduling_is_scoped_to_mt256():
    """The partitioned block schedule is built at MT256x256 and at no other tile.

    ``plsinBlockSchedTile`` scopes block scheduling with an equality rather than a
    bound, so MT128x128 is PLSIN-eligible yet keeps its single-partition schedule.
    Both tiles here are in scope for PLSIN, which is what isolates the tile
    condition from the type condition.
    """
    tiles = _by_macro_tile(_emit(_IN_SCOPE))
    assert set(tiles) == {"MT256x256", "MT128x128"}, f"unexpected tiles: {sorted(tiles)}"

    def partitions(src):
        return {int(n) for n in re.findall(r"/\* partition=(\d+) subIterK=", src)}

    # Eight partitions, not four: the tail splits M as well as N on this tile,
    # giving a 2x4 grid. The N split alone floors at four.
    assert partitions(tiles["MT256x256"]) == {0, 1, 2, 3, 4, 5, 6, 7}, (
        "MT256x256 should carry the partitioned K reduction, got "
        f"{sorted(partitions(tiles['MT256x256']))}"
    )
    assert partitions(tiles["MT128x128"]) == {0}, (
        "MT128x128 is outside plsinBlockSchedTile and should keep its single-partition "
        f"schedule, got {sorted(partitions(tiles['MT128x128']))}"
    )


def test_arm_split_is_built_only_with_block_scheduling():
    """The NGLL/NLL arm split rides with block scheduling, not with PLSIN.

    Fusing the store into the no-load-loop means the NGLL and NLL each need a
    second, unfused arm for the cases the fused one cannot take. Those arms cost
    instruction cache, so they are built only where the staged store needs them,
    which is the block-scheduled tile -- MT128x128 runs PLSIN without them.
    """
    tiles = _by_macro_tile(_emit(_IN_SCOPE))
    for arm in (r"/\* NGLL_C1 \(non-PLSIN arm\) \*/", r"/\* NLL_C1 \(non-PLSIN arm\) \*/"):
        assert re.search(arm, tiles["MT256x256"]), (
            f"block-scheduled MT256x256 is missing its unfused arm: /{arm}/"
        )
        assert not re.search(arm, tiles["MT128x128"]), (
            f"MT128x128 is not block-scheduled and should not carry an unfused arm: /{arm}/"
        )


def test_paired_store_repack_is_scoped():
    """The 16-bit paired-store repack is built only on the MXF4 subtile path.

    Both lane-exchange implementations appear in scope: ``plsinStorePermlane16Active``
    picks ``v_permlane16_swap`` for the arm where last-K MFMAs separate the pairs,
    and the remaining arms keep ``ds_bpermute`` as the vPack WAR fence. Matching
    the repack's own ``swap dwords`` operands rather than the bare instruction
    keeps this off the unrelated ``v_permlane16_swap`` the MX scale path emits,
    which is present on develop and in the control.
    """
    in_scope = "\n".join(src for _b, src, _e in _emit(_IN_SCOPE))
    control = "\n".join(src for _b, src, _e in _emit(_CONTROL))

    repack = r"v_permlane16_swap_b32 .*// swap dwords"
    assert re.search(repack, in_scope), "MXF4->bf16 is missing the permlane16 store repack"
    assert not re.search(repack, control), (
        "MXF4->fp32 is outside isMxf4SubtilePath but carries the permlane16 store repack"
    )

    assert "ds_bpermute" in in_scope, (
        "MXF4->bf16 should still use ds_bpermute on the arms permlane16 cannot take"
    )
    assert "ds_bpermute" not in control, (
        "MXF4->fp32 is outside isMxf4SubtilePath but carries the ds_bpermute store repack"
    )


def test_preloop_wait_is_partial_and_drains_the_first_batch():
    """The pre-loop waits on half its loads, and on exactly the right half.

    This is the headline of the pre-loop change. Off the path the pre-loop issues
    one batch of global reads and drains it with ``vmcnt(0)``; in scope the MT1
    batch is hoisted in front of the wait so both batches are in flight at once,
    and the wait drops to ``vmcnt(N)`` with N the size of the batch that stays
    outstanding. The local reads after the barrier consume MT0, so the wait is
    correct only if N leaves MT1 and nothing else behind -- a wait one too weak
    reads LDS before the write lands.

    The two problem groups differ only in ``DestDataType``, so they issue the
    same global reads. That is what lets the in-scope batch be checked against
    the control's rather than against a number written down here.
    """
    for tile, in_scope, control in _pairs_by_macro_tile():
        scope_loads, scope_wait, counts = _preloop_wait(in_scope)
        control_loads, control_wait, _ = _preloop_wait(control)

        assert control_wait == 0, (
            f"{tile}: the fp32 control is outside the gate and must keep develop's "
            f"full drain, got vmcnt({control_wait})"
        )
        assert scope_wait > 0, f"{tile}: in scope the pre-loop wait is still a full drain"
        assert scope_loads == 2 * control_loads, (
            f"{tile}: the reorder should put a second batch in front of the wait; "
            f"in scope issues {scope_loads} loads against the control's {control_loads}"
        )
        assert scope_wait == control_loads, (
            f"{tile}: vmcnt({scope_wait}) should leave exactly the hoisted batch of "
            f"{control_loads} loads in flight, so the batch the local reads consume is drained"
        )
        assert counts and sum(counts) == scope_wait, (
            f"{tile}: vmcnt({scope_wait}) does not match its own per-tensor count {counts}, "
            f"so it is not derived from the loads it is covering"
        )


def test_preloop_holds_the_srd_only_in_scope():
    """The pre-loop SRD advance is conditional in scope and immediate outside it.

    Its unit-level behavior is pinned in ``test_plsin_srd_hold.py``; what this
    adds is that the hold is built where it is supposed to be. Only the four
    pre-loop advances are conditional -- the mainloop's advance always has a
    K-block to read, and making it conditional would stall every iteration.
    """
    held = re.compile(r"s_cselect_b32 \S+, 0, \d+\s*//\s*(\w+): hold SRD")
    for tile, in_scope, control in _pairs_by_macro_tile():
        tensors = sorted(m.group(1) for m in held.finditer(in_scope))
        assert tensors == ["A", "B", "MXSA", "MXSB"], (
            f"{tile}: expected the hold on each of the four pre-loop SRDs, got {tensors}"
        )
        assert not held.search(control), (
            f"{tile}: the fp32 control is outside the gate but carries the SRD hold"
        )


def test_block_schedule_splits_the_tile_two_by_four():
    """The eight partitions are a 2x4 M/N split that covers the tile exactly once.

    Partition count alone does not say the split is right. A partitioning that
    dropped a tile would leave its accumulator unstored, and one that repeated a
    tile would double its K reduction; both pass a count. So this asserts the
    shape of the split and that the partitions are disjoint and complete.
    """
    tiles, _stores = _partitions(_block_scheduled_arm(_by_macro_tile(_emit(_IN_SCOPE))["MT256x256"]))
    assert sorted(tiles) == list(range(8)), f"expected 8 partitions, got {sorted(tiles)}"

    m_sets = {frozenset(m for m, _n in part) for part in tiles.values()}
    n_sets = {frozenset(n for _m, n in part) for part in tiles.values()}
    assert len(m_sets) == 2 and len(n_sets) == 4, (
        f"expected a 2x4 M/N split, got {len(m_sets)}x{len(n_sets)}"
    )

    covered = [t for part in tiles.values() for t in part]
    assert len(covered) == len(set(covered)), "a tile is reduced in more than one partition"
    assert set(covered) == {(m, n) for m in range(8) for n in range(8)}, (
        "the partitions do not cover the 8x8 MFMA tile grid"
    )


def test_fused_stores_never_outrun_their_k_reduction():
    """Stores lag the partitions, which is the whole point of staging them.

    The fused store issues a tile's D writes inside a *later* partition's MFMA
    stream, so the stores hide under MFMA latency instead of forming a block at
    the end. The constraint is that a tile cannot be written before its K
    reduction finishes, so at every partition boundary the stores issued so far
    must cover no more of the tile than the MFMAs have completed. Equality is
    allowed only at the end, where the last partition drains the remainder.
    """
    tiles, stores = _partitions(_block_scheduled_arm(_by_macro_tile(_emit(_IN_SCOPE))["MT256x256"]))
    total_tiles = sum(len(t) for t in tiles.values())
    total_stores = sum(stores.values())
    assert total_stores, "the block-scheduled arm issued no D stores"

    reduced = stored = 0
    for part in sorted(tiles):
        reduced += len(tiles[part])
        stored += stores[part]
        assert stored * total_tiles <= reduced * total_stores, (
            f"partition {part}: {stored}/{total_stores} of the tile is stored but only "
            f"{reduced}/{total_tiles} is reduced -- a store ran ahead of its MFMAs"
        )
        if part < max(tiles):
            assert stored * total_tiles < reduced * total_stores, (
                f"partition {part}: the stores have caught up with the MFMAs, so they are "
                f"not staged across partitions"
            )
    assert stored == total_stores and reduced == total_tiles


def test_ngll_init_arm_hoists_the_store_coords():
    """At 256x256 and below, the owner's NGLL arm carries the store's coord math.

    The NGLL splits into an arm for the workgroup that owns the fused store and
    a lean one for everyone else. The owner's arm is where the store's
    data-independent coordinate VALU goes: it is interleaved into the MFMA gaps
    so it issues under MFMA latency instead of running as an exposed serial block
    before the first accumulator read. So the owner's arm must come out with more
    VALU than the lean arm, and with the same MFMAs -- hoisting work in must not
    cost K reduction.

    Above 256x256 the hoist is off and the two arms match;
    ``test_ngll_init_arm_is_unwoven_above_256x256`` in
    ``test_mt320x256_plsin_weave_char.py`` holds that half. Together they pin a
    branch that a source comment described as never taken and a review read as
    dead, which the library build then disproved.
    """
    for tile, in_scope, _control in _pairs_by_macro_tile():
        for init, plain in _ngll_arms(in_scope):
            init_valu, plain_valu = _valu(init), _valu(plain)
            assert len(init_valu) > len(plain_valu), (
                f"{tile}: the owner's NGLL arm has {len(init_valu)} VALU against the plain "
                f"arm's {len(plain_valu)}, so the store coords were not hoisted into it"
            )
            mfma = [i for i in init if i.startswith("v_mfma")]
            assert mfma and len(mfma) == len([i for i in plain if i.startswith("v_mfma")]), (
                f"{tile}: weaving the coords in changed how much K the arm reduces"
            )


def test_fused_store_flag_is_hoisted_and_only_ever_narrowed():
    """``PostLoopFusedStore`` is decided before the mainloop and never widened.

    The guard hoist replaces a chain of epilogue branches with one SGPR settled
    in the pre-loop, so two things have to hold. Every write must land before the
    mainloop -- a write inside it would be re-evaluating a guard per iteration,
    which is the cost the hoist exists to remove. And after the flag is
    established, every write must be able only to clear it: the arms fold in with
    ``s_cselect_b32 flag, flag, 0``, so a guard that failed cannot be undone by a
    later one. A write of the ``flag, 1, 0`` form anywhere but first would let the
    last guard decide alone and fuse a tile the others rejected.
    """
    for tile, in_scope, control in _pairs_by_macro_tile():
        writes, reads = _flag_writes_and_reads(in_scope)
        assert writes, f"{tile}: the hoisted fused-store flag is never written"

        mainloop = next(i for i, l in enumerate(in_scope.splitlines())
                        if l.strip() == _MAINLOOP_LABEL)
        late = [l for i, l in writes if i >= mainloop]
        assert not late, f"{tile}: the flag is written inside the mainloop: {late}"

        establish, folds = writes[0][1], [l for _i, l in writes[1:]]
        assert establish.endswith(", 1, 0"), (
            f"{tile}: the first write should establish the flag, got {establish!r}"
        )
        for fold in folds:
            assert fold == f"s_cselect_b32 {_FUSED_STORE_FLAG}, {_FUSED_STORE_FLAG}, 0", (
                f"{tile}: a guard writes the flag without narrowing it: {fold!r}"
            )

        assert reads, f"{tile}: nothing branches on the hoisted flag"
        assert all(l == f"s_cmp_eq_u32 {_FUSED_STORE_FLAG}, 1" for _i, l in reads), (
            f"{tile}: the flag is tested some way other than against 1: {set(l for _i, l in reads)}"
        )

        assert _FUSED_STORE_FLAG not in control, (
            f"{tile}: the fp32 control is outside the gate but carries the fused-store flag"
        )


def test_mxf4_bf16_subtile_epilogue_emits_assembly():
    """The epilogue group generates kernels at all."""
    results = _emit(_EPILOGUE)
    assert len(results) == 2, f"expected 2 kernels, got {len(results)}"
    for basename, src, err in results:
        assert err == 0, f"{basename}: err={err}"
        assert "gfx950" in src, f"{basename}: wrong arch in assembly"
        assert "_Bias_" in basename, (
            f"{basename}: the epilogue group must emit bias-carrying kernels; "
            "without them the fused store has no kernargs to load"
        )
        assert "v_mfma" in src, f"{basename}: no MFMA in the emitted body"


def test_fused_store_defers_the_runtime_epilogue_pointers():
    """Bias and ScaleAlphaVec are resolved at runtime, not compile time.

    A kernel compiled with those epilogues can still be *called* with null
    pointers, so eligibility cannot be settled by the problem type alone. The
    fused store folds a runtime test of each pointer into the same flag as the
    compile-time terms, sending a call that actually passes one down the plain
    arm. Groups 0 and 1 have no such pointers, so they must carry neither test
    -- a guard that appeared there would be spending SALU on a constant.
    """
    epilogue = "\n".join(src for _b, src, _e in _emit(_EPILOGUE))
    bare = "\n".join(src for _b, src, _e in _emit(_IN_SCOPE))

    for what, marker in (
        ("the bias pointer",
         "/* PLSIN guard-hoist: runtime AddressBias -> PLAIN NLL */"),
        ("the scale-alpha-vector pointer",
         "/* PLSIN guard-hoist: runtime AddressScaleAlphaVec -> PLAIN NLL */"),
    ):
        assert marker in epilogue, (
            f"{what} is not folded into the fused-store flag; a call passing it "
            "would take the fused arm and skip the epilogue"
        )
        assert marker not in bare, (
            f"{what} is guarded in a kernel that has no such pointer"
        )


def _fused_arm_kernarg_loads(results):
    """Per kernel, the dword width of every kernarg load inside a fused arm."""
    widths = {}
    for basename, src, _err in results:
        got = []
        for arm in _fused_arms(src):
            for line in arm:
                load = _EPILOGUE_LOAD.search(line.split("//")[0])
                if load:
                    got.append(int(load.group(1) or 1))
        widths[basename] = got
    return widths


def test_fused_epilogue_kernargs_load_in_small_chunks():
    """The fused arm loads its epilogue kernargs in requests of at most 4 SGPRs.

    The epilogue block is ~13 SGPRs contiguous. Requesting it in one piece needs
    13 consecutive free SGPRs, which MT256x256 fp4 does not have -- it overflowed
    at sgprs=105. Splitting it per slot (scaleAB / scaleCD / SAV / bias / E /
    activation) caps the largest request at 4, which is what fits in the holes
    the drain lends the store. Asserting the width is asserting that budget: an
    s_load_dwordx8 here is the overflow coming back.
    """
    for basename, widths in _fused_arm_kernarg_loads(_emit(_EPILOGUE)).items():
        assert widths, (
            f"{basename}: the fused arm loads no kernargs at all, so this config "
            "no longer reaches the loader it exists to cover"
        )
        assert max(widths) <= 4, (
            f"{basename}: kernargs requested {max(widths)} SGPRs at once; the "
            "per-slot split exists to keep that at 4 or fewer"
        )


def test_the_epilogue_loader_is_driven_by_the_epilogue():
    """The fused arm loads kernargs only for what the problem actually carries.

    Groups 0 and 2 differ only in the epilogue features, so comparing what their
    fused arms load isolates the loader. The bare group still loads its batch
    offset -- every fused store needs that -- so the control is that it loads
    strictly less, not nothing. Emitting the epilogue block unconditionally would
    spend the SGPRs on every PLSIN kernel in the library, which is the budget
    this path was split up to protect.
    """
    bare = max(sum(w) for w in _fused_arm_kernarg_loads(_emit(_IN_SCOPE)).values())
    rich = min(sum(w) for w in _fused_arm_kernarg_loads(_emit(_EPILOGUE)).values())
    assert rich > bare, (
        f"the epilogue group loads {rich} kernarg dwords in its fused arm and the "
        f"bare group {bare}; the loader is not tracking the epilogue"
    )


def test_mxf4_bf16_subtile_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for the in-scope kernels."""
    assert golden_digest(_emit(_IN_SCOPE)) == snapshot


def test_mxf4_fp32_control_golden(snapshot):
    """Order-invariant golden for the scoping control.

    These basenames embed the solution-name hash, so the golden also pins that no
    parameter reaches the name of a kernel outside ``isMxf4SubtilePath`` -- the
    failure mode that would renumber every non-MXF4 kernel in the library.
    """
    assert golden_digest(_emit(_CONTROL)) == snapshot


def test_mxf4_bf16_subtile_epilogue_golden(snapshot):
    """Order-invariant golden for the epilogue group."""
    assert golden_digest(_emit(_EPILOGUE)) == snapshot
