################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx950 MXF4->bf16 subtile characterization: PostLoopStoreInNll + block scheduling.

Drives ``data/_designed/gfx950/mxf4_bf16_subtile.yaml`` through the config-driven
emit harness. The config has two problem groups that differ only in
``DestDataType``, which is what makes them a scoping pair:

  group 0  MXF4 -> bf16, in scope for ``isMxf4SubtilePath``
  group 1  MXF4 -> fp32, out of scope, otherwise identical

Target: the ``isMxf4SubtilePath`` gate in ``Tensile/Common/Utilities.py`` and
everything hanging off it -- the fused post-loop store (``PostLoopStoreInNll``),
its guard hoisting, and the partitioned block schedule. None of it had unit-level
coverage: the only other gfx950 MXF4 config in the tree is ``DestDataType: S``,
so ``plsinSubtileTypes`` is False for it and the gate never opened.

Beyond the ``{basename, err}`` goldens, two tests assert the *scope* of the
optimizations rather than their content, since that scope is the property the
rest of the library depends on:

  - PLSIN markers appear for MXF4->bf16 and are wholly absent from the fp32
    control, so the fused store cannot silently widen to other destination types.
  - The four-way partitioned schedule appears only at MT256x256, pinning the
    ``MacroTile0 == 256 and MacroTile1 == 256`` equality in ``plsinBlockSchedTile``
    that keeps every other tile on its shipped schedule.

CPU-only. No GPU, no compile, no hardware access.
"""

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


def _emit(problem_index):
    return emit_kernels_from_config(
        _CONFIG, limit=8, arch=_ARCH,
        problem_index=problem_index, expected_fork_count=2,
    )


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

    assert partitions(tiles["MT256x256"]) == {0, 1, 2, 3}, (
        "MT256x256 should carry the four-way partitioned K reduction, got "
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
