################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx950 MXF4 subtile characterization: host-pre-swizzled B.

Drives ``data/_designed/gfx950/mxf4_swizzleb_subtile.yaml`` through the
config-driven emit harness. The config has two problem groups that differ only in
``SwizzleTensorB``, which is what makes them a scoping pair:

  group 0  SwizzleTensorB: True   -> ``TileInfo.isPreShuffled`` True for B
  group 1  SwizzleTensorB absent  -> ``isPreShuffled`` False, otherwise identical

Target: every arm the pre-shuffled layout adds to the subtile emitters. When the
host has already interleaved B into the MFMA-friendly 16-row layout, the kernel
must not swizzle it again, must advance B's SRD by a whole 16-row block per
DepthU, must address rows by DepthU rather than by ``StrideB1J``, and must fold
the aligned-row HBM padding back into the global-read offsets. None of that had
unit-level coverage: the existing ``SwizzleTensorB`` tests
(``s09_assignderivedparameters_swizzlet.yaml``, ``test_r7_gsu_reduce_char.py``)
are all on the classic DirectToVgpr path, which never builds a subtile
``TileInfo``, so ``isPreShuffled`` was False everywhere in the suite and every
arm below emitted untested.

The asymmetry is the point of the pair. Only B is pre-shuffled, so A has to keep
its ordinary stride-scaled addressing in the very same kernel -- a per-kernel
rather than per-tensor check would pass a markers-present test and still be
wrong, so ``test_pre_shuffle_applies_per_tensor`` pins both sides.

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
    "mxf4_swizzleb_subtile.yaml",
)

# Problem-group indices within the config. They are a matched pair: same solution
# parameters, different B layout.
_SWIZZLED = 0   # SwizzleTensorB: True  -> isPreShuffled True for B
_CONTROL = 1    # ordinary B            -> isPreShuffled False

# B is fp4 with DepthU 256, so one K window is 128 bytes, and the MFMA tile is 16
# rows deep. Pre-shuffled, one DepthU step therefore consumes 16 contiguous row
# fragments: 128 * 16.
_DEPTHU_BYTES = 128
_ROWS_PER_BLOCK = 16
_PRESHUFFLED_INC = _DEPTHU_BYTES * _ROWS_PER_BLOCK

# Emitted markers of the pre-shuffled arms. These are the comments the emitters
# write, which keeps the test tied to the feature rather than to the full,
# compiler-dependent assembly text.
_PRESHUFFLE_MARKERS = (
    ("the aligned-row padding correction",
     r"// aligned pre-shuffled row"),
    ("the row-padding stride computation",
     r"// stride - DepthU"),
    ("the fp4 element-to-byte narrowing of the correction",
     r"// FP4 elements to bytes"),
    ("the global-read offset correction",
     r"// pre-shuffled B GR\[\d+\] correction"),
    ("the row-block scaling of the buffer-load limit",
     r"// numLine \* stride \* \d+ \(row-block stride\)"),
    ("the row-block scaling of the tile offset",
     r"// pre-shuffled: scale by \d+ rows per block"),
    ("the tile-boundary buffer-load limit",
     r"// buffer_load limit for B \(pre-shuffled, tile-boundary\)"),
    ("the linear local-read offset",
     r"// B: linear pre-shuffled LR offset \d+"),
)


@functools.lru_cache(maxsize=None)
def _emit(problem_index):
    """Emit one problem group. Cached: each call costs a full solution derivation."""
    return emit_kernels_from_config(
        _CONFIG, limit=8, arch=_ARCH,
        problem_index=problem_index, expected_fork_count=1,
    )


def _asm(problem_index):
    return "\n".join(src for _b, src, _e in _emit(problem_index))


def test_mxf4_swizzleb_emits_assembly():
    """The pre-swizzled-B config emits real gfx950 assembly, all err==0."""
    results = _emit(_SWIZZLED)
    assert len(results) == 1, f"expected 1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert len(src.splitlines()) > 100, f"kernel {base!r}: suspiciously short assembly"
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"


def test_ordinary_b_control_emits_assembly():
    """The ordinary-B control emits real gfx950 assembly, all err==0."""
    results = _emit(_CONTROL)
    assert len(results) == 1, f"expected 1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )


def test_swizzled_b_does_not_require_direct_to_vgpr():
    """SwizzleTensorB is accepted on the subtile path without DirectToVgprB.

    On the classic path B has to be staged through DirectToVgpr to be read in the
    swizzled layout, and Solution.py rejects ``SwizzleTensorB`` without it. The
    subtile global-read emitter walks the layout itself, so it is exempt. The
    config sets no DirectToVgprB, so the exemption is the only reason a solution
    survives derivation at all -- if it regresses, this emits nothing.
    """
    assert _emit(_SWIZZLED), (
        "SwizzleTensorB with UseSubtileImpl produced no solution; the "
        "DirectToVgprB requirement is no longer waived for the subtile path"
    )


def test_pre_shuffle_is_scoped_to_swizzled_b():
    """The pre-shuffled arms are built for swizzled B and for nothing else.

    Both halves matter. Losing the markers on the left means the layout handling
    stopped being built, and B would be read as though the host had not shuffled
    it; gaining any on the right means it escaped its gate and is now mis-addressing
    ordinary B tensors.
    """
    swizzled = _asm(_SWIZZLED)
    control = _asm(_CONTROL)

    for description, pattern in _PRESHUFFLE_MARKERS:
        assert re.search(pattern, swizzled), (
            f"pre-swizzled-B assembly is missing {description}: /{pattern}/"
        )
        assert not re.search(pattern, control), (
            f"ordinary B is not pre-shuffled but its assembly contains "
            f"{description}: /{pattern}/"
        )


def test_pre_shuffle_applies_per_tensor():
    """Only B takes the pre-shuffled addressing; A keeps its stride-scaled form.

    This config swizzles B alone, so both forms have to coexist in one kernel. A
    check written per kernel instead of per tensor would still show the B markers
    and would still be wrong, which is what this pins.
    """
    swizzled = _asm(_SWIZZLED)
    control = _asm(_CONTROL)

    assert "B: rowId * depthUBytes" in swizzled, (
        "pre-shuffled B should address rows by DepthU, not by StrideB1J"
    )
    assert "B: rowId * stride" not in swizzled, (
        "pre-shuffled B still carries the stride-scaled row offset"
    )
    assert "A: rowId * stride" in swizzled, (
        "A is not pre-shuffled and must keep its stride-scaled row offset, but the "
        "pre-shuffled arm was applied to it too"
    )
    assert "A: rowId * depthUBytes" not in swizzled, (
        "A is not pre-shuffled but took the pre-shuffled row addressing"
    )

    # The control pins that B's ordinary form is what changed, not something else.
    assert "B: rowId * stride" in control
    assert "B: rowId * depthUBytes" not in control


def test_gr_pointer_advance_scales_by_row_block():
    """Pre-shuffled B advances its SRD a whole 16-row block per DepthU.

    One DepthU of pre-shuffled data is interleaved across all 16 rows of the MFMA
    tile, so the pointer has to step ``depthUBytes * 16`` where ordinary B steps
    ``depthUBytes``. A is unshuffled in the same kernel and keeps the plain step,
    which is what separates this from a global change to the advance.
    """
    swizzled = _asm(_SWIZZLED)
    control = _asm(_CONTROL)

    preshuffled = f"advance SRD by {_PRESHUFFLED_INC} bytes"
    plain = f"advance SRD by {_DEPTHU_BYTES} bytes"

    assert preshuffled in swizzled, (
        f"pre-shuffled B should advance by {_PRESHUFFLED_INC} bytes "
        f"({_DEPTHU_BYTES} * {_ROWS_PER_BLOCK})"
    )
    assert plain in swizzled, "A is not pre-shuffled and should keep the plain advance"
    assert preshuffled not in control, (
        f"ordinary B advanced by the pre-shuffled {_PRESHUFFLED_INC} bytes"
    )


def test_swizzled_b_skips_the_lds_swizzle():
    """Pre-shuffled B is not swizzled a second time on the way through LDS.

    The host layout already is the swizzle, so re-applying it would permute B
    twice. The local-read offset for a pre-shuffled tensor is linear instead.
    """
    swizzled = _asm(_SWIZZLED)
    assert re.search(r"// B: linear pre-shuffled LR offset \d+", swizzled), (
        "pre-shuffled B should take the linear local-read offset"
    )
    assert "// colIdB base" in swizzled, (
        "with B skipping the swizzle, colIdB is no longer derived from colIdA and "
        "must be seeded on its own"
    )
    assert "// colIdB base" not in _asm(_CONTROL), (
        "ordinary B derives colIdB through the swizzle and needs no separate seed"
    )


def test_mxf4_swizzleb_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for the pre-swizzled kernels."""
    assert golden_digest(_emit(_SWIZZLED)) == snapshot


def test_ordinary_b_control_golden(snapshot):
    """Order-invariant golden for the scoping control.

    The basename embeds the solution-name hash, so this also pins that nothing
    about the pre-shuffled work reaches the name of a kernel that does not use it.
    """
    assert golden_digest(_emit(_CONTROL)) == snapshot
