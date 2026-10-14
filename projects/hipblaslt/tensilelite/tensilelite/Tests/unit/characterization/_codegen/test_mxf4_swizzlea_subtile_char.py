################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx950 MXF4 subtile characterization: host-pre-swizzled A.

The mirror of ``test_mxf4_swizzleb_subtile_char.py``, driving the same
``data/_designed/gfx950/mxf4_swizzle_subtile.yaml`` through the config-driven
emit harness:

  group 2  SwizzleTensorA: True   -> ``TileInfo.isPreShuffled`` True for A
  group 1  neither flag set       -> ``isPreShuffled`` False, otherwise identical

Target: that the pre-shuffled arms read the per-tensor flag rather than assuming
B. Every marker the emitters write is formatted with the tensor char -- for
instance ``"%s: rowId * depthUBytes" % tc`` and
``"pre-shuffled %s GR[%u] correction" % (tc, i)`` -- so an arm that hardcoded B
would still satisfy the swizzled-B tests while emitting nothing here, and an arm
that applied the layout per kernel rather than per tensor would wrongly rewrite
B's addressing in this group. The two are distinguishable only by driving A.

The asymmetry runs the opposite way from the B module: here A is pre-shuffled
and B must keep its ordinary stride-scaled addressing in the same kernel.

A and B are never swizzled together. The shipped libraries carry a problem
predicate per tensor, so a caller gets one or the other and the combined shape is
not offered.

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
    "mxf4_swizzle_subtile.yaml",
)

# Problem-group indices within the config. Same solution parameters, different A
# layout; group 1 is shared with the swizzled-B module as the common control.
_SWIZZLED = 2   # SwizzleTensorA: True  -> isPreShuffled True for A
_CONTROL = 1    # ordinary A            -> isPreShuffled False

# A is fp4 with DepthU 256, so one K window is 128 bytes, and the MFMA tile is 16
# rows deep. Pre-shuffled, one DepthU step therefore consumes 16 contiguous row
# fragments: 128 * 16.
_DEPTHU_BYTES = 128
_ROWS_PER_BLOCK = 16
_PRESHUFFLED_INC = _DEPTHU_BYTES * _ROWS_PER_BLOCK

# Emitted markers of the pre-shuffled arms, spelled for A. These are the comments
# the emitters write, which keeps the test tied to the feature rather than to the
# full, compiler-dependent assembly text. The two entries without a tensor char
# come from code that runs once per swizzled tensor, so they are present here for
# the same reason they are present in the swizzled-B module.
_PRESHUFFLE_MARKERS = (
    ("the aligned-row padding correction",
     r"// aligned pre-shuffled row"),
    ("the row-padding stride computation",
     r"// stride - DepthU"),
    ("the fp4 element-to-byte narrowing of the correction",
     r"// FP4 elements to bytes"),
    ("the global-read offset correction",
     r"// pre-shuffled A GR\[\d+\] correction"),
    ("the row-block scaling of the buffer-load limit",
     r"// numLine \* stride \* \d+ \(row-block stride\)"),
    ("the row-block scaling of the tile offset",
     r"// pre-shuffled: scale by \d+ rows per block"),
    ("the tile-boundary buffer-load limit",
     r"// buffer_load limit for A \(pre-shuffled, tile-boundary\)"),
    ("the linear local-read offset",
     r"// A: linear pre-shuffled LR offset \d+"),
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


def test_mxf4_swizzlea_emits_assembly():
    """The pre-swizzled-A config emits real gfx950 assembly, all err==0."""
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


def test_swizzled_a_does_not_require_direct_to_vgpr():
    """SwizzleTensorA is accepted on the subtile path without DirectToVgprA.

    On the classic path A has to be staged through DirectToVgpr to be read in the
    swizzled layout, and Solution.py rejects ``SwizzleTensorA`` without it. That
    requirement is unsatisfiable for fp4 in the first place --
    ``isDirectToVgprSupportDataType`` has no Float4 arm -- so without the subtile
    exemption this combination cannot produce a kernel at all. The config sets no
    DirectToVgprA, so the exemption is the only reason a solution survives
    derivation; if it regresses, this emits nothing.
    """
    assert _emit(_SWIZZLED), (
        "SwizzleTensorA with UseSubtileImpl produced no solution; the "
        "DirectToVgprA requirement is no longer waived for the subtile path"
    )


def test_pre_shuffle_is_scoped_to_swizzled_a():
    """The pre-shuffled arms are built for swizzled A and for nothing else.

    Both halves matter. Losing the markers on the left means the layout handling
    stopped being built for A, and A would be read as though the host had not
    shuffled it; gaining any on the right means it escaped its gate and is now
    mis-addressing ordinary tensors.
    """
    swizzled = _asm(_SWIZZLED)
    control = _asm(_CONTROL)

    for description, pattern in _PRESHUFFLE_MARKERS:
        assert re.search(pattern, swizzled), (
            f"pre-swizzled-A assembly is missing {description}: /{pattern}/"
        )
        assert not re.search(pattern, control), (
            f"ordinary A is not pre-shuffled but its assembly contains "
            f"{description}: /{pattern}/"
        )


def test_pre_shuffle_applies_per_tensor():
    """Only A takes the pre-shuffled addressing; B keeps its stride-scaled form.

    The exact inverse of the swizzled-B module's per-tensor check. An arm that
    applied the layout per kernel would pass there and fail here, because here it
    would rewrite B's addressing too.
    """
    swizzled = _asm(_SWIZZLED)
    control = _asm(_CONTROL)

    assert "A: rowId * depthUBytes" in swizzled, (
        "pre-shuffled A should address rows by DepthU, not by StrideA0I"
    )
    assert "A: rowId * stride" not in swizzled, (
        "pre-shuffled A still carries the stride-scaled row offset"
    )
    assert "B: rowId * stride" in swizzled, (
        "B is not pre-shuffled and must keep its stride-scaled row offset, but the "
        "pre-shuffled arm was applied to it too"
    )
    assert "B: rowId * depthUBytes" not in swizzled, (
        "B is not pre-shuffled but took the pre-shuffled row addressing"
    )

    # The control pins that A's ordinary form is what changed, not something else.
    assert "A: rowId * stride" in control
    assert "A: rowId * depthUBytes" not in control


def test_gr_pointer_advance_scales_by_row_block():
    """Pre-shuffled A advances its SRD a whole 16-row block per DepthU.

    One DepthU of pre-shuffled data is interleaved across all 16 rows of the MFMA
    tile, so the pointer has to step ``depthUBytes * 16`` where ordinary A steps
    ``depthUBytes``. B is unshuffled in the same kernel and keeps the plain step,
    which is what separates this from a global change to the advance.
    """
    swizzled = _asm(_SWIZZLED)
    control = _asm(_CONTROL)

    assert f"A: advance SRD by {_PRESHUFFLED_INC} bytes" in swizzled, (
        f"pre-shuffled A should advance by {_PRESHUFFLED_INC} bytes "
        f"({_DEPTHU_BYTES} * {_ROWS_PER_BLOCK})"
    )
    assert f"B: advance SRD by {_DEPTHU_BYTES} bytes" in swizzled, (
        "B is not pre-shuffled and should keep the plain advance"
    )
    assert f"A: advance SRD by {_PRESHUFFLED_INC} bytes" not in control, (
        f"ordinary A advanced by the pre-shuffled {_PRESHUFFLED_INC} bytes"
    )


def test_swizzled_a_skips_the_lds_swizzle():
    """Pre-shuffled A is not swizzled a second time on the way through LDS.

    The host layout already is the swizzle, so re-applying it would permute A
    twice. The local-read offset for a pre-shuffled tensor is linear instead.
    """
    swizzled = _asm(_SWIZZLED)
    assert re.search(r"// A: linear pre-shuffled LR offset \d+", swizzled), (
        "pre-shuffled A should take the linear local-read offset"
    )


def test_mxf4_swizzlea_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for the pre-swizzled-A kernels.

    The basename embeds the solution-name hash, so this also pins that the STA
    token reaches the kernel name and that it differs from the swizzled-B kernel.
    """
    assert golden_digest(_emit(_SWIZZLED)) == snapshot
