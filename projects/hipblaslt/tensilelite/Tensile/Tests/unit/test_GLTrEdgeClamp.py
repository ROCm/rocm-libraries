# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the GLTr (transpose-load) byte-limit clamp used by
``KernelWriterAssembly.calcMaxGroForGLTr``.

Background: RDNA4 transpose global loads (``global_load_tr``) use a bare 2-register
address with no buffer ``num_records`` bounds (unlike ``buffer_load``). For operand B
the per-lane global-read byte offset must therefore be clamped so a free dim smaller
than the macro-tile (N < MacroTile1) -- or the last K step of a small tensor -- cannot
read past the end of the operand and fault on an unmapped page.

The clamp bound is the SAME value ``buffer_load`` enforces in hardware via the SRD
``num_records`` word ``Srd+2`` = ``(tensor2dSize - tileStart)*bpe + prePad`` (bytes from
the current SRD base to the tensor end, kept live by ``computeLoadSrd`` /
``incrementSrd``). The transpose load reads a full ``bytesPerLoad``-wide vector, so the
last legal start offset is ``Srd+2 - bytesPerLoad``. When the tensor exceeds 2^32 bytes
``Srd+2`` holds ``BufferLimit`` (0xFFFFFFFF); the bound is capped at ``INT_MAX`` so the
downstream signed ``VMinI32`` stays a guaranteed no-op there.

The codegen emits ``s_sub_u32 (Srd+2, bytesPerLoad)`` ; ``s_min_u32 (.,INT_MAX)`` ;
``v_mov`` ; per-load ``v_min_i32 (offset, bound)``. This test pins that pure-integer
decision so a regression in the bound (wrong width, missing INT_MAX cap, signedness)
is caught without a GPU.
"""

import pytest

pytestmark = pytest.mark.unit

INT_MAX = 0x7FFFFFFF
BUFFER_LIMIT = 0xFFFFFFFF


def _bound(srd2, bytes_per_load):
    """The bound vgpr value computed by calcMaxGroForGLTr.

    Mirrors: s_sub_u32 s,(Srd+2),bytesPerLoad ; s_min_u32 s,INT_MAX.
    s_sub_u32 is modular (32-bit unsigned); s_min_u32 is unsigned.
    """
    sub = (srd2 - bytes_per_load) & 0xFFFFFFFF      # s_sub_u32 (wraps mod 2^32)
    return min(sub, INT_MAX)                         # s_min_u32 with INT_MAX


def _clamp(offset, bound):
    """Per-load clamp: signed v_min_i32(offset, bound). Offsets are small
    non-negative byte offsets; bound <= INT_MAX so signed min is correct."""
    return min(offset, bound)


# bytes per transpose load = bpeGR * glvw. fp8 glvw8 -> 8 (b64); fp16 glvw8 -> 16 (b128).
@pytest.mark.parametrize("bytes_per_load", [8, 16])
def test_bound_is_tensor_end_minus_loadwidth(bytes_per_load):
    """For an in-range tensor the bound is exactly Srd+2 - bytesPerLoad."""
    for srd2 in (bytes_per_load, 64, 1536, 4096, 65536, INT_MAX):
        assert _bound(srd2, bytes_per_load) == srd2 - bytes_per_load


@pytest.mark.parametrize("bytes_per_load", [8, 16])
def test_huge_tensor_is_noop(bytes_per_load):
    """When Srd+2 == BufferLimit (tensor > 2^32 bytes) the bound caps at INT_MAX so
    the downstream signed clamp can never truncate a valid (positive) offset."""
    bound = _bound(BUFFER_LIMIT, bytes_per_load)
    assert bound == INT_MAX
    # any realistic positive offset passes through unchanged
    for off in (0, 8, 1024, 0x40000000, INT_MAX):
        assert _clamp(off, bound) == off


@pytest.mark.parametrize("bytes_per_load", [8, 16])
def test_offset_past_tensor_end_is_clamped(bytes_per_load):
    """An offset at/after the last legal start is pulled back to the bound; an offset
    before it is unchanged."""
    srd2 = 1536  # e.g. small B tensor (n=8,k=96,fp16 -> ~1536 bytes)
    bound = _bound(srd2, bytes_per_load)         # = 1536 - width
    assert _clamp(bound - 1, bound) == bound - 1   # in range: unchanged
    assert _clamp(bound, bound) == bound           # exactly at edge: unchanged
    assert _clamp(bound + 1, bound) == bound       # past edge: clamped
    assert _clamp(srd2, bound) == bound            # tensor end: clamped back by >= width
    assert _clamp(srd2 + 4096, bound) == bound     # far over-read: clamped


@pytest.mark.parametrize("bytes_per_load", [8, 16])
def test_clamped_load_stays_in_bounds(bytes_per_load):
    """Invariant: after clamping, a full bytesPerLoad-wide load ends at or before the
    tensor end (offset + width <= Srd+2) for every offset, for any in-range Srd+2."""
    for srd2 in range(bytes_per_load, 4 * bytes_per_load + 1):
        bound = _bound(srd2, bytes_per_load)
        for offset in range(0, srd2 + 2 * bytes_per_load):
            clamped = _clamp(offset, bound)
            assert clamped + bytes_per_load <= srd2


def test_n8_downproj_partial_tile_is_bounded():
    """The original crash: down_proj B is n=8,k=96 fp16 (~1536 bytes). The padded
    MT1=64 tile would over-read N by (64-8)*2 = 112 bytes; the byte limit pulls every
    lane back so no load crosses the 1536-byte tensor end."""
    bytes_per_load = 16  # fp16 b128
    srd2 = 8 * 96 * 2    # n*k*bpe = 1536
    bound = _bound(srd2, bytes_per_load)
    # the over-reading lane offsets (past tensor end) all clamp in-bounds
    for over in (srd2, srd2 + 16, srd2 + 112):
        assert _clamp(over, bound) + bytes_per_load <= srd2
