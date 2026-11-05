# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Format the final matrix-staging LDS allocation without changing solution state."""

from ..Components.DecouplePGR import decouplePGRBlocks
from .segment_interleave import SEG, _footprint, _tensor_footprint


def format_lds_layout(state, kernel_name):
    """Return physical byte ranges and, on gfx1250, their 64 KiB segment map.

    Ranges include internal padding. This describes matrix staging, not the
    contents of LDS at runtime; reductions and the epilogue can reuse the memory.
    """
    offsets = {}
    if state.get("LDSSegmentInterleave") == 1:
        offsets = state.get("LDSSegInterleaveOffsets") or {}
    paired = offsets.get("pairedBuffers", False)
    single_buffer = state.get("1LDSBuffer") == 1 or not state.get("PrefetchGlobalRead")
    count = 1 if single_buffer else max(1, state["NumLdsBlk"])
    decoupled, count_a, count_b = decouplePGRBlocks(state)
    divergent = decoupled and count_a != count_b
    regions = []
    strides = []
    for tc in ("A", "B", "MXSA", "MXSB", "Metadata"):
        direct = state.get("DirectToVgpr" + tc)
        if tc == "Metadata":
            direct = state.get("DirectToVgprSparseMetadata")
        if direct:
            continue
        if offsets and tc in ("A", "B"):
            whole = offsets.get("aBaseline" if tc == "A" else "bBaseline", False)
            size = _tensor_footprint(state, tc) if whole else _footprint(state, tc)
            components = 1 if whole else 2
        else:
            size_key = "LdsNumElementsAligned" if offsets else "ldsNumBytes"
            size = state.get(size_key + tc, 0)
            components = 1
        if size <= 0:
            continue
        base = offsets.get("ldsBase" + tc, state.get("LdsOffset" + tc, 0))
        copies = count
        if paired:
            stride = offsets["bufferStride" + tc]
        elif divergent:
            side = "A" if tc in ("A", "MXSA") else "B"
            copies = count_a if side == "A" else count_b
            stride = state["LdsOffsetBlk" + side]
        else:
            stride = state.get("LdsOffsetA_Blk", 0)
        strides.append(f"{tc}={stride if copies > 1 else 0}")
        for component in range(components):
            group = tc
            if components > 1:
                group += f" group{component}" if state.get("TDMSplit") else str(component)
            for buffer in range(copies):
                phase = ("ping", "pong")[buffer] if copies == 2 else f"buffer{buffer}"
                start = int(base + component * offsets.get("writeStrideBytes", 0) + buffer * stride)
                regions.append((start, start + int(size), f"{group} {phase}"))
    regions.sort()
    layout = "paired" if paired else "interleaved" if offsets else "ordinary"
    lines = [f"[LDS layout] {kernel_name}",
             f"  layout={layout}, MIWaveGroup={state.get('MIWaveGroup')}, "
             f"TDMSplit={bool(state.get('TDMSplit'))}, BRotate={bool(state.get('LDSSegmentBRotate'))}",
             f"  Reserved LDS: {state['LdsNumBytes']} bytes; ranges are [start, end), including padding.",
             "  Buffer strides (bytes): " + ", ".join(strides),
             "  Matrix staging regions (ascending address):"]
    for start, end, name in regions:
        lines.append(f"    [{start:6d}, {end:6d}) {name} ({end - start} bytes)")
    if tuple(state.get("ISA", ())) == (12, 5, 0):
        lines.append("  64 KiB segments (regions crossing a boundary appear in both):")
        for segment in range((state["LdsNumBytes"] + SEG - 1) // SEG):
            lo, hi = segment * SEG, min((segment + 1) * SEG, state["LdsNumBytes"])
            cursor = lo
            blocks = []
            for start, end, name in regions:
                if end <= lo or start >= hi:
                    continue
                if start > cursor:
                    blocks.append(f"[gap {start - cursor} B]")
                blocks.append(f"[{name}]")
                cursor = max(cursor, min(end, hi))
            if cursor < hi:
                blocks.append(f"[gap {hi - cursor} B]")
            lines.append(f"    S{segment} [{lo}, {hi}): " + "".join(blocks))
    if offsets and state.get("TDMSplit"):
        lines.append("  group0/1 name physical read groups; "
                     "see segment_interleave.py for matrix-slice mapping.")
    return "\n".join(lines)
