#!/usr/bin/env python3
"""gemm_wt-style segmented LDS layout for the Tensile DU512 MXF4 subtile kernel.

Tensile already partitions LR by gemm_wt pairs=2:
  A: waveId % 2  -> M-half  (rows 0-127 vs 128-255)
  B: waveId // 2 -> N-half

This layout places those halves in distinct 64 KB segments (A0/A1/B0/B1) so one
issuer wave can fill a whole operand via ITERATE x2 while concurrent pair-0/pair-1
LRs hit different segments. Within each half the row-major pad-272 map is
unchanged, so existing ds_load tile immediates stay valid.

Scale rings sit after the four data groups (shared, not per-group). Depth-4
with SC_BUF=4096 (actual scale footprint ~2 KB) fits two rings under the
320 KB cap — matching Tensile LdsV7Layout / gemm_wt mx_sbuf=4.
"""
from __future__ import annotations

from dataclasses import dataclass

SEG = 65536
STEP = 256
CAP = 327680
ROW = 272
HALF_ROWS = 128
BUF = HALF_ROWS * ROW  # 34816
SC_BUF = 4096
SC_NBUF = 4  # gemm_wt BEST mx_sbuf; match Tensile LdsV7Layout
SC_RING = SC_BUF * SC_NBUF  # 16384 per operand


@dataclass(frozen=True)
class Layout:
    a0: int
    a1: int
    b0: int
    b1: int
    buf: int
    inc_a: int
    inc_b: int
    sc_mxsa: int
    sc_mxsb: int
    sc_buf: int
    sc_nbuf: int
    sc_ring: int
    end: int

    def base_a(self, half: int) -> int:
        return self.a0 if half == 0 else self.a1

    def base_b(self, half: int) -> int:
        return self.b0 if half == 0 else self.b1


def make_layout() -> Layout:
    """Greedy place A0,A1,B0,B1 at max(segment_g, prev_end), then scale rings."""
    bases: list[int] = []
    prev = 0
    for g in range(4):
        b = max(g * SEG, prev)
        b = (b + STEP - 1) // STEP * STEP
        need = 2 * BUF  # buf0 + buf1 stacked
        if b + need > CAP:
            raise RuntimeError(f"group {g} needs {need} B at {b}, past cap {CAP}")
        bases.append(b)
        prev = b + need
    a0, a1, b0, b1 = bases
    # Scales after data: MXSA then MXSB, each depth-4 ring.
    sc_mxsa = (prev + STEP - 1) // STEP * STEP
    sc_mxsb = sc_mxsa + SC_RING
    end = sc_mxsb + SC_RING
    if end > CAP:
        raise RuntimeError(f"layout end {end} > cap {CAP}")
    return Layout(
        a0=a0, a1=a1, b0=b0, b1=b1,
        buf=BUF, inc_a=a1 - a0, inc_b=b1 - b0,
        sc_mxsa=sc_mxsa, sc_mxsb=sc_mxsb,
        sc_buf=SC_BUF, sc_nbuf=SC_NBUF, sc_ring=SC_RING,
        end=end,
    )


def main() -> None:
    L = make_layout()
    print(f"A0={L.a0} (seg {L.a0 // SEG})  A1={L.a1} (seg {L.a1 // SEG})")
    print(f"B0={L.b0} (seg {L.b0 // SEG})  B1={L.b1} (seg {L.b1 // SEG})")
    print(f"buf={L.buf}  inc_a={L.inc_a}  inc_b={L.inc_b}")
    print(f"MXSA={L.sc_mxsa}  MXSB={L.sc_mxsb}  ring={L.sc_ring}  end={L.end}")
    assert L.end <= CAP
    assert L.a0 // SEG != L.a1 // SEG
    assert L.b0 // SEG != L.b1 // SEG


if __name__ == "__main__":
    main()
