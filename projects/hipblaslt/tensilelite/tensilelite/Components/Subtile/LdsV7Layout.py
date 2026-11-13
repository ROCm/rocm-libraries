# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gemm_wt-style segmented LDS layout for subtile TDMSingleIssuer.

Places A0/A1/B0/B1 in distinct 64 KB segments so one issuer wave can fill a
whole operand via ITERATE x2 while concurrent pair-0/pair-1 LRs hit different
segments. Scale rings sit after the four data groups (shared, depth 2 or 4 —
gemm_wt ``mx_sbuf``) so the per-phase mainloop bodies can bake the scale buf
into ds immediates and drop the mid-loop rotate. Fits the 320 KB gfx1250 LDS cap.

Tile geometry this targets: MT256, MI32x16x128_f4, WT 4x8, 2x2 waves, row pad
272, DU512. See bench/gfx1250_mxf4/lds_v7_layout.py for the asm-patch twin.
"""

from __future__ import annotations

from dataclasses import dataclass

SEG = 65536
STEP = 256
CAP = 327680
ROW = 272
HALF_ROWS = 128
BUF = HALF_ROWS * ROW  # 34816
SC_NBUF = 2  # gemm_wt default mx_sbuf; enables even/odd bake (was 3 + runtime rotate)
SC_RING = 16384  # per operand; MXScaleLdsBuffers splits it into sc_nbuf buffers
SC_SLICE = 4096  # one operand's scales per DU512 slice: MT256 x 512/32 bytes

# Wave ownership for TDMSingleIssuer — match gemm_wt mx_v7:
#   TDM A/B on waves 0/2; scale TDM + GL2 PF on waves 1/3 (mx_wave = pf_wave).
# Dual-issue A+MX on 0/2 failed mainloop validate; keep MX on PF.
ISSUER_A = 0
ISSUER_B = 2
ISSUER_MXSA = 1
ISSUER_MXSB = 3
PREFETCH_A = 1
PREFETCH_B = 3


@dataclass(frozen=True)
class LdsV7Layout:
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

  def issuer(self, tc: str) -> int:
    return {
      "A": ISSUER_A, "B": ISSUER_B,
      "MXSA": ISSUER_MXSA, "MXSB": ISSUER_MXSB,
    }[tc]

  def prefetch(self, tc: str) -> int:
    return {"A": PREFETCH_A, "B": PREFETCH_B}[tc]


def make_lds_v7_layout(sc_nbuf: int = SC_NBUF) -> LdsV7Layout:
  """Greedy place A0,A1,B0,B1 at max(segment_g, prev_end), then scale rings."""
  sc_buf = SC_RING // sc_nbuf
  if sc_buf < SC_SLICE:
    raise RuntimeError(f"LdsV7 scale buffer {sc_buf} B < one slice ({SC_SLICE} B)")
  bases: list[int] = []
  prev = 0
  for g in range(4):
    b = max(g * SEG, prev)
    b = (b + STEP - 1) // STEP * STEP
    need = 2 * BUF  # buf0 + buf1 stacked
    if b + need > CAP:
      raise RuntimeError(f"LdsV7 group {g} needs {need} B at {b}, past cap {CAP}")
    bases.append(b)
    prev = b + need
  a0, a1, b0, b1 = bases
  sc_mxsa = (prev + STEP - 1) // STEP * STEP
  sc_mxsb = sc_mxsa + SC_RING
  end = sc_mxsb + SC_RING
  if end > CAP:
    raise RuntimeError(f"LdsV7 layout end {end} > cap {CAP}")
  return LdsV7Layout(
      a0=a0, a1=a1, b0=b0, b1=b1,
      buf=BUF, inc_a=a1 - a0, inc_b=b1 - b0,
      sc_mxsa=sc_mxsa, sc_mxsb=sc_mxsb,
      sc_buf=sc_buf, sc_nbuf=sc_nbuf, sc_ring=SC_RING,
      end=end,
  )


def tdmSingleIssuerApplicable(state) -> tuple[bool, str]:
  """Return (ok, reason). Only the DU512 MXF4 subtile [2,2] shape is supported."""
  if not state.get("TDMSingleIssuer"):
    return False, "TDMSingleIssuer off"
  if not state.get("UseSubtileImpl"):
    return False, "needs UseSubtileImpl"
  if not (state.get("enableTDMA") and state.get("enableTDMB")):
    return False, "needs TDMInst=3"
  wg = state.get("MIWaveGroup")
  if list(wg) != [2, 2]:
    return False, "needs MIWaveGroup [2,2]"
  pt = state["ProblemType"]
  if not (pt.get("MXBlockA") and pt.get("MXBlockB")):
    return False, "needs MX scales on A and B"
  if state.get("MacroTile0") != 256 or state.get("MacroTile1") != 256:
    return False, "needs MT 256x256"
  if state.get("DepthU") != 512:
    return False, "needs DepthU=512"
  if state.get("LDSSegmentInterleave", 0) not in (0, -1):
    return False, "incompatible with LDSSegmentInterleave"
  return True, ""
