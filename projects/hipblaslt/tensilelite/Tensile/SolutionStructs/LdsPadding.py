################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Transpose LDS padding config for gfx1250 bank-conflict avoidance.

Computes (LdsBlockSizePerPad, ldsPad, shift) for tile-major LDS layouts
used with ds_load_tr* instructions on gfx1250. One ds_load_tr variant
per data type:

  FP4  -- ds_load_tr4_b64   (64-bit, 2 banks/thread)
  FP8  -- ds_load_tr8_b64   (64-bit, 2 banks/thread)
  FP16 -- ds_load_tr16_b128 (128-bit, 4 banks/thread)
  FP32 -- ds_load_b32        (32-bit, 1 bank/thread)

Public API (see each `def` for the full signature):
  get_fp4_mt_config  -- FP4       ds_load_tr4_b64   padding
  get_fp8_mt_config  -- FP8       ds_load_tr8_b64   padding
  get_fp16_mt_config -- FP16      ds_load_tr16_b128 padding
  get_fp32_mt_config -- FP32/TF32 ds_load_b32       padding
  get_mxs_mt_config  -- MX scale tensor padding

  key is one of "perBlock", "pad", "shift" (FP4/FP8 only for "shift").
"""

from functools import lru_cache
from typing import Dict

# TDM hardware encoding caps for LdsBlockSizePerPad / LdsPad:
#   pad_interval = log2(LdsBlockSizePerPad // 4) - 1, must fit in 3 bits => <=7
#   pad_amount   = LdsPad // 4 - 1,                  must fit in 7 bits => <=127
# Therefore valid LdsBlockSizePerPad values (in bytes) are powers of 2 in
# [8, 1024], and LdsPad in bytes is a positive multiple of 4 up to 512.
_TDM_VALID_BLOCK_BYTES = (8, 16, 32, 64, 128, 256, 512, 1024)
_TDM_MAX_PAD_BYTES     = 512

# FP4/FP8 share ds_load_tr*_b64 (64-bit/thread, 2 banks/thread). HW
# picks half-wave vs full-wave per instruction by per-thread address
# alignment.

def _b64_base_addrs_fp4(mt: int) -> tuple:
  """Return (half0_16, half1_16) per-lane base addresses for FP4."""
  bpeDS = 0.5
  addrs = []
  for w in range(32):
    t = (w // 16) * 8 + ((w % 16) // 8) * 32 + w % 8
    addrs.append(int(t * mt * bpeDS))
  return addrs[:16], addrs[16:]

def _b64_base_addrs_fp8(mt: int) -> tuple:
  """Return (half0_16, half1_16) per-lane base addresses for FP8."""
  kH0 = (0,1,2,3,0,1,2,3,4,5,6,7,4,5,6,7)
  kH1 = (16,17,18,19,16,17,18,19,20,21,22,23,20,21,22,23)
  tH  = (0,0,0,0,8,8,8,8,0,0,0,0,8,8,8,8)
  half0 = [tH[i] + kH0[i] * mt for i in range(16)]
  half1 = [tH[i] + kH1[i] * mt for i in range(16)]
  return half0, half1

def _b64_check(half0, half1, B, P, instOffs, wOffsets, shift,
               require_pure_full_wave=False,
               require_some_full_wave=False):
  """Verify (B, P, shift) is bank-conflict-free for every emitted load.

  HW address per thread:  pad(natural[t] + wOff) + shift + pad(instOff).
  HW picks full-wave when all 32 addrs are 8-aligned, else half-wave.
  Each 64-bit thread occupies 2 consecutive banks; a wave-cycle is
  conflict-free iff every bank-slot is unique.

  If `require_pure_full_wave` is True, also reject configs where any
  instruction would land at half-wave alignment (used by tier-1 search).
  If `require_some_full_wave` is True, accept a config only when at
  least one emitted instruction issues as full-wave.
  """
  def pad(x):
    return x + (x // B) * P
  def conflict_free(addrs):
    banks = []
    for p in addrs:
      sb = (p >> 2) % 64
      banks.append(sb)
      banks.append((sb + 1) % 64)
    return len(set(banks)) == len(banks)
  any_full_wave = False
  for wOff in wOffsets:
    base_h0 = [pad(a + wOff) for a in half0]
    base_h1 = [pad(a + wOff) for a in half1]
    for instOff in instOffs:
      delta = shift + pad(instOff)
      addr_h0 = [a + delta for a in base_h0]
      addr_h1 = [a + delta for a in base_h1]
      addr_all = addr_h0 + addr_h1
      if not all(p % 4 == 0 for p in addr_all):
        return False  # per-thread address not dword-aligned
      all_8aligned = all(p % 8 == 0 for p in addr_all)
      if require_pure_full_wave and not all_8aligned:
        return False
      if all_8aligned:
        any_full_wave = True
        if not conflict_free(addr_all): return False  # full-wave
      else:
        if not conflict_free(addr_h0): return False   # half-wave
        if not conflict_free(addr_h1): return False
  if require_some_full_wave and not any_full_wave:
    return False
  return True

def _b64_compute_config(mt: int, bpeDS: float,
                        addrFn, minB: int,
                        instOffs, wOffsets) -> Dict[str, int]:
  """Pick the (B, P, shift) padding config using the 3-tier policy:
  pure-full-wave > mixed-mode > half-wave (shift=4). Within a tier,
  smaller P/B overhead wins.

  Returns {"perBlock", "pad", "shift"} with pad in bytes (not elements).
  All-zero return = no valid config (caller disables block padding).
  """
  mtBytes = int(mt * bpeDS)
  validB = [b for b in _TDM_VALID_BLOCK_BYTES if b >= minB]
  half0, half1 = addrFn(mt)

  def _search(shift, require_pure_full_wave=False,
              require_some_full_wave=False):
    # Smallest P per B passes first; pick lowest overhead across B's.
    best = None
    for B in validB:
      if mtBytes % B != 0 and B % mtBytes != 0:
        continue
      for P in range(4, _TDM_MAX_PAD_BYTES + 1, 4):
        if _b64_check(half0, half1, B, P, instOffs, wOffsets, shift,
                      require_pure_full_wave, require_some_full_wave):
          overhead = P / B
          if best is None or overhead < best[0]:
            best = (overhead, B, P)
          break
    return best

  # Tier 1: pure full-wave at shift=0 (every load 1-cycle).
  # Tier 2: mixed mode at shift=0, but only accept it when at least one
  #         instruction actually issues full-wave -- otherwise this tier
  #         degenerates to all-half-wave and just over-pads vs. tier 3.
  # Tier 3: shift=4 forces every load to half-wave (2-cycle), smaller P
  #         since only h0 and h1 need to be conflict-free independently.
  for shift, pure, some_full in [(0, True,  False),
                                  (0, False, True),
                                  (4, False, False)]:
    best = _search(shift, pure, some_full)
    if best:
      return {"perBlock": best[1], "pad": best[2], "shift": shift}
  return {"perBlock": 0, "pad": 0, "shift": 0}

# Mirrors LocalRead.py per-instruction offset formulas.
# (vwTrLoad, outerInc, innerInc) per type:
_B64_EMIT_PARAMS = {
  0.5: (16, 64, 16),
  1.0: ( 8, 32,  8),
}

def _b64_emit_instOffs(mt, bpeDS, lrvw, miInputPerThread, miWaveTile):
  vwTrLoad, outerInc, innerInc = _B64_EMIT_PARAMS[bpeDS]
  outer_n = max(miInputPerThread // max(lrvw, 1), 1)
  inner_n = max(lrvw // vwTrLoad, 1)
  miWaveGroupShape = mt // max(miWaveTile, 1)
  instOffs = set()
  for tIdx in range(max(miWaveTile, 1)):
    constOff = int(miWaveGroupShape * tIdx * bpeDS)
    for outerIdx in range(outer_n):
      for innerIdx in range(inner_n):
        step = innerIdx * innerInc + outerIdx * outerInc
        instOffs.add(constOff + int(step * mt * bpeDS))
  return tuple(sorted(instOffs))

def _b64_w_offsets(miWaveGroup, matrixInstMBytes):
  """Per-wave LDS shifts (logical bytes) for MIWaveGroup waves stacked
  in one tile dimension: wave w reads from lroA + w * matrixInstMBytes
  before block padding. matrixInstMBytes = MFMA_M (elements) * bpeDS."""
  return tuple(w * matrixInstMBytes for w in range(max(miWaveGroup, 1)))

@lru_cache(maxsize=None)
def _compute_fp4_config(mt: int, miWaveTile: int, miWaveGroup: int,
                        lrvw: int = 32,
                        miInputPerThread: int = 64,
                        matrixInstM: int = 16) -> Dict[str, int]:
  # FP4: bpeDS=0.5, minB=8 (GRVW=16 * 0.5)
  # Per-wave M shift in BYTES = matrixInstM (elements) * bpeDS = 16 * 0.5 = 8.
  instOffs = _b64_emit_instOffs(mt, 0.5, lrvw, miInputPerThread, miWaveTile)
  wOffsets = _b64_w_offsets(miWaveGroup, int(matrixInstM * 0.5))
  result = _b64_compute_config(mt, 0.5, _b64_base_addrs_fp4, 8, instOffs, wOffsets)
  # bpeDS=0.5 -> convert pad from bytes to elements
  return {"perBlock": result["perBlock"], "pad": result["pad"] * 2, "shift": result["shift"]}

def get_fp4_mt_config(mt: int, key: str, miWaveTile: int, miWaveGroup: int) -> int:
  return _compute_fp4_config(mt, miWaveTile, miWaveGroup)[key]

@lru_cache(maxsize=None)
def _compute_fp8_config(mt: int, miWaveTile: int, miWaveGroup: int,
                        lrvw: int = 16,
                        miInputPerThread: int = 64,
                        matrixInstM: int = 16) -> Dict[str, int]:
  # FP8: bpeDS=1, minB=16 (GRVW=16). pad already in bytes.
  # Per-wave M shift in BYTES = matrixInstM (elements) * bpeDS = 16 * 1 = 16.
  instOffs = _b64_emit_instOffs(mt, 1.0, lrvw, miInputPerThread, miWaveTile)
  wOffsets = _b64_w_offsets(miWaveGroup, matrixInstM * 1)
  return _b64_compute_config(mt, 1.0, _b64_base_addrs_fp8, 16, instOffs, wOffsets)

def get_fp8_mt_config(mt: int, key: str, miWaveTile: int, miWaveGroup: int) -> int:
  return _compute_fp8_config(mt, miWaveTile, miWaveGroup)[key]

# -- FP16 b128 padding ---------------------------------------------

def _b128_base_addrs_fp16(mt: int) -> tuple:
  """Per-thread byte addresses for ds_load_tr16_b128 half-wave (16 threads)."""
  return [k * 2 * mt for k in range(8)] + [16 + k * 2 * mt for k in range(8)]

def _b128_check(half, B, P, wOffsets, instOffs=(0,)):
  """True if (B, P) yields 16-byte aligned, conflict-free b128 reads
  for every (wOff, instOff) combination (each thread occupies 4 banks)."""
  def pad(x): return x + (x // B) * P if B else x
  offs = instOffs if instOffs else (0,)
  for wOff in wOffsets:
    for io in offs:
      addrs = [pad(a + wOff + io) for a in half]
      if not all(a % 16 == 0 for a in addrs):
        return False
      banks = set()
      for a in addrs:
        base = (a >> 2) % 64
        for b in range(4):
          bk = (base + b) % 64
          if bk in banks:
            return False
          banks.add(bk)
  return True

def _build_fp16_instOffs(mt: int, miInputPerThUnroll: int, lrvw: int,
                         miWaveTile: int, miWaveGroup: int, vw: int,
                         matrixInstM: int = 16) -> tuple:
  # Mirror LocalRead.py FP16 LDSTr ds_load_tr16_b128 emit
  numberLRVWPerMIInput = miInputPerThUnroll // lrvw
  incrementBytes = numberLRVWPerMIInput * lrvw * mt * 2  # bpeDS = 2
  miWaveGroupShape = matrixInstM * miWaveGroup * vw
  return tuple(sorted({
    kRead + tIdx * miWaveGroupShape * 2
    for tIdx in range(max(miWaveTile, 1))
    for kRead in (0, incrementBytes)
  }))

@lru_cache(maxsize=None)
def _compute_fp16_config(mt: int, miWaveGroup: int,
                         miInputPerThUnroll: int,
                         lrvw: int,
                         miWaveTile: int,
                         vw: int,
                         matrixInstM: int = 16) -> Dict[str, int]:
  """FP16 ds_load_tr16_b128 padding pick. Tries the closed-form
  (B = 16*2^v2(MT/16), P = 8 elements) first; if it fails the
  (wave + per-instruction)-aware bank check, searches (B, P) for the
  lowest-overhead passing combination."""
  half = _b128_base_addrs_fp16(mt)
  wOffsets = tuple(w * matrixInstM * 2 for w in range(max(miWaveGroup, 1)))
  instOffs = _build_fp16_instOffs(mt, miInputPerThUnroll, lrvw,
                                  miWaveTile, miWaveGroup, vw)

  # Closed-form pick
  n = mt // 16
  if n == 0 or n % 2 == 1:
    cfg = {"perBlock": 0, "pad": 0}
  else:
    a = 0
    tmp = n
    while tmp % 2 == 0:
        a += 1
        tmp //= 2
    B = 16 * (1 << a)
    cfg = {"perBlock": B, "pad": 8} if B <= 1024 else {"perBlock": 0, "pad": 0}

  B, padE = cfg["perBlock"], cfg["pad"]
  if B == 0 or _b128_check(half, B, padE * 2, wOffsets, instOffs):
    return cfg

  # Search fallback: lowest-overhead (B, P) that passes. b128 needs B >= 16.
  mtBytes = mt * 2  # FP16 bpeDS = 2
  best = None
  for Bcand in [b for b in _TDM_VALID_BLOCK_BYTES if b >= 16]:
    if mtBytes % Bcand != 0 and Bcand % mtBytes != 0:
      continue
    for padBytes in range(4, _TDM_MAX_PAD_BYTES + 1, 4):
      if _b128_check(half, Bcand, padBytes, wOffsets, instOffs):
        overhead = padBytes / Bcand
        if best is None or overhead < best[0]:
          best = (overhead, Bcand, padBytes)
        break
  if best:
    return {"perBlock": best[1], "pad": best[2] // 2}  # pad in elements
  return {"perBlock": 0, "pad": 0}

def get_fp16_mt_config(mt: int, key: str, miWaveGroup: int,
                       miInputPerThUnroll: int, lrvw: int,
                       miWaveTile: int, vw: int) -> int:
  return _compute_fp16_config(mt, miWaveGroup,
                              miInputPerThUnroll=miInputPerThUnroll,
                              lrvw=lrvw,
                              miWaveTile=miWaveTile,
                              vw=vw)[key]

# -- FP32 b32 padding ------------------------------------------------

def _b32_check(rawAddrs, B, P, wOffsets, instOffs=(0,)):
  """True if (B, P) gives 32 distinct banks for every (wOff, instOff)
  combination."""
  def pad(x): return x + (x // B) * P if B else x
  offs = instOffs if instOffs else (0,)
  for wOff in wOffsets:
    for io in offs:
      padded = [pad(a + wOff + io) for a in rawAddrs]
      if len({(p // 4) % 64 for p in padded}) != 32:
        return False
  return True

def _b32_search_padding(mt: int, rawAddrs: list, wOffsets, instOffs=(0,)):
  """Lowest-overhead (B, P) that passes _b32_check across all (wOff, instOff)."""
  mtBytes = mt * 4
  validB = sorted([b for b in _TDM_VALID_BLOCK_BYTES
                   if mtBytes % b == 0 or b % mtBytes == 0], reverse=True)
  best = None
  for B in validB:
    for padDw in range(1, _TDM_MAX_PAD_BYTES // 4 + 1):
      if _b32_check(rawAddrs, B, padDw * 4, wOffsets, instOffs):
        overhead = (padDw * 4) / B
        if best is None or overhead < best[0]:
          best = (overhead, B, padDw)
        break
  return {"perBlock": best[1], "pad": best[2]} if best else None

def _build_fp32_instOffs(mt: int, vw: int, lrvw: int,
                         miInputPerThread: int, miWaveTile: int,
                         miWaveGroup: int,
                         xf32EmuPack: bool,
                         matrixInstM: int = 16) -> tuple:
  # Mirror LocalRead.py FP32 / XF32 ds_load_b32 emit
  nRPU = miInputPerThread // lrvw
  numVectorsPerTile = max(miWaveTile // vw, 1)
  numReadsPerVector = max(vw, 1)
  miWaveGroupShape  = matrixInstM * miWaveGroup * vw
  unrollStrideBytes = mt * 4
  if xf32EmuPack:
    kFn = lambda r: ((r // lrvw) * lrvw + r * lrvw) * unrollStrideBytes
  else:
    kFn = lambda r: r * lrvw * unrollStrideBytes
  return tuple(sorted({
    kFn(r) + v * miWaveGroupShape * 4 + e * 4
    for v in range(numVectorsPerTile)
    for e in range(numReadsPerVector)
    for r in range(nRPU)
  }))

@lru_cache(maxsize=None)
def _compute_fp32_config(mt: int, vw: int, lrvw: int,
                         miWaveGroup: int,
                         miInputPerThread: int,
                         miWaveTile: int,
                         xf32EmuPack: bool = False,
                         matrixInstM: int = 16) -> Dict[str, int]:
  rawAddrs = [(t % 16 * vw + t // 16 * mt * lrvw) * 4 for t in range(32)]
  wOffsets = tuple(w * matrixInstM * vw * 4 for w in range(max(miWaveGroup, 1)))
  instOffs = _build_fp32_instOffs(mt, vw, lrvw, miInputPerThread, miWaveTile,
                                  miWaveGroup, xf32EmuPack)
  if _b32_check(rawAddrs, B=0, P=0, wOffsets=wOffsets, instOffs=instOffs):
    return {"perBlock": 0, "pad": 0}
  cfg = _b32_search_padding(mt, rawAddrs, wOffsets, instOffs=instOffs)
  return cfg if cfg else {"perBlock": 0, "pad": 0}

def get_fp32_mt_config(mt: int, key: str, vw: int, lrvw: int,
                       miWaveGroup: int,
                       miInputPerThread: int,
                       miWaveTile: int,
                       xf32EmuPack: bool = False) -> int:
  return _compute_fp32_config(mt, vw, lrvw, miWaveGroup,
                              miInputPerThread=miInputPerThread,
                              miWaveTile=miWaveTile,
                              xf32EmuPack=xf32EmuPack)[key]

@lru_cache(maxsize=None)
def _compute_mxs_config(matrixInstK: int, mxBlock: int, vw: int) -> Dict[str, int]:
  if mxBlock <= 0 or vw <= 0:
    return {"perBlock": 0, "pad": 0}
  d = (matrixInstK // mxBlock) * vw
  if vw < 4 or d == 0 or d % 16 != 0 or (d // 16) & 1:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": 256, "pad": 16}

def get_mxs_mt_config(matrixInstK: int, mxBlock: int, vw: int, key: str) -> int:
  return _compute_mxs_config(matrixInstK, mxBlock, vw)[key]
