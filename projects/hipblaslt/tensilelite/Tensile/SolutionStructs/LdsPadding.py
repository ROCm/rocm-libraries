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

"""LDS padding for gfx1250 tile-major layouts.

Returns (LdsBlockSizePerPad, ldsPad, shift) for the ds_load_tr* read paths:

  FP4  -- ds_load_tr4_b64   (2 banks per thread)
  FP8  -- ds_load_tr8_b64   (2 banks per thread)
  FP16 -- ds_load_tr16_b128 (4 banks per thread)
  FP32 -- ds_load_b32       (1 bank per thread)

Each path builds every legal (block, pad) candidate, including no padding,
and takes the lowest rank: largest per-wave cost, summed cost, LDS overhead
pad / block, then pad. Cost of one instruction is the largest number of
threads on one bank.

ldsPad in bytes is always an even number of dwords.

The MX scale table (get_mxs_mt_config) is a fixed lookup, not a search.

Entry points: get_fp4_mt_config, get_fp8_mt_config, get_fp16_mt_config,
get_fp32_mt_config, get_mxs_mt_config, get_metadata_mt_config. key is
"perBlock", "pad", or "shift" (FP4/FP8 only).
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

# LdsPad in bytes must be an even number of dwords. Odd-dword padding leaves
# the hardware in a state we do not understand, so it is never a candidate.
_TDM_PAD_STEP_BYTES = 8

def _pad_candidate_bytes(step: int = _TDM_PAD_STEP_BYTES):
  """Legal LdsPad values in bytes, smallest first.

  step must be a multiple of 8. The b128 path passes 16 because a pad that
  is a multiple of 8 but not 16 breaks its 16-byte alignment.
  """
  return range(step, _TDM_MAX_PAD_BYTES + 1, step)

def _even_dword_only(cfg: Dict[str, int], bpeDS: float) -> Dict[str, int]:
  """Return cfg, or an all-zero config if pad is an odd number of dwords.

  cfg["pad"] counts elements, so 8 bytes is 8 / bpeDS elements.
  """
  padStepElements = int(_TDM_PAD_STEP_BYTES / bpeDS)
  if cfg["pad"] % padStepElements:
    return {key: 0 for key in cfg}
  return cfg

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

def _max_threads_per_bank(addrs, banksPerThread: int) -> int:
  """Largest number of threads landing on any one of the 64 LDS banks.

  A thread whose load is `banksPerThread` dwords wide occupies that many
  consecutive banks starting at its own address.
  """
  counts = {}
  for a in addrs:
    first = (a >> 2) % 64
    for i in range(banksPerThread):
      bank = (first + i) % 64
      counts[bank] = counts.get(bank, 0) + 1
  return max(counts.values())

def _valid_blocks(mtBytes, bases, instOffs, minB):
  """Block sizes that keep the padded address the code generator computes.

  The generator pads the base register and the ds instruction offset
  separately and adds them, so a read only lands on the byte that was
  written when the row span and the block size divide one another.
  """
  return [b for b in _TDM_VALID_BLOCK_BYTES
          if b >= minB and (mtBytes % b == 0 or b % mtBytes == 0)]

def _search_padding(validB, padStep, shifts, floorFn, costFn):
  """Rank every legal (B, P, shift) candidate and return the best one.

  Rank: largest per-wave cost, summed cost, LDS overhead P / B, then P. No
  padding, (0, 0), is a candidate at every shift.

  floorFn(shift) gives the lowest cost a wave can reach at that shift, or
  None when every candidate at that shift is illegal. Growing P past the
  floor can only cost more LDS, never less bank pressure, so the search
  drops the rest of that (B, shift).

  costFn(cand) gives the per-wave costs of one candidate, or None when it
  is illegal. Returns None if every candidate is illegal.
  """
  candidates = [(0, 0, shift) for shift in shifts]
  cache = {}
  for shift in shifts:
    floor = floorFn(shift)
    if floor is None:
      continue
    for B in validB:
      for P in _pad_candidate_bytes(padStep):
        cand = (B, P, shift)
        candidates.append(cand)
        costs = costFn(cand)
        cache[cand] = costs
        if costs is not None and max(costs) <= floor:
          break
  return _pick_best(candidates,
                    lambda cand: cache[cand] if cand in cache else costFn(cand))

def _pick_best(candidates, costFn):
  """Return the candidate with the lowest rank, or None if all are illegal.

  Rank: largest per-wave cost, summed cost, LDS overhead P / B, then P.
  Every candidate starts with (B, P). costFn returns the per-wave costs of
  one candidate, or None when it is illegal.
  """
  best = None
  bestKey = None
  for cand in candidates:
    B, P = cand[0], cand[1]
    costs = costFn(cand)
    if costs is None:
      continue
    # The no-padding candidate is (0, 0), so B == 0 means zero overhead.
    key = (max(costs), sum(costs), P / B if B else 0, P)
    if bestKey is None or key < bestKey:
      bestKey = key
      best = cand
  return best

def _b64_wave_costs(half0, half1, B, P, instOffs, wOffsets, shift):
  """Cost of one (B, P, shift) candidate, one entry per wave.

  A thread reads pad(natural + wOff) + shift + pad(instOff), matching the
  code generator.

  When all 32 addresses are 8-byte aligned the hardware takes the wave in
  one batch, and the cost is the largest number of threads on one bank.
  Otherwise it takes two batches of 16, and the cost is twice the larger
  batch. Costs are summed over the instructions a wave issues.

  Returns None when an address is below dword alignment.
  """
  def pad(x):
    return x + (x // B) * P if B else x
  perWave = []
  for wOff in wOffsets:
    base0 = [pad(a + wOff) for a in half0]
    base1 = [pad(a + wOff) for a in half1]
    total = 0
    for instOff in instOffs:
      delta = shift + pad(instOff)
      addr0 = [a + delta for a in base0]
      addr1 = [a + delta for a in base1]
      addrAll = addr0 + addr1
      if any(p % 4 for p in addrAll):
        return None
      if all(p % 8 == 0 for p in addrAll):
        total += _max_threads_per_bank(addrAll, 2)
      else:
        total += 2 * max(_max_threads_per_bank(addr0, 2),
                         _max_threads_per_bank(addr1, 2))
    perWave.append(total)
  return perWave

def _b64_min_possible_cost(half0, half1, instOffs, wOffsets, shift):
  """Lowest per-wave cost any (B, P) can reach at this shift.

  P is always a multiple of 8, so pad(x) keeps x mod 4 and x mod 8. Dword
  alignment and the choice of batching mode therefore do not depend on
  (B, P), and the floor -- 1 per instruction for one batch, 2 for two --
  can be computed once from the unpadded addresses.

  Returns None when an instruction is illegal at this shift for every
  (B, P).
  """
  best = 0
  for wOff in wOffsets:
    total = 0
    for instOff in instOffs:
      delta = shift + instOff
      addrAll = [a + wOff + delta for a in half0] + [a + wOff + delta for a in half1]
      if any(p % 4 for p in addrAll):
        return None
      total += 1 if all(p % 8 == 0 for p in addrAll) else 2
    best = max(best, total)
  return best

def _b64_compute_config(mt: int, bpeDS: float,
                        addrFn, minB: int,
                        instOffs, wOffsets) -> Dict[str, int]:
  """Pick (B, P, shift) by ranking every legal candidate.

  No block padding is one of the candidates. Returns {"perBlock", "pad",
  "shift"} with pad in bytes.
  """
  mtBytes = int(mt * bpeDS)
  half0, half1 = addrFn(mt)
  bases = [a + wOff for a in half0 + half1 for wOff in wOffsets]
  best = _search_padding(
    _valid_blocks(mtBytes, bases, instOffs, minB),
    _TDM_PAD_STEP_BYTES, (0, 4),
    lambda shift: _b64_min_possible_cost(half0, half1, instOffs, wOffsets, shift),
    lambda cand: _b64_wave_costs(half0, half1, cand[0], cand[1],
                                 instOffs, wOffsets, cand[2]))
  # The no-padding candidates are always legal here, so best is never None
  # today. Kept in case a future change narrows the candidate set.
  if best is None:
    return {"perBlock": 0, "pad": 0, "shift": 0}
  return {"perBlock": best[0], "pad": best[1], "shift": best[2]}

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
  """Per-wave LDS shift in bytes before block padding: wave w reads from
  lroA + w * matrixInstMBytes, where matrixInstMBytes is MFMA_M * bpeDS."""
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
  return _even_dword_only(_compute_fp4_config(mt, miWaveTile, miWaveGroup), 0.5)[key]

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
  return _even_dword_only(_compute_fp8_config(mt, miWaveTile, miWaveGroup), 1.0)[key]

def get_metadata_mt_config(mt: int, key: str, miWaveTile: int, miWaveGroup: int,
                           lrvwBytes: int, miInputPerThreadBytes: int) -> int:
  """Padding for sparse metadata TileMajor (enableLDSTrMetadata).

  Metadata is byte-typed, so the FP8 search applies unchanged. Only the
  instruction-offset counts differ: pass LocalReadVectorWidthMetadata and
  MIInputPerThreadMetadata already converted to metadata bytes.
  """
  return _even_dword_only(
    _compute_fp8_config(mt, miWaveTile, miWaveGroup,
                        lrvw=max(lrvwBytes, 1),
                        miInputPerThread=max(miInputPerThreadBytes, 1)), 1.0)[key]

# -- FP16 b128 padding ---------------------------------------------

def _b128_base_addrs_fp16(mt: int) -> tuple:
  """Per-thread byte addresses for ds_load_tr16_b128 half-wave (16 threads)."""
  return [k * 2 * mt for k in range(8)] + [16 + k * 2 * mt for k in range(8)]

def _b128_wave_costs(half, B, P, wOffsets, instOffs=(0,)):
  """Cost of one (B, P) candidate for ds_load_tr16_b128, one entry per wave.

  A thread reads pad(natural + wOff) + pad(instOff) and covers 4 banks.
  Cost per instruction is the largest number of threads on one bank, summed
  over the instructions a wave issues.

  Returns None when an address is below 16-byte alignment.
  """
  def pad(x):
    return x + (x // B) * P if B else x
  offs = instOffs if instOffs else (0,)
  perWave = []
  for wOff in wOffsets:
    base = [pad(a + wOff) for a in half]
    total = 0
    for instOff in offs:
      addrs = [a + pad(instOff) for a in base]
      if any(a % 16 for a in addrs):
        return None
      total += _max_threads_per_bank(addrs, 4)
    perWave.append(total)
  return perWave

def _build_fp16_instOffs(mt: int, miInputPerThUnroll: int, lrvw: int,
                         miWaveTile: int, miWaveGroup: int, vw: int,
                         matrixInstM: int = 16) -> tuple:
  # Mirror LocalRead.py FP16 LDSTr ds_load_tr16_b128 emit
  numberLRVWPerMIInput = max(miInputPerThUnroll // max(lrvw, 1), 1)
  incrementBytes = numberLRVWPerMIInput * max(lrvw, 1) * mt * 2  # bpeDS = 2
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
  """Pick (B, P) for ds_load_tr16_b128 by ranking every legal candidate.

  No block padding is one of the candidates. Returns {"perBlock", "pad"}
  with pad in elements.

  P steps by 16 bytes here, see _pad_candidate_bytes, and stops growing
  once a candidate reaches one thread per bank on every instruction.
  """
  half = _b128_base_addrs_fp16(mt)
  wOffsets = tuple(w * matrixInstM * 2 for w in range(max(miWaveGroup, 1)))
  instOffs = _build_fp16_instOffs(mt, miInputPerThUnroll, lrvw,
                                  miWaveTile, miWaveGroup, vw)
  offs = instOffs if instOffs else (0,)
  bases = [a + wOff for a in half for wOff in wOffsets]
  best = _search_padding(
    _valid_blocks(mt * 2, bases, offs, 16),
    16, (0,),
    lambda shift: len(offs),   # one thread per bank on every instruction
    lambda cand: _b128_wave_costs(half, cand[0], cand[1], wOffsets, instOffs))

  if best is None:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": best[0], "pad": best[1] // 2}

def get_fp16_mt_config(mt: int, key: str, miWaveGroup: int,
                       miInputPerThUnroll: int, lrvw: int,
                       miWaveTile: int, vw: int) -> int:
  return _even_dword_only(_compute_fp16_config(mt, miWaveGroup,
                                               miInputPerThUnroll=miInputPerThUnroll,
                                               lrvw=lrvw,
                                               miWaveTile=miWaveTile,
                                               vw=vw), 2.0)[key]

# -- FP32 b32 padding ------------------------------------------------

def _b32_wave_costs(rawAddrs, B, P, wOffsets, instOffs=(0,)):
  """Cost of one (B, P) candidate for ds_load_b32, one entry per wave.

  A thread reads pad(natural + wOff) + pad(instOff) and occupies one bank.
  Cost per instruction is the largest number of threads on one bank, summed
  over the instructions a wave issues.

  Returns None when an address is below dword alignment.
  """
  def pad(x):
    return x + (x // B) * P if B else x
  offs = instOffs if instOffs else (0,)
  perWave = []
  for wOff in wOffsets:
    base = [pad(a + wOff) for a in rawAddrs]
    total = 0
    for instOff in offs:
      addrs = [a + pad(instOff) for a in base]
      if any(a % 4 for a in addrs):
        return None
      total += _max_threads_per_bank(addrs, 1)
    perWave.append(total)
  return perWave

def _build_fp32_instOffs(mt: int, vw: int, lrvw: int,
                         miInputPerThread: int, miWaveTile: int,
                         miWaveGroup: int,
                         xf32EmuPack: bool,
                         matrixInstM: int = 16) -> tuple:
  # Mirror LocalRead.py FP32 / XF32 ds_load_b32 emit
  nRPU = max(miInputPerThread // max(lrvw, 1), 1)
  numVectorsPerTile = max(miWaveTile // max(vw, 1), 1)
  numReadsPerVector = max(vw, 1)
  miWaveGroupShape  = matrixInstM * miWaveGroup * vw
  unrollStrideBytes = mt * 4
  if xf32EmuPack:
    kFn = lambda r: ((r // max(lrvw, 1)) * max(lrvw, 1) + r * max(lrvw, 1)) * unrollStrideBytes
  else:
    kFn = lambda r: r * max(lrvw, 1) * unrollStrideBytes
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
  """Pick (B, P) for ds_load_b32 by ranking every legal candidate.

  No block padding is one of the candidates. Returns {"perBlock", "pad"}
  with pad in dwords.
  """
  rawAddrs = [(t % 16 * vw + t // 16 * mt * lrvw) * 4 for t in range(32)]
  wOffsets = tuple(w * matrixInstM * vw * 4 for w in range(max(miWaveGroup, 1)))
  instOffs = _build_fp32_instOffs(mt, vw, lrvw, miInputPerThread, miWaveTile,
                                  miWaveGroup, xf32EmuPack)
  offs = instOffs if instOffs else (0,)
  bases = [a + wOff for a in rawAddrs for wOff in wOffsets]
  best = _search_padding(
    _valid_blocks(mt * 4, bases, offs, 0),
    _TDM_PAD_STEP_BYTES, (0,),
    lambda shift: len(offs),   # one thread per bank on every instruction
    lambda cand: _b32_wave_costs(rawAddrs, cand[0], cand[1],
                                 wOffsets, instOffs))
  if best is None:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": best[0], "pad": best[1] // 4}

def get_fp32_mt_config(mt: int, key: str, vw: int, lrvw: int,
                       miWaveGroup: int,
                       miInputPerThread: int,
                       miWaveTile: int,
                       xf32EmuPack: bool = False) -> int:
  return _even_dword_only(_compute_fp32_config(mt, vw, lrvw, miWaveGroup,
                                               miInputPerThread=miInputPerThread,
                                               miWaveTile=miWaveTile,
                                               xf32EmuPack=xf32EmuPack), 4.0)[key]

@lru_cache(maxsize=None)
def _compute_mxs_config(matrixInstK: int, mxBlock: int, vw: int) -> Dict[str, int]:
  # Both entry points return before calling this when MXBlock is 0.
  d = (matrixInstK // mxBlock) * vw
  if vw < 4 or d == 0 or d % 16 != 0 or (d // 16) & 1:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": 256, "pad": 16}

def get_mxs_mt_config(matrixInstK: int, mxBlock: int, vw: int, key: str) -> int:
  # The pad is a fixed 16 bytes, already an even number of dwords, so the
  # check the other entry points carry has nothing to do here.
  return _compute_mxs_config(matrixInstK, mxBlock, vw)[key]
