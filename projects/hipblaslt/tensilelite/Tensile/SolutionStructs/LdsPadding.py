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

Returns (LdsBlockSizePerPad, ldsPad) for the ds_load_tr* read paths:

  FP4  -- ds_load_tr4_b64   (2 banks per thread)
  FP8  -- ds_load_tr8_b64   (2 banks per thread)
  FP16 -- ds_load_tr16_b128 (4 banks per thread)
  FP32 -- ds_load_b32       (1 bank per thread)

Each path builds every legal (block, pad) candidate, including no padding,
and takes the lowest rank: largest per-wave cost, summed cost, LDS overhead
pad / block, then the larger block. Cost of one instruction is the largest
number of threads on one bank.

ldsPad in bytes is always an even number of dwords.

The MX scale table (get_mxs_mt_config) is a fixed lookup, not a search.

Entry points: get_fp4_mt_config, get_fp8_mt_config, get_fp16_mt_config,
get_fp32_mt_config, get_mxs_mt_config, get_metadata_mt_config. key is
"perBlock" or "pad".
"""

from collections import namedtuple
from functools import lru_cache
from typing import Dict

# Common.ValidParameters reads these too, so a yaml can name what is picked here.
from ..Common.LdsPaddingLimits import (
    LDS_MAX_PAD_BYTES as _LDS_MAX_PAD_BYTES,
    LDS_PAD_STEP_BYTES as _LDS_PAD_STEP_BYTES,
    LDS_PAD_BLOCK_BYTES as _LDS_PAD_BLOCK_BYTES,
)

def _pad_candidate_bytes(step: int = _LDS_PAD_STEP_BYTES):
  """Legal LdsPad values in bytes, smallest first.

  step must be a multiple of 8. The b128 path passes 16 because a pad that
  is a multiple of 8 but not 16 breaks its 16-byte alignment.
  """
  return range(step, _LDS_MAX_PAD_BYTES + 1, step)

def _even_dword_only(cfg: Dict[str, int], bpeDS: float) -> Dict[str, int]:
  """Return cfg, or an all-zero config if pad is an odd number of dwords.

  cfg["pad"] counts elements, so 8 bytes is 8 / bpeDS elements.
  """
  padStepElements = int(_LDS_PAD_STEP_BYTES / bpeDS)
  if cfg["pad"] % padStepElements:
    return {key: 0 for key in cfg}
  return cfg

# FP4/FP8 share ds_load_tr*_b64 (64-bit/thread, 2 banks/thread).

def _b64_base_addrs_fp4(mt: int) -> list:
  """Per-lane base addresses for FP4, one entry per lane of the wave."""
  bpeDS = 0.5
  addrs = []
  for w in range(32):
    t = (w // 16) * 8 + ((w % 16) // 8) * 32 + w % 8
    addrs.append(int(t * mt * bpeDS))
  return addrs

def _b64_base_addrs_fp8(mt: int) -> list:
  """Per-lane base addresses for FP8, one entry per lane of the wave."""
  k  = (0,1,2,3,0,1,2,3,4,5,6,7,4,5,6,7,
        16,17,18,19,16,17,18,19,20,21,22,23,20,21,22,23)
  t  = (0,0,0,0,8,8,8,8,0,0,0,0,8,8,8,8,
        0,0,0,0,8,8,8,8,0,0,0,0,8,8,8,8)
  return [t[i] + k[i] * mt for i in range(32)]

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

def _no_block_carry(bases, instOffs, B) -> bool:
  """True when no (base, instOff) pair carries across a block boundary at B.

  With pad(x) = x + (x // B) * P, pad(a) + pad(b) equals pad(a + b) only
  when (a mod B) + (b mod B) < B. The code generator pads the base and the
  ds immediate offset separately and adds them, so a carrying pair reads the
  wrong bytes. The carry depends on B only, not on P.

  bases and instOffs are in the same byte units as B.
  """
  return all((a % B) + (b % B) < B for a in bases for b in instOffs)

def _min_block_bytes(usesTDM: bool) -> int:
  """Smallest block the local write can tolerate.

  Padding goes in at block boundaries, never inside one store, so a block
  smaller than one ds_write would split it. b128 is the widest store the
  generator emits, wider ones are split into b128, so 16 bytes covers it.

  0 for TDM, which emits no local write.
  """
  return 0 if usesTDM else 16

def _write_row_bytes(mt: int, bpeDS: float, usesTDM: bool) -> int:
  """Row stride of the local write. 0 for TDM, which has no per-thread write."""
  return 0 if usesTDM else int(mt * bpeDS)

def _valid_blocks(incBytes, readBases, readOffs, writeMinBytes, writeRowBytes):
  """Block sizes the code generator can address correctly.

  Padding is phi(x) = x + (x // b) * P, applied to the base and to the
  instruction offset separately. A pair that carries across a block boundary
  then lands one pad short.

  incBytes       the tail loop step; b has to divide it
  readBases,
  readOffs       local read addresses, checked for that carry
  writeRowBytes  the LDS row; keep it and the block from cutting each other
  writeMinBytes  one ds_store has to fit inside a block

  Both write terms are 0 under TDM, which emits no local write.
  """
  def usable(b):
    if b < writeMinBytes:
      return False
    if incBytes % b:
      return False
    if writeRowBytes and writeRowBytes % b and b % writeRowBytes:
      return False
    return _no_block_carry(readBases, readOffs, b)
  return [b for b in _LDS_PAD_BLOCK_BYTES if usable(b)]

def _search_padding(validB, padStep, floorFn, costFn):
  """Rank every legal (B, P) candidate and return the best one.

  Rank: largest per-wave cost, summed cost, LDS overhead P / B, then the
  larger block. No padding, (0, 0), is always a candidate.

  floorFn() gives the lowest cost a wave can reach, or None when every
  candidate is illegal. Growing P past the floor can only cost more LDS,
  never less bank pressure, so the search drops the rest of that B.

  costFn(cand) gives the per-wave costs of one candidate, or None when it
  is illegal. Returns None if every candidate is illegal.
  """
  candidates = [(0, 0)]
  cache = {}
  floor = floorFn()
  if floor is not None:
    for B in validB:
      for P in _pad_candidate_bytes(padStep):
        cand = (B, P)
        candidates.append(cand)
        costs = costFn(cand)
        cache[cand] = costs
        if costs is not None and max(costs) <= floor:
          break
  return _pick_best(candidates,
                    lambda cand: cache[cand] if cand in cache else costFn(cand))

def _pick_best(candidates, costFn):
  """Return the candidate with the lowest rank, or None if all are illegal.

  Rank: largest per-wave cost, summed cost, LDS overhead P / B, then the
  larger block. Every candidate starts with (B, P). costFn returns the
  per-wave costs of one candidate, or None when it is illegal.

  The last key prefers the larger block because the block also bounds how
  wide a chunk TDM writes to LDS at a time, which the read-side cost above
  does not see. Two candidates tie on LDS overhead when P / B is equal, and
  the larger block then carries a proportionally larger P, so preferring it
  costs no extra LDS.
  """
  best = None
  bestKey = None
  for cand in candidates:
    B, P = cand[0], cand[1]
    costs = costFn(cand)
    if costs is None:
      continue
    # The no-padding candidate is (0, 0), so B == 0 means zero overhead.
    key = (max(costs), sum(costs), P / B if B else 0, -B)
    if bestKey is None or key < bestKey:
      bestKey = key
      best = cand
  return best

def _b64_wave_costs(rawAddrs, B, P, instOffs, wOffsets):
  """Cost of one (B, P) candidate, one entry per wave.

  A thread reads pad(natural + wOff) + pad(instOff), matching the code
  generator. Cost per instruction is the largest number of threads on one
  bank, summed over the instructions a wave issues.

  Every address is 8-byte aligned -- bases, per-wave offsets, instruction
  offsets and P are all multiples of 8 -- so the hardware takes each wave in
  one batch. Returns None if a candidate breaks that.
  """
  def pad(x):
    return x + (x // B) * P if B else x
  perWave = []
  for wOff in wOffsets:
    base = [pad(a + wOff) for a in rawAddrs]
    total = 0
    for instOff in instOffs:
      delta = pad(instOff)
      addrs = [a + delta for a in base]
      if any(a % 8 for a in addrs):
        return None
      total += _max_threads_per_bank(addrs, 2)
    perWave.append(total)
  return perWave

# One operand's local read pattern, in bytes. _valid_blocks turns it into the
# blocks the generator can address; Solution.py checks a hand-written block
# against the same list.
_Shape = namedtuple("_Shape",
                    "rawAddrs instOffs wOffsets incBytes minBlockBytes writeRowBytes")

def _valid_blocks_for(shape: _Shape) -> list:
  """Block sizes the code generator can address correctly for this operand."""
  bases = [a + wOff for a in shape.rawAddrs for wOff in shape.wOffsets]
  return _valid_blocks(shape.incBytes, bases, shape.instOffs,
                       shape.minBlockBytes, shape.writeRowBytes)

def _b64_compute_config(shape: _Shape) -> Dict[str, int]:
  """Pick (B, P) by ranking every legal candidate.

  No block padding is one of the candidates. Returns {"perBlock", "pad"}
  with pad in bytes.
  """
  best = _search_padding(
    _valid_blocks_for(shape),
    _LDS_PAD_STEP_BYTES,
    lambda: len(shape.instOffs),   # one thread per bank on every instruction
    lambda cand: _b64_wave_costs(shape.rawAddrs, cand[0], cand[1],
                                 shape.instOffs, shape.wOffsets))
  # The no-padding candidate is always legal here, so best is never None
  # today. Kept in case a future change narrows the candidate set.
  if best is None:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": best[0], "pad": best[1]}

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
def _fp4_shape(mt: int, miWaveTile: int, miWaveGroup: int,
               matrixInstK: int, usesTDM: bool,
               lrvw: int = 32, miInputPerThread: int = 64,
               matrixInstM: int = 16) -> _Shape:
  # FP4: bpeDS=0.5. Per-wave M shift in BYTES = matrixInstM * bpeDS = 8.
  return _Shape(_b64_base_addrs_fp4(mt),
                _b64_emit_instOffs(mt, 0.5, lrvw, miInputPerThread, miWaveTile),
                _b64_w_offsets(miWaveGroup, int(matrixInstM * 0.5)),
                int(mt * 0.5) * matrixInstK,
                _min_block_bytes(usesTDM),
                _write_row_bytes(mt, 0.5, usesTDM))

@lru_cache(maxsize=None)
def _compute_fp4_config(mt: int, miWaveTile: int, miWaveGroup: int,
                        matrixInstK: int, usesTDM: bool) -> Dict[str, int]:
  result = _b64_compute_config(
    _fp4_shape(mt, miWaveTile, miWaveGroup, matrixInstK, usesTDM))
  # bpeDS=0.5 -> convert pad from bytes to elements
  return {"perBlock": result["perBlock"], "pad": result["pad"] * 2}

def get_fp4_mt_config(mt: int, key: str, miWaveTile: int, miWaveGroup: int,
                      matrixInstK: int, usesTDM: bool) -> int:
  return _even_dword_only(
    _compute_fp4_config(mt, miWaveTile, miWaveGroup, matrixInstK, usesTDM), 0.5)[key]

def get_fp4_valid_blocks(mt: int, miWaveTile: int, miWaveGroup: int,
                         matrixInstK: int, usesTDM: bool) -> tuple:
  return tuple(_valid_blocks_for(
    _fp4_shape(mt, miWaveTile, miWaveGroup, matrixInstK, usesTDM)))

@lru_cache(maxsize=None)
def _fp8_shape(mt: int, miWaveTile: int, miWaveGroup: int,
               matrixInstK: int, usesTDM: bool,
               incDivisor: int = 1, lrvw: int = 16,
               miInputPerThread: int = 64, matrixInstM: int = 16) -> _Shape:
  # FP8: bpeDS=1, pad already in bytes. Per-wave M shift = matrixInstM = 16.
  # incDivisor matches the tail loop, which divides its step for metadata.
  return _Shape(_b64_base_addrs_fp8(mt),
                _b64_emit_instOffs(mt, 1.0, lrvw, miInputPerThread, miWaveTile),
                _b64_w_offsets(miWaveGroup, matrixInstM * 1),
                mt * matrixInstK // incDivisor,
                _min_block_bytes(usesTDM),
                _write_row_bytes(mt, 1.0, usesTDM))

@lru_cache(maxsize=None)
def _compute_fp8_config(mt: int, miWaveTile: int, miWaveGroup: int,
                        matrixInstK: int, usesTDM: bool,
                        incDivisor: int = 1,
                        lrvw: int = 16,
                        miInputPerThread: int = 64) -> Dict[str, int]:
  return _b64_compute_config(
    _fp8_shape(mt, miWaveTile, miWaveGroup, matrixInstK, usesTDM,
               incDivisor, lrvw, miInputPerThread))

def get_fp8_mt_config(mt: int, key: str, miWaveTile: int, miWaveGroup: int,
                      matrixInstK: int, usesTDM: bool) -> int:
  return _even_dword_only(
    _compute_fp8_config(mt, miWaveTile, miWaveGroup, matrixInstK, usesTDM), 1.0)[key]

def get_fp8_valid_blocks(mt: int, miWaveTile: int, miWaveGroup: int,
                         matrixInstK: int, usesTDM: bool) -> tuple:
  return tuple(_valid_blocks_for(
    _fp8_shape(mt, miWaveTile, miWaveGroup, matrixInstK, usesTDM)))

def get_metadata_mt_config(mt: int, key: str, miWaveTile: int, miWaveGroup: int,
                           lrvwBytes: int, miInputPerThreadBytes: int,
                           matrixInstK: int, usesTDM: bool) -> int:
  """Padding for sparse metadata TileMajor (enableLDSTrMetadata).

  Metadata is byte-typed, so the FP8 search applies unchanged. Only the
  instruction-offset counts differ: pass LocalReadVectorWidthMetadata and
  MIInputPerThreadMetadata already converted to metadata bytes. The tail
  loop step is an eighth of the dense one.
  """
  return _even_dword_only(
    _compute_fp8_config(mt, miWaveTile, miWaveGroup, matrixInstK, usesTDM,
                        incDivisor=8,
                        lrvw=max(lrvwBytes, 1),
                        miInputPerThread=max(miInputPerThreadBytes, 1)), 1.0)[key]

# -- FP16 b128 padding ---------------------------------------------

def _b128_base_addrs_fp16(mt: int) -> list:
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
def _fp16_shape(mt: int, miWaveGroup: int, miInputPerThUnroll: int, lrvw: int,
                miWaveTile: int, vw: int, matrixInstK: int, usesTDM: bool,
                matrixInstM: int = 16) -> _Shape:
  instOffs = _build_fp16_instOffs(mt, miInputPerThUnroll, lrvw,
                                  miWaveTile, miWaveGroup, vw)
  return _Shape(_b128_base_addrs_fp16(mt),
                instOffs if instOffs else (0,),
                tuple(w * matrixInstM * 2 for w in range(max(miWaveGroup, 1))),
                mt * 2 * matrixInstK,
                _min_block_bytes(usesTDM),
                _write_row_bytes(mt, 2.0, usesTDM))

@lru_cache(maxsize=None)
def _compute_fp16_config(mt: int, miWaveGroup: int,
                         miInputPerThUnroll: int,
                         lrvw: int,
                         miWaveTile: int,
                         vw: int,
                         matrixInstK: int, usesTDM: bool) -> Dict[str, int]:
  """Pick (B, P) for ds_load_tr16_b128 by ranking every legal candidate.

  No block padding is one of the candidates. Returns {"perBlock", "pad"}
  with pad in elements.

  P steps by 16 bytes here, see _pad_candidate_bytes, and stops growing
  once a candidate reaches one thread per bank on every instruction.
  """
  shape = _fp16_shape(mt, miWaveGroup, miInputPerThUnroll, lrvw,
                      miWaveTile, vw, matrixInstK, usesTDM)
  best = _search_padding(
    _valid_blocks_for(shape),
    16,
    lambda: len(shape.instOffs),   # one thread per bank on every instruction
    lambda cand: _b128_wave_costs(shape.rawAddrs, cand[0], cand[1],
                                  shape.wOffsets, shape.instOffs))
  if best is None:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": best[0], "pad": best[1] // 2}

def get_fp16_mt_config(mt: int, key: str, miWaveGroup: int,
                       miInputPerThUnroll: int, lrvw: int,
                       miWaveTile: int, vw: int, matrixInstK: int,
                       usesTDM: bool) -> int:
  return _even_dword_only(_compute_fp16_config(mt, miWaveGroup,
                                               miInputPerThUnroll=miInputPerThUnroll,
                                               lrvw=lrvw,
                                               miWaveTile=miWaveTile,
                                               vw=vw,
                                               matrixInstK=matrixInstK,
                                               usesTDM=usesTDM), 2.0)[key]

def get_fp16_valid_blocks(mt: int, miWaveGroup: int, miInputPerThUnroll: int,
                          lrvw: int, miWaveTile: int, vw: int,
                          matrixInstK: int, usesTDM: bool) -> tuple:
  return tuple(_valid_blocks_for(
    _fp16_shape(mt, miWaveGroup, miInputPerThUnroll, lrvw,
                miWaveTile, vw, matrixInstK, usesTDM)))

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
def _fp32_shape(mt: int, vw: int, lrvw: int, miWaveGroup: int,
                miInputPerThread: int, miWaveTile: int,
                matrixInstK: int, usesTDM: bool,
                xf32EmuPack: bool = False, matrixInstM: int = 16) -> _Shape:
  instOffs = _build_fp32_instOffs(mt, vw, lrvw, miInputPerThread, miWaveTile,
                                  miWaveGroup, xf32EmuPack)
  return _Shape([(t % 16 * vw + t // 16 * mt * lrvw) * 4 for t in range(32)],
                instOffs if instOffs else (0,),
                tuple(w * matrixInstM * vw * 4 for w in range(max(miWaveGroup, 1))),
                mt * 4 * matrixInstK,
                _min_block_bytes(usesTDM),
                _write_row_bytes(mt, 4.0, usesTDM))

@lru_cache(maxsize=None)
def _compute_fp32_config(mt: int, vw: int, lrvw: int,
                         miWaveGroup: int,
                         miInputPerThread: int,
                         miWaveTile: int,
                         matrixInstK: int, usesTDM: bool,
                         xf32EmuPack: bool = False) -> Dict[str, int]:
  """Pick (B, P) for ds_load_b32 by ranking every legal candidate.

  No block padding is one of the candidates. Returns {"perBlock", "pad"}
  with pad in dwords.
  """
  shape = _fp32_shape(mt, vw, lrvw, miWaveGroup, miInputPerThread, miWaveTile,
                      matrixInstK, usesTDM, xf32EmuPack)
  best = _search_padding(
    _valid_blocks_for(shape),
    _LDS_PAD_STEP_BYTES,
    lambda: len(shape.instOffs),   # one thread per bank on every instruction
    lambda cand: _b32_wave_costs(shape.rawAddrs, cand[0], cand[1],
                                 shape.wOffsets, shape.instOffs))
  if best is None:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": best[0], "pad": best[1] // 4}

def get_fp32_mt_config(mt: int, key: str, vw: int, lrvw: int,
                       miWaveGroup: int,
                       miInputPerThread: int,
                       miWaveTile: int,
                       matrixInstK: int, usesTDM: bool,
                       xf32EmuPack: bool = False) -> int:
  return _even_dword_only(_compute_fp32_config(mt, vw, lrvw, miWaveGroup,
                                               miInputPerThread=miInputPerThread,
                                               miWaveTile=miWaveTile,
                                               matrixInstK=matrixInstK,
                                               usesTDM=usesTDM,
                                               xf32EmuPack=xf32EmuPack), 4.0)[key]

def get_fp32_valid_blocks(mt: int, vw: int, lrvw: int, miWaveGroup: int,
                          miInputPerThread: int, miWaveTile: int,
                          matrixInstK: int, usesTDM: bool,
                          xf32EmuPack: bool = False) -> tuple:
  return tuple(_valid_blocks_for(
    _fp32_shape(mt, vw, lrvw, miWaveGroup, miInputPerThread, miWaveTile,
                matrixInstK, usesTDM, xf32EmuPack)))

# The one pair the MX scale layout uses when it pads at all.
MXS_LDS_BLOCK_BYTES = 256
MXS_LDS_PAD_BYTES   = 16

@lru_cache(maxsize=None)
def _compute_mxs_config(matrixInstK: int, mxBlock: int, vw: int) -> Dict[str, int]:
  if mxBlock <= 0:
    return {"perBlock": 0, "pad": 0}
  d = (matrixInstK // mxBlock) * vw
  if vw < 4 or d == 0 or d % 16 != 0 or (d // 16) & 1:
    return {"perBlock": 0, "pad": 0}
  return {"perBlock": MXS_LDS_BLOCK_BYTES, "pad": MXS_LDS_PAD_BYTES}

def get_mxs_mt_config(matrixInstK: int, mxBlock: int, vw: int, key: str) -> int:
  # The pad is a fixed 16 bytes, already an even number of dwords, so the
  # check the other entry points carry has nothing to do here.
  return _compute_mxs_config(matrixInstK, mxBlock, vw)[key]
