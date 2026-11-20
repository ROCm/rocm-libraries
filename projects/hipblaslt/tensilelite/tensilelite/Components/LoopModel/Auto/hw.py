# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hardware costs the LoopModel parameter search prices a candidate with.

The numbers MIRROR StinkyTofu, which schedules the kernel and is the source of truth; they are
copied rather than re-derived so the search and the scheduler agree on what a cycle is:

  WMMA issue/latency     shared/stinkytofu/hardware/src/gfx/Gfx1250/Gfx1250Formats.def (.cost)
  ds_load latency, rate  shared/stinkytofu/hardware/src/gfx/Gfx1250/Gfx1250Instructions.def
                         (.cost, .dsThroughput, .dsMaxDrain)
  LDS return queue       shared/stinkytofu/src/hardware/HWModel.cpp (kGfx1250Model.lds)
  LDS return rate        shared/stinkytofu/include/stinkytofu/bindings/python/Module.hpp
                         (DsReadCyclesPerLoad, the DAG scheduler's default)

The WMMA pipe is modelled the way `WmmaQueueTimeline` models it: each WMMA occupies the matrix
queue for its latency, back to back, and everything else co-issues in that shadow.

`dsEffectiveLatency`, `asmDsLatency`, `asmTdmLatency`, `tdmLatency` and the register terms below
are CALIBRATED on measured kernels, not copied: see `calibration` for the data each came from.
`dsEffectiveLatency` (300) is far above StinkyTofu's per-instruction ds_load latency (56) because
it prices a load under the steady loop's whole traffic, not an idle LDS pipe.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gfx1250Costs:
  #: Cycles one WMMA holds the matrix queue, by (M, N, K, mx).  Formats.def: WMMA {1, 8},
  #: WMMA_FLOAT8MIX_16x16x128 {1, 8}, WMMA_FLOAT8MIX_16x16x64 {1, 4}, MXWMMA {1, 8}.
  wmmaCycles: dict = field(default_factory=lambda: {
      (16, 16, 128, True): 8, (16, 16, 128, False): 8,
      (16, 16, 64, True): 4, (16, 16, 64, False): 4,
      (16, 16, 32, False): 8,
  })
  wmmaDefaultCycles: int = 8
  #: ds_load latency by bytes per lane.  Instructions.def: ds_load_b128 {1, 56}; DS format 52.
  dsLatency: dict = field(default_factory=lambda: {16: 56})
  dsDefaultLatency: int = 52
  #: LDS return rate in cycles per ds_load, the DAG scheduler's `DsReadCyclesPerLoad` default.
  dsCyclesPerLoad: int = 6
  #: HWModel.lds.readQueueDepth -- ds_loads one wave may have in flight before it stalls.
  dsQueueDepth: int = 16
  #: Widest ds_load, in bytes per lane.
  dsMaxBytes: int = 16
  #: Cycles from a unit's ds_loads issuing to its first WMMA being able to read them, under the
  #: steady loop's load.  Charged per unit whatever its size: a smaller VgprGroup sub-unit does
  #: not land sooner on hardware.  CALIBRATED (see `calibration`).
  dsEffectiveLatency: int = 300
  #: Tiers 0 and 2: tensor_load_to_lds issue to data in LDS, timed from where GIR places the
  #: load -- later than StinkyTofu schedules it, so below the hardware latency.  CALIBRATED.
  tdmLatency: int = 800
  #: Tiers 0 and 2: issue cycles of one tensor_load_to_lds, its descriptor's scalar setup
  #: (~70 SALU per load) included.  CALIBRATED (see `calibration`).
  tdmIssueCycles: int = 40
  #: Tier 3 (`asmsim`): ds_load and tensor_load_to_lds latency when timing the emitted loop.
  #: CALIBRATED (see `calibration`).
  asmDsLatency: int = 250
  asmTdmLatency: int = 2500
  #: Per-trip overhead outside the WMMA shadow: loop branch, s_barrier signal/wait, waitcnts.
  #: HWModel.barrier.signalToWaitLatency (11) + jumpOverheadCycles (6).
  tripOverhead: int = 17
  #: Steady-body instructions before the I-cache starts to miss.  CALIBRATED.
  icacheBodyInstructions: int = 16384
  #: Register file.  VGPRs are counted without alignment: StinkyTofuRegisterAllocation packs.
  maxVgpr: int = 1024
  maxSgpr: int = 106
  #: Bytes one LDS address register reaches through ds instruction offsets (regCaps
  #: maxLDSConstOffset); each further 64 KiB of a tensor's LDS extent takes one more.
  ldsConstOffsetReach: int = 65536
  #: The writer moves the B ring off A's VGPR bank (archCaps VgprBank on ISA 12.5): one or two
  #: registers after an unpacked A ring.
  vgprBankPadsB: bool = True
  #: SGPRs that do not depend on the searched parameters, and those each split operand adds.
  fixedSgpr: int = 86
  sgprPerSplitOperand: int = 2

  def wmma(self, m, n, k, mx) -> int:
    return self.wmmaCycles.get((int(m), int(n), int(k), bool(mx)), self.wmmaDefaultCycles)

  def dsLoadLatency(self, bytesPerLane) -> int:
    return self.dsLatency.get(int(bytesPerLane), self.dsDefaultLatency)


GFX1250 = Gfx1250Costs()

#: Where each calibrated constant came from, so a refit knows what to rerun.
calibration = {
    "fixedSgpr": (
        "config/maf/mxf8_loop_model.yaml and register variants, emitted CPU-only with "
        "StinkyTofuRegisterAllocation=3 (2026-11-17): .amdhsa_next_free_sgpr 86 for every "
        "register variant"),
    "dsEffectiveLatency": (
        "17 kernels of the same family measured on one gfx1250 card (4096x5120x65536, "
        "2026-11-17): the tuned config, the search's first top-8 and a one-factor sweep around "
        "the tuned config.  Exposure at 300 cycles ranks them with Spearman +0.87 against "
        "GFLOPS and puts the measured best alone at 0; 250-400 rank within 0.03 of it."),
    "asmDsLatency": (
        "27 kernels of the same family (the 17 above plus a held-out top-12 of the recalibrated "
        "search), emitted CPU-only and their LoopBeginL bodies timed: Spearman +0.90 against "
        "GFLOPS at ds 250 / tdm 2500, >= +0.88 over ds 200-300 x tdm 2000-3000, and the three "
        "fastest predicted are the three fastest measured.  Tier 0 ranks the same set at +0.44."),
    "asmTdmLatency": (
        "fitted jointly with asmDsLatency, same data.  StinkyTofu has no TDM latency to copy: "
        "EstimateAsmCyclesPass charges tensor_load_to_lds and s_wait_tensorcnt their issue "
        "cycles only (its TODO).  A tensor_load_to_lds latency microbenchmark should replace "
        "the fit."),
    "OPEN: what dsEffectiveLatency stands for": (
        "Both latencies above are fits to kernel rankings and may overfit.  With StinkyTofu's "
        "own LDS model instead -- ds_load_b128 56 cycles, returns 6 apart, depth 16, one queue "
        "across A, B and the MX scales -- tier 0 leaves no latency exposed on any of the 27 "
        "kernels (~158 loads x 6 < 1280 WMMA cycles; the shortest distance, 20 WMMAs, covers "
        "56), so every kernel scores the 1297 floor and the tie-break picks PLRA1/VAA2 "
        "(8.67) over the tuned kernel (9.33).  Ablating tier 3 at ds 56 locates the real "
        "signal: Spearman +0.82 with every wait, +0.79 without s_wait_dscnt, +0.585 without "
        "s_wait_tensorcnt.  The fast kernels keep two TDM loads in flight (s_wait_tensorcnt 2), "
        "the slowest drain them every other copy (s_wait_tensorcnt 1/0).  So the 300 stands "
        "in for TDM stalls, not ds latency.  Next: derive from the code generator what sets "
        "the tensorcnt depth of the steady loop, model it in tier 0, and price ds with "
        "StinkyTofu's numbers (perf.drainLatency is its computeDynamicDrainLatency already).  "
        "Prototype: ~/loopmodel-autoselect-scratch/queue_fit.py; ablation over bodies_*.pkl."),
    "tdmIssueCycles": (
        "40 kernels of the same family measured on one gfx1250 card (4096x5120x65536, "
        "2026-11-19): 27 unsplit and 13 TDMSplitA/B variants of 7 loop configurations.  12 of "
        "the 13 splits are slower than their unsplit kernel; the median slowdown per extra "
        "tensor_load per trip is 43 cycles at tier 0's scale and 42 at tier 2's.  The emitted "
        "loops agree: each extra load adds ~70 scalar instructions.  Any value > 0 orders the "
        "22 same-configuration split pairs 75% right at tier 0 (ties 50% without it)."),
    "tdmLatency": (
        "Same 40 kernels, tier 2 swept 0-2000 at tdmIssueCycles 40: 600-1000 is a plateau "
        "where only the two slowest kernels (WOO0 WFN1, a tensor_load GIR places 31 WMMAs "
        "before its first ds_load) wait, raising Spearman from -0.06 to +0.05; from 1100 most "
        "kernels wait and it falls to -0.30 at 2000.  The emitted slack (tensor_load to the "
        "s_wait_tensorcnt that retires it) ranks GFLOPS at +0.57 but GIR's does not predict it "
        "(-0.21): StinkyTofu hoists the loads and StinkyWaitCntInsertionPass sets the counts "
        "on the scheduled loop, e.g. tensorcnt 0 every trip for a one-copy body.  Only tier 3 "
        "(+0.82 on these 40) sees that.  With both TDM terms tier 0 ranks them +0.52 (+0.54 "
        "without, its split pairs tied) and tier 2 +0.05 (-0.07 without).  Tier 0's "
        "`perf.tdmExposed` reuses it: 6 variants (MXF8/F8 16x16x128 8x10 and 4x4, F8 16x16x64 "
        "8x10, BF16 16x16x32 8x8 and 4x4; 14-24 kernels each, measured twice, 2026-11-19) "
        "rank at a mean +0.41 within-variant Spearman (+0.19 without), flat from 600 up, "
        "and the 40 above stay at +0.51."),
    "icacheBodyInstructions": "not yet calibrated on hardware; P5",
}


def costsFor(isa):
  """The cost table for `isa`, or None when the search has none for it."""
  if tuple(isa or ()) == (12, 5, 0):
    return GFX1250
  return None
