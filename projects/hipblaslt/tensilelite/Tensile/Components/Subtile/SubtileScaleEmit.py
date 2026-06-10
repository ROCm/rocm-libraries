# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# Scale GR/LR emit for MX scale factor operands (MXSA/MXSB).
#
# Scale factors use a simpler access pattern than data tiles:
#   GR: DTL with linear offset (serial * loadWidth), one buffer_load per wave
#   LR: ds_read_b32 per scale group (2 M-adjacent subtiles per group)
#
# This module is now a minimal boundary facade. The rocisa construction of the
# scale GR/LR *data-movement* leaves (DTL buffer_load, ds_read_b32, scale SRD
# pointer update, GR/LR LDS swap) lives in the C++ ModuleBuilder
# (cpp/include/tensile_writer/rocisa_module_builder.hpp); the functions below
# resolve writer-owned register state (sharedVgprGROffset / sharedVgprLROffset,
# destination tile VGPRs) and the MXBlock guard, then delegate construction to
# C++. See cpp_migration/docs/rocisa_module_builder_boundary.md.
#
# The scale *offset-assignment* emission (graTileAssignmentScaleSwizzled /
# lraTileAssignmentScaleSwizzled / globalReadScaleSwizzledDTLInitCommonSgpr)
# stays in Python because it allocates from the writer's register pools, which
# the boundary contract keeps authoritative in Python. Those functions already
# source their scalar math from the C++ MXScaleTileInfoQuery offset-assign plans
# (TileInfo.scaleGrOffsetAssignPlan / scaleLrOffsetAssignPlan).
################################################################################

import math

from rocisa.code import Module
from rocisa.container import vgpr, sgpr
from rocisa.instruction import (
    SAddU32, SLShiftLeftB32, SMovB32, SNop, SXorB32,
    VAddU32, VAndB32, VMulLOU32, VReadfirstlaneB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
)

from tensile_writer.subtile.module_builder import ModuleBuilder

# Single cached C++ rocisa module-builder. The builder owns no writer state; it
# only assembles rocisa Items from primitive ints/strings the boundary functions
# below resolve from the writer's register pools. See
# cpp_migration/docs/rocisa_module_builder_boundary.md.
_MODULE_BUILDER = None


def _builder():
  global _MODULE_BUILDER
  if _MODULE_BUILDER is None:
    _MODULE_BUILDER = ModuleBuilder()
  return _MODULE_BUILDER


# ---------------------------------------------------------------------------
# Scale GR/LR data-movement boundary leaves (rocisa construction in C++)
# ---------------------------------------------------------------------------

def globalReadDoScaleSubtile(tc, writer, kernel):
  """Scale GR: load scale bytes global -> LDS via DTL BufferLoadB128.

  Boundary call: M0 is set to the scale LDS base and sharedVgprGROffset[0]
  serves as both the global read offset (from SRD) and the LDS write offset
  (from M0). The rocisa construction lives in the C++ ModuleBuilder.
  """
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return Module()

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  assert len(tileInfo.sharedVgprGROffset) > 0, "Scale GR requires at least 1 GR offset VGPR"

  return _builder().scale_gr_load(tc, isGlc, isSlc, isNT, tileInfo.sharedVgprGROffset[0])


def emitScaleDsRead(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k=-1):
  """Scale LR: read 4 scale bytes (one E8M0 group) from LDS via ds_read_b32.

  Boundary call: the destination tile VGPR (vdst), the sharedVgprLROffset[0]
  address VGPR (addrVgpr), and the constant DS offset are writer-resolved; the
  rocisa construction lives in the C++ ModuleBuilder. ``k`` carries the K index
  into the comment for the scheduler emit path (k<0 omits it for the PGR=0
  path).
  """
  return _builder().scale_ds_read(tc, vdst, addrVgpr, dsOffset, scaleGroupIdx, k)


def localReadDoScaleSubtile(tc, writer, kernel):
  """Emit scale ds_reads for all scale groups (PGR=0 path)."""
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  if tileInfo.mxBlock == 0:
    return module

  # TileInfo LR subtile (2,2) already spans 2 M-adjacent tiles -> stride = lrSubtileSize.
  # Legacy TileInfo subtile (1,2) spans 1 M-tile -> stride = 2 * subtileSize.
  if hasattr(tileInfo, 'lrSubtileSize'):
    groupStride = int(tileInfo.lrSubtileSize)
  else:
    groupStride = 2 * tileInfo.subtileSize

  # Iterate over scale groups: one ds_read per 2 M-adjacent subtiles
  numScaleGroups = math.ceil(tileInfo.localSubtileGrid[0] / 2) * tileInfo.localSubtileGrid[1]
  for gid in range(numScaleGroups):
    dsOffset = groupStride * gid
    vdst = tileInfo.vgprTiles[4 * gid].regList.indices[0]
    module.add(emitScaleDsRead(tc, vdst, tileInfo.sharedVgprLROffset[0], dsOffset, gid))

  return module


def globalReadScalePtrUpdates(tc, writer, kernel):
  """Advance scale SRD base pointer by one depthU iteration.

  Boundary call: the byte increment is a writer-resolved scalar; the rocisa
  SAddU32/SAddCU32 construction lives in the C++ ModuleBuilder.
  """
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  inc = int(ti_.lrSubtileSize * ti_.lrGlobalSubtileGrid[1])
  return _builder().scale_gr_ptr_update(tc, inc)


def emitScaleGRLDSSwap(ti, writer, kernel):
  """Toggle scale GR DTL write target between double-buffer halves.

  Byte-identical to the AB GR LDS swap; reuses the C++ gr_lds_buffer_swap leaf
  with the MXSA/MXSB component tag.
  """
  return _builder().gr_lds_buffer_swap(ti.tc)


def emitScaleLRLDSSwap(ti, writer, kernel):
  """Toggle scale LR read offsets between double-buffer halves.

  Byte-identical to the AB LR LDS swap; reuses the C++ lr_lds_buffer_swap leaf.
  The sharedVgprLROffset / sharedVgprLROffsetSwap index lists are writer-owned
  register state, resolved here.
  """
  return _builder().lr_lds_buffer_swap(
      ti.tc, list(ti.sharedVgprLROffset), list(ti.sharedVgprLROffsetSwap))


# =========================================================================
# Scale GR/LR offset assignment (swizzled scale path)
#
# The scalar offset-assignment math (threads-per-group, partition stride, wave
# count) is computed by the C++ MXScaleTileInfoQuery via
# TileInfo.scaleGrOffsetAssignPlan / scaleLrOffsetAssignPlan for the gfx950
# scale geometries (MXFP4 / MXFP8). There is no Python scalar-math twin; the
# rocisa emission stays here because it allocates from the writer's register
# pools. The plan is integer-typed, which also fixes the legacy float-immediate
# crash (numThreadsPerGroup derived from the float lrSubtileSize was fed to
# hex(... - 1) and raised TypeError, so the legacy swizzled-scale GR offset path
# never actually ran).
# =========================================================================

##################################################
# Compute the per-thread global-read (DTL) vaddr for scale tensor tc.
#
# With DTL (buffer_load lds=True) the same vaddr serves as:
#   - global byte offset from the SRD base  (where to read from global memory)
#   - LDS byte offset from M0               (where to write in LDS)
#
# Threads within a wave are split into groups of numThreadsPerGroup.
# Each group loads one contiguous subtile-column worth of scale bytes:
#
#   groupId  = serial / numThreadsPerGroup          (which scale column)
#   threadId = serial % numThreadsPerGroup           (position within group)
#
#   grOffset = groupId  * stride_bpe                (column byte offset via tensor stride)
#            + threadId * loadWidth                  (byte offset within column)
#
# Output: sharedVgprGROffset[0] = grOffset (used as vaddr in DTL load)
#
def _graScaleOffset_cpp(tc, writer, kernel):
  module = Module()

  module.addComment("Computing GR Offset for %s"%tc)

  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  plan = ti_.scaleGrOffsetAssignPlan()
  loadWidth = plan.loadWidth
  loadWidthShift = loadWidth.bit_length() - 1
  numThreadsPerGroup = plan.numThreadsPerGroup
  bpe = plan.bpe

  vtmp = writer.vgprPool.checkOut(1)

  stmp = writer.sgprPool.checkOut(1)

  module.add(VLShiftRightB32(dst=vgpr(vtmp),
                            shiftHex=hex(int(math.log2(numThreadsPerGroup))), src=vgpr("Serial"),
                            comment="%s: grOffset = serial / %d" % (tc, loadWidth)))
  module.add(SLShiftLeftB32(sgpr(stmp), int(math.log2(bpe)), sgpr("Strides%s"%tc), comment="*= bpe (%d)"%bpe))

  module.add(VMulLOU32(dst=vgpr(vtmp), src1=vgpr(vtmp), src0=sgpr(stmp), comment="Apply scale%s stride to each group"%tc))
  module.add(VAndB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                     src0=hex(numThreadsPerGroup - 1), src1=vgpr("Serial"),
                     comment="%s: grOffset = serial %% %d" % (tc, loadWidth)))
  module.add(VLShiftLeftB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                            shiftHex=hex(loadWidthShift), src=vgpr(ti_.sharedVgprGROffset[0]),
                            comment="Scale by load width for each thread in group"))
  module.add(VAddU32(dst=vgpr(ti_.sharedVgprGROffset[0]), src0=vgpr(ti_.sharedVgprGROffset[0]), src1=vgpr(vtmp), comment="Final offset calc"))
  writer.vgprPool.checkIn(vtmp)
  writer.sgprPool.checkIn(stmp)

  return module

##################################################
# Generate GR offset calculation for scaleA/B (DTL).
#
# With DTL, vaddr serves as both the global read offset (from SRD)
# and the LDS write offset (from M0). Simple linear access:
#   grOffset = serial * scaleLoadWidth
#
def graTileAssignmentScaleSwizzled(writer, kernel):
  module = Module()
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module
  module.add(_graScaleOffset_cpp('MXSA', writer, kernel))
  module.add(_graScaleOffset_cpp('MXSB', writer, kernel))
  return module


##################################################
# Apply wave partition offset for scale LR.
#
# Each wave reads from its assigned LDS partition for scale A or B.
#
#   MXSA: partition index = waveId % MIWaveGroup[0]  (M-direction wave index)
#   MXSB: partition index = waveId / MIWaveGroup[0]  (N-direction wave index)
#         Using MIWaveGroup[0] (not [1]) correctly handles asymmetric configs
#         (e.g. 4x1: all 4 M-waves share the same N partition -> index = 0).
#
# Output: sharedVgprLROffset[0] = partitionIndex * totalScaleBytes
#
def _applyScaleWavePartitionLROffset_cpp(module, writer, ti_, plan, waveId):
  tc = ti_.tc

  # totalScaleBytes (bytes per wave partition in LDS for this scale tensor),
  # mWavesM (M-direction wave count), and the partition-axis selector are
  # sourced from the C++ scaleLrOffsetAssignPlan. Register state and rocisa
  # emission stay here.
  tmpSgpr = writer.sgprPool.checkOut(1)
  tmp = writer.vgprPool.checkOut(2)

  if plan.isA:
    module.add(VAndB32(dst=vgpr(tmp), src0=plan.mWavesM-1, src1=vgpr(waveId), comment="scale%s: waveId %% %d"%(tc, plan.mWavesM)))
  else:
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=int(math.log2(plan.mWavesM)), src=vgpr(waveId), comment="scale%s: waveId / numWavesM"%tc))

  module.add(SMovB32(dst=sgpr(tmpSgpr), src=plan.totalScaleBytes, comment="scale%s: scale region"%tc))
  module.add(VMulLOU32(dst=vgpr(ti_.sharedVgprLROffset[0]), src0=sgpr(tmpSgpr), src1=vgpr(tmp), comment="scale%s: partition offset"%tc))

  writer.vgprPool.checkIn(tmp)
  writer.sgprPool.checkIn(tmpSgpr)


##################################################
# Generate LR offset calculation for scaleA/B.
#
# Computes the per-lane LDS read offset for scale tensors. Called once
# during kernel setup; the resulting VGPRs are used every loop iteration.
#
# Final LR offset per lane:
#   lrOffset[lane] = wavePartitionOffset + laneId * 4 + ldsStartOffset
#
# where:
#   wavePartitionOffset  = partitionIndex * totalScaleBytes
#     MXSA partitionIndex = waveId % MIWaveGroup[0]   (M-direction)
#     MXSB partitionIndex = waveId / MIWaveGroup[0]   (N-direction)
#   laneId               = serial & (wavesize - 1)
#   ldsStartOffset       = writer.ldsStartOffsetMXSA/B
#
# LDS layout (double-buffered, one buffer shown):
#   [ DataA | DataB | ScaleA | ScaleB ]
#   ScaleA starts at ldsStartOffsetMXSA, ScaleB at ldsStartOffsetMXSB.
#
# After the LR offset is fully computed, the double-buffer swap VGPR is
# initialised here (not in localReadDTLInitCommonSwapVgpr, which runs
# before this function and would use uninitialised values):
#   swapVgpr = lrOffset XOR (lrOffset + ldsTotalSize)
# This lets localReadLDSBufferSwap toggle between buffer 0 and buffer 1.
#
def lraTileAssignmentScaleSwizzled(writer, kernel):
  module = Module()
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module
  tiA_ = writer.states.mxsa.tileInfo
  tiB_ = writer.states.mxsb.tileInfo
  planA = tiA_.scaleLrOffsetAssignPlan(kernel)
  planB = tiB_.scaleLrOffsetAssignPlan(kernel)
  module.addComment0("LR Offset Calculation for Scale Tensors")
  wavesize = kernel["WavefrontSize"]
  waveIdVgpr = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="scale: waveId"))
  _applyScaleWavePartitionLROffset_cpp(module, writer, tiA_, planA, waveIdVgpr)
  _applyScaleWavePartitionLROffset_cpp(module, writer, tiB_, planB, waveIdVgpr)
  writer.vgprPool.checkIn(waveIdVgpr)
  laneOffset = writer.vgprPool.checkOut(1)
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1, comment="scale: laneId"))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset), comment="scale: laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiA_.sharedVgprLROffset[0]), comment="scaleA: lrOffset = laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiB_.sharedVgprLROffset[0]), comment="scaleB: lrOffset = laneId * 4"))
  writer.vgprPool.checkIn(laneOffset)
  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSA), comment="scale: LDS offset for A scale"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(tiA_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleA: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSB), comment="scale: LDS offset for B scale"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(tiB_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleB: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=writer.ldsTotalSize, comment="scale: total LDS size for swap"))
  for ti_ in [tiA_, tiB_]:
    for i in range(len(ti_.sharedVgprLROffset)):
      vgprId     = ti_.sharedVgprLROffset[i]
      vgprSwapId = ti_.sharedVgprLROffsetSwap[i]
      module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(tmpSgpr), comment="scale%s: LR swap"%ti_.tc))
      module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment="scale%s: LR swap"%ti_.tc))
  writer.sgprPool.checkIn(tmpSgpr)
  return module

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
# For Swizzled Scales each wave will collectively stream
# the scale values
#
def globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel):
  module = Module()

  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1)
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))

  tiMXSA_ = writer.states.mxsa.tileInfo
  tiMXSB_ = writer.states.mxsb.tileInfo

  loadWidth = tiMXSA_.loadWidthGR

  bytesPerLoad = loadWidth * wavesize
  module.add(VLShiftLeftB32(dst=vgpr(vgprWaveId), shiftHex=hex((bytesPerLoad).bit_length()-1), src=vgpr(vgprWaveId), comment="Apply wave-specific common offset (%u) for A/B"%bytesPerLoad))

  module.add(SNop(waitState=0, comment="Wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrMXSA"), src=vgpr(vgprWaveId), comment="Store base LDS offset, will be modified"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrMXSB"), src=vgpr(vgprWaveId), comment="Store base LDS offset, will be modified"))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=hex(writer.ldsStartOffsetMXSA), comment=""))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=hex(writer.ldsStartOffsetMXSB), comment=""))

  module.add(SAddU32(dst=sgpr("SwapMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=sgpr("SwapMXSA"), comment=""))
  module.add(SAddU32(dst=sgpr("SwapMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=sgpr("SwapMXSB"), comment=""))

  writer.vgprPool.checkIn(vgprWaveId)
  return module
