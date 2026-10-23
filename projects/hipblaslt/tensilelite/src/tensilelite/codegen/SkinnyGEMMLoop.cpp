// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The K loop of one tile (@run_tile): the prologue that fills the prefetch
// ring, the steady loop and the remainder arms, and the per-slot work
// (@process_slot and its helpers). The vmcnt and lgkmcnt waits are derived
// here from the issue order of the loads.

#include "SkinnyGEMMEmitter.hpp"

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"

#include <algorithm>
#include <memory>

namespace rocm::tensilelite::codegen::detail {

rocisa::MUBUFModifiers offen() { return rocisa::MUBUFModifiers(true, 0); }

namespace {

constexpr int exponentMask = 0x7F800000;

} // namespace

RegisterRef SkinnyGEMMEmitter::dataChunk(int slot, int chunk,
                                         int column) const {
  int index = ((slot * SkinnyDims::chunks + chunk) * dims.columns + column) * 4;
  return ringData.ref(index, 4);
}

RegisterRef SkinnyGEMMEmitter::rawScales(int slot) const {
  return ringRaw->ref(slot * dims.columns, dims.columns);
}

RegisterRef SkinnyGEMMEmitter::dataRsrc(int column) const {
  return column < dims.bm ? sRsrc.ref() : wRsrc.ref();
}

RegisterRef SkinnyGEMMEmitter::scaleRsrc(int column) const {
  return column < dims.bm ? sSclRsrc.ref() : wSclRsrc.ref();
}

void SkinnyGEMMEmitter::runTile() {
  if (dims.upIters > 0)
    issueUpFrontScales();
  if (dims.prefetch == 0)
    runNoPrefetchLoop();
  else
    runPrefetchedLoop();
  // The steps run on ring slot 0 and read the up-front scales, so they precede
  // the release of both.
  if (dims.partialSteps == 1)
    runTailStep();
  else if (dims.partialSteps == 2)
    runHeadAndTailSteps();
  scaleGroup.reset();
  // The ring and the scale vector are dead, so the epilogue reuses their
  // registers.
  ringData.release();
  if (ringRaw)
    ringRaw->release();
  scaleVector.reset();
  hoistedScales.clear();
}

// dst = sum >= NIfull ? sum - NIfull : sum, the physical iteration of a
// logical one, with the rotation added once.
void SkinnyGEMMEmitter::wrapIteration(RegisterRef dst, RegisterRef sum) {
  RegisterRange wrapped = sgpr();
  builder.subi32(wrapped.ref(), sum, nIterFull.ref());
  builder.create<rocisa::SCmpGeU32>(sum.container(), nIterFull.container());
  builder.create<rocisa::SCSelectB32>(dst.container(), wrapped.container(),
                                      sum.container());
}

// dst = the wrapped physical iteration of the logical iteration 0 without
// rotation.
void SkinnyGEMMEmitter::wrapZero(RegisterRef dst) {
  RegisterRange wrapped = sgpr();
  builder.subi32(wrapped.ref(), 0, nIterFull.ref());
  builder.create<rocisa::SCmpLeU32>(nIterFull.container(), 0);
  builder.create<rocisa::SCSelectB32>(dst.container(), wrapped.container(), 0);
}

// chunkBase[u] = (kBaseIter << log2KI) + u * 16 * S, the part of the chunk
// soffsets that does not depend on the iteration.
void SkinnyGEMMEmitter::ensureChunkBase(int chunk) {
  if (!chunkBaseReady[0]) {
    shiftLeft(builder, chunkBase.ref(0), kBaseIter.ref(), dims.log2Ki);
    chunkBaseReady[0] = true;
  }
  if (chunk == 0 || chunkBaseReady[chunk])
    return;
  builder.addi32(chunkBase.ref(chunk), chunkBase.ref(0),
                 chunk * dims.chunkStride);
  chunkBaseReady[chunk] = true;
}

void SkinnyGEMMEmitter::hoistChunkBases() {
  for (int chunk = 1; chunk < SkinnyDims::chunks; ++chunk)
    ensureChunkBase(chunk);
}

// Loads the scale dword of every column of the iteration at soffset into the
// NC registers of raw. With neg (0 or -1 per lane) the voffset is ORed with it
// first, into one temporary that every column reuses since a load reads its
// voffset when it issues.
void SkinnyGEMMEmitter::issueScales(RegisterRef raw, RegisterRef soffset,
                                    std::optional<RegisterRef> neg) {
  std::optional<RegisterRange> steered;
  if (neg)
    steered = vgpr();
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef dst = raw.slice(c);
    std::shared_ptr<rocisa::RegisterContainer> voffset = sBase.sub(c);
    if (neg) {
      builder.create<rocisa::VOrB32>(steered->container(), voffset,
                                     neg->container());
      voffset = steered->container();
    }
    builder.create<rocisa::BufferLoadB32>(dst.container(), voffset,
                                          scaleRsrc(c).container(),
                                          soffset.container(), offen());
    if (recordLoads)
      prologueLoads.push_back(dst);
  }
}

// Loads chunk u of every column of the ring slot, steered by neg as the scales
// are. The steered voffset never goes into the destination registers, which
// may still have a prologue load in flight.
void SkinnyGEMMEmitter::issueChunk(int slot, int chunk, RegisterRef soffset,
                                   std::optional<RegisterRef> neg) {
  std::optional<RegisterRange> steered;
  if (neg)
    steered = vgpr();
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef dst = dataChunk(slot, chunk, c);
    std::shared_ptr<rocisa::RegisterContainer> voffset = dBase.sub(c);
    if (neg) {
      builder.create<rocisa::VOrB32>(steered->container(), voffset,
                                     neg->container());
      voffset = steered->container();
    }
    builder.create<rocisa::BufferLoadB128>(dst.container(), voffset,
                                           dataRsrc(c).container(),
                                           soffset.container(), offen());
    if (recordLoads)
      prologueLoads.push_back(dst);
  }
}

// soffset = (kBaseIter + iteration) << log2KI for the refill iteration, which
// chunk 0 uses as its own.
void SkinnyGEMMEmitter::refillOffset(const SlotContext &context,
                                     RegisterRef soffset) {
  builder.addi32(soffset, kBaseIter.ref(), *context.refillIteration);
  shiftLeft(builder, soffset, soffset, dims.log2Ki);
}

// Computes the offset of the refill iteration and issues its ring scales;
// up-front scales are loaded once, before the loop.
void SkinnyGEMMEmitter::startRefill(const SlotContext &context,
                                    RegisterRef soffset) {
  refillOffset(context, soffset);
  if (dims.upIters == 0)
    issueScales(rawScales(context.slot), soffset);
}

// Chunk u >= 1 is at chunkBase[u] + (iteration << log2KI) in the loops, whose
// chunk bases are hoisted, and at the iteration's soffset plus u * 16 * S
// elsewhere. The shifted iteration replaces the unshifted one on first use.
void SkinnyGEMMEmitter::refillChunk(const SlotContext &context, int chunk,
                                    RegisterRef soffset) {
  if (chunk == 0) {
    issueChunk(context.slot, 0, soffset);
    return;
  }
  RegisterRange chunkOffset = sgpr();
  if (context.hoistedOffsets) {
    ensureChunkBase(chunk);
    RegisterRef iteration = *context.refillIteration;
    if (chunk == 1)
      shiftLeft(builder, iteration, iteration, dims.log2Ki);
    builder.addi32(chunkOffset.ref(), chunkBase.ref(chunk), iteration);
  } else {
    builder.addi32(chunkOffset.ref(), soffset, chunk * dims.chunkStride);
  }
  issueChunk(context.slot, chunk, chunkOffset.ref());
}

// The load at where is complete once at most the loads issued after it are
// outstanding, as vmcnt counts the loads of a wave in issue order.
void SkinnyGEMMEmitter::waitForLoad(SlotLoadPosition where) {
  waitForVectorLoads(loadsIssuedAfter(dims, where));
}

// Waits until at most outstanding vector memory loads are in flight, capped at
// the counter's range.
void SkinnyGEMMEmitter::waitForVectorLoads(int outstanding) {
  builder.createWaitCnt(WaitCounts{
      .vectorLoads = std::min(outstanding, kernel.target().maxVmcnt())});
}

// Fetches the two scale dwords of every column from the lanes that hold them
// with ds_bpermute_b32: the first goes to lo, the second over the raw dword,
// which is dead once both permutes have issued. The raw dword of the first
// column has firstOutstanding vector loads issued after it.
void SkinnyGEMMEmitter::gatherSlotScales(RegisterRef raw, RegisterRef lo,
                                         int firstOutstanding) {
  for (int c = 0; c < dims.columns; ++c) {
    waitForVectorLoads(firstOutstanding - c);
    RegisterRef dword = raw.slice(c);
    builder.create<rocisa::DSBPermuteB32>(lo.slice(c).container(),
                                          addresses.sub(0), dword.container());
    builder.create<rocisa::DSBPermuteB32>(dword.container(), addresses.sub(1),
                                          dword.container());
  }
}

RegisterRange SkinnyGEMMEmitter::gatherSlotScales(const SlotContext &context) {
  RegisterRange packed = vgpr(dims.columns);
  // A step issues only its own scale loads on every path: its chunk loads may
  // all be skipped.
  int firstOutstanding =
      context.partialStep
          ? dims.columns - 1
          : loadsIssuedAfter(dims, {0, context.groupsAfter, /*ownRefill=*/0});
  gatherSlotScales(rawScales(context.slot), packed.ref(), firstOutstanding);
  return packed;
}

// Packs the two dwords of every column into one whose byte u is the scale of
// chunk u. Permute c is the (NC - 1 - c)-th newest of the 2 * NC in flight.
void SkinnyGEMMEmitter::permSlotScales(RegisterRef hi, RegisterRef lo,
                                       RegisterRef out) {
  for (int c = 0; c < dims.columns; ++c) {
    int outstanding = 2 * (dims.columns - 1 - c);
    builder.createWaitCnt(
        WaitCounts{.lds = std::min(outstanding, kernel.target().maxLgkmcnt())});
    builder.create<rocisa::VPermB32>(
        out.slice(c).container(), hi.slice(c).container(),
        lo.slice(c).container(), selector(c).container());
  }
}

// Decodes the E8M0 scale of chunk u to an f32: (dword << (23 - 8u)) &
// 0x7F800000, or a shift right by 1 for u = 3.
void SkinnyGEMMEmitter::decodeScales(RegisterRef dst, RegisterRef packed,
                                     int chunk) {
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef scale = dst.slice(c);
    if (chunk < SkinnyDims::chunks - 1)
      builder.create<rocisa::VLShiftLeftB32>(scale.container(), 23 - 8 * chunk,
                                             packed.slice(c).container());
    else
      builder.create<rocisa::VLShiftRightB32>(scale.container(), 1,
                                              packed.slice(c).container());
    builder.create<rocisa::VAndB32>(scale.container(), exponentMask,
                                    scale.container());
  }
}

// The first convert of column c of the chunk waits for its load. In a partial
// step every chunk load may be skipped, so only the later columns of the chunk
// itself are certain to be in flight; with ring scales the gather already
// waited for every load, as the chunk loads follow the scale loads.
void SkinnyGEMMEmitter::waitBeforeFirstUse(const SlotContext &context,
                                           int chunk, int column,
                                           int ownRefill) {
  if (!context.partialStep) {
    waitForLoad({dims.scaleLoads + chunk * dims.columns + column,
                 context.groupsAfter, ownRefill});
    return;
  }
  if (dims.scaleLoads == 0)
    waitForVectorLoads(dims.columns - 1 - column);
}

// Consumes chunk u of the slot: 4 dwords of 2 fp8 pairs per column, converted
// with the decoded scale and accumulated with v_dot2_f32_bf16.
void SkinnyGEMMEmitter::computeChunk(const SlotContext &context, int chunk,
                                     RegisterRef scales) {
  RegisterRange pairs = vgpr(dims.columns);
  int ownRefill =
      context.refillIteration ? dims.scaleLoads + chunk * dims.columns : 0;
  for (int dword = 0; dword < 4; ++dword) {
    for (int word = 0; word < 2; ++word) {
      std::optional<rocisa::VOP3PModifiers> highWord;
      if (word == 1)
        highWord = rocisa::VOP3PModifiers({1, 0, 0});
      for (int c = 0; c < dims.columns; ++c) {
        if (dword == 0 && word == 0)
          waitBeforeFirstUse(context, chunk, c, ownRefill);
        builder.create<rocisa::VCvtScalePkFP8toBF16>(
            pairs.sub(c),
            dataChunk(context.slot, chunk, c).slice(dword).container(),
            scales.slice(c).container(), std::nullopt, highWord);
      }
      for (int r = 0; r < dims.bm; ++r) {
        for (int t = 0; t < dims.tn; ++t) {
          RegisterRef total = acc.ref(r * dims.tn + t);
          builder.create<rocisa::VDot2F32BF16>(total.container(), pairs.sub(r),
                                               pairs.sub(dims.bm + t),
                                               total.container());
        }
      }
    }
  }
}

// Staged schedule (PF = 1 and the PF = 0 loop): refill the scales, then per
// chunk decode its scale, compute and refill the chunk, so that the new loads
// follow the consumption of the old ones.
void SkinnyGEMMEmitter::processStaged(const SlotContext &context,
                                      RegisterRef packed) {
  RegisterRange soffset = sgpr();
  if (context.refillIteration)
    startRefill(context, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    RegisterRange scales = vgpr(dims.columns);
    decodeScales(scales.ref(), packed, chunk);
    computeChunk(context, chunk, scales.ref());
    scales.release();
    if (context.refillIteration)
      refillChunk(context, chunk, soffset.ref());
  }
}

// Decode-first schedule (PF >= 2): decode the four chunk scales before the
// scales are refilled, then per chunk compute and refill. The packed dwords
// die after the last decode. A partial step decodes chunk 3 in place over the
// packed dwords, as the reference does, to fit the register budget; the loop
// keeps its allocation.
void SkinnyGEMMEmitter::processDecodeFirst(const SlotContext &context,
                                           RegisterRange &packed) {
  std::vector<RegisterRange> scales;
  std::vector<RegisterRef> chunkScales;
  int separate =
      context.partialStep ? SkinnyDims::chunks - 1 : SkinnyDims::chunks;
  for (int chunk = 0; chunk < separate; ++chunk) {
    scales.push_back(vgpr(dims.columns));
    chunkScales.push_back(scales.back().ref());
    decodeScales(chunkScales.back(), packed.ref(), chunk);
  }
  if (context.partialStep) {
    decodeScales(packed.ref(), packed.ref(), SkinnyDims::chunks - 1);
    chunkScales.push_back(packed.ref());
  } else {
    packed.release();
  }
  RegisterRange soffset = sgpr();
  if (context.refillIteration)
    startRefill(context, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    computeChunk(context, chunk, chunkScales[chunk]);
    if (chunk < separate)
      scales[chunk].release();
    else
      packed.release();
    if (context.refillIteration)
      refillChunk(context, chunk, soffset.ref());
  }
}

// Up-front schedule: per chunk decode its scale, compute and refill the
// chunk; the scales are not refilled. The last chunk decodes over the packed
// dwords, which no chunk reads afterwards. Without packed dwords (one entry)
// the decoded scales were hoisted out of the loop.
void SkinnyGEMMEmitter::processUpFront(const SlotContext &context,
                                       std::optional<RegisterRange> &packed) {
  RegisterRange soffset = sgpr();
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    std::optional<RegisterRange> decoded;
    RegisterRef scales;
    if (!packed) {
      scales = hoistedScales[chunk].ref();
    } else if (chunk == SkinnyDims::chunks - 1) {
      decodeScales(packed->ref(), packed->ref(), chunk);
      scales = packed->ref();
    } else {
      decoded = vgpr(dims.columns);
      decodeScales(decoded->ref(), packed->ref(), chunk);
      scales = decoded->ref();
    }
    computeChunk(context, chunk, scales);
    if (!context.refillIteration)
      continue;
    if (chunk == 0)
      refillOffset(context, soffset.ref());
    refillChunk(context, chunk, soffset.ref());
  }
}

void SkinnyGEMMEmitter::processSlot(const SlotContext &context) {
  if (dims.upIters > 0) {
    std::optional<RegisterRange> packed = selectScales(context);
    processUpFront(context, packed);
    return;
  }
  RegisterRange packed = gatherSlotScales(context);
  permSlotScales(rawScales(context.slot), packed.ref(), packed.ref());
  if (dims.prefetch == 0)
    ringRaw->release();
  // The MLIR decodes first for every depth but 1; the PF = 0 loop keeps the
  // staged schedule, which its steps do not use.
  bool decodeFirst =
      dims.prefetch >= 2 || (dims.prefetch == 0 && context.partialStep);
  if (decodeFirst)
    processDecodeFirst(context, packed);
  else
    processStaged(context, packed.ref());
}

// PF = 0: each iteration issues its chunks, and its scales with ring scales,
// then consumes them.
void SkinnyGEMMEmitter::runNoPrefetchLoop() {
  if (dims.upIters > 0)
    packScaleVectors();
  if (dims.upIters == 1)
    decodeHoistedScales();
  hoistChunkBases();
  // Nothing of the ring is loop carried, so the raw scales live only from the
  // loads of an iteration to its permutes.
  if (ringRaw)
    ringRaw->release();
  RegisterRange iteration = sgpr();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(nIterFull.container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  if (ringRaw)
    ringRaw = vgpr(dims.columns);
  RegisterRange physical = sgpr();
  if (fitRot) {
    builder.addi32(physical.ref(), iteration.ref(), fitRot->ref());
    wrapIteration(physical.ref(), physical.ref());
  } else {
    wrapIteration(physical.ref(), iteration.ref());
  }
  // The select precedes the loads so that its VALU work hides their latency.
  std::optional<RegisterRange> packed;
  std::optional<RegisterRange> entry;
  if (dims.upIters > 0)
    packed = selectScales(SlotContext{0, 0, std::nullopt, true,
                                      vectorEntry(physical.ref(), entry)});
  RegisterRange soffset = sgpr();
  SlotContext loads{0, 0, physical.ref()};
  startRefill(loads, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    refillChunk(loads, chunk, soffset.ref());
  SlotContext consume{0, 0, std::nullopt};
  if (dims.upIters > 0)
    processUpFront(consume, packed);
  else
    processSlot(consume);
  builder.addi32(iteration.ref(), iteration.ref(), 1);
  builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                    nIterFull.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

void SkinnyGEMMEmitter::runPrefetchedLoop() {
  emitPrologue();
  if (dims.upIters > 0)
    packScaleVectors();
  RegisterRange steadyEnd = sgpr();
  computeSteadyEnd(steadyEnd.ref());
  if (dims.upIters == 1)
    decodeHoistedScales();
  emitSteadyLoop(steadyEnd.ref());
  emitRemainderArms(steadyEnd.ref());
}

// Issues the iterations [0, PF) into the ring. A split with fewer full
// iterations reloads its last one through the clamp: a load dropped by the
// range check would still take a VMEM issue slot, and the extra slots are
// never consumed.
void SkinnyGEMMEmitter::emitPrologue() {
  recordLoads = true;
  std::optional<RegisterRange> lastIteration;
  if (dims.prefetch > 1) {
    lastIteration = sgpr();
    builder.addi32(lastIteration->ref(), nIterFull.ref(), -1);
    builder.create<rocisa::SMaxI32>(lastIteration->container(),
                                    lastIteration->container(), 0);
  }
  // LLVM hoists the chunk bases out of the prologue from PF = 4 on.
  bool hoisted = dims.prefetch >= 4;
  for (int slot = 0; slot < dims.prefetch; ++slot) {
    RegisterRange physical = sgpr();
    if (slot > 0) {
      builder.create<rocisa::SMinU32>(physical.container(),
                                      lastIteration->container(), slot);
      if (fitRot)
        builder.addi32(physical.ref(), physical.ref(), fitRot->ref());
      wrapIteration(physical.ref(), physical.ref());
    } else if (fitRot) {
      wrapIteration(physical.ref(), fitRot->ref());
    } else {
      wrapZero(physical.ref());
    }
    SlotContext context{slot, 0, physical.ref(), hoisted};
    RegisterRange soffset = sgpr();
    startRefill(context, soffset.ref());
    for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
      refillChunk(context, chunk, soffset.ref());
  }
  recordLoads = false;
}

// steadyEnd = max((NIfull / PF - 1) * PF, 0), so every refill of the steady
// loop targets an iteration below NIfull.
void SkinnyGEMMEmitter::computeSteadyEnd(RegisterRef steadyEnd) {
  int pf = dims.prefetch;
  if (pf == 1) {
    builder.addi32(steadyEnd, nIterFull.ref(), -1);
  } else if ((pf & (pf - 1)) == 0) {
    int log2Pf = log2Exact(pf);
    builder.create<rocisa::SAShiftRightI32>(steadyEnd.container(), log2Pf,
                                            nIterFull.container());
    shiftLeft(builder, steadyEnd, steadyEnd, log2Pf);
    builder.addi32(steadyEnd, steadyEnd, -pf);
  } else {
    divideByConstant(builder, steadyEnd, nIterFull.ref(), pf);
    builder.muli32(steadyEnd, steadyEnd, pf);
    builder.addi32(steadyEnd, steadyEnd, -pf);
  }
  builder.create<rocisa::SMaxI32>(steadyEnd.container(), steadyEnd.container(),
                                  0);
}

// The loop body is unrolled over the slots, so every slot index is a
// constant. Slot p refills with the iteration it0 + PF + p and, with several
// up-front entries, selects the scales of the physical iteration it0 + p. The
// bases fold the rotation: refillBase[p] = fitRot + PF + p and physBase[p] =
// fitRot + p. Without rotation they are the literals.
void SkinnyGEMMEmitter::emitSteadyLoop(RegisterRef steadyEnd) {
  int pf = dims.prefetch;
  bool vectorized = dims.upIters >= 2;
  std::optional<RegisterRange> refillBase;
  std::optional<RegisterRange> physBase;
  if (fitRot) {
    refillBase = sgpr(pf);
    builder.addi32(refillBase->ref(0), fitRot->ref(), pf);
  }
  hoistChunkBases();
  if (vectorized)
    physBase = sgpr(pf);
  for (int slot = 1; fitRot && slot < pf; ++slot) {
    builder.addi32(refillBase->ref(slot), fitRot->ref(), pf + slot);
    if (vectorized)
      builder.addi32(physBase->ref(slot), fitRot->ref(), slot);
  }
  RegisterRange iteration = sgpr();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(steadyEnd.container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  for (int slot = 0; slot < pf; ++slot) {
    RegisterRange refill = sgpr();
    if (refillBase)
      builder.addi32(refill.ref(), iteration.ref(), refillBase->ref(slot));
    else
      builder.addi32(refill.ref(), iteration.ref(), pf + slot);
    wrapIteration(refill.ref(), refill.ref());
    SlotContext context{slot, pf - 1, refill.ref()};
    std::optional<RegisterRange> physical;
    std::optional<RegisterRange> entry;
    if (vectorized) {
      physical = sgpr();
      if (slot == 0)
        builder.addi32(physical->ref(), fitRot->ref(), iteration.ref());
      else
        builder.addi32(physical->ref(), iteration.ref(), physBase->ref(slot));
      wrapIteration(physical->ref(), physical->ref());
      context.vectorIndex = vectorEntry(physical->ref(), entry);
    }
    processSlot(context);
  }
  builder.addi32(iteration.ref(), iteration.ref(), pf);
  builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                    steadyEnd.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

// The left = NIfull - steadyEnd (1 to 2 * PF - 1) remaining iterations run in
// the arm of a chain of independent ifs on left.
void SkinnyGEMMEmitter::emitRemainderArms(RegisterRef steadyEnd) {
  RegisterRange left = sgpr();
  builder.subi32(left.ref(), nIterFull.ref(), steadyEnd);
  for (int remaining = 1; remaining < 2 * dims.prefetch; ++remaining)
    emitRemainderArm(left.ref(), steadyEnd, remaining);
}

// Iteration i of the arm uses ring slot i mod PF and refills it only when
// iteration i + PF exists. Of the loads issued after the slot's group, the
// PF - 1 - i initial groups and the refills of the earlier iterations remain.
// With several up-front entries it also selects the scales of its physical
// iteration, which is the refill iteration of i - PF from i = PF on.
void SkinnyGEMMEmitter::emitRemainderArm(RegisterRef left,
                                         RegisterRef steadyEnd, int remaining) {
  int pf = dims.prefetch;
  bool vectorized = dims.upIters >= 2;
  std::shared_ptr<rocisa::Label> next = builder.createLabel("arm_end");
  builder.create<rocisa::SCmpEQI32>(left.container(), remaining);
  builder.create<rocisa::SCBranchSCC0>(next->getLabelName());
  // The logical iteration of a refill is fitRot + steadyEnd + i + PF.
  std::optional<RegisterRange> base;
  std::vector<RegisterRange> keptRefills;
  for (int i = 0; i < remaining; ++i) {
    int refilled = std::clamp(remaining - pf, 0, i);
    SlotContext context{i % pf, pf - 1 + refilled - i, std::nullopt, false};
    RegisterRange refill = sgpr();
    bool hasRefill = i + pf < remaining;
    if (!base && fitRot && (vectorized || hasRefill)) {
      base = sgpr();
      builder.addi32(base->ref(), fitRot->ref(), steadyEnd);
    }
    RegisterRef baseRef = base ? base->ref() : steadyEnd;
    if (hasRefill) {
      builder.addi32(refill.ref(), baseRef, i + pf);
      wrapIteration(refill.ref(), refill.ref());
      context.refillIteration = refill.ref();
    }
    std::optional<RegisterRange> physical;
    std::optional<RegisterRange> entry;
    if (vectorized && i >= pf) {
      context.vectorIndex = vectorEntry(keptRefills[i - pf].ref(), entry);
    } else if (vectorized) {
      physical = sgpr();
      if (i > 0)
        builder.addi32(physical->ref(), baseRef, i);
      wrapIteration(physical->ref(), i > 0 ? physical->ref() : baseRef);
      context.vectorIndex = vectorEntry(physical->ref(), entry);
    }
    processSlot(context);
    if (vectorized && hasRefill)
      keptRefills.push_back(std::move(refill));
  }
  builder.insert(next);
}

} // namespace rocm::tensilelite::codegen::detail
