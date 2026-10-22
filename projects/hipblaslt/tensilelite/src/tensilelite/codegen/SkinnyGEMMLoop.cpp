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
#include <cstdint>

namespace rocm::tensilelite::codegen::detail {

namespace {

constexpr int exponentMask = 0x7F800000;

// Multiplier and shift of LLVM's signed division by a positive constant.
struct SignedMagic {
  int32_t multiplier;
  int shift;
};

// Hacker's Delight, signed magic numbers for a divisor d >= 2.
SignedMagic signedMagic(int32_t d) {
  constexpr uint32_t two31 = 0x80000000u;
  uint32_t ad = static_cast<uint32_t>(d);
  uint32_t t = two31;
  uint32_t anc = t - 1 - t % ad;
  int p = 31;
  uint32_t q1 = two31 / anc;
  uint32_t r1 = two31 - q1 * anc;
  uint32_t q2 = two31 / ad;
  uint32_t r2 = two31 - q2 * ad;
  uint32_t delta = 0;
  do {
    ++p;
    q1 *= 2;
    r1 *= 2;
    if (r1 >= anc) {
      ++q1;
      r1 -= anc;
    }
    q2 *= 2;
    r2 *= 2;
    if (r2 >= ad) {
      ++q2;
      r2 -= ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  return {static_cast<int32_t>(q2 + 1), p - 32};
}

rocisa::MUBUFModifiers offen() { return rocisa::MUBUFModifiers(true, 0); }

} // namespace

RegisterRef SkinnyGEMMEmitter::dataChunk(int slot, int chunk,
                                         int column) const {
  int index = ((slot * SkinnyDims::chunks + chunk) * dims.columns + column) * 4;
  return ringData.ref(index, 4);
}

RegisterRef SkinnyGEMMEmitter::rawScale(int slot, int column) const {
  return ringRaw.ref(slot * dims.columns + column);
}

RegisterRef SkinnyGEMMEmitter::dataRsrc(int column) const {
  return column < dims.bm ? sRsrc.ref() : wRsrc.ref();
}

RegisterRef SkinnyGEMMEmitter::scaleRsrc(int column) const {
  return column < dims.bm ? sSclRsrc.ref() : wSclRsrc.ref();
}

void SkinnyGEMMEmitter::runTile() {
  if (dims.prefetch == 0)
    runNoPrefetchLoop();
  else
    runPrefetchedLoop();
  // The ring is dead, so the epilogue reuses its registers.
  ringData.release();
  ringRaw.release();
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

void SkinnyGEMMEmitter::issueScales(int slot, RegisterRef soffset) {
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef dst = rawScale(slot, c);
    builder.create<rocisa::BufferLoadB32>(dst.container(), sBase.sub(c),
                                          scaleRsrc(c).container(),
                                          soffset.container(), offen());
    if (recordLoads)
      prologueLoads.push_back(dst);
  }
}

void SkinnyGEMMEmitter::issueChunk(int slot, int chunk, RegisterRef soffset) {
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef dst = dataChunk(slot, chunk, c);
    builder.create<rocisa::BufferLoadB128>(dst.container(), dBase.sub(c),
                                           dataRsrc(c).container(),
                                           soffset.container(), offen());
    if (recordLoads)
      prologueLoads.push_back(dst);
  }
}

// Issues the scales of the refill iteration with the soffset of the
// iteration, (kBaseIter + iteration) << log2KI, which chunk 0 reuses.
void SkinnyGEMMEmitter::startRefill(const SlotContext &context,
                                    RegisterRef soffset) {
  builder.addi32(soffset, kBaseIter.ref(), *context.refillIteration);
  shiftLeft(builder, soffset, soffset, dims.log2Ki);
  issueScales(context.slot, soffset);
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
  int outstanding = loadsIssuedAfter(dims, where);
  builder.createWaitCnt(WaitCounts{
      .vectorLoads = std::min(outstanding, kernel.target().maxVmcnt())});
}

// Fetches the two scale dwords of every column from the lanes that hold them
// with ds_bpermute_b32: the first goes to a new register, the second over the
// raw dword, which is dead once both permutes have issued.
RegisterRange SkinnyGEMMEmitter::gatherSlotScales(const SlotContext &context) {
  RegisterRange packed = vgpr(dims.columns);
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef raw = rawScale(context.slot, c);
    waitForLoad({c, context.groupsAfter, 0});
    builder.create<rocisa::DSBPermuteB32>(packed.sub(c), addresses.sub(0),
                                          raw.container());
    builder.create<rocisa::DSBPermuteB32>(raw.container(), addresses.sub(1),
                                          raw.container());
  }
  return packed;
}

// Packs the two dwords of every column into one whose byte u is the scale of
// chunk u. Permute c is the (NC - 1 - c)-th newest of the 2 * NC in flight.
void SkinnyGEMMEmitter::permSlotScales(int slot, RegisterRef packed) {
  for (int c = 0; c < dims.columns; ++c) {
    int outstanding = 2 * (dims.columns - 1 - c);
    builder.createWaitCnt(
        WaitCounts{.lds = std::min(outstanding, kernel.target().maxLgkmcnt())});
    builder.create<rocisa::VPermB32>(
        packed.slice(c).container(), rawScale(slot, c).container(),
        packed.slice(c).container(), selector(c).container());
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

// Consumes chunk u of the slot: 4 dwords of 2 fp8 pairs per column, converted
// with the decoded scale and accumulated with v_dot2_f32_bf16.
void SkinnyGEMMEmitter::computeChunk(const SlotContext &context, int chunk,
                                     RegisterRef scales) {
  RegisterRange pairs = vgpr(dims.columns);
  int ownRefill =
      context.refillIteration ? dims.columns + chunk * dims.columns : 0;
  for (int dword = 0; dword < 4; ++dword) {
    for (int word = 0; word < 2; ++word) {
      std::optional<rocisa::VOP3PModifiers> highWord;
      if (word == 1)
        highWord = rocisa::VOP3PModifiers({1, 0, 0});
      for (int c = 0; c < dims.columns; ++c) {
        if (dword == 0 && word == 0)
          waitForLoad({dims.columns + chunk * dims.columns + c,
                       context.groupsAfter, ownRefill});
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
// die after the last decode.
void SkinnyGEMMEmitter::processDecodeFirst(const SlotContext &context,
                                           RegisterRange &packed) {
  std::vector<RegisterRange> scales;
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    scales.push_back(vgpr(dims.columns));
    decodeScales(scales.back().ref(), packed.ref(), chunk);
  }
  packed.release();
  RegisterRange soffset = sgpr();
  if (context.refillIteration)
    startRefill(context, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    computeChunk(context, chunk, scales[chunk].ref());
    scales[chunk].release();
    if (context.refillIteration)
      refillChunk(context, chunk, soffset.ref());
  }
}

void SkinnyGEMMEmitter::processSlot(const SlotContext &context) {
  RegisterRange packed = gatherSlotScales(context);
  permSlotScales(context.slot, packed.ref());
  if (dims.prefetch == 0)
    ringRaw.release();
  if (dims.prefetch >= 2)
    processDecodeFirst(context, packed);
  else
    processStaged(context, packed.ref());
}

// PF = 0: each iteration issues its scales and chunks, then consumes them.
void SkinnyGEMMEmitter::runNoPrefetchLoop() {
  hoistChunkBases();
  // Nothing of the ring is loop carried, so the raw scales live only from the
  // loads of an iteration to its permutes.
  ringRaw.release();
  RegisterRange iteration = sgpr();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(nIterFull.container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  ringRaw = vgpr(dims.columns);
  RegisterRange physical = sgpr();
  builder.addi32(physical.ref(), iteration.ref(), fitRot.ref());
  wrapIteration(physical.ref(), physical.ref());
  RegisterRange soffset = sgpr();
  SlotContext loads{0, 0, physical.ref()};
  startRefill(loads, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    refillChunk(loads, chunk, soffset.ref());
  processSlot(SlotContext{0, 0, std::nullopt});
  builder.addi32(iteration.ref(), iteration.ref(), 1);
  builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                    nIterFull.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

void SkinnyGEMMEmitter::runPrefetchedLoop() {
  emitPrologue();
  RegisterRange steadyEnd = sgpr();
  computeSteadyEnd(steadyEnd.ref());
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
    if (slot == 0) {
      wrapIteration(physical.ref(), fitRot.ref());
    } else {
      builder.create<rocisa::SMinU32>(physical.container(),
                                      lastIteration->container(), slot);
      builder.addi32(physical.ref(), physical.ref(), fitRot.ref());
      wrapIteration(physical.ref(), physical.ref());
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
    // LLVM's expansion of the signed division by the constant.
    SignedMagic magic = signedMagic(pf);
    RegisterRange sign = sgpr();
    builder.create<rocisa::SMulHII32>(steadyEnd.container(),
                                      nIterFull.container(), magic.multiplier);
    if (magic.multiplier < 0)
      builder.addi32(steadyEnd, steadyEnd, nIterFull.ref());
    if (magic.shift > 0)
      builder.create<rocisa::SAShiftRightI32>(
          steadyEnd.container(), magic.shift, steadyEnd.container());
    shiftRight(builder, sign.ref(), steadyEnd, 31);
    builder.addi32(steadyEnd, steadyEnd, sign.ref());
    builder.muli32(steadyEnd, steadyEnd, pf);
    builder.addi32(steadyEnd, steadyEnd, -pf);
  }
  builder.create<rocisa::SMaxI32>(steadyEnd.container(), steadyEnd.container(),
                                  0);
}

// The loop body is unrolled over the slots, so every slot index is a
// constant. Slot p refills with the iteration it0 + PF + p.
void SkinnyGEMMEmitter::emitSteadyLoop(RegisterRef steadyEnd) {
  int pf = dims.prefetch;
  RegisterRange refillBase = sgpr(pf);
  builder.addi32(refillBase.ref(0), fitRot.ref(), pf);
  hoistChunkBases();
  for (int slot = 1; slot < pf; ++slot)
    builder.addi32(refillBase.ref(slot), fitRot.ref(), pf + slot);
  RegisterRange iteration = sgpr();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(steadyEnd.container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  for (int slot = 0; slot < pf; ++slot) {
    RegisterRange refill = sgpr();
    builder.addi32(refill.ref(), iteration.ref(), refillBase.ref(slot));
    wrapIteration(refill.ref(), refill.ref());
    processSlot(SlotContext{slot, pf - 1, refill.ref()});
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
void SkinnyGEMMEmitter::emitRemainderArm(RegisterRef left,
                                         RegisterRef steadyEnd, int remaining) {
  int pf = dims.prefetch;
  std::shared_ptr<rocisa::Label> next = builder.createLabel("arm_end");
  builder.create<rocisa::SCmpEQI32>(left.container(), remaining);
  builder.create<rocisa::SCBranchSCC0>(next->getLabelName());
  // The logical iteration of a refill is fitRot + steadyEnd + i + PF.
  std::optional<RegisterRange> refillBase;
  for (int i = 0; i < remaining; ++i) {
    int refilled = std::clamp(remaining - pf, 0, i);
    SlotContext context{i % pf, pf - 1 + refilled - i, std::nullopt, false};
    RegisterRange refill = sgpr();
    if (i + pf < remaining) {
      if (!refillBase) {
        refillBase = sgpr();
        builder.addi32(refillBase->ref(), fitRot.ref(), steadyEnd);
      }
      builder.addi32(refill.ref(), refillBase->ref(), i + pf);
      wrapIteration(refill.ref(), refill.ref());
      context.refillIteration = refill.ref();
    }
    processSlot(context);
  }
  builder.insert(next);
}

} // namespace rocm::tensilelite::codegen::detail
