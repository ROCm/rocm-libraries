// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The partial K steps (@run_partial_step and its helpers) and the scale loads
// that stop at k (@issue_scales_within_k).

#include "SkinnyGEMMEmitter.hpp"

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <memory>

namespace rocm::tensilelite::codegen::detail {

namespace {

constexpr int dwordBytes = 4;
constexpr int dwordsPerChunk = 4;
constexpr int partBytes = dwordBytes * dwordsPerChunk;
// The largest byte shift of a dword mask: a 32-bit shift is poison in the MLIR,
// and a dword that needs a shift of 4 bytes is steered anyway.
constexpr int maxMaskBytes = 3;

// The register of an optional range, if any.
std::optional<RegisterRef>
optionalRef(const std::optional<RegisterRange> &range) {
  if (!range)
    return std::nullopt;
  return range->ref();
}

// dst = -1 when src + delta is negative, else 0.
void signOfSum(IRBuilder &builder, RegisterRef dst, RegisterRef src,
               int delta) {
  builder.addi32(dst, src, delta);
  builder.create<rocisa::VAShiftRightI32>(dst.container(), 31, dst.container(),
                                          "");
}

// dst = -1 when delta - src is negative, else 0.
void signOfDifference(IRBuilder &builder, RegisterRef dst, int delta,
                      RegisterRef src) {
  builder.subi32(dst, delta, src);
  builder.create<rocisa::VAShiftRightI32>(dst.container(), 31, dst.container(),
                                          "");
}

// dst = 8 * clamp(src, 0, 3), the shift in bits of a mask of whole bytes.
void clampedBitShift(IRBuilder &builder, RegisterRef dst, RegisterRef src) {
  builder.create<rocisa::VMaxI32>(dst.container(), 0, src.container());
  builder.create<rocisa::VMinI32>(dst.container(), maxMaskBytes,
                                  dst.container());
  builder.create<rocisa::VLShiftLeftB32>(dst.container(), 3, dst.container());
}

} // namespace

// Loads the raw scale dword of every column of the iteration at soffset into
// raw, steering to -1 the lanes whose role group starts at or after k: past k a
// group belongs to the next row group of the scale matrix (other rows' scales,
// 255 decodes to +inf and 0 * inf is NaN). kAvail = k - iterOff, an SGPR, may
// be zero or negative for vector entries past k.
void SkinnyGEMMEmitter::issueScalesWithinK(RegisterRef raw, RegisterRef soffset,
                                           RegisterRef kAvail) {
  RegisterRange neg = vgpr();
  builder.subi32(neg.ref(), kAvail, scaleGroup->ref());
  builder.addi32(neg.ref(), neg.ref(), -1);
  builder.create<rocisa::VAShiftRightI32>(neg.container(), 31, neg.container(),
                                          "");
  issueScales(raw, soffset, neg.ref());
}

// Bit u of dst is @straddles_in_chunk(bound, u): the bound is not a multiple of
// 16 and lies in the chunk (bound / (16 S)) mod 4, whose part of one lane per
// column holds it.
void SkinnyGEMMEmitter::straddlesInChunk(RegisterRef dst, RegisterRef bound) {
  RegisterRange misaligned = sgpr();
  shiftRight(builder, dst, bound, log2Exact(dims.chunkStride));
  bitAnd(builder, dst, dst, SkinnyDims::chunks - 1);
  builder.create<rocisa::SLShiftLeftB32>(dst.container(), dst.container(), 1);
  bitAnd(builder, misaligned.ref(), bound, partBytes - 1);
  builder.create<rocisa::SCSelectB32>(dst.container(), dst.container(), 0);
}

// The PF = 0 loop released its raw scales, and the step takes their registers
// back before anything else, so that a smaller value does not split the hole
// they leave.
void SkinnyGEMMEmitter::allocateStepRawScales() {
  if (dims.upIters == 0 && dims.prefetch == 0)
    ringRaw = vgpr(dims.columns);
}

// The tail step of PARTIAL_STEPS = 1: ks is a multiple of KI, so the split has
// no head and the tail keeps the bytes [0, ke mod KI) of its iteration.
void SkinnyGEMMEmitter::runTailStep() {
  std::shared_ptr<rocisa::Label> stepEnd = builder.createLabel("step_end");
  RegisterRange hi = sgpr();
  bitAnd(builder, hi.ref(), splitEnd->ref(), dims.ki - 1);
  builder.create<rocisa::SCBranchSCC0>(stepEnd->getLabelName());
  RegisterRange iterOff = sgpr();
  builder.subi32(iterOff.ref(), splitEnd->ref(), hi.ref());
  std::optional<RegisterRange> kAvail;
  if (dims.upIters == 0) {
    kAvail = sgpr();
    builder.subi32(kAvail->ref(), k.ref(), iterOff.ref());
  }
  // The up-front vector starts at the split's first iteration, kBaseIter.
  std::optional<RegisterRange> vectorIndex;
  if (dims.upIters >= 2) {
    vectorIndex = sgpr();
    shiftRight(builder, vectorIndex->ref(), iterOff.ref(), dims.log2Ki);
    builder.subi32(vectorIndex->ref(), vectorIndex->ref(), kBaseIter.ref());
  }
  // v0 holds the lane's index in its cluster after the setup.
  allocateStepRawScales();
  RegisterRange laneBytes = computeLaneBytes(kernel.workitemId().ref());
  PartialStep step{iterOff.ref(),
                   std::nullopt,
                   hi.ref(),
                   optionalRef(kAvail),
                   optionalRef(vectorIndex),
                   laneBytes.ref()};
  loadPartialStep(step);
  // The only step runs under the runtime branch and nothing after it, the
  // epilogue included, reads the load state, so its registers are free for the
  // compute to fit the register budget.
  dBase.release();
  sBase.release();
  scaleGroup.reset();
  laneBytes.release();
  computePartialStep(step);
  builder.insert(stepEnd);
}

// The head and tail steps of PARTIAL_STEPS = 2 in one dynamic loop over j in
// [first, end), so that the large step body exists once: j = 0 is the head,
// which holds ks, and j = 1 the tail, which holds ke. The tail is skipped when
// the head covers both bounds.
void SkinnyGEMMEmitter::runHeadAndTailSteps() {
  RegisterRange endIter = sgpr();
  RegisterRange j = sgpr();
  RegisterRange end = sgpr();
  RegisterRange coverage = sgpr();
  shiftRight(builder, endIter.ref(), splitEnd->ref(), dims.log2Ki);
  // j starts at the head, or past it when the split starts on an iteration.
  bitAnd(builder, coverage.ref(), splitStart->ref(), dims.ki - 1);
  builder.create<rocisa::SCSelectB32>(j.container(), 0, 1);
  bitAnd(builder, coverage.ref(), splitEnd->ref(), dims.ki - 1);
  builder.create<rocisa::SCSelectB32>(end.container(), 2, 1);
  // coverage is 0 exactly when there is a head and it holds ke too.
  builder.create<rocisa::SCmpEQU32>(startIter->container(),
                                    endIter.container());
  builder.create<rocisa::SCSelectB32>(coverage.container(), j.container(), 1);
  builder.create<rocisa::SCmpEQU32>(coverage.container(), 0);
  builder.create<rocisa::SCSelectB32>(end.container(), 1, end.container());
  coverage.release();
  allocateStepRawScales();
  RegisterRange laneBytes = computeLaneBytes(kernel.workitemId().ref());

  std::shared_ptr<rocisa::Label> body = builder.createLabel("step_loop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("step_loop_end");
  builder.create<rocisa::SCmpLtI32>(j.container(), end.container());
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  {
    RegisterRange iter = sgpr();
    RegisterRange iterOff = sgpr();
    RegisterRange lo = sgpr();
    RegisterRange hi = sgpr();
    builder.create<rocisa::SCmpEQU32>(j.container(), 0);
    builder.create<rocisa::SCSelectB32>(
        iter.container(), startIter->container(), endIter.container());
    shiftLeft(builder, iterOff.ref(), iter.ref(), dims.log2Ki);
    builder.subi32(lo.ref(), splitStart->ref(), iterOff.ref());
    builder.create<rocisa::SMaxI32>(lo.container(), lo.container(), 0);
    builder.subi32(hi.ref(), splitEnd->ref(), iterOff.ref());
    builder.create<rocisa::SMinI32>(hi.container(), hi.container(), dims.ki);
    std::optional<RegisterRange> kAvail;
    if (dims.upIters == 0) {
      kAvail = sgpr();
      builder.subi32(kAvail->ref(), k.ref(), iterOff.ref());
    }
    std::optional<RegisterRange> vectorIndex;
    if (dims.upIters >= 2) {
      vectorIndex = sgpr();
      builder.subi32(vectorIndex->ref(), iter.ref(), startIter->ref());
    }
    runPartialStep(PartialStep{iterOff.ref(), lo.ref(), hi.ref(),
                               optionalRef(kAvail), optionalRef(vectorIndex),
                               laneBytes.ref()});
  }
  builder.addi32(j.ref(), j.ref(), 1);
  builder.create<rocisa::SCmpLtI32>(j.container(), end.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

// A partial step: the iteration at K offset iterOff, of which the split keeps
// only the bytes [lo, hi), run on ring slot 0, which the K loop left free. Ring
// scales load as in the loop with the groups at or after k steered to -1, and
// every chunk loads behind a dynamic branch, so that the slot holds exact
// zeros outside [lo, hi). The compute is @process_slot without refill.
void SkinnyGEMMEmitter::runPartialStep(const PartialStep &step) {
  loadPartialStep(step);
  computePartialStep(step);
}

// The scale loads and the chunk loads of a step.
void SkinnyGEMMEmitter::loadPartialStep(const PartialStep &step) {
  if (dims.upIters == 0)
    issueScalesWithinK(rawScales(0), step.iterOff, *step.kAvail);
  RegisterRange partialChunks = sgpr();
  straddlesInChunk(partialChunks.ref(), step.hi);
  if (step.lo) {
    RegisterRange loChunks = sgpr();
    straddlesInChunk(loChunks.ref(), *step.lo);
    builder.create<rocisa::SOrB32>(partialChunks.container(),
                                   partialChunks.container(),
                                   loChunks.container());
  }
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    loadStepChunk(step, chunk, partialChunks.ref());
  partialChunks.release();
}

void SkinnyGEMMEmitter::computePartialStep(const PartialStep &step) {
  processSlot(SlotContext{0, 0, std::nullopt, true, step.vectorIndex,
                          /*partialStep=*/true});
}

// Loads chunk u of the step for every column, in one of three arms. At most one
// lane per bound straddles it, in the chunk of its bound: that chunk loads
// dwords (@load_tail_dwords). In every other chunk each lane lies fully inside
// or outside [lo, hi), so steering the voffset of the outside lanes to -1
// suffices, and a chunk with no live lane is zeroed.
void SkinnyGEMMEmitter::loadStepChunk(const PartialStep &step, int chunk,
                                      RegisterRef partialChunks) {
  int offset = chunk * dims.chunkStride;
  // All SALU first: s_add writes SCC, which the branches read.
  std::optional<RegisterRange> soffsetStorage;
  std::optional<RegisterRange> hiStorage;
  std::optional<RegisterRange> loStorage;
  RegisterRef soffset = step.iterOff;
  RegisterRef chunkHi = step.hi;
  std::optional<RegisterRef> chunkLo = step.lo;
  if (chunk > 0) {
    soffsetStorage = sgpr();
    soffset = soffsetStorage->ref();
    builder.addi32(soffset, step.iterOff, offset);
    hiStorage = sgpr();
    chunkHi = hiStorage->ref();
    builder.addi32(chunkHi, step.hi, -offset);
    if (step.lo) {
      loStorage = sgpr();
      chunkLo = loStorage->ref();
      builder.addi32(*chunkLo, *step.lo, -offset);
    }
  }
  // The bytes of the lane's 16 below hi and below lo.
  RegisterRange laneRem = vgpr();
  builder.subi32(laneRem.ref(), chunkHi, step.laneBytes);
  std::optional<RegisterRange> laneLo;
  if (chunkLo) {
    laneLo = vgpr();
    builder.subi32(laneLo->ref(), *chunkLo, step.laneBytes);
  }
  std::shared_ptr<rocisa::Label> notPartial = builder.createLabel("step_load");
  std::shared_ptr<rocisa::Label> zero = builder.createLabel("step_zero");
  std::shared_ptr<rocisa::Label> chunkEnd =
      builder.createLabel("step_chunk_end");
  builder.create<rocisa::SBitcmp1B32>(partialChunks.container(), chunk);
  builder.create<rocisa::SCBranchSCC0>(notPartial->getLabelName());
  loadTailDwords(step, chunk, soffset, laneRem.ref(),
                 laneLo ? std::optional<RegisterRef>(laneLo->ref())
                        : std::nullopt);
  builder.create<rocisa::SBranch>(chunkEnd->getLabelName());
  builder.insert(notPartial);
  // A necessary condition for any lane of the chunk to lie inside [lo, hi);
  // when it holds but no lane qualifies, the steered loads read 0.
  builder.create<rocisa::SCmpGeI32>(chunkHi.container(), partBytes);
  builder.create<rocisa::SCBranchSCC0>(zero->getLabelName());
  if (step.lo) {
    int chunkLast = dims.ki - 3 * dims.chunkStride + offset - partBytes;
    builder.create<rocisa::SCmpLeI32>(step.lo->container(), chunkLast);
    builder.create<rocisa::SCBranchSCC0>(zero->getLabelName());
  }
  {
    RegisterRange neg = vgpr();
    signOfSum(builder, neg.ref(), laneRem.ref(), -partBytes);
    if (laneLo) {
      RegisterRange belowLo = vgpr();
      signOfDifference(builder, belowLo.ref(), partBytes - 1, laneLo->ref());
      builder.create<rocisa::VOrB32>(neg.container(), neg.container(),
                                     belowLo.container());
    }
    issueChunk(0, chunk, soffset, neg.ref());
  }
  builder.create<rocisa::SBranch>(chunkEnd->getLabelName());
  builder.insert(zero);
  // The slot's registers may still have a prologue load in flight when the
  // split has no full iteration; the step's scale loads are issued on every
  // path and complete after it.
  waitForPrologueLoads({ringData.ref(chunk * dims.columns * dwordsPerChunk,
                                     dims.columns * dwordsPerChunk)},
                       dims.scaleLoads);
  for (int column = 0; column < dims.columns; ++column)
    for (int dword = 0; dword < dwordsPerChunk; ++dword)
      builder.movb32(dataChunk(0, chunk, column).slice(dword), 0);
  builder.insert(chunkEnd);
}

// Loads dword d in [0, 4) of chunk u for every column. A lane reads dword d
// only when it overlaps [lo, hi) (4d < laneRem and 4d + 4 > laneLo), else its
// voffset is the out-of-range -1. The dwords that straddle lo or hi are loaded
// at their aligned addresses and their bytes outside [lo, hi) cleared with an
// AND mask (both masks when lo and hi share a dword). Those bytes lie inside
// the column's storage, since every A and B column is padded to 4 bytes, but
// belong to another split or to the padding. The steering, mask and soffset of
// a dword do not depend on the column, so the loop over d is outermost and
// only one steering and mask pair is live.
void SkinnyGEMMEmitter::loadTailDwords(const PartialStep &step, int chunk,
                                       RegisterRef soffset, RegisterRef laneRem,
                                       std::optional<RegisterRef> laneLo) {
  RegisterRange voffset = vgpr();
  for (int d = 0; d < dwordsPerChunk; ++d) {
    int start = d * dwordBytes;
    RegisterRange steer = vgpr();
    RegisterRange mask = vgpr();
    // Only a step with a lower bound needs a second temporary.
    std::optional<RegisterRange> hiTemp;
    if (laneLo)
      hiTemp = vgpr();
    // Steering: 0 when the dword overlaps [lo, hi), else -1. The mask register
    // holds the lo side until the final AND.
    signOfSum(builder, steer.ref(), laneRem, -(start + 1));
    if (laneLo) {
      signOfDifference(builder, mask.ref(), start + dwordBytes - 1, *laneLo);
      builder.create<rocisa::VOrB32>(steer.container(), steer.container(),
                                     mask.container());
    }
    // Clear the bytes of the dword below lo and at or after hi.
    if (laneLo) {
      RegisterRef below = *laneLo;
      if (start != 0) {
        builder.addi32(mask.ref(), *laneLo, -start);
        below = mask.ref();
      }
      clampedBitShift(builder, mask.ref(), below);
      builder.create<rocisa::VLShiftLeftB32>(mask.container(), mask.container(),
                                             -1);
    }
    RegisterRef hiMask = laneLo ? hiTemp->ref() : mask.ref();
    builder.subi32(hiMask, start + dwordBytes, laneRem);
    clampedBitShift(builder, hiMask, hiMask);
    builder.create<rocisa::VLShiftRightB32>(hiMask.container(),
                                            hiMask.container(), -1);
    if (laneLo)
      builder.create<rocisa::VAndB32>(mask.container(), mask.container(),
                                      hiMask.container());
    std::optional<RegisterRange> dwordOffset;
    RegisterRef dwordSoffset = soffset;
    if (d > 0) {
      dwordOffset = sgpr();
      dwordSoffset = dwordOffset->ref();
      builder.addi32(dwordSoffset, soffset, start);
    }
    TailDword tail{steer.ref(), dwordSoffset, mask.ref()};
    for (int column = 0; column < dims.columns; ++column)
      loadMaskedDword(chunk, column, d, voffset.ref(), tail);
  }
}

// Loads one dword of a column, steered, and clears the bytes outside [lo, hi).
// The AND consumes the load just issued, so nothing may stay in flight.
void SkinnyGEMMEmitter::loadMaskedDword(int chunk, int column, int dword,
                                        RegisterRef voffset,
                                        const TailDword &tail) {
  RegisterRef dst = dataChunk(0, chunk, column).slice(dword);
  builder.create<rocisa::VOrB32>(voffset.container(), dBase.sub(column),
                                 tail.steer.container());
  builder.create<rocisa::BufferLoadB32>(dst.container(), voffset.container(),
                                        dataRsrc(column).container(),
                                        tail.soffset.container(), offen());
  waitForVectorLoads(0);
  builder.create<rocisa::VAndB32>(dst.container(), dst.container(),
                                  tail.mask.container());
}

} // namespace rocm::tensilelite::codegen::detail
