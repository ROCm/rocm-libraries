// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The parts of the K phase that the VALU and the MFMA paths share: the load
// order model of a ring slot, the emission of its loads, the vmcnt wait and the
// loop header alignment, and the ring schedule that drives a path.

#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Registers.hpp>

#include "SkinnyGEMMEmitter.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::skinny;

int32_t RingLoads::issuedAfter(int32_t position, int32_t groupsAfter,
                               int32_t extra) const {
  return perSlot - 1 - position + groupsAfter * perSlot + extra;
}

void skinny::emitSlotLoads(IRBuilder &builder,
                           const std::vector<SlotLoad> &loads,
                           std::vector<RegisterRef> *record) {
  std::optional<RegisterRange> steered;
  for (const SlotLoad &load : loads) {
    assert(load.dst.has_value() != load.lds.has_value() &&
           "a slot load has one destination");
    RegisterRef voffset = load.voffset;
    if (load.steer) {
      if (!steered)
        steered = builder.allocateVgpr(1);
      builder.ori32(steered->ref(), voffset, *load.steer);
      voffset = steered->ref();
    }
    BufferAccess access{.offset = load.immediate};
    if (load.lds) {
      builder.loadBufferToLds(load.rsrc, voffset, load.soffset, load.lds->base,
                              load.lds->offset, load.width, access);
      continue;
    }
    builder.loadBuffer(*load.dst, load.rsrc, voffset, load.soffset, load.width,
                       access);
    if (record)
      record->push_back(*load.dst);
  }
}

// Waits until at most outstanding vector memory loads are in flight, capped at
// the counter's range.
void skinny::waitForVectorLoads(IRBuilder &builder, int32_t outstanding) {
  builder.createWaitCnt(WaitCounts{
      .vectorLoads = std::min(outstanding, builder.target().maxVmcnt())});
}

// Aligns a hot loop header to 16 bytes: a header at 4 mod 8 made the loop 5 to
// 7 percent slower on some points, and one at 8 mod 16 made another 4 percent
// slower, as measured by the perf A/B harness.
void skinny::alignLoopHeader(IRBuilder &builder) {
  builder.create<rocisa::TextBlock>(".p2align 4\n");
}

// The PF = 0 form: one iteration per round, issued and then consumed.
static void emitSingleStepLoop(IRBuilder &builder, RingPath &path) {
  path.afterPrologue();
  path.beginLoop();
  RegisterRange iteration = builder.allocateSgpr(1);
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(path.iterations().container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  alignLoopHeader(builder);
  builder.insert(body);
  RingStep step{0, 0, false, false, iteration.ref(), 0, true};
  path.fill(step);
  path.consume(step);
  builder.addi32(iteration.ref(), iteration.ref(), 1);
  builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                    path.iterations().container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
  path.endLoop();
}

// steadyEnd = max((NI / PF - 1) * PF, 0), so that every refill of the steady
// loop targets an iteration below NI.
static RegisterRange steadyEndOf(IRBuilder &builder, RegisterRef iterations,
                                 int32_t pf) {
  RegisterRange steadyEnd = builder.allocateSgpr(1);
  if (pf == 1)
    builder.addi32(steadyEnd.ref(), iterations, -1);
  else if (isPowerOfTwo(pf)) {
    int32_t log2Pf = log2Exact(pf);
    builder.shrsi32(steadyEnd.ref(), iterations, log2Pf);
    builder.shli32(steadyEnd.ref(), steadyEnd.ref(), log2Pf);
    builder.addi32(steadyEnd.ref(), steadyEnd.ref(), -pf);
  } else {
    builder.divsi32(steadyEnd.ref(), iterations, pf, /*useIEEE=*/true);
    builder.muli32(steadyEnd.ref(), steadyEnd.ref(), pf);
    builder.addi32(steadyEnd.ref(), steadyEnd.ref(), -pf);
  }
  builder.maxsi32(steadyEnd.ref(), steadyEnd.ref(), 0);
  return steadyEnd;
}

// The steady loop, unrolled over the slots so that every slot index is a
// constant: slot p consumes the iteration it0 + p and refills with it0 + PF +
// p.
static void emitSteadyLoop(IRBuilder &builder, RingPath &path,
                           RegisterRef steadyEnd) {
  int32_t pf = path.slots();
  path.beginLoop();
  RegisterRange iteration = builder.allocateSgpr(1);
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(steadyEnd.container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  alignLoopHeader(builder);
  builder.insert(body);
  for (int32_t slot = 0; slot < pf; ++slot)
    path.consume(RingStep{slot, pf - 1, true, true, iteration.ref(), slot,
                          /*inLoop=*/true});
  builder.addi32(iteration.ref(), iteration.ref(), pf);
  builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                    steadyEnd.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

// The left = NI - steadyEnd (1 to 2 * PF - 1) remaining iterations run in the
// arm of a chain of independent ifs on left. Iteration i of the arm uses ring
// slot i mod PF and refills it only when iteration i + PF exists. Of the loads
// issued after the slot's group, the PF - 1 - i initial groups and the refills
// of the earlier iterations remain.
static void emitRemainderArms(IRBuilder &builder, RingPath &path,
                              RegisterRef steadyEnd) {
  int32_t pf = path.slots();
  RegisterRange left = builder.allocateSgpr(1);
  builder.subi32(left.ref(), path.iterations(), steadyEnd);
  for (int32_t remaining = 1; remaining < 2 * pf; ++remaining) {
    std::shared_ptr<rocisa::Label> next = builder.createLabel("arm_end");
    builder.create<rocisa::SCmpEQI32>(left.container(), remaining);
    builder.create<rocisa::SCBranchSCC0>(next->getLabelName());
    for (int32_t i = 0; i < remaining; ++i) {
      int32_t refilled = std::clamp(remaining - pf, 0, i);
      path.consume(RingStep{i % pf, pf - 1 + refilled - i, i + pf < remaining,
                            i + 1 < remaining, steadyEnd, i,
                            /*inLoop=*/false});
    }
    path.endArm();
    builder.insert(next);
  }
}

void skinny::emitRingLoop(IRBuilder &builder, RingPath &path) {
  int32_t pf = path.slots();
  if (pf == 0) {
    emitSingleStepLoop(builder, path);
    path.drain();
    return;
  }
  // The prologue issues the iterations [0, PF) into the ring.
  for (int32_t slot = 0; slot < pf; ++slot)
    path.fill(RingStep{slot, 0, false, true, path.iterations(), slot,
                       /*inLoop=*/false});
  RegisterRange steadyEnd = steadyEndOf(builder, path.iterations(), pf);
  path.afterPrologue();
  emitSteadyLoop(builder, path, steadyEnd.ref());
  path.endLoop();
  emitRemainderArms(builder, path, steadyEnd.ref());
  path.drain();
}
