// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The up-front scales (UP_ITERS != 0): the raw scale loads before the K loop,
// their packing (@pack_scale_vectors, through LDS when the waves of a block
// share them), and the choice of the entry of an iteration (@select_scales).

#include "SkinnyGEMMEmitter.hpp"

#include "container.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <algorithm>
#include <memory>

namespace rocm::tensilelite::codegen::detail {

namespace {

// The largest value of the 16-bit offset field of a ds instruction.
constexpr int maxDsOffset = 65535;

// A ds address register and the offset field that, together, name base +
// offset.
struct LdsAddress {
  RegisterRef base;
  int offset;
};

// Names base + offset, adding an offset beyond the field to base in scratch,
// which the address of every ds instruction of the caller can share since a ds
// instruction reads its address when it issues.
LdsAddress ldsAddress(IRBuilder &builder, RegisterRef base, int offset,
                      std::optional<RegisterRange> &scratch) {
  if (offset <= maxDsOffset)
    return {base, offset};
  if (!scratch)
    scratch = builder.allocateVgpr(1);
  builder.addi32(scratch->ref(), offset, base);
  return {scratch->ref(), 0};
}

} // namespace

// Loads every entry of the up-front vector, in order, before anything else
// that reads memory, so that the in-order vmcnt wait of the first use covers
// only the scales. Entry j is the iteration base + j, with base the split's
// first full iteration, or with PARTIAL_STEPS = 2 its first touched one, which
// the head step owns. The first two entries compute their offset from the base
// iteration and the others from the base offset. With partial steps the entries
// are steered at k, as the vector may hold iterations that reach or pass it.
void SkinnyGEMMEmitter::issueUpFrontScales() {
  scaleVector = vgpr(dims.upIters * dims.columns);
  RegisterRef baseIter = startIter ? startIter->ref() : kBaseIter.ref();
  std::optional<RegisterRange> headBase;
  RegisterRef baseOffset = chunkBase.ref(0);
  if (startIter) {
    headBase = sgpr();
    shiftLeft(builder, headBase->ref(), baseIter, dims.log2Ki);
    baseOffset = headBase->ref();
  } else {
    ensureChunkBase(0);
  }
  for (int entry = 0; entry < dims.upIters; ++entry) {
    RegisterRef soffset = baseOffset;
    std::optional<RegisterRange> entryOffset;
    if (entry > 0) {
      entryOffset = sgpr();
      soffset = entryOffset->ref();
    }
    if (entry == 1) {
      builder.addi32(soffset, baseIter, 1);
      shiftLeft(builder, soffset, soffset, dims.log2Ki);
    } else if (entry > 1) {
      builder.addi32(soffset, baseOffset, entry * dims.ki);
    }
    RegisterRef raw = scaleVector->ref(entry * dims.columns, dims.columns);
    if (dims.partialSteps == 0) {
      issueScales(raw, soffset);
      continue;
    }
    RegisterRange kAvail = sgpr();
    builder.subi32(kAvail.ref(), k.ref(), soffset);
    issueScalesWithinK(raw, soffset, kAvail.ref());
  }
  // No scale load remains in the loop or the steps.
  sBase.release();
  scaleGroup.reset();
}

// The up-front vector entry of a physical iteration, vecOff + it (vecOff is 1
// when PARTIAL_STEPS = 2 and the split starts inside an iteration, whose entry
// 0 belongs to the head step). storage receives the sum when there is one.
RegisterRef
SkinnyGEMMEmitter::vectorEntry(RegisterRef physical,
                               std::optional<RegisterRange> &storage) {
  if (!vecOff)
    return physical;
  storage = sgpr();
  builder.addi32(storage->ref(), physical, vecOff->ref());
  return storage->ref();
}

// Loads issued after the up-front load at index: the later up-front loads and
// the PF slots of the prologue.
int SkinnyGEMMEmitter::upFrontLoadsAfter(int index) const {
  return dims.upIters * dims.columns - 1 - index +
         dims.prefetch * dims.loadsPerSlot;
}

// Packs the raw dwords of every entry into one dword per column whose byte u
// is the scale of chunk u, in place.
void SkinnyGEMMEmitter::packScaleVectors() {
  if (dims.scalesShared)
    packShared();
  else
    packPerWave();
  selectors.release();
  addresses.release();
}

// An entry is packed in two phases, the 2 * NC fetches back to back and then
// the permutes, and the entries stay in sequence: fetching all of them first
// would keep two dwords per column and entry live.
void SkinnyGEMMEmitter::packPerWave() {
  for (int entry = 0; entry < dims.upIters; ++entry) {
    RegisterRange lo = vgpr(dims.columns);
    RegisterRef raw = scaleVector->ref(entry * dims.columns, dims.columns);
    gatherSlotScales(raw, lo.ref(), upFrontLoadsAfter(entry * dims.columns));
    permSlotScales(raw, lo.ref(), raw);
  }
}

// Publishes the raw dwords through LDS, then every wave reads the two dwords
// of its entries.
void SkinnyGEMMEmitter::packShared() {
  stageScalesInLds();
  ldsStage.reset();
  for (int entry = 0; entry < dims.upIters; ++entry)
    readSharedEntry(entry);
}

// The dword (e * NC + c) of thread tid goes to the LDS dword (e * NC + c) *
// WAVES * 64 + tid. Each write waits only for its own load, as vmcnt counts in
// order, and the barrier does not wait for vmcnt, unlike gpu.barrier, so the
// prologue data loads stay in flight.
void SkinnyGEMMEmitter::stageScalesInLds() {
  std::optional<RegisterRange> scratch;
  int dwords = dims.upIters * dims.columns;
  for (int index = 0; index < dwords; ++index) {
    builder.createWaitCnt(
        WaitCounts{.vectorLoads = std::min(upFrontLoadsAfter(index),
                                           kernel.target().maxVmcnt())});
    LdsAddress address =
        ldsAddress(builder, ldsStage->ref(), index * dims.waves * 256, scratch);
    builder.create<rocisa::DSStoreB32>(address.base.container(),
                                       scaleVector->sub(index),
                                       rocisa::DSModifiers(1, address.offset));
  }
  builder.createWaitCnt(WaitCounts{.lds = 0});
  builder.create<rocisa::SBarrier>();
}

// The lane reads the dwords d = 0 and 1 of every column of the entry at the
// byte offsets of its address, the second 4 * R * S / 2 bytes after the first.
void SkinnyGEMMEmitter::readSharedEntry(int entry) {
  std::optional<RegisterRange> scratch;
  RegisterRange hi = vgpr(dims.columns);
  RegisterRef lo = scaleVector->ref(entry * dims.columns, dims.columns);
  int hiDelta = 4 * dims.roleSlots * dims.halfSubgroup;
  for (int c = 0; c < dims.columns; ++c) {
    int offset = (entry * dims.columns + c) * dims.waves * 256;
    LdsAddress first = ldsAddress(builder, addresses.ref(), offset, scratch);
    builder.create<rocisa::DSLoadB32>(lo.slice(c).container(),
                                      first.base.container(),
                                      rocisa::DSModifiers(1, first.offset));
    LdsAddress second =
        ldsAddress(builder, addresses.ref(), offset + hiDelta, scratch);
    builder.create<rocisa::DSLoadB32>(hi.sub(c), second.base.container(),
                                      rocisa::DSModifiers(1, second.offset));
  }
  permSlotScales(hi.ref(), lo, lo);
}

// Moves the packed dwords of the entry of the physical iteration into a new
// register per column. VGPRs cannot be indexed dynamically, so this is a chain
// of UP - 1 selects per column; the entry is the outer loop so that each mask
// dies after its sweep. A single entry needs no selection.
std::optional<RegisterRange>
SkinnyGEMMEmitter::selectScales(const SlotContext &context) {
  if (dims.upIters == 1)
    return std::nullopt;
  RegisterRange packed = vgpr(dims.columns);
  for (int entry = 1; entry < dims.upIters; ++entry) {
    builder.create<rocisa::SCmpEQI32>(context.vectorIndex->container(), entry);
    builder.create<rocisa::SCSelectB64>(std::make_shared<rocisa::VCC>(), -1, 0);
    for (int c = 0; c < dims.columns; ++c) {
      // The first select reads entry 0 itself instead of a copy of it.
      std::shared_ptr<rocisa::RegisterContainer> current =
          entry == 1 ? scaleVector->sub(c) : packed.sub(c);
      builder.create<rocisa::VCndMaskB32>(
          packed.sub(c), current, scaleVector->sub(entry * dims.columns + c),
          std::make_shared<rocisa::VCC>());
    }
  }
  return packed;
}

// The single entry has a loop-invariant scale per chunk, which the loop reads
// instead of decoding it every iteration.
void SkinnyGEMMEmitter::decodeHoistedScales() {
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    hoistedScales.push_back(vgpr(dims.columns));
    decodeScales(hoistedScales.back().ref(), scaleVector->ref(0, dims.columns),
                 chunk);
  }
  scaleVector.reset();
}

} // namespace rocm::tensilelite::codegen::detail
