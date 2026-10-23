// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The epilogue of one tile (@reduce_and_store, @store_tile and @store_run):
// the cross-lane reduction of the accumulators and their stores, one variant
// per orientation when the vector stores exist.

#include "SkinnyGEMMEmitter.hpp"

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <algorithm>
#include <cassert>
#include <memory>

namespace rocm::tensilelite::codegen::detail {

namespace {

constexpr int outOfRange = -1;
// Wait states between a v_dot2 write and a DPP read of its result (LLVM's
// DotWriteDifferentVALURead) and between a VALU write and a DPP read of it.
constexpr int dotToDppWaitStates = 3;
constexpr int valuToDppWaitStates = 2;
// Wait states of v_readlane_b32 after a VALU wrote its source, and of a VALU
// read of the SGPR that v_readlane_b32 wrote (gfx940, per LLVM's hazard
// recognizer; the reference omits the second).
constexpr int valuToReadlaneWaitStates = 1;
constexpr int readlaneToValuWaitStates = 2;
constexpr int bankMaskAll = 0xf;

bool overlaps(RegisterRef a, RegisterRef b) {
  if (a.kind != b.kind)
    return false;
  return a.start < b.start + b.count && b.start < a.start + a.count;
}

} // namespace

// An orientation stores runs of consecutive elements: the BM skinny elements
// of a wide column when the skinny dimension is contiguous in the output, the
// TN wide elements of a skinny column otherwise.
int SkinnyGEMMEmitter::runLength(bool skinnyContiguous) const {
  return skinnyContiguous ? dims.bm : dims.tn;
}

// A chunk must start dword aligned: an f32 run of at least 2 elements or an
// even bf16 run.
bool SkinnyGEMMEmitter::hasVector(bool skinnyContiguous, bool bf16) const {
  int length = runLength(skinnyContiguous);
  return bf16 ? length % 2 == 0 : length >= 2;
}

SkinnyGEMMEmitter::RedLayout
SkinnyGEMMEmitter::redLayout(bool skinnyContiguous) const {
  int length = runLength(skinnyContiguous);
  if (length < 2)
    return RedLayout{false, skinnyContiguous, 0, dims.accumulators};
  // Every run starts at an even register, as the x2 and x4 stores need.
  int stride = (length + 1) & ~1;
  int runs = skinnyContiguous ? dims.tn : dims.bm;
  return RedLayout{true, skinnyContiguous, stride, runs * stride};
}

int SkinnyGEMMEmitter::redIndex(const RedLayout &layout, int r, int t) const {
  if (!layout.runs)
    return r * dims.tn + t;
  return layout.skinnyContiguous ? t * layout.stride + r
                                 : r * layout.stride + t;
}

void SkinnyGEMMEmitter::emitEpilogue() {
  bool vectorA = runLength(true) >= 2;
  bool vectorB = runLength(false) >= 2;
  // Without vector stores the two orientations are the same code.
  if (!vectorA && !vectorB) {
    emitEpilogueVariant(true);
    return;
  }
  std::shared_ptr<rocisa::Label> orientationB =
      builder.createLabel("orientation_b");
  std::shared_ptr<rocisa::Label> end = builder.createLabel("epilogue_end");
  builder.create<rocisa::SCmpEQU32>(contigS.container(), 1);
  builder.create<rocisa::SCBranchSCC0>(orientationB->getLabelName());
  emitEpilogueVariant(true);
  builder.create<rocisa::SBranch>(end->getLabelName());
  builder.insert(orientationB);
  emitEpilogueVariant(false);
  builder.insert(end);
}

void SkinnyGEMMEmitter::emitEpilogueVariant(bool skinnyContiguous) {
  RedLayout layout = redLayout(skinnyContiguous);
  RegisterRef kl = kernel.workitemId().ref();
  // The registers are allocated up front so that one wait covers every
  // prologue load whose destination they reuse.
  RegisterRange red = vgpr(layout.total, layout.runs ? 2 : 1);
  RegisterRange off = vgpr();
  std::optional<RegisterRange> wBase;
  if (!uniformWide())
    wBase = vgpr();
  RegisterRange leader = sgpr(2, 2);
  RegisterRange tileFull = sgpr();
  bool anyVector =
      hasVector(skinnyContiguous, true) || hasVector(skinnyContiguous, false);
  EpilogueState state{
      layout,
      red.ref(),
      off.ref(),
      wBase ? std::optional<RegisterRef>(wBase->ref()) : std::nullopt,
      leader.ref(),
      anyVector ? std::optional<RegisterRef>(tileFull.ref()) : std::nullopt};
  waitForEpilogueRegisters(state);
  reduceAccumulators(state);
  if (anyVector)
    computeTileFull(tileFull.ref());
  builder.create<rocisa::VCmpEQI32>(leader.container(), 0, kl.container());
  // A single split stores bf16 straight to D, else each split stores its f32
  // partial slice into the workspace.
  std::shared_ptr<rocisa::Label> workspace = builder.createLabel("workspace");
  std::shared_ptr<rocisa::Label> done = builder.createLabel("stored");
  builder.create<rocisa::SCmpLeU32>(k.container(), spk.container());
  builder.create<rocisa::SCBranchSCC0>(workspace->getLabelName());
  storeTile(state, StoreTarget{dRsrc.ref(), dStrideS.ref(), dStrideW.ref(),
                               std::nullopt, true});
  builder.create<rocisa::SBranch>(done->getLabelName());
  builder.insert(workspace);
  RegisterRange slice = sgpr();
  builder.muli32(slice.ref(), tile.kz, mn.ref());
  storeTile(state, StoreTarget{wsRsrc.ref(), wsStrideS.ref(), wsStrideW.ref(),
                               slice.ref(), false});
  builder.insert(done);
}

// The epilogue overwrites the ring registers, and a prologue load that was
// never consumed (a split with fewer than PF iterations) may still be in
// flight.
void SkinnyGEMMEmitter::waitForEpilogueRegisters(const EpilogueState &state) {
  std::vector<RegisterRef> written = {state.red, state.off};
  if (state.wBase)
    written.push_back(*state.wBase);
  waitForPrologueLoads(written, /*issuedSince=*/0);
}

// Waits until no prologue load that writes one of the registers in written is
// in flight. A prologue load stays in flight when the split has fewer full
// iterations than PF and its slot is never consumed; loads complete in order,
// so the wait allows the later prologue loads and the issuedSince loads that
// every path issued after the prologue.
void SkinnyGEMMEmitter::waitForPrologueLoads(
    const std::vector<RegisterRef> &written, int issuedSince) {
  int count = static_cast<int>(prologueLoads.size());
  for (int newest = count - 1; newest >= 0; --newest) {
    bool conflict =
        std::any_of(written.begin(), written.end(), [&](RegisterRef reg) {
          return overlaps(reg, prologueLoads[newest]);
        });
    if (!conflict)
      continue;
    waitForVectorLoads(count - 1 - newest + issuedSince);
    return;
  }
}

void SkinnyGEMMEmitter::emitNops(int count) {
  if (count > 0)
    builder.create<rocisa::VNop>(count);
}

void SkinnyGEMMEmitter::emitDpp(RegisterRef dst, RegisterRef src,
                                const DppStep &step) {
  rocisa::DPPModifiers dpp(-1, step.rowBcast, step.boundCtrl, step.quadPerm, -1,
                           step.rowMirror, step.rowHalfMirror, step.rowMask,
                           bankMaskAll);
  builder.create<rocisa::VAddF32>(dst.container(), src.container(),
                                  src.container(), dpp);
}

// The DPP adds that follow the first one for a cluster of LANES_K lanes: the
// quad permutes, the half-row and row mirrors, and for 64 lanes the row
// broadcasts.
std::vector<SkinnyGEMMEmitter::DppStep>
SkinnyGEMMEmitter::reductionSteps() const {
  std::vector<DppStep> steps;
  if (dims.lanesK >= 4)
    steps.push_back({-1, {2, 3, 0, 1}, false, false, 0xf, 1});
  if (dims.lanesK >= 8)
    steps.push_back({-1, {}, false, true, 0xf, 1});
  if (dims.lanesK >= 16)
    steps.push_back({-1, {}, true, false, 0xf, 1});
  if (dims.lanesK == 64) {
    steps.push_back({15, {}, false, false, 0xa, -1});
    steps.push_back({31, {}, false, false, 0xf, 1});
  }
  return steps;
}

// Reduces every accumulator over its LANES_K lanes into the red registers:
// DPP adds that leave the cluster sum in every lane, which a v_readlane of
// lane 63 broadcasts for 64 lanes.
void SkinnyGEMMEmitter::reduceAccumulators(const EpilogueState &state) {
  int valuSinceReduce = 0;
  for (int p = 0; p < dims.accumulators; ++p) {
    // The last chunk ends with the dots of all accumulators in order, so p has
    // P - 1 - p VALU instructions after its last dot.
    int sinceDot = dims.accumulators - 1 - p + valuSinceReduce;
    emitNops(dotToDppWaitStates - sinceDot);
    valuSinceReduce += std::max(0, dotToDppWaitStates - sinceDot);
    valuSinceReduce += reduceAccumulator(state, p / dims.tn, p % dims.tn);
  }
}

// Returns the number of VALU instructions emitted, nops included.
int SkinnyGEMMEmitter::reduceAccumulator(const EpilogueState &state, int r,
                                         int t) {
  RegisterRef source = acc.ref(r * dims.tn + t);
  RegisterRef dst = state.red.slice(redIndex(state.layout, r, t));
  emitDpp(dst, source, DppStep{-1, {1, 0, 3, 2}, false, false, 0xf, 1});
  int emitted = 1;
  for (const DppStep &step : reductionSteps()) {
    emitNops(valuToDppWaitStates);
    emitDpp(dst, dst, step);
    emitted += valuToDppWaitStates + 1;
  }
  if (dims.lanesK < 64)
    return emitted;
  // Lane 63 holds the total; every lane takes it.
  RegisterRange total = sgpr();
  emitNops(valuToReadlaneWaitStates);
  builder.create<rocisa::VReadlaneB32>(total.container(), dst.container(), 63);
  emitNops(readlaneToValuWaitStates);
  builder.movb32(dst, total.ref());
  return emitted + valuToReadlaneWaitStates + 1 + readlaneToValuWaitStates + 1;
}

// tileFull = the tile lies inside the output, ((ty + 1) * BM <= skinnyCols) &&
// ((tx + 1) * BN <= wideCols).
void SkinnyGEMMEmitter::computeTileFull(RegisterRef tileFull) {
  RegisterRange skinnyEnd = sgpr();
  RegisterRange wideEnd = sgpr();
  RegisterRange skinnyIn = sgpr();
  RegisterRange wideIn = sgpr();
  builder.addi32(skinnyEnd.ref(), tile.ty, 1);
  multiplyBy(builder, skinnyEnd.ref(), skinnyEnd.ref(), dims.bm);
  builder.addi32(wideEnd.ref(), tile.tx, 1);
  multiplyBy(builder, wideEnd.ref(), wideEnd.ref(), dims.tileWide);
  builder.cmpi32(Predicate::ule, skinnyIn.ref(), skinnyEnd.ref(),
                 skinnyCols.ref());
  builder.cmpi32(Predicate::ule, wideIn.ref(), wideEnd.ref(), wideCols.ref());
  builder.create<rocisa::SAndB32>(tileFull.container(), skinnyIn.container(),
                                  wideIn.container());
}

// Stores the tile into target: through the vector stores when the tile lies
// inside the output and the orientation has them, else per element with the
// out-of-range offset masking the tail columns and the non-leader lanes.
void SkinnyGEMMEmitter::storeTile(const EpilogueState &state,
                                  const StoreTarget &target) {
  bool vector =
      state.tileFull && hasVector(state.layout.skinnyContiguous, target.bf16);
  if (!vector) {
    storeMasked(state, target);
    return;
  }
  std::shared_ptr<rocisa::Label> vectorPath = builder.createLabel("vector");
  std::shared_ptr<rocisa::Label> done = builder.createLabel("tile_stored");
  builder.create<rocisa::SCmpEQU32>(state.tileFull->container(), 1);
  builder.create<rocisa::SCBranchSCC1>(vectorPath->getLabelName());
  storeMasked(state, target);
  builder.create<rocisa::SBranch>(done->getLabelName());
  builder.insert(vectorPath);
  storeVector(state, target);
  builder.insert(done);
}

// The scalars of one store path, in elements: the offset of the tile's first
// element without the per-lane part of the wide column, which goes to the
// wBase register of the state (or into the scalar offset when w0 is uniform).
SkinnyGEMMEmitter::StoreScalars
SkinnyGEMMEmitter::makeStoreScalars(const EpilogueState &state,
                                    const StoreTarget &target) {
  StoreScalars scalars{sgpr(), sgpr(), sgpr()};
  builder.muli32(scalars.base.ref(), s0.ref(), target.strideS);
  if (target.sliceBase)
    builder.addi32(scalars.base.ref(), scalars.base.ref(), *target.sliceBase);
  if (state.wBase) {
    builder.muli32(*state.wBase, target.strideW, w0.ref());
    return scalars;
  }
  builder.muli32(scalars.scratch.ref(), w0.ref(), target.strideW);
  builder.addi32(scalars.base.ref(), scalars.base.ref(), scalars.scratch.ref());
  return scalars;
}

// off = the byte offset of tile element (r, t) for this lane, ((s0 + r) *
// strideS + (w0 + t) * strideW + sliceBase) * elementBytes.
void SkinnyGEMMEmitter::computeOffset(const EpilogueState &state,
                                      const StoreTarget &target,
                                      const StoreScalars &scalars, int r,
                                      int t) {
  int log2Element = target.bf16 ? 1 : 2;
  RegisterRef uniform = scalars.base.ref();
  if (r != 0 || t != 0) {
    uniform = scalars.offset.ref();
    builder.movb32(uniform, scalars.base.ref());
    if (r != 0) {
      multiplyBy(builder, scalars.scratch.ref(), target.strideS, r);
      builder.addi32(uniform, uniform, scalars.scratch.ref());
    }
    if (t != 0) {
      multiplyBy(builder, scalars.scratch.ref(), target.strideW, t);
      builder.addi32(uniform, uniform, scalars.scratch.ref());
    }
  }
  if (state.wBase) {
    builder.addi32(state.off, uniform, *state.wBase);
    shiftLeft(builder, state.off, state.off, log2Element);
    return;
  }
  // The offset is uniform: shift it in the SALU and copy it to the VGPR.
  RegisterRef shifted = scalars.scratch.ref();
  shiftLeft(builder, shifted, uniform, log2Element);
  builder.movb32(state.off, shifted);
}

// One store per element: bf16 is the truncation, the top 16 bits of the f32.
void SkinnyGEMMEmitter::storeMasked(const EpilogueState &state,
                                    const StoreTarget &target) {
  StoreScalars scalars = makeStoreScalars(state, target);
  RegisterRange mask = sgpr(2, 2);
  for (int r = 0; r < dims.bm; ++r) {
    for (int t = 0; t < dims.tn; ++t) {
      RegisterRef value = state.red.slice(redIndex(state.layout, r, t));
      if (target.bf16)
        shiftRight(builder, value, value, 16);
      computeOffset(state, target, scalars, r, t);
      builder.create<rocisa::SAndB64>(mask.container(), sValid.sub(2 * r, 2),
                                      wValid.sub(2 * t, 2));
      builder.create<rocisa::SAndB64>(mask.container(), mask.container(),
                                      state.leader.container());
      builder.selectb32(state.off, mask.ref(), state.off, outOfRange);
      if (target.bf16)
        builder.create<rocisa::BufferStoreB16>(
            value.container(), state.off.container(), target.rsrc.container(),
            0, rocisa::MUBUFModifiers(true, 0));
      else
        builder.create<rocisa::BufferStoreB32>(
            value.container(), state.off.container(), target.rsrc.container(),
            0, rocisa::MUBUFModifiers(true, 0));
    }
  }
}

void SkinnyGEMMEmitter::storeVector(const EpilogueState &state,
                                    const StoreTarget &target) {
  StoreScalars scalars = makeStoreScalars(state, target);
  bool skinnyContiguous = state.layout.skinnyContiguous;
  int runs = skinnyContiguous ? dims.tn : dims.bm;
  for (int run = 0; run < runs; ++run) {
    if (skinnyContiguous)
      storeRun(state, target, scalars, 0, run, dims.bm);
    else
      storeRun(state, target, scalars, run, 0, dims.tn);
  }
}

// Stores len consecutive elements from tile position (r, t) with the widest
// chunks first: dwordx4, dwordx2, dword. A bf16 run has an even length.
void SkinnyGEMMEmitter::storeRun(const EpilogueState &state,
                                 const StoreTarget &target,
                                 const StoreScalars &scalars, int r, int t,
                                 int length) {
  int per16 = target.bf16 ? 8 : 4;
  int first = 0;
  for (int chunk : {per16, per16 / 2, per16 / 4}) {
    for (; length - first >= chunk; first += chunk)
      storeChunk(state, target, scalars, r, t, first, chunk);
  }
  assert(first == length && "run length is not a multiple of the chunk");
}

// Stores elements [first, first + elements) of the run that starts at (r, t).
void SkinnyGEMMEmitter::storeChunk(const EpilogueState &state,
                                   const StoreTarget &target,
                                   const StoreScalars &scalars, int r, int t,
                                   int first, int elements) {
  bool skinnyContiguous = state.layout.skinnyContiguous;
  int rc = skinnyContiguous ? r + first : r;
  int tc = skinnyContiguous ? t : t + first;
  computeOffset(state, target, scalars, rc, tc);
  builder.selectb32(state.off, state.leader, state.off, outOfRange);
  int start = redIndex(state.layout, rc, tc);
  int dwords = elements;
  if (target.bf16) {
    // Pack pairs of truncated values in place; dword k reads the registers
    // 2k and 2k + 1, which no earlier dword has overwritten.
    dwords = elements / 2;
    for (int j = 0; j < elements; ++j) {
      RegisterRef value = state.red.slice(start + j);
      shiftRight(builder, value, value, 16);
    }
    for (int k = 0; k < dwords; ++k)
      builder.create<rocisa::VPackF16toB32>(
          state.red.slice(start + k).container(),
          state.red.slice(start + 2 * k).container(),
          state.red.slice(start + 2 * k + 1).container());
  }
  std::shared_ptr<rocisa::RegisterContainer> data =
      state.red.slice(start, dwords).container();
  rocisa::MUBUFModifiers offenOnly(true, 0);
  RegisterRef off = state.off;
  switch (dwords) {
  case 1:
    builder.create<rocisa::BufferStoreB32>(
        data, off.container(), target.rsrc.container(), 0, offenOnly);
    return;
  case 2:
    builder.create<rocisa::BufferStoreB64>(
        data, off.container(), target.rsrc.container(), 0, offenOnly);
    return;
  default:
    builder.create<rocisa::BufferStoreB128>(
        data, off.container(), target.rsrc.container(), 0, offenOnly);
  }
}

} // namespace rocm::tensilelite::codegen::detail
