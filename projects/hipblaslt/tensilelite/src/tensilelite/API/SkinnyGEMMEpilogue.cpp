// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The output epilogue of the skinny MXFP8 GEMM emitter, shared by the VALU and
// the MFMA compute paths: the C prefetch, alpha, beta * C, the split-K
// workspace store and the bf16 D store, each through vector accesses where the
// tile lies inside the output and the orientation has runs of consecutive
// elements, else per element with the out-of-range offset masking. Only the
// production of the final values and the addresses of the elements differ per
// path; they come through OutputMap.

#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Kernel.hpp>
#include <tensilelite/Codegen/Registers.hpp>

#include "SkinnyGEMMEmitter.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"

#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::skinny;

namespace {

// Which variant of a vector-or-masked access follows the branch.
enum class FallThrough { masked, vector };

} // namespace

OutputContext SkinnyGEMMEmitter::outputContext() {
  auto optionalRef = [](const std::optional<RegisterRange> &range) {
    return range ? std::optional<RegisterRef>(range->ref()) : std::nullopt;
  };
  std::optional<StoreTarget> cTarget;
  if (cRsrc)
    cTarget = StoreTarget{cRsrc->ref(), cStrideS->ref(), cStrideW->ref(),
                          std::nullopt, true};
  return OutputContext{builder,
                       dims,
                       alpha.ref(),
                       optionalRef(beta),
                       k.ref(),
                       spk.ref(),
                       mn.ref(),
                       skinnyCols.ref(),
                       wideCols.ref(),
                       contigS.ref(),
                       StoreTarget{dRsrc.ref(), dStrideS.ref(), dStrideW.ref(),
                                   std::nullopt, true},
                       StoreTarget{wsRsrc.ref(), wsStrideS.ref(),
                                   wsStrideW.ref(), std::nullopt, false},
                       cTarget};
}

// Elements of a bf16 run in the C registers pack in pairs, each run starting
// at an even register so that every dwordx2 and dwordx4 load is aligned.
static int32_t cRunStride(int32_t length) { return (length / 2 + 1) & ~1; }

// A chunk must start dword aligned: an f32 run of at least 2 elements or an
// even bf16 run.
static bool hasVectorRuns(const OutputMap &map, bool bf16) {
  return bf16 ? map.runLength % 2 == 0 : map.runLength >= 2;
}

static MemoryWidth widthOfDwords(int32_t dwords) {
  if (dwords == 1)
    return MemoryWidth::b32;
  return dwords == 2 ? MemoryWidth::b64 : MemoryWidth::b128;
}

// Calls emit(first, chunk) for the chunks of a run of length elements, the
// widest first: unit, unit / 2 and unit / 4 elements.
static void
forEachRunChunk(int32_t length, int32_t unit,
                const std::function<void(int32_t first, int32_t chunk)> &emit) {
  int32_t first = 0;
  for (int32_t chunk : {unit, unit / 2, unit / 4}) {
    for (; length - first >= chunk; first += chunk)
      emit(first, chunk);
  }
  assert(first == length && "run length is not a multiple of the chunk");
}

// Calls emit(i, j) with the first element of every run, the runs lying along
// the contiguous dimension.
static void forEachRun(const OutputMap &map,
                       const std::function<void(int32_t i, int32_t j)> &emit) {
  int32_t length = map.runLength;
  if (map.skinnyContiguous) {
    for (int32_t j = 0; j < map.wideCount; ++j)
      for (int32_t i = 0; i < map.skinnyCount; i += length)
        emit(i, j);
    return;
  }
  for (int32_t i = 0; i < map.skinnyCount; ++i)
    for (int32_t j = 0; j < map.wideCount; j += length)
      emit(i, j);
}

// The number of the run of element (i, j) and its position in the run.
static int32_t runOrdinal(const OutputMap &map, int32_t i, int32_t j) {
  int32_t length = map.runLength;
  if (map.skinnyContiguous)
    return j * (map.skinnyCount / length) + i / length;
  return i * (map.wideCount / length) + j / length;
}

static int32_t runPosition(const OutputMap &map, int32_t i, int32_t j) {
  return (map.skinnyContiguous ? i : j) % map.runLength;
}

void skinny::forEachOrientation(
    const OutputContext &ctx, bool vectorStores,
    const std::function<void(bool skinnyContiguous)> &emitVariant) {
  if (!vectorStores) {
    emitVariant(true);
    return;
  }
  IRBuilder &builder = ctx.builder;
  std::shared_ptr<rocisa::Label> orientationB =
      builder.createLabel("orientation_b");
  std::shared_ptr<rocisa::Label> end = builder.createLabel("epilogue_end");
  builder.create<rocisa::SCmpEQU32>(ctx.contigS.container(), 1);
  builder.create<rocisa::SCBranchSCC0>(orientationB->getLabelName());
  emitVariant(true);
  builder.create<rocisa::SBranch>(end->getLabelName());
  builder.insert(orientationB);
  emitVariant(false);
  builder.insert(end);
}

void skinny::branchUnlessCIsUsed(const OutputContext &ctx,
                                 const std::shared_ptr<rocisa::Label> &skip) {
  IRBuilder &builder = ctx.builder;
  builder.create<rocisa::SCmpLeU32>(ctx.k.container(), ctx.spk.container());
  builder.create<rocisa::SCBranchSCC0>(skip->getLabelName());
  RegisterRange magnitude = builder.allocateSgpr(1);
  branchIfBetaZero(builder, *ctx.beta, magnitude.ref(), skip);
}

// Emits the masked or the vector variant of a tile access. The vector one runs
// when the tile lies inside the output; fallThrough names the variant that
// follows the branch, the other sits behind targetLabel.
static void emitVectorOrMasked(IRBuilder &builder,
                               std::optional<RegisterRef> tileFull,
                               bool vectorSupported, FallThrough fallThrough,
                               const char *targetLabel, const char *doneLabel,
                               const std::function<void()> &emitMasked,
                               const std::function<void()> &emitVector) {
  if (!tileFull || !vectorSupported) {
    emitMasked();
    return;
  }
  std::shared_ptr<rocisa::Label> target = builder.createLabel(targetLabel);
  std::shared_ptr<rocisa::Label> done = builder.createLabel(doneLabel);
  builder.create<rocisa::SCmpEQU32>(tileFull->container(), 1);
  bool maskedFirst = fallThrough == FallThrough::masked;
  if (maskedFirst)
    builder.create<rocisa::SCBranchSCC1>(target->getLabelName());
  else
    builder.create<rocisa::SCBranchSCC0>(target->getLabelName());
  (maskedFirst ? emitMasked : emitVector)();
  builder.create<rocisa::SBranch>(done->getLabelName());
  builder.insert(target);
  (maskedFirst ? emitVector : emitMasked)();
  builder.insert(done);
}

// tileFull = the tile lies inside the output, ((ty + 1) * BM <= skinnyCols) &&
// ((tx + 1) * BN <= wideCols).
static void computeTileFull(const OutputContext &ctx, const TileIds &ids,
                            RegisterRef tileFull) {
  IRBuilder &builder = ctx.builder;
  RegisterRange skinnyEnd = builder.allocateSgpr(1);
  RegisterRange wideEnd = builder.allocateSgpr(1);
  RegisterRange skinnyIn = builder.allocateSgpr(1);
  RegisterRange wideIn = builder.allocateSgpr(1);
  builder.addi32(skinnyEnd.ref(), ids.ty, 1);
  builder.muli32(skinnyEnd.ref(), skinnyEnd.ref(), ctx.dims.bm);
  builder.addi32(wideEnd.ref(), ids.tx, 1);
  builder.muli32(wideEnd.ref(), wideEnd.ref(), ctx.dims.tileWide);
  builder.cmpi32(Predicate::ule, skinnyIn.ref(), skinnyEnd.ref(),
                 ctx.skinnyCols);
  builder.cmpi32(Predicate::ule, wideIn.ref(), wideEnd.ref(), ctx.wideCols);
  builder.andi32(tileFull, skinnyIn.ref(), wideIn.ref());
}

// Sets the offset of the element (i, j) to out of range unless the lane
// accesses it, and returns it.
static RegisterRef maskedOffset(OutputMap &map, IRBuilder &builder,
                                RegisterRef mask, int32_t i, int32_t j) {
  RegisterRef off = map.offset(i, j);
  map.accessLanes(mask, i, j);
  builder.selectb32(off, mask, off, outOfRange);
  return off;
}

// The C prefetch of one orientation.
void skinny::loadCTile(const OutputContext &ctx, OutputMap &map,
                       const TileIds &ids) {
  IRBuilder &builder = ctx.builder;
  const StoreTarget &target = *ctx.c;
  bool vector = hasVectorRuns(map, true);
  RegisterRange tileFull = builder.allocateSgpr(1);
  if (vector)
    computeTileFull(ctx, ids, tileFull.ref());
  map.prepare(target);
  // One zero-extending 16-bit load per element, into dword valueIndex of the C
  // registers; the lanes that do not access the element load out of range, so
  // they cause no traffic.
  auto loadMasked = [&] {
    RegisterRange mask = builder.allocateSgpr(2, 2);
    for (int32_t i = 0; i < map.skinnyCount; ++i) {
      for (int32_t j = 0; j < map.wideCount; ++j) {
        RegisterRef off = maskedOffset(map, builder, mask.ref(), i, j);
        builder.loadBuffer(map.values.slice(map.valueIndex(i, j)), target.rsrc,
                           off, 0, MemoryWidth::b16);
      }
    }
  };
  // The runs load with the widest chunks first, as the D store does: dwordx4,
  // dwordx2, dword. Run q lands at dword q * cRunStride of the registers.
  auto loadVector = [&] {
    int32_t stride = cRunStride(map.runLength);
    forEachRun(map, [&](int32_t i, int32_t j) {
      int32_t run = runOrdinal(map, i, j);
      forEachRunChunk(map.runLength, 8, [&](int32_t first, int32_t chunk) {
        RegisterRef off = map.offset(map.skinnyContiguous ? i + first : i,
                                     map.skinnyContiguous ? j : j + first);
        std::optional<RegisterRef> lanes = map.storeLanes();
        if (lanes)
          builder.selectb32(off, *lanes, off, outOfRange);
        int32_t dwords = chunk / 2;
        builder.loadBuffer(map.values.slice(run * stride + first / 2, dwords),
                           target.rsrc, off, 0, widthOfDwords(dwords));
      });
    });
  };
  emitVectorOrMasked(builder,
                     vector ? std::optional<RegisterRef>(tileFull.ref())
                            : std::nullopt,
                     vector, FallThrough::masked, "c_vector", "c_loaded",
                     loadMasked, loadVector);
}

// Stores the tile into target: through the vector stores when the tile lies
// inside the output and the orientation has them, else per element with the
// out-of-range offset masking the tail columns and the lanes that do not
// store.
static void storeTile(const OutputContext &ctx, OutputMap &map,
                      const StoreTarget &target,
                      std::optional<RegisterRef> tileFull) {
  IRBuilder &builder = ctx.builder;
  // One store per element: bf16 is the truncation, the top 16 bits of the f32.
  auto storeMasked = [&] {
    map.prepare(target);
    RegisterRange mask = builder.allocateSgpr(2, 2);
    for (int32_t i = 0; i < map.skinnyCount; ++i) {
      for (int32_t j = 0; j < map.wideCount; ++j) {
        RegisterRef value = map.values.slice(map.valueIndex(i, j));
        if (target.bf16)
          builder.shrui32(value, value, 16);
        RegisterRef off = maskedOffset(map, builder, mask.ref(), i, j);
        builder.storeBuffer(value, target.rsrc, off, 0,
                            target.bf16 ? MemoryWidth::b16 : MemoryWidth::b32);
      }
    }
  };
  // Stores the elements [first, first + elements) of the run that starts at
  // (i, j) with the widest chunks first: dwordx4, dwordx2, dword. A bf16 run
  // has an even length.
  auto storeChunk = [&](int32_t i, int32_t j, int32_t first, int32_t elements) {
    int32_t ic = map.skinnyContiguous ? i + first : i;
    int32_t jc = map.skinnyContiguous ? j : j + first;
    RegisterRef off = map.offset(ic, jc);
    std::optional<RegisterRef> lanes = map.storeLanes();
    if (lanes)
      builder.selectb32(off, *lanes, off, outOfRange);
    int32_t start = map.valueIndex(ic, jc);
    int32_t dwords = elements;
    if (target.bf16) {
      // Pack pairs of truncated values in place; dword k reads the registers
      // 2k and 2k + 1, which no earlier dword has overwritten.
      dwords = elements / 2;
      for (int32_t e = 0; e < elements; ++e) {
        RegisterRef value = map.values.slice(start + e);
        builder.shrui32(value, value, 16);
      }
      for (int32_t d = 0; d < dwords; ++d)
        builder.create<rocisa::VPackF16toB32>(
            map.values.slice(start + d).container(),
            map.values.slice(start + 2 * d).container(),
            map.values.slice(start + 2 * d + 1).container());
    }
    builder.storeBuffer(map.values.slice(start, dwords), target.rsrc, off, 0,
                        widthOfDwords(dwords));
  };
  auto storeVector = [&] {
    map.prepare(target);
    forEachRun(map, [&](int32_t i, int32_t j) {
      forEachRunChunk(map.runLength, target.bf16 ? 8 : 4,
                      [&](int32_t first, int32_t chunk) {
                        storeChunk(i, j, first, chunk);
                      });
    });
  };
  emitVectorOrMasked(builder, tileFull, hasVectorRuns(map, target.bf16),
                     FallThrough::masked, "vector", "tile_stored", storeMasked,
                     storeVector);
}

// Multiplies the values by alpha, in f32 and before the bf16 truncation, two at
// a time with v_pk_mul_f32 where the pair is even aligned. The pair reads the
// low dword of alpha for both halves (op_sel_hi).
static void scaleByAlpha(const OutputContext &ctx, const OutputMap &map) {
  IRBuilder &builder = ctx.builder;
  for (int32_t segment = 0; segment < map.segmentCount; ++segment) {
    for (int32_t i = 0; i < map.segmentLength;) {
      RegisterRef values = map.values.slice(segment * map.segmentStride + i);
      bool pair = i + 1 < map.segmentLength && values.start % 2 == 0;
      if (!pair) {
        builder.create<rocisa::VMulF32>(values.container(),
                                        ctx.alpha.slice(0, 1).container(),
                                        values.container());
        ++i;
        continue;
      }
      values = map.values.slice(segment * map.segmentStride + i, 2);
      builder.create<rocisa::VMulPKF32>(
          values.container(), ctx.alpha.container(), values.container(),
          rocisa::VOP3PModifiers({}, {0, 1}));
      i += 2;
    }
  }
}

// The C halves of a bf16 pair live in one dword, low half first.
static constexpr int32_t highHalfMask = static_cast<int32_t>(0xFFFF0000);

// c = the f32 of the bf16 in the low or the high half of src.
static void unpackBf16(IRBuilder &builder, RegisterRef c, RegisterRef src,
                       bool high) {
  if (high)
    builder.andi32(c, highHalfMask, src);
  else
    builder.shli32(c, src, 16);
}

// values = fma(beta, C, values) for the prefetched C, skipped when beta is zero
// as the prefetch is. The C loads are older than every other load of the tile,
// so one vmcnt(0) completes them; the temporary is allocated after it because
// it may reuse a ring register that an unconsumed prologue load writes.
static void addBetaC(const OutputContext &ctx, const OutputMap &map,
                     RegisterRef cData, std::optional<RegisterRef> tileFull) {
  IRBuilder &builder = ctx.builder;
  std::shared_ptr<rocisa::Label> noBeta = builder.createLabel("no_beta");
  {
    RegisterRange magnitude = builder.allocateSgpr(1);
    branchIfBetaZero(builder, *ctx.beta, magnitude.ref(), noBeta);
  }
  waitForVectorLoads(builder, 0);
  RegisterRange c = builder.allocateVgpr(1);
  int32_t stride = cRunStride(map.runLength);
  // The vector path loaded runs of dword pairs: element p of run q is the half
  // p % 2 of dword q * stride + p / 2. The masked path loaded one zero-extended
  // element per dword.
  auto addC = [&](bool vector) {
    for (int32_t i = 0; i < map.skinnyCount; ++i) {
      for (int32_t j = 0; j < map.wideCount; ++j) {
        int32_t index = map.valueIndex(i, j);
        int32_t position = runPosition(map, i, j);
        RegisterRef src =
            vector ? cData.slice(runOrdinal(map, i, j) * stride + position / 2)
                   : cData.slice(index);
        unpackBf16(builder, c.ref(), src, vector && position % 2 == 1);
        builder.create<rocisa::VMacF32>(map.values.slice(index).container(),
                                        ctx.beta->container(), c.container());
      }
    }
  };
  emitVectorOrMasked(
      builder, tileFull, hasVectorRuns(map, true), FallThrough::vector,
      "c_masked", "c_added", [&] { addC(false); }, [&] { addC(true); });
  builder.insert(noBeta);
}

void skinny::emitOutput(const OutputContext &ctx, OutputMap &map,
                        const TileIds &ids, std::optional<RegisterRef> cData) {
  IRBuilder &builder = ctx.builder;
  bool anyVector = hasVectorRuns(map, true) || hasVectorRuns(map, false);
  std::optional<RegisterRange> tileFullRange;
  std::optional<RegisterRef> tileFull;
  if (anyVector) {
    tileFullRange = builder.allocateSgpr(1);
    tileFull = tileFullRange->ref();
    computeTileFull(ctx, ids, *tileFull);
  }
  // A single split stores bf16 straight to D, else each split stores its f32
  // partial slice into the workspace.
  std::shared_ptr<rocisa::Label> workspace = builder.createLabel("workspace");
  std::shared_ptr<rocisa::Label> done = builder.createLabel("stored");
  builder.create<rocisa::SCmpLeU32>(ctx.k.container(), ctx.spk.container());
  builder.create<rocisa::SCBranchSCC0>(workspace->getLabelName());
  scaleByAlpha(ctx, map);
  if (cData)
    addBetaC(ctx, map, *cData, tileFull);
  storeTile(ctx, map, ctx.d, tileFull);
  builder.create<rocisa::SBranch>(done->getLabelName());
  builder.insert(workspace);
  RegisterRange slice = builder.allocateSgpr(1);
  builder.muli32(slice.ref(), ids.kz, ctx.mn);
  builder.shli32(slice.ref(), slice.ref(), 2);
  StoreTarget target = ctx.workspace;
  target.sliceBase = slice.ref();
  storeTile(ctx, map, target, tileFull);
  builder.insert(done);
}
