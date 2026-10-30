// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The scaled-MFMA compute path of the skinny MXFP8 GEMM emitter, which uses
// v_mfma_scale_f32_16x16x128_f8f6f4 instead of the VALU conversions and dot2.
// emitMfma emits the whole kernel: the setup shared with the VALU path
// (emitArguments), the tile of the 3-D launch or the tile loop of the
// persistent launch, and per tile the C prefetch, the K loop (the ring schedule
// of SkinnyGEMMKLoop.cpp over MfmaRingPath), the guarded tail iteration of a
// partial K (the MFMA half of SkinnyGEMMTail.cpp), the K reduction through LDS
// and the epilogue.

#include <tensilelite/API/SkinnyGEMM.hpp>
#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Kernel.hpp>
#include <tensilelite/Codegen/Registers.hpp>

#include "SkinnyGEMMEmitter.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mfma.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::skinny;

// The voffset sentinel 0x80000000 of the accesses that must be out of range:
// with the soffset and the immediate added it stays at or above num_records,
// unlike -1, which wraps. It only works for buffers smaller than 2 GiB, which
// the host must guarantee (see emitSkinnyGEMM).
static constexpr int32_t farOutOfRange = std::numeric_limits<int32_t>::min();

// K bytes of one iteration, two MFMA steps of 128.
static constexpr int32_t iterationBytes = 256;
static constexpr int32_t stepBytes = 128;
// A lane loads its two 16-byte pieces of a step with two dwordx4 loads, 64
// bytes apart.
static constexpr int32_t loadsPerStepTile = 2;
static constexpr int32_t chunkBytes = 64;
// VGPRs of the data of one 16-column tile and one MFMA step.
static constexpr int32_t stepRegs = 8;
static constexpr int32_t accRegs = 4;
static constexpr int32_t ldsTileBytes = 1024;
// The staged variant: one wave load covers 8 columns of a K step, a 1 KiB blob
// in lane order, and a ds offset reaches 64 KiB from its address register.
static constexpr int32_t blobBytes = 1024;
static constexpr int32_t readSegmentBytes = 65536;
// The wait states after the MFMA hazards, from s_nop. They are conservative
// guesses, not measured on gfx950: the code passes the GPU matrix with them,
// and no hazard table was available to derive tighter values.
static constexpr int32_t maxNop = 15;
static constexpr int32_t valuToMfmaNop = 7;

namespace {

// The ring slots and the scale plan. A slot holds the data of every wide and
// skinny tile for both MFMA steps (8 VGPRs per tile and step), then the scale
// dwords. With lds the data is not in the ring slots: the tile data of the two
// MFMA steps is read from LDS into two step buffers that follow the slots and
// are shared by all of them. A scale dword serves two tiles ("paired": the wide
// operand with an even tn, the skinny one with bm = 32, the tile selecting its
// row half with op_sel) or one tile ("single": the dword is shifted so that its
// byte 0 or 2 holds the tile's scales). With lds and partial steps one more
// slot, after the ring, holds the scales and the LDS data of the tail iteration,
// which is loaded in the prologue.
struct MfmaLayout {
  explicit MfmaLayout(const SkinnyDims &dims)
      : wideTiles(dims.wideTiles), skinnyTiles(dims.skinnyTiles),
        slots(dims.prefetch), lds(dims.lds),
        widePaired(dims.wideTiles % 2 == 0),
        skinnyPaired(dims.skinnyTiles == 2),
        wideScales(widePaired ? wideTiles / 2 : wideTiles), skinnyScales(1) {
    slotRegs = (dataRegs() + wideScales + skinnyScales + 1) & ~1;
    tailSlot = lds && dims.partialSteps ? slots : 0;
    stepBuffersBase = (slots + (tailSlot != 0 ? 1 : 0)) * slotRegs;
    stepLoads = loadsPerStepTile * (wideTiles + skinnyTiles);
    scaleLoads = wideScales + skinnyScales;
    loadsPerSlot = scaleLoads + 2 * stepLoads;
  }

  int32_t tiles() const { return wideTiles * skinnyTiles; }
  // The voffsets of a tile: one, or with lds one per column half of a load.
  int32_t basesPerTile() const { return lds ? 2 : 1; }
  // The data registers of a ring slot, none with lds.
  int32_t dataRegs() const {
    return lds ? 0 : 2 * stepRegs * (wideTiles + skinnyTiles);
  }
  // The VGPRs of the ring: the slots, then with lds the step buffers.
  int32_t ringRegs() const {
    return stepBuffersBase +
           (lds ? 2 * stepRegs * (wideTiles + skinnyTiles) : 0);
  }
  RegisterRef wideData(RegisterRef ring, int32_t slot, int32_t tile,
                       int32_t step) const {
    if (lds)
      return stepBuffer(ring, step, tile);
    return ring.slice(slot * slotRegs + (2 * tile + step) * stepRegs, stepRegs);
  }
  RegisterRef skinnyData(RegisterRef ring, int32_t slot, int32_t tile,
                         int32_t step) const {
    if (lds)
      return stepBuffer(ring, step, wideTiles + tile);
    return ring.slice(
        slot * slotRegs + (2 * (wideTiles + tile) + step) * stepRegs, stepRegs);
  }
  // The data of a tile (the wide tiles, then the skinny ones) at a K step.
  RegisterRef stepBuffer(RegisterRef ring, int32_t step, int32_t tile) const {
    return ring.slice(stepBuffersBase +
                          (step * (wideTiles + skinnyTiles) + tile) * stepRegs,
                      stepRegs);
  }
  RegisterRef wideScale(RegisterRef ring, int32_t slot, int32_t index) const {
    return ring.slice(slot * slotRegs + dataRegs() + index);
  }
  RegisterRef skinnyScale(RegisterRef ring, int32_t slot, int32_t index) const {
    return ring.slice(slot * slotRegs + dataRegs() + wideScales + index);
  }

  int32_t wideTiles, skinnyTiles, slots;
  bool lds;
  bool widePaired, skinnyPaired;
  int32_t wideScales, skinnyScales;
  int32_t slotRegs = 0;
  // The slot of the tail iteration: the one after the ring with lds and partial
  // steps, else the first.
  int32_t tailSlot = 0;
  // With lds, the first VGPR of the step buffers.
  int32_t stepBuffersBase = 0;
  // Loads of one MFMA step (data only), of the scales, and of a whole slot.
  int32_t stepLoads = 0, scaleLoads = 0, loadsPerSlot = 0;
};

// The lane values of the staged variant. A load covers 8 columns, so the lane
// loads the column loadColumn (lane / 8) of the half, at the K byte offset
// chunkBytes of the half (the lane's slot of the 128-byte row, swizzled). The
// operand reads address the LDS through readAddress, two VGPRs per 64 KiB
// segment of the ring (the chunks lg and 4 + lg of the lane's column), from the
// ring of the wave that starts at the SGPR ringBase.
struct StagedLane {
  RegisterRange loadColumn, chunkBytes, readAddress, ringBase;
};

// The per-lane values, which only depend on the lane in the wave: the column
// of the lane within a 16-column tile, its K byte offset within a step, the
// first of its four D rows, its scale dword offset and its LDS address.
struct MfmaLane {
  RegisterRange column, kBytes, rowBase, scaleBytes;
  std::optional<RegisterRange> ldsAddress;
  std::optional<StagedLane> staged;
};

// What every phase of the kernel reads.
struct MfmaContext {
  IRBuilder &builder;
  const SkinnyDims &dims;
  const MfmaLayout &layout;
  const MfmaLane &lane;
  RegisterRef sRsrc, wRsrc, sSclRsrc, wSclRsrc;
  // The num_records of the four descriptors above, in that order; only with
  // the tail slot, whose loads zero them for the waves without tail bytes.
  std::optional<RegisterRef> savedSizes;
  RegisterRef skinnyCols, wideCols, rowGroup, ldS, ldW, k, spk, waveId;
  // What the shared epilogue and the C prefetch read.
  OutputContext output;
};

// The registers of one tile.
struct MfmaTile {
  MfmaTile(IRBuilder &builder, const MfmaLayout &layout, bool useBeta)
      : acc(builder.allocateVgpr(accRegs * layout.tiles(), 2)),
        wideBase(
            builder.allocateVgpr(layout.wideTiles * layout.basesPerTile())),
        skinnyBase(
            builder.allocateVgpr(layout.skinnyTiles * layout.basesPerTile())),
        wideScaleBase(builder.allocateVgpr(layout.wideScales)),
        skinnyScaleBase(builder.allocateVgpr(layout.skinnyScales)),
        s0(builder.allocateSgpr(1)), w0(builder.allocateSgpr(1)) {
    if (useBeta)
      cData = builder.allocateVgpr(accRegs * layout.tiles(), 2);
    if (!layout.widePaired)
      wideShift = builder.allocateSgpr(layout.wideTiles);
    if (!layout.skinnyPaired)
      skinnyShift = builder.allocateSgpr(1);
  }
  RegisterRange acc;
  // The voffsets of the data (per column half with lds) and of the scale dword
  // of every tile.
  RegisterRange wideBase, skinnyBase, wideScaleBase, skinnyScaleBase;
  // The first skinny and wide column of the tile.
  RegisterRange s0, w0;
  // The bit shifts that bring the tile's scale bytes to the byte 0 and 2 of a
  // single dword.
  std::optional<RegisterRange> wideShift, skinnyShift;
  // The prefetched C elements, with useBeta.
  std::optional<RegisterRange> cData;
};

// The SGPR state of the K loop of one wave. Iterations are numbered per wave:
// iteration i is the K iteration waveId + waves * i of the split.
struct MfmaLoop {
  MfmaLoop(IRBuilder &builder, const SkinnyDims &dims)
      : waveIterations(builder.allocateSgpr(1)),
        issueOffset(builder.allocateSgpr(1)) {
    if (!dims.partialSteps)
      return;
    tailBytes = builder.allocateSgpr(1);
    tailOffset = builder.allocateSgpr(1);
  }
  RegisterRange waveIterations;
  // The K byte offset of the next iteration to issue.
  RegisterRange issueOffset;
  // With partial steps, the bytes of the tail iteration that this wave runs, 0
  // when it does not run the tail, and the K byte offset of that iteration.
  std::optional<RegisterRange> tailBytes, tailOffset;
};

// The offsets of the elements of a tile into an output matrix.
struct OutputOffsets {
  OutputOffsets(IRBuilder &builder, const MfmaLayout &layout)
      : skinnyOffset(builder.allocateVgpr(layout.skinnyTiles)),
        skinnyValid(builder.allocateSgpr(2 * layout.skinnyTiles, 2)),
        wideColumn(builder.allocateVgpr(layout.wideTiles)),
        wideOffset(builder.allocateVgpr(layout.wideTiles)),
        elementStride(builder.allocateSgpr(3)),
        wideLimit(builder.allocateSgpr(4)) {}
  // The byte offset of the lane's skinny column of every skinny tile, and
  // whether the column is inside the matrix.
  RegisterRange skinnyOffset, skinnyValid;
  // The first wide row of the lane of every wide tile and its byte offset.
  RegisterRange wideColumn, wideOffset;
  // e * strideW for e = 1..3, and wideCols - e for e = 0..3.
  RegisterRange elementStride, wideLimit;
};

} // namespace

// Lane values.

// dst = f(column) = column / 2, the chunk swizzle of the staged variant: a
// column c of a 16-column tile keeps its 16-byte chunk x of a 128-byte K row at
// the slot x xor f(c) of the row in LDS. The lanes of an operand read are
// served in groups of 8 (consecutive columns of a lane group), and the columns
// that share a bank row (c and c + 2, + 4, + 6 for 64 banks of 16-byte
// groups) land on distinct slots with this f; rocprofv3 shows no
// SQ_LDS_BANK_CONFLICT for it, against 12 cycles per read without a swizzle.
static void chunkSwizzle(IRBuilder &builder, RegisterRef dst,
                         RegisterRef column) {
  builder.shrui32(dst, column, 1);
}

// The VGPR and SGPR values of the staged variant.
static StagedLane makeStagedLane(IRBuilder &builder, const SkinnyDims &dims,
                                 RegisterRef laneId, RegisterRef waveId) {
  int32_t segments = (dims.ringBytes + readSegmentBytes - 1) / readSegmentBytes;
  StagedLane staged{builder.allocateVgpr(1), builder.allocateVgpr(2),
                    builder.allocateVgpr(2 * segments),
                    builder.allocateSgpr(1)};
  builder.shrui32(staged.loadColumn.ref(), laneId, 3);
  builder.muli32(staged.ringBase.ref(), waveId, dims.ringBytes);
  RegisterRange slot = builder.allocateVgpr(1);
  RegisterRange swizzle = builder.allocateVgpr(1);
  RegisterRange column = builder.allocateVgpr(1);
  // The loads: lane l holds the column 8 h + l / 8 of the half h, in the slot
  // l % 8 of its row, which holds the chunk (l % 8) xor f(column).
  builder.andi32(slot.ref(), 7, laneId);
  for (int32_t half = 0; half < 2; ++half) {
    RegisterRef chunk = staged.chunkBytes.ref(half);
    builder.addi32(column.ref(), staged.loadColumn.ref(), 8 * half);
    chunkSwizzle(builder, swizzle.ref(), column.ref());
    builder.create<rocisa::VXorB32>(
        chunk.container(), swizzle.ref().container(), slot.ref().container());
    builder.shli32(chunk, chunk, 4);
  }
  // The reads: the lane (column c = l % 16, lane group lg = l / 16) reads the
  // chunk lg at the slot lg xor f(c) of the row c % 8 of the blob c / 8, and
  // the chunk 4 + lg, which differs in the slot bit 2, so its address differs
  // in bit 6.
  RegisterRange row = builder.allocateVgpr(1);
  builder.andi32(column.ref(), 15, laneId);
  builder.shrui32(row.ref(), column.ref(), 3);
  builder.shli32(row.ref(), row.ref(), 10);
  builder.andi32(slot.ref(), 7, column.ref());
  builder.shli32(slot.ref(), slot.ref(), 7);
  builder.addi32(row.ref(), row.ref(), slot.ref());
  chunkSwizzle(builder, swizzle.ref(), column.ref());
  builder.shrui32(slot.ref(), laneId, 4);
  builder.create<rocisa::VXorB32>(slot.ref().container(),
                                  swizzle.ref().container(),
                                  slot.ref().container());
  builder.shli32(slot.ref(), slot.ref(), 4);
  builder.addi32(row.ref(), row.ref(), slot.ref());
  RegisterRange base = builder.allocateSgpr(1);
  for (int32_t segment = 0; segment < segments; ++segment) {
    builder.addi32(base.ref(), staged.ringBase.ref(),
                   segment * readSegmentBytes);
    builder.addi32(staged.readAddress.ref(2 * segment), row.ref(), base.ref());
    builder.create<rocisa::VXorB32>(
        staged.readAddress.ref(2 * segment + 1).container(), 64,
        staged.readAddress.ref(2 * segment).container());
  }
  return staged;
}

static MfmaLane makeLane(IRBuilder &builder, const SkinnyDims &dims,
                         RegisterRef tid, RegisterRef waveId) {
  MfmaLane lane{builder.allocateVgpr(1), builder.allocateVgpr(1),
                builder.allocateVgpr(1), builder.allocateVgpr(1), std::nullopt};
  RegisterRange laneId = builder.allocateVgpr(1);
  builder.andi32(laneId.ref(), 63, tid);
  builder.andi32(lane.column.ref(), 15, laneId.ref());
  // The lane group lg = lane / 16 holds the K bytes [16 lg, 16 lg + 16) and
  // [64 + 16 lg, 64 + 16 lg + 16) of the step, as the hardware lays out src0
  // and src1.
  builder.andi32(lane.kBytes.ref(), 48, laneId.ref());
  builder.andi32(lane.rowBase.ref(), 48, laneId.ref());
  builder.shrui32(lane.rowBase.ref(), lane.rowBase.ref(), 2);
  builder.shli32(lane.scaleBytes.ref(), laneId.ref(), 2);
  if (dims.lds)
    lane.staged = makeStagedLane(builder, dims, laneId.ref(), waveId);
  if (dims.waves == 1)
    return lane;
  // Wave w > 0 writes the slice w - 1 of the reduction area; wave 0 reads every
  // slice from the base of the area with immediate offsets.
  lane.ldsAddress = builder.allocateVgpr(1);
  RegisterRange slice = builder.allocateSgpr(1);
  builder.addi32(slice.ref(), waveId, -1);
  builder.maxsi32(slice.ref(), slice.ref(), 0);
  builder.muli32(slice.ref(), slice.ref(),
                 dims.wideTiles * dims.skinnyTiles * ldsTileBytes);
  // The area follows the rings of the waves.
  if (dims.lds)
    builder.addi32(slice.ref(), slice.ref(), dims.waves * dims.ringBytes);
  builder.shli32(lane.ldsAddress->ref(), laneId.ref(), 4);
  builder.addi32(lane.ldsAddress->ref(), lane.ldsAddress->ref(), slice.ref());
  return lane;
}

// waveId = the wave of the block, as a uniform value.
static void computeWaveId(IRBuilder &builder, const SkinnyDims &dims,
                          RegisterRef waveId, RegisterRef tid) {
  if (dims.waves == 1) {
    builder.movb32(waveId, 0);
    return;
  }
  RegisterRange wave = builder.allocateVgpr(1);
  builder.shrui32(wave.ref(), tid, 6);
  // Observed on gfx950: without wait states a v_readfirstlane right after the
  // VALU write of its source returned the stale register, so the waves computed
  // wrong ids and the kernel hung. s_nop 4 fixed it, and it is the only value
  // tried.
  builder.create<rocisa::SNop>(4);
  builder.create<rocisa::VReadfirstlaneB32>(waveId.container(),
                                            wave.container());
}

// Tile bases.

// base[i] = the voffset of the column first + 16 * i for the lane, or far out
// of range when the column is past cols.
static void dataBases(const MfmaContext &ctx, RegisterRef base, int32_t count,
                      RegisterRef first, RegisterRef cols, RegisterRef ld) {
  IRBuilder &builder = ctx.builder;
  for (int32_t i = 0; i < count; ++i) {
    RegisterRange start = builder.allocateSgpr(1);
    RegisterRange column = builder.allocateVgpr(1);
    RegisterRange valid = builder.allocateSgpr(2, 2);
    builder.addi32(start.ref(), first, 16 * i);
    builder.addi32(column.ref(), ctx.lane.column.ref(), start.ref());
    builder.cmpi32(Predicate::ult, valid.ref(), column.ref(), cols);
    RegisterRef voffset = base.slice(i);
    builder.muli32(voffset, column.ref(), ld);
    builder.addi32(voffset, voffset, ctx.lane.kBytes.ref());
    builder.selectb32(voffset, valid.ref(), voffset, farOutOfRange);
  }
}

// base[2 i + h] = the voffset of the lane's chunk of the column first + 16 i +
// 8 h + lane / 8 for the load of the column half h, or far out of range when
// the column is past cols.
static void stagedBases(const MfmaContext &ctx, RegisterRef base, int32_t count,
                        RegisterRef first, RegisterRef cols, RegisterRef ld) {
  IRBuilder &builder = ctx.builder;
  const StagedLane &staged = *ctx.lane.staged;
  for (int32_t i = 0; i < count; ++i)
    for (int32_t half = 0; half < 2; ++half) {
      RegisterRange start = builder.allocateSgpr(1);
      RegisterRange column = builder.allocateVgpr(1);
      RegisterRange valid = builder.allocateSgpr(2, 2);
      builder.addi32(start.ref(), first, 16 * i + 8 * half);
      builder.addi32(column.ref(), staged.loadColumn.ref(), start.ref());
      builder.cmpi32(Predicate::ult, valid.ref(), column.ref(), cols);
      RegisterRef voffset = base.slice(2 * i + half);
      builder.muli32(voffset, column.ref(), ld);
      builder.addi32(voffset, voffset, staged.chunkBytes.ref(half));
      builder.selectb32(voffset, valid.ref(), voffset, farOutOfRange);
    }
}

// base = the voffset of the lane's dword of the scale tile of the 32-column
// group that holds the column first; with a shift, also the bit shift that
// moves the tile's half of the dword to its low bytes.
static void scaleBase(const MfmaContext &ctx, RegisterRef base,
                      std::optional<RegisterRef> shift, RegisterRef first) {
  IRBuilder &builder = ctx.builder;
  RegisterRange group = builder.allocateSgpr(1);
  builder.shrui32(group.ref(), first, 5);
  builder.muli32(group.ref(), group.ref(), ctx.rowGroup);
  builder.addi32(base, ctx.lane.scaleBytes.ref(), group.ref());
  if (!shift)
    return;
  builder.andi32(*shift, 16, first);
  builder.shrui32(*shift, *shift, 1);
}

static void computeTileBases(const MfmaContext &ctx, const MfmaTile &tile) {
  IRBuilder &builder = ctx.builder;
  const MfmaLayout &layout = ctx.layout;
  if (layout.lds) {
    stagedBases(ctx, tile.skinnyBase.ref(), layout.skinnyTiles, tile.s0.ref(),
                ctx.skinnyCols, ctx.ldS);
    stagedBases(ctx, tile.wideBase.ref(), layout.wideTiles, tile.w0.ref(),
                ctx.wideCols, ctx.ldW);
  } else {
    dataBases(ctx, tile.skinnyBase.ref(), layout.skinnyTiles, tile.s0.ref(),
              ctx.skinnyCols, ctx.ldS);
    dataBases(ctx, tile.wideBase.ref(), layout.wideTiles, tile.w0.ref(),
              ctx.wideCols, ctx.ldW);
  }
  scaleBase(ctx, tile.skinnyScaleBase.ref(),
            tile.skinnyShift
                ? std::optional<RegisterRef>(tile.skinnyShift->ref())
                : std::nullopt,
            tile.s0.ref());
  for (int32_t d = 0; d < layout.wideScales; ++d) {
    // A paired dword covers two tiles, 32 columns.
    RegisterRange first = builder.allocateSgpr(1);
    builder.addi32(first.ref(), tile.w0.ref(),
                   (layout.widePaired ? 32 : 16) * d);
    scaleBase(ctx, tile.wideScaleBase.ref(d),
              tile.wideShift
                  ? std::optional<RegisterRef>(tile.wideShift->ref(d))
                  : std::nullopt,
              first.ref());
  }
}

// The K loop.

// Places the tail iteration after the extent's whole iterations, at the K
// offset ks + 256 * wholeIterations. The wave that the round-robin gives the
// next iteration runs it; the others get no tail bytes.
static void startTail(const MfmaContext &ctx, const MfmaLoop &loop,
                      RegisterRef start, RegisterRef wholeIterations) {
  IRBuilder &builder = ctx.builder;
  builder.shli32(loop.tailOffset->ref(), wholeIterations, 8);
  builder.addi32(loop.tailOffset->ref(), loop.tailOffset->ref(), start);
  if (ctx.dims.waves == 1)
    return;
  RegisterRange tailWave = builder.allocateSgpr(1);
  RegisterRange isTailWave = builder.allocateSgpr(1);
  moduloByLiteral(builder, tailWave.ref(), wholeIterations, ctx.dims.waves);
  builder.cmpi32(Predicate::eq, isTailWave.ref(), ctx.waveId, tailWave.ref());
  builder.selectb32(loop.tailBytes->ref(), isTailWave.ref(),
                    loop.tailBytes->ref(), 0);
}

// Starts the loop state of the wave: its iteration count and the first K offset
// of the split, ks + 256 * waveId.
static void startLoop(const MfmaContext &ctx, const MfmaLoop &loop,
                      RegisterRef kz) {
  IRBuilder &builder = ctx.builder;
  const SkinnyDims &dims = ctx.dims;
  RegisterRange start = builder.allocateSgpr(1);
  RegisterRange extent = builder.allocateSgpr(1);
  builder.muli32(start.ref(), kz, ctx.spk);
  builder.subi32(extent.ref(), ctx.k, start.ref());
  builder.minsi32(extent.ref(), extent.ref(), ctx.spk);
  if (dims.partialSteps)
    builder.andi32(loop.tailBytes->ref(), extent.ref(), iterationBytes - 1);
  builder.shrui32(extent.ref(), extent.ref(), 8);
  // Wave w runs the iterations w, w + NW, ... of the split's NI iterations.
  RegisterRange share = builder.allocateSgpr(1);
  builder.addi32(share.ref(), extent.ref(), dims.waves - 1);
  builder.subi32(share.ref(), share.ref(), ctx.waveId);
  divideByLiteral(builder, loop.waveIterations.ref(), share.ref(), dims.waves);
  builder.shli32(loop.issueOffset.ref(), ctx.waveId, 8);
  builder.addi32(loop.issueOffset.ref(), loop.issueOffset.ref(), start.ref());
  if (dims.partialSteps)
    startTail(ctx, loop, start.ref(), extent.ref());
}

// The loads of the scale dwords of the K iteration at the soffset into a ring
// slot.
static std::vector<SlotLoad> scaleLoads(const MfmaContext &ctx,
                                        const MfmaTile &tile, RegisterRef ring,
                                        int32_t slot, RegisterRef soffset) {
  const MfmaLayout &layout = ctx.layout;
  std::vector<SlotLoad> loads;
  for (int32_t d = 0; d < layout.wideScales; ++d)
    loads.push_back({.rsrc = ctx.wSclRsrc,
                     .voffset = tile.wideScaleBase.ref(d),
                     .soffset = soffset,
                     .width = MemoryWidth::b32,
                     .dst = layout.wideScale(ring, slot, d)});
  loads.push_back({.rsrc = ctx.sSclRsrc,
                   .voffset = tile.skinnyScaleBase.ref(),
                   .soffset = soffset,
                   .width = MemoryWidth::b32,
                   .dst = layout.skinnyScale(ring, slot, 0)});
  return loads;
}

// The loads of the data of the K iteration at the soffset into a ring slot's
// VGPRs, the two chunks of every tile for each of the two MFMA steps.
static std::vector<SlotLoad> slotDataLoads(const MfmaContext &ctx,
                                           const MfmaTile &tile,
                                           RegisterRef ring, int32_t slot,
                                           RegisterRef offset) {
  const MfmaLayout &layout = ctx.layout;
  std::vector<SlotLoad> loads;
  for (int32_t step = 0; step < 2; ++step) {
    for (int32_t half = 0; half < 2; ++half) {
      int32_t immediate = step * stepBytes + half * chunkBytes;
      for (int32_t t = 0; t < layout.wideTiles; ++t)
        loads.push_back(
            {.rsrc = ctx.wRsrc,
             .voffset = tile.wideBase.ref(t),
             .soffset = offset,
             .immediate = immediate,
             .width = MemoryWidth::b128,
             .dst = layout.wideData(ring, slot, t, step).slice(4 * half, 4)});
      for (int32_t u = 0; u < layout.skinnyTiles; ++u)
        loads.push_back(
            {.rsrc = ctx.sRsrc,
             .voffset = tile.skinnyBase.ref(u),
             .soffset = offset,
             .immediate = immediate,
             .width = MemoryWidth::b128,
             .dst = layout.skinnyData(ring, slot, u, step).slice(4 * half, 4)});
    }
  }
  return loads;
}

// The loads of the data of the K iteration at the soffset straight into a slot
// of the wave's LDS ring, with the same order as slotDataLoads: per step and
// column half, one load of 8 columns x 128 contiguous bytes for every tile. The
// tile index counts the wide tiles, then the skinny ones. The immediate offset
// of the step moves the global address, and loadBufferToLds keeps the LDS
// address of the load unaffected by it.
static std::vector<SlotLoad> stagedDataLoads(const MfmaContext &ctx,
                                             const MfmaTile &tile, int32_t slot,
                                             RegisterRef offset) {
  const MfmaLayout &layout = ctx.layout;
  int32_t tiles = layout.wideTiles + layout.skinnyTiles;
  RegisterRef ringBase = ctx.lane.staged->ringBase.ref();
  std::vector<SlotLoad> loads;
  for (int32_t step = 0; step < 2; ++step)
    for (int32_t half = 0; half < 2; ++half)
      for (int32_t index = 0; index < tiles; ++index) {
        bool wide = index < layout.wideTiles;
        RegisterRef voffset =
            wide ? tile.wideBase.ref(2 * index + half)
                 : tile.skinnyBase.ref(2 * (index - layout.wideTiles) + half);
        int32_t ldsOffset =
            (slot * tiles + index) * SkinnyDims::stagedTileBytes +
            (2 * step + half) * blobBytes;
        loads.push_back({.rsrc = wide ? ctx.wRsrc : ctx.sRsrc,
                         .voffset = voffset,
                         .soffset = offset,
                         .immediate = step * stepBytes,
                         .width = MemoryWidth::b128,
                         .lds = LdsTarget{ringBase, ldsOffset}});
      }
  return loads;
}

// Gives the descriptors their size where the condition holds and num_records 0
// where it does not, so that every access is out of range.
static void selectSizes(const MfmaContext &ctx, RegisterRef condition) {
  int32_t index = 0;
  for (RegisterRef rsrc : {ctx.sRsrc, ctx.wRsrc, ctx.sSclRsrc, ctx.wSclRsrc})
    ctx.builder.selectb32(rsrc.slice(2), condition,
                          ctx.savedSizes->slice(index++), 0);
}

// Gives the descriptors their size back.
static void restoreSizes(const MfmaContext &ctx) {
  int32_t index = 0;
  for (RegisterRef rsrc : {ctx.sRsrc, ctx.wRsrc, ctx.sSclRsrc, ctx.wSclRsrc})
    ctx.builder.movb32(rsrc.slice(2), ctx.savedSizes->slice(index++));
}

// Moves the issue to the next iteration.
static void advanceSlot(const MfmaContext &ctx, const MfmaLoop &loop) {
  ctx.builder.addi32(loop.issueOffset.ref(), loop.issueOffset.ref(),
                     iterationBytes * ctx.dims.waves);
}

// Issues the loads of the next iteration into a ring slot. With lds the data
// comes before the scales, as the K loop refills the slot in two parts.
static void issueSlot(const MfmaContext &ctx, const MfmaTile &tile,
                      const MfmaLoop &loop, RegisterRef ring, int32_t slot) {
  RegisterRef offset = loop.issueOffset.ref();
  if (ctx.layout.lds) {
    emitSlotLoads(ctx.builder, stagedDataLoads(ctx, tile, slot, offset));
    emitSlotLoads(ctx.builder, scaleLoads(ctx, tile, ring, slot, offset));
  } else {
    emitSlotLoads(ctx.builder, scaleLoads(ctx, tile, ring, slot, offset));
    emitSlotLoads(ctx.builder, slotDataLoads(ctx, tile, ring, slot, offset));
  }
  advanceSlot(ctx, loop);
}

// Brings the tile's scale bytes of every single dword to the byte 0 and 2.
static void shiftScales(const MfmaContext &ctx, const MfmaTile &tile,
                        RegisterRef ring, int32_t slot) {
  IRBuilder &builder = ctx.builder;
  const MfmaLayout &layout = ctx.layout;
  if (tile.wideShift)
    for (int32_t t = 0; t < layout.wideTiles; ++t)
      builder.create<rocisa::VLShiftRightB32>(
          layout.wideScale(ring, slot, t).container(),
          tile.wideShift->ref(t).container(),
          layout.wideScale(ring, slot, t).container());
  if (tile.skinnyShift)
    builder.create<rocisa::VLShiftRightB32>(
        layout.skinnyScale(ring, slot, 0).container(),
        tile.skinnyShift->ref().container(),
        layout.skinnyScale(ring, slot, 0).container());
  builder.create<rocisa::SNop>(valuToMfmaNop);
}

// One MFMA of the 16x16 tile (t, u) at a K step: the wide tile is src0 and the
// skinny tile src1. A scale operand selects its byte with op_sel (bit 0) and
// op_sel_hi (bit 1): byte 2 * step + the row half of a paired tile. Dependent
// MFMAs on the same accumulator get no wait states, as their destination and
// srcC overlap exactly; that is the usual srcC forwarding rule, not verified on
// gfx950 beyond the passing GPU matrix.
static void mfma(const MfmaContext &ctx, const MfmaTile &tile, RegisterRef ring,
                 int32_t slot, int32_t step, int32_t t, int32_t u) {
  const MfmaLayout &layout = ctx.layout;
  RegisterRef acc =
      tile.acc.ref((t * layout.skinnyTiles + u) * accRegs, accRegs);
  RegisterRef scaleA =
      layout.wideScale(ring, slot, layout.widePaired ? t / 2 : t);
  RegisterRef scaleB = layout.skinnyScale(ring, slot, 0);
  int32_t byteA = 2 * step + (layout.widePaired ? t % 2 : 0);
  int32_t byteB = 2 * step + (layout.skinnyPaired ? u % 2 : 0);
  ctx.builder.create<rocisa::MXMFMAInstruction>(
      rocisa::InstType::INST_F8, rocisa::InstType::INST_F32,
      std::vector<int>{16, 16, 128, 1}, acc.container(),
      layout.wideData(ring, slot, t, step).container(),
      layout.skinnyData(ring, slot, u, step).container(), acc.container(),
      scaleA.container(), scaleB.container(),
      rocisa::VOP3PModifiers(std::vector<int>{byteA & 1, byteB & 1, 0},
                             std::vector<int>{byteA >> 1, byteB >> 1, 0}));
}

static void mfmaStep(const MfmaContext &ctx, const MfmaTile &tile,
                     RegisterRef ring, int32_t slot, int32_t step) {
  for (int32_t t = 0; t < ctx.layout.wideTiles; ++t)
    for (int32_t u = 0; u < ctx.layout.skinnyTiles; ++u)
      mfma(ctx, tile, ring, slot, step, t, u);
}

// Consumes a slot and, with a refill, refills it. The slots issue in order, so
// when the slot is consumed the step's groupsAfter slots are in flight behind
// its loads: the first step waits for all but those and the step-1 loads, the
// second for all but those. vmcnt counts the older loads of the tile, such as
// C, first.
static void consumeSlot(const MfmaContext &ctx, const MfmaTile &tile,
                        const MfmaLoop &loop, RegisterRef ring,
                        const RingStep &step) {
  IRBuilder &builder = ctx.builder;
  const MfmaLayout &layout = ctx.layout;
  RingLoads loads{layout.loadsPerSlot};
  waitForVectorLoads(builder,
                     loads.issuedAfter(layout.scaleLoads + layout.stepLoads - 1,
                                       step.groupsAfter));
  shiftScales(ctx, tile, ring, step.slot);
  mfmaStep(ctx, tile, ring, step.slot, 0);
  waitForVectorLoads(
      builder, loads.issuedAfter(layout.loadsPerSlot - 1, step.groupsAfter));
  mfmaStep(ctx, tile, ring, step.slot, 1);
  if (step.refill)
    issueSlot(ctx, tile, loop, ring, step.slot);
}

// The staged K loop.

// Waits until at most outstanding LDS reads are in flight, capped at the
// counter's range, so that a larger count only waits longer.
static void waitForLds(IRBuilder &builder, int32_t outstanding) {
  builder.createWaitCnt(
      WaitCounts{.lds = std::min(outstanding, builder.target().maxLgkmcnt())});
}

// Reads the MFMA operands of a K step of a ring slot from the wave's LDS into
// the step buffer: for every tile, the chunks lg and 4 + lg of the lane's
// column as two ds_read_b128.
static void readStep(const MfmaContext &ctx, RegisterRef ring, int32_t slot,
                     int32_t step) {
  const MfmaLayout &layout = ctx.layout;
  const StagedLane &staged = *ctx.lane.staged;
  int32_t tiles = layout.wideTiles + layout.skinnyTiles;
  for (int32_t index = 0; index < tiles; ++index) {
    bool wide = index < layout.wideTiles;
    RegisterRef data =
        wide ? layout.wideData(ring, slot, index, step)
             : layout.skinnyData(ring, slot, index - layout.wideTiles, step);
    int32_t offset = (slot * tiles + index) * SkinnyDims::stagedTileBytes +
                     2 * step * blobBytes;
    int32_t segment = offset / readSegmentBytes;
    ctx.builder.loadDs(data.slice(0, 4), staged.readAddress.ref(2 * segment),
                       MemoryWidth::b128, offset % readSegmentBytes);
    ctx.builder.loadDs(data.slice(4, 4),
                       staged.readAddress.ref(2 * segment + 1),
                       MemoryWidth::b128, offset % readSegmentBytes);
  }
}

// Reads the first step of the slot that follows the slot being consumed, once
// its first step's loads landed. Of the loads issued after them, the rest of
// the slot, the slots behind it (one fewer than behind the slot being
// consumed, which is older) and the data refill of the slot being consumed
// remain. Before the loop the bound assumes the first step of the loop, which
// is conservative: it also waits for the scales of the slot.
static void readFirstStep(const MfmaContext &ctx, RegisterRef ring,
                          int32_t slot, int32_t groupsAfter, bool refill) {
  const MfmaLayout &layout = ctx.layout;
  RingLoads loads{layout.loadsPerSlot};
  waitForVectorLoads(ctx.builder,
                     loads.issuedAfter(layout.stepLoads - 1, groupsAfter - 1,
                                       refill ? 2 * layout.stepLoads : 0));
  readStep(ctx, ring, slot, 0);
}

// Consumes a slot whose first step is already being read, and refills it. The
// step buffers are shared by all slots: the second step is read while the first
// one computes, and the first step of the next slot while the second computes.
// The slot's data is refilled as soon as its reads completed, before the second
// step computes, as the load overwrites its LDS bytes and every cycle that the
// slot stays empty shortens the ring; its scales are refilled after the second
// step, which reads them. The loads complete in order, so the slot's loads wait
// for all but the slots behind it. A step without a refill skips its loads, and
// one without a next slot (the last of the wave) skips the read-ahead.
static void consumeStagedSlot(const MfmaContext &ctx, const MfmaTile &tile,
                              const MfmaLoop &loop, RegisterRef ring,
                              const RingStep &step) {
  IRBuilder &builder = ctx.builder;
  const MfmaLayout &layout = ctx.layout;
  int32_t stepReads = 2 * (layout.wideTiles + layout.skinnyTiles);
  waitForVectorLoads(builder, RingLoads{layout.loadsPerSlot}.issuedAfter(
                                  layout.loadsPerSlot - 1, step.groupsAfter));
  readStep(ctx, ring, step.slot, 1);
  shiftScales(ctx, tile, ring, step.slot);
  waitForLds(builder, stepReads);
  mfmaStep(ctx, tile, ring, step.slot, 0);
  waitForLds(builder, 0);
  RegisterRef offset = loop.issueOffset.ref();
  if (step.refill)
    emitSlotLoads(builder, stagedDataLoads(ctx, tile, step.slot, offset));
  if (step.hasNext)
    readFirstStep(ctx, ring, (step.slot + 1) % layout.slots, step.groupsAfter,
                  step.refill);
  mfmaStep(ctx, tile, ring, step.slot, 1);
  if (!step.refill)
    return;
  emitSlotLoads(builder, scaleLoads(ctx, tile, ring, step.slot, offset));
  advanceSlot(ctx, loop);
}

// The tail iteration.

// The tail iteration of the staged variant is loaded in the prologue (see
// issueTail), its data into the LDS of the tail slot, so that its latency
// overlaps the loop. This reads both steps into the registers, where the masks
// apply as on the other path, once the loads have landed. The loads are in
// range up to the end of the buffer, and the bytes at or past the end of the
// split are masked after the read, so no lane needs steering. A load of the
// last column may straddle num_records: as measured on gfx950, a straddling
// buffer-to-LDS load writes its in-range dwords and zeros the others, as a load
// into registers returns them, and the masks need no more.
static void readStagedTail(const MfmaContext &ctx, RegisterRef ring) {
  readStep(ctx, ring, ctx.layout.tailSlot, 0);
  readStep(ctx, ring, ctx.layout.tailSlot, 1);
  waitForLds(ctx.builder, 0);
}

// Sets the scale bytes of the K blocks that start at or past the end of the
// split to 127 (2^0), as the loaded ones may be NaNs that would poison the
// zeroed products. The lane group lg supplies the block lg (bytes 0 and 1 of
// the dword) and 4 + lg (bytes 2 and 3) of 32 K bytes.
static void overrideTailScales(const MfmaContext &ctx, const MfmaTile &tile,
                               const MfmaLoop &loop, RegisterRef ring) {
  IRBuilder &builder = ctx.builder;
  const MfmaLayout &layout = ctx.layout;
  constexpr int32_t scaleOne = 0x7F7F7F7F;
  RegisterRange rem = builder.allocateVgpr(1);
  RegisterRange keep = builder.allocateVgpr(1);
  RegisterRange fill = builder.allocateVgpr(1);
  RegisterRange live = builder.allocateSgpr(2, 2);
  // The lane's block lg starts at 32 lg = 2 kBytes.
  builder.subi32(rem.ref(), loop.tailBytes->ref(), ctx.lane.kBytes.ref());
  builder.subi32(rem.ref(), rem.ref(), ctx.lane.kBytes.ref());
  builder.cmpi32(Predicate::sgt, live.ref(), rem.ref(), 0);
  builder.selectb32(keep.ref(), live.ref(), 0x0000FFFF, 0);
  builder.addi32(rem.ref(), rem.ref(), -stepBytes);
  builder.cmpi32(Predicate::sgt, live.ref(), rem.ref(), 0);
  builder.selectb32(rem.ref(), live.ref(), -65536, 0);
  builder.ori32(keep.ref(), keep.ref(), rem.ref());
  // The bytes of keep are 0x00 or 0xFF, so this leaves 0x7F where keep is 0.
  builder.andi32(fill.ref(), keep.ref(), scaleOne);
  builder.subi32(fill.ref(), scaleOne, fill.ref());
  std::vector<RegisterRef> scales;
  for (int32_t d = 0; d < layout.wideScales; ++d)
    scales.push_back(layout.wideScale(ring, layout.tailSlot, d));
  scales.push_back(layout.skinnyScale(ring, layout.tailSlot, 0));
  for (RegisterRef scale : scales) {
    builder.andi32(scale, scale, keep.ref());
    builder.ori32(scale, scale, fill.ref());
  }
}

// Issues the loads of the tail iteration of the wave that holds the tail bytes
// into the tail slot: the other waves get out-of-range descriptors and load
// nothing. The sizes are restored for the loop.
static void issueTail(const MfmaContext &ctx, const MfmaTile &tile,
                      const MfmaLoop &loop, RegisterRef ring) {
  RegisterRef offset = loop.tailOffset->ref();
  RegisterRange hasTail = ctx.builder.allocateSgpr(1);
  ctx.builder.cmpi32(Predicate::ne, hasTail.ref(), loop.tailBytes->ref(), 0);
  selectSizes(ctx, hasTail.ref());
  emitSlotLoads(ctx.builder,
                stagedDataLoads(ctx, tile, ctx.layout.tailSlot, offset));
  emitSlotLoads(ctx.builder,
                scaleLoads(ctx, tile, ring, ctx.layout.tailSlot, offset));
  restoreSizes(ctx);
}

namespace {

// The MFMA half of the tail iteration: a lane's chunk u is the 16 bytes at
// kBytes + 64 u of the iteration, in the MFMA operand registers of the tail
// slot, which is slot 0 without lds.
class MfmaTailPath final : public TailPath {
public:
  MfmaTailPath(const MfmaContext &ctx, const MfmaTile &tile,
               const MfmaLoop &loop, RegisterRef ring)
      : ctx(ctx), tile(tile), loop(loop), ring(ring) {}

  TailLanes lanes() const override {
    return {loop.tailBytes->ref(), ctx.lane.kBytes.ref(), chunkBytes,
            chunkBytes};
  }
  std::vector<RegisterRef> chunkData(int32_t chunk) const override {
    const MfmaLayout &layout = ctx.layout;
    int32_t step = chunk / 2;
    int32_t offset = 4 * (chunk % 2);
    std::vector<RegisterRef> data;
    for (int32_t t = 0; t < layout.wideTiles; ++t)
      data.push_back(
          layout.wideData(ring, layout.tailSlot, t, step).slice(offset, 4));
    for (int32_t u = 0; u < layout.skinnyTiles; ++u)
      data.push_back(
          layout.skinnyData(ring, layout.tailSlot, u, step).slice(offset, 4));
    return data;
  }
  // With lds the data is in the registers after readStagedTail.
  bool loadsChunks() const override { return !ctx.layout.lds; }
  // The chunk offset goes in the soffset: with an immediate, the steered
  // voffset -1 would wrap into range.
  void loadChunk(int32_t chunk, std::optional<RegisterRef> steer) override {
    const MfmaLayout &layout = ctx.layout;
    RegisterRange soffset = ctx.builder.allocateSgpr(1);
    ctx.builder.addi32(soffset.ref(), loop.tailOffset->ref(),
                       chunk * chunkBytes);
    std::vector<RegisterRef> data = chunkData(chunk);
    std::vector<SlotLoad> loads;
    for (int32_t t = 0; t < layout.wideTiles; ++t)
      loads.push_back({.rsrc = ctx.wRsrc,
                       .voffset = tile.wideBase.ref(t),
                       .soffset = soffset.ref(),
                       .width = MemoryWidth::b128,
                       .dst = data[t],
                       .steer = steer});
    for (int32_t u = 0; u < layout.skinnyTiles; ++u)
      loads.push_back({.rsrc = ctx.sRsrc,
                       .voffset = tile.skinnyBase.ref(u),
                       .soffset = soffset.ref(),
                       .width = MemoryWidth::b128,
                       .dst = data[layout.wideTiles + u],
                       .steer = steer});
    emitSlotLoads(ctx.builder, loads);
  }

private:
  const MfmaContext &ctx;
  const MfmaTile &tile;
  const MfmaLoop &loop;
  RegisterRef ring;
};

} // namespace

// The tail iteration of PARTIAL_STEPS = 1: the wave that holds the tail bytes
// runs one more iteration, with the bytes at or past the end of the split
// zeroed in both operands by the shared three-arm tail (emitTailChunks), on
// ring slot 0, which the K loop left free, or with lds on the tail slot, whose
// loads the prologue issued and the exit of the loop waited for. The scales
// load into the slot without steering, and their bytes past the end are
// overridden.
static void emitTail(const MfmaContext &ctx, const MfmaTile &tile,
                     const MfmaLoop &loop, RegisterRef ring) {
  IRBuilder &builder = ctx.builder;
  int32_t slot = ctx.layout.tailSlot;
  emitTailGuard(builder, loop.tailBytes->ref(), [&] {
    if (ctx.layout.lds)
      readStagedTail(ctx, ring);
    else
      emitSlotLoads(builder,
                    scaleLoads(ctx, tile, ring, slot, loop.tailOffset->ref()));
    MfmaTailPath path(ctx, tile, loop, ring);
    emitTailChunks(builder, path);
    overrideTailScales(ctx, tile, loop, ring);
    shiftScales(ctx, tile, ring, slot);
    mfmaStep(ctx, tile, ring, slot, 0);
    mfmaStep(ctx, tile, ring, slot, 1);
  });
}

namespace {

// The MFMA half of the ring schedule over the iterations of one wave: the
// wave's iteration counter and the K offset of the next load advance in program
// order, so the path ignores the iteration of a step.
class MfmaRingPath final : public RingPath {
public:
  MfmaRingPath(const MfmaContext &ctx, const MfmaTile &tile,
               const MfmaLoop &loop, RegisterRef ring)
      : ctx(ctx), tile(tile), loop(loop), ring(ring) {}

  int32_t slots() const override { return ctx.layout.slots; }
  RegisterRef iterations() const override { return loop.waveIterations.ref(); }
  void fill(const RingStep &step) override {
    issueSlot(ctx, tile, loop, ring, step.slot);
  }
  // The tail loads are not part of RingLoads: the steady loop body is shared by
  // the first round, where they are in flight and newer than the prologue
  // slots, and by the later rounds, where they have long completed, so no one
  // count is exact for both. The counts ignore them, which can only make the
  // waits of the first round longer, never unsafe.
  void afterPrologue() override {
    if (ctx.layout.tailSlot != 0)
      issueTail(ctx, tile, loop, ring);
  }
  // With lds the first step of slot 0 is read before the loop or, when the
  // loop does not run, before the first arm; every step then reads the first
  // step of the next slot.
  void beginLoop() override {
    if (ctx.layout.lds)
      readFirstStep(ctx, ring, 0, ctx.layout.slots - 1, /*refill=*/true);
  }
  // The hooks exist for the VALU path, which hoists registers around the loop
  // and its arms.
  void endLoop() override {}
  void endArm() override {}
  void consume(const RingStep &step) override {
    if (ctx.layout.lds)
      consumeStagedSlot(ctx, tile, loop, ring, step);
    else
      consumeSlot(ctx, tile, loop, ring, step);
  }
  // The prologue slots past the wave's last iteration are never consumed, so
  // their loads must land before the ring is reused.
  void drain() override {
    waitForVectorLoads(ctx.builder, 0);
    if (ctx.layout.lds)
      waitForLds(ctx.builder, 0);
  }

private:
  const MfmaContext &ctx;
  const MfmaTile &tile;
  const MfmaLoop &loop;
  RegisterRef ring;
};

} // namespace

// Runs the wave's iterations through the ring schedule, then the tail iteration
// of a partial K.
static void emitKLoop(const MfmaContext &ctx, const MfmaTile &tile,
                      RegisterRef kz) {
  IRBuilder &builder = ctx.builder;
  MfmaLoop loop(builder, ctx.dims);
  startLoop(ctx, loop, kz);
  RegisterRange ringStorage = builder.allocateVgpr(ctx.layout.ringRegs(), 2);
  MfmaRingPath path(ctx, tile, loop, ringStorage.ref());
  emitRingLoop(builder, path);
  if (ctx.dims.partialSteps)
    emitTail(ctx, tile, loop, ringStorage.ref());
}

// The K reduction.

// Waits for the results of the last MFMAs before a VALU, DS or VMEM instruction
// reads the accumulators. Two s_nop 15 are a conservative guess, not measured.
static void waitForMfma(IRBuilder &builder) {
  builder.create<rocisa::SNop>(maxNop);
  builder.create<rocisa::SNop>(maxNop);
}

// Waves other than 0 store their accumulators to their slice of LDS and leave;
// wave 0 adds the slices of every other wave in increasing wave order. Both
// paths execute exactly one s_barrier.
static void reduceWaves(const MfmaContext &ctx, const MfmaTile &tile,
                        const std::shared_ptr<rocisa::Label> &tileEnd) {
  IRBuilder &builder = ctx.builder;
  const MfmaLayout &layout = ctx.layout;
  int32_t tiles = layout.tiles();
  RegisterRef address = ctx.lane.ldsAddress->ref();
  std::shared_ptr<rocisa::Label> leader = builder.createLabel("wave_zero");
  builder.create<rocisa::SCmpEQU32>(ctx.waveId.container(), 0);
  builder.create<rocisa::SCBranchSCC1>(leader->getLabelName());
  for (int32_t tileIndex = 0; tileIndex < tiles; ++tileIndex)
    builder.storeDs(address, tile.acc.ref(accRegs * tileIndex, accRegs),
                    MemoryWidth::b128, tileIndex * ldsTileBytes);
  builder.createWaitCnt(WaitCounts{.lds = 0});
  builder.create<rocisa::SBarrier>();
  builder.create<rocisa::SBranch>(tileEnd->getLabelName());
  builder.insert(leader);
  builder.create<rocisa::SBarrier>();
  RegisterRange partial = builder.allocateVgpr(accRegs * tiles, 2);
  for (int32_t wave = 1; wave < ctx.dims.waves; ++wave) {
    for (int32_t tileIndex = 0; tileIndex < tiles; ++tileIndex)
      builder.loadDs(partial.ref(accRegs * tileIndex, accRegs), address,
                     MemoryWidth::b128,
                     ((wave - 1) * tiles + tileIndex) * ldsTileBytes);
    builder.createWaitCnt(WaitCounts{.lds = 0});
    for (int32_t r = 0; r < accRegs * tiles; ++r)
      builder.create<rocisa::VAddF32>(tile.acc.ref(r).container(),
                                      tile.acc.ref(r).container(),
                                      partial.ref(r).container());
  }
}

// Output elements.

// The MFMA path's view of the output after the K reduction, on wave 0: the
// element (i, j) of the lane is the skinny column s0 + 16 i + lane % 16 and the
// wide row w0 + 16 (j / 4) + 4 (lane / 16) + j % 4, in register
// ((j / 4) * Ts + i) * 4 + j % 4, which is the accumulator (t * Ts + u) * 4 + e
// of the tile (t, u). The 4 rows a lane holds are consecutive wide elements, so
// they form a run when the wide dimension is contiguous; in the other
// orientation the lanes of 16 consecutive skinny columns already write
// contiguous elements and a per-lane run would need a cross-lane transpose.
namespace {

class MfmaOutputMap final : public OutputMap {
public:
  MfmaOutputMap(const MfmaContext &ctx, const MfmaTile &tile,
                bool skinnyContiguous, RegisterRef values)
      : ctx(ctx), tile(tile), off(ctx.builder.allocateVgpr(1)) {
    this->skinnyContiguous = skinnyContiguous;
    this->values = values;
    skinnyCount = ctx.layout.skinnyTiles;
    wideCount = 4 * ctx.layout.wideTiles;
    runLength = skinnyContiguous ? 1 : 4;
    segmentLength = accRegs * ctx.layout.tiles();
    segmentStride = segmentLength;
  }

  int32_t valueIndex(int32_t i, int32_t j) const override {
    return ((j / 4) * skinnyCount + i) * 4 + j % 4;
  }

  // Every lane of wave 0 stores.
  std::optional<RegisterRef> storeLanes() const override {
    return std::nullopt;
  }

  // The offsets of the lane's skinny column and wide rows, with the workspace
  // slice folded into the skinny part.
  void prepare(const StoreTarget &target) override {
    IRBuilder &builder = ctx.builder;
    offsets.reset();
    offsets.emplace(builder, ctx.layout);
    for (int32_t u = 0; u < ctx.layout.skinnyTiles; ++u) {
      RegisterRange start = builder.allocateSgpr(1);
      RegisterRange column = builder.allocateVgpr(1);
      builder.addi32(start.ref(), tile.s0.ref(), 16 * u);
      builder.addi32(column.ref(), ctx.lane.column.ref(), start.ref());
      builder.cmpi32(Predicate::ult, offsets->skinnyValid.ref(2 * u, 2),
                     column.ref(), ctx.skinnyCols);
      builder.muli32(offsets->skinnyOffset.ref(u), column.ref(),
                     target.strideS);
      if (target.sliceBase)
        builder.addi32(offsets->skinnyOffset.ref(u),
                       offsets->skinnyOffset.ref(u), *target.sliceBase);
    }
    for (int32_t t = 0; t < ctx.layout.wideTiles; ++t) {
      RegisterRange start = builder.allocateSgpr(1);
      builder.addi32(start.ref(), tile.w0.ref(), 16 * t);
      builder.addi32(offsets->wideColumn.ref(t), ctx.lane.rowBase.ref(),
                     start.ref());
      builder.muli32(offsets->wideOffset.ref(t), offsets->wideColumn.ref(t),
                     target.strideW);
    }
    for (int32_t e = 1; e < 4; ++e)
      builder.muli32(offsets->elementStride.ref(e - 1), target.strideW, e);
    for (int32_t e = 0; e < 4; ++e)
      builder.subi32(offsets->wideLimit.ref(e), ctx.wideCols, e);
  }

  RegisterRef offset(int32_t i, int32_t j) override {
    int32_t e = j % 4;
    RegisterRef result = off.ref();
    ctx.builder.addi32(result, offsets->skinnyOffset.ref(i),
                       offsets->wideOffset.ref(j / 4));
    if (e != 0)
      ctx.builder.addi32(result, result, offsets->elementStride.ref(e - 1));
    return result;
  }

  // The wide row is valid when it is below wideCols, compared signed against
  // wideCols - e, which can be negative.
  void accessLanes(RegisterRef mask, int32_t i, int32_t j) override {
    ctx.builder.cmpi32(Predicate::slt, mask, offsets->wideColumn.ref(j / 4),
                       offsets->wideLimit.ref(j % 4));
    ctx.builder.andi64(mask, mask, offsets->skinnyValid.ref(2 * i, 2));
  }

private:
  const MfmaContext &ctx;
  const MfmaTile &tile;
  RegisterRange off;
  std::optional<OutputOffsets> offsets;
};

} // namespace

// The tile.

// Prefetches the C elements of the tile for the epilogue's beta * C, on wave 0
// only and only when the tile stores D directly (k <= spk) and beta is not
// zero, as the epilogue's tests. The loads precede every ring load, so any
// wait of the K loop covers them first. C may alias D.
static void emitCPrefetch(const MfmaContext &ctx, const MfmaTile &tile,
                          const TileIds &ids) {
  IRBuilder &builder = ctx.builder;
  std::shared_ptr<rocisa::Label> skip = builder.createLabel("c_prefetch_skip");
  builder.create<rocisa::SCmpEQU32>(ctx.waveId.container(), 0);
  builder.create<rocisa::SCBranchSCC0>(skip->getLabelName());
  branchUnlessCIsUsed(ctx.output, skip);
  forEachOrientation(
      ctx.output, /*vectorStores=*/true, [&](bool skinnyContiguous) {
        MfmaOutputMap map(ctx, tile, skinnyContiguous, tile.cData->ref());
        loadCTile(ctx.output, map, ids);
      });
  builder.insert(skip);
}

// One tile up to and including its epilogue.
static void emitMfmaTile(const MfmaContext &ctx, const TileIds &ids) {
  IRBuilder &builder = ctx.builder;
  const SkinnyDims &dims = ctx.dims;
  MfmaTile tile(builder, ctx.layout, dims.useBeta);
  builder.muli32(tile.s0.ref(), ids.ty, dims.bm);
  builder.muli32(tile.w0.ref(), ids.tx, dims.tileWide);
  computeTileBases(ctx, tile);
  for (int32_t r = 0; r < accRegs * ctx.layout.tiles(); ++r)
    builder.movb32(tile.acc.ref(r), 0);
  builder.create<rocisa::SNop>(valuToMfmaNop);
  if (tile.cData)
    emitCPrefetch(ctx, tile, ids);
  emitKLoop(ctx, tile, ids.kz);
  waitForMfma(builder);
  std::shared_ptr<rocisa::Label> tileEnd = builder.createLabel("tile_end");
  if (dims.waves > 1)
    reduceWaves(ctx, tile, tileEnd);
  forEachOrientation(
      ctx.output, /*vectorStores=*/true, [&](bool skinnyContiguous) {
        MfmaOutputMap map(ctx, tile, skinnyContiguous, tile.acc.ref());
        emitOutput(ctx.output, map, ids,
                   tile.cData ? std::optional<RegisterRef>(tile.cData->ref())
                              : std::nullopt);
      });
  builder.insert(tileEnd);
}

// The persistent launch: the values of the tile loop that do not depend on the
// tile, as the VALU path computes them without the K rotation.
static TileLoop makeTileLoop(const MfmaContext &ctx, RegisterRef splits) {
  IRBuilder &builder = ctx.builder;
  const SkinnyDims &dims = ctx.dims;
  TileLoop loop{builder.allocateSgpr(1),
                builder.allocateSgpr(1),
                builder.allocateSgpr(1),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt};
  RegisterRange tilesY = builder.allocateSgpr(1);
  ceilDivByLiteral(builder, loop.tilesX.ref(), ctx.wideCols, dims.tileWide);
  ceilDivByLiteral(builder, tilesY.ref(), ctx.skinnyCols, dims.bm);
  builder.muli32(loop.plane.ref(), loop.tilesX.ref(), tilesY.ref());
  builder.muli32(loop.numTiles.ref(), loop.plane.ref(), splits);
  if (dims.numXcds != 1) {
    // perXcd = ceilDiv(numTiles, X) and tall = (numTiles - 1) mod X + 1, the
    // XCDs that hold perXcd tiles, the others hold one less.
    loop.perXcd = builder.allocateSgpr(1);
    ceilDivByLiteral(builder, loop.perXcd->ref(), loop.numTiles.ref(),
                     dims.numXcds);
    loop.tall = builder.allocateSgpr(1);
    RegisterRange last = builder.allocateSgpr(1);
    builder.addi32(last.ref(), loop.numTiles.ref(), -1);
    moduloByLiteral(builder, loop.tall->ref(), last.ref(), dims.numXcds);
    builder.addi32(loop.tall->ref(), loop.tall->ref(), 1);
  }
  if (dims.group != 1) {
    loop.tilesY = std::move(tilesY);
    loop.bandSize = builder.allocateSgpr(1);
    builder.muli32(loop.bandSize->ref(), loop.tilesX.ref(), dims.group);
  }
  return loop;
}

// Block b runs the tiles b, b + B, ... of the flattened tile space; the LDS
// area of the K reduction is free again at the top of every tile once all waves
// have read it.
static void emitTileLoop(const MfmaContext &ctx, TileLoop loop,
                         RegisterRef blockId, RegisterRef blocksX) {
  IRBuilder &builder = ctx.builder;
  std::shared_ptr<rocisa::Label> body = builder.createLabel("tile_loop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("tile_loop_end");
  builder.create<rocisa::SCmpLtI32>(blockId.container(),
                                    loop.numTiles.container());
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  if (ctx.dims.waves > 1) {
    builder.createWaitCnt(WaitCounts{.lds = 0});
    builder.create<rocisa::SBarrier>();
  }
  {
    RegisterRange tx = builder.allocateSgpr(1);
    RegisterRange ty = builder.allocateSgpr(1);
    RegisterRange kz = builder.allocateSgpr(1);
    TileIds ids{tx.ref(), ty.ref(), kz.ref()};
    decodeTile(builder, ctx.dims, blockId, ids, loop);
    emitMfmaTile(ctx, ids);
  }
  builder.addi32(blockId, blockId, blocksX);
  builder.create<rocisa::SCmpLtI32>(blockId.container(),
                                    loop.numTiles.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

// The kernel.

void SkinnyGEMMEmitter::emitMfma() {
  SetupScalars scalars(builder, dims.grid3d);
  emitArguments(scalars);
  // The descriptors lose their size for the loads of the tail slot that the
  // waves without tail bytes issue, so the sizes wait in SGPRs.
  std::optional<RegisterRange> savedSizes;
  if (dims.lds && dims.partialSteps) {
    savedSizes = builder.allocateSgpr(4);
    int32_t index = 0;
    for (RegisterRange *rsrc : {&sRsrc, &wRsrc, &sSclRsrc, &wSclRsrc})
      builder.movb32(savedSizes->ref(index++), rsrc->ref(2));
  }
  RegisterRef tid = kernel.workitemId().ref();
  RegisterRange waveId = builder.allocateSgpr(1);
  computeWaveId(builder, dims, waveId.ref(), tid);
  MfmaLayout layout(dims);
  MfmaLane lane = makeLane(builder, dims, tid, waveId.ref());
  MfmaContext ctx{.builder = builder,
                  .dims = dims,
                  .layout = layout,
                  .lane = lane,
                  .sRsrc = sRsrc.ref(),
                  .wRsrc = wRsrc.ref(),
                  .sSclRsrc = sSclRsrc.ref(),
                  .wSclRsrc = wSclRsrc.ref(),
                  .savedSizes =
                      savedSizes ? std::optional<RegisterRef>(savedSizes->ref())
                                 : std::nullopt,
                  .skinnyCols = skinnyCols.ref(),
                  .wideCols = wideCols.ref(),
                  .rowGroup = scalars.rowGroup.ref(),
                  .ldS = scalars.ldS.ref(),
                  .ldW = scalars.ldW.ref(),
                  .k = k.ref(),
                  .spk = spk.ref(),
                  .waveId = waveId.ref(),
                  .output = outputContext()};
  if (dims.grid3d) {
    TileIds ids{kernel.workgroupIdX().ref(), kernel.workgroupIdY().ref(),
                kernel.workgroupIdZ().ref()};
    if (dims.xcdRun > 1)
      groupTilesPerXcd(builder, dims, kernel.target().xcdCount(), ids.tx,
                       scalars.blocksX.ref());
    emitMfmaTile(ctx, ids);
    builder.createEndProgram();
    return;
  }
  TileLoop loop = makeTileLoop(ctx, scalars.splits->ref());
  // The split count only sizes the workspace and the tile space.
  scalars.splits->release();
  emitTileLoop(ctx, std::move(loop), kernel.workgroupIdX().ref(),
               scalars.blocksX.ref());
  builder.createEndProgram();
}
