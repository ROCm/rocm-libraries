// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The VALU compute path of the skinny MXFP8 GEMM emitter, for both launch forms
// (GRID_3D = 1 and the persistent GRID_3D = 0), any PARTIAL_STEPS and ring
// scales (UP_ITERS = 0) or up-front scales. emitValu drives emitSetup and
// emitTile, which runs emitCPrefetch, emitUpFrontScales, emitKLoop (the VALU
// half of the ring schedule that emitRingLoop drives, with emitCompute),
// emitTail (the VALU half of the tail iteration of SkinnyGEMMTail.cpp), the
// accumulator reduction and emitEpilogue.

#include <tensilelite/API/SkinnyGEMM.hpp>
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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::skinny;

static bool isScalar(RegisterRef ref) { return ref.kind == RegisterKind::sgpr; }

namespace {

// Layout of the reduced accumulators: the natural p order, or runs of
// consecutive output elements, each starting at an even register, for the
// vector stores of one orientation.
struct RedLayout {
  bool runs;
  bool skinnyContiguous;
  int32_t stride;
  int32_t total;
};

} // namespace

// An orientation stores runs of consecutive elements: the BM skinny elements
// of a wide column when the skinny dimension is contiguous in the output, the
// TN wide elements of a skinny column otherwise.
static int32_t outputRunLength(const SkinnyDims &dims, bool skinnyContiguous) {
  return skinnyContiguous ? dims.bm : dims.tn;
}

// A chunk must start dword aligned: an f32 run of at least 2 elements or an
// even bf16 run.
static bool hasVector(const SkinnyDims &dims, bool skinnyContiguous,
                      bool bf16) {
  int32_t length = outputRunLength(dims, skinnyContiguous);
  return bf16 ? length % 2 == 0 : length >= 2;
}

static RedLayout redLayout(const SkinnyDims &dims, bool skinnyContiguous) {
  int32_t length = outputRunLength(dims, skinnyContiguous);
  if (length < 2)
    return RedLayout{false, skinnyContiguous, 0, dims.accumulators};
  // Every run starts at an even register, as the x2 and x4 stores need.
  int32_t stride = (length + 1) & ~1;
  int32_t runs = skinnyContiguous ? dims.tn : dims.bm;
  return RedLayout{true, skinnyContiguous, stride, runs * stride};
}

static int32_t redIndex(const SkinnyDims &dims, RedLayout layout, int32_t r,
                        int32_t t) {
  if (!layout.runs)
    return r * dims.tn + t;
  return layout.skinnyContiguous ? t * layout.stride + r
                                 : r * layout.stride + t;
}

// The VGPRs the C prefetch needs: the larger of the two orientations' layouts.
static int32_t cDwords(const SkinnyDims &dims) {
  return std::max(redLayout(dims, true).total, redLayout(dims, false).total);
}

// Setup.

static constexpr int32_t selectorMaskStep = 0x01010101;

// The scale base of a column of a row group matrix: (col / 32) * rowGroup +
// (col mod 16) * 4, before the lane's role offset. Overwrites col.
static void scaleRowOffset(IRBuilder &builder, RegisterRef dst, RegisterRef col,
                           RegisterRef rowGroup) {
  builder.shrui32(dst, col, 5);
  builder.muli32(dst, dst, rowGroup);
  builder.andi32(col, 15, col);
  builder.shli32(col, col, 2);
  builder.addi32(dst, dst, col);
}

Ring::Ring(IRBuilder &builder, const SkinnyDims &dims)
    : data(builder.allocateVgpr(
          dims.slots * SkinnyDims::chunks * dims.columns * 4, 2)),
      raw(dims.upIters == 0 ? std::optional<RegisterRange>(builder.allocateVgpr(
                                  dims.slots * dims.columns))
                            : std::nullopt) {}

TileState::TileState(IRBuilder &builder, const SkinnyDims &dims)
    : w0(dims.uniformWide ? builder.allocateSgpr(1) : builder.allocateVgpr(1)),
      acc(builder.allocateVgpr(dims.accumulators)),
      dBase(builder.allocateVgpr(dims.columns)),
      sBase(builder.allocateVgpr(dims.columns)),
      selectors(builder.allocateVgpr((16 % dims.bm == 0 ? 1 : dims.bm) +
                                     (16 % dims.tn == 0 ? 1 : dims.tn))),
      s0(builder.allocateSgpr(1)), nIterFull(builder.allocateSgpr(1)),
      fitRot(dims.upIters != 1
                 ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                 : std::nullopt),
      tailBytes(dims.partialSteps != 0
                    ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                    : std::nullopt),
      chunkBase{builder.allocateSgpr(4), std::vector<bool>(4, false)},
      sValid(builder.allocateSgpr(2 * dims.bm, 2)),
      wValid(builder.allocateSgpr(2 * dims.tn, 2)) {}

void TileState::allocateOptionalVectors(IRBuilder &builder,
                                        const SkinnyDims &dims,
                                        int32_t cDwords) {
  if (dims.partialSteps != 0)
    scaleGroup = builder.allocateVgpr(1);
  if (dims.useBeta)
    cData = builder.allocateVgpr(cDwords, 2);
}

ScaleGather::ScaleGather(IRBuilder &builder, const SkinnyDims &dims)
    : addresses(builder.allocateVgpr(dims.scalesShared ? 1 : 2)),
      ldsStage(dims.scalesShared
                   ? std::optional<RegisterRange>(builder.allocateVgpr(1))
                   : std::nullopt) {}

// rot = (key + key / 16) mod NI, or mod the literal UP_ITERS without ni. Shared
// by the 3-D rotation, the loop invariant and the slab rotation of the
// persistent launch.
static void blockRotation(IRBuilder &builder, const SkinnyDims &dims,
                          RegisterRef rot, RegisterRef key,
                          std::optional<RegisterRef> ni) {
  RegisterRange sum = builder.allocateSgpr(1);
  builder.shrui32(sum.ref(), key, 4);
  builder.addi32(sum.ref(), sum.ref(), key);
  if (ni) {
    RegisterRange quotient = builder.allocateSgpr(1);
    builder.divsi32(quotient.ref(), sum.ref(), *ni, /*useIEEE=*/true);
    builder.muli32(quotient.ref(), quotient.ref(), *ni);
    builder.subi32(rot, sum.ref(), quotient.ref());
    return;
  }
  if (isPowerOfTwo(dims.upIters)) {
    builder.andi32(rot, sum.ref(), dims.upIters - 1);
    return;
  }
  RegisterRange quotient = builder.allocateSgpr(1);
  builder.divsi32(quotient.ref(), sum.ref(), dims.upIters, /*useIEEE=*/true);
  builder.muli32(quotient.ref(), quotient.ref(), dims.upIters);
  builder.subi32(rot, sum.ref(), quotient.ref());
}

// Gather addresses of the scale permutes: the ds_bpermute byte addresses of the
// lanes that hold the scale dwords of this lane, or the LDS read address with
// shared scales. A function as its derivation is independent of the other lane
// values.
static void computeAddresses(IRBuilder &builder, const SkinnyDims &dims,
                             RegisterRef addresses, RegisterRef klv,
                             RegisterRef lane, RegisterRef tid) {
  if (dims.scalesShared) {
    // Shared scales: the LDS byte offset 4 * (j * R + wave) of the lowest dword
    // of this lane, with the slot j(kl, 0) = (kl / 8) * 8 + (kl mod 8) / 2 of
    // LANES_K = 64. The other dword is a constant offset away, so one address
    // suffices.
    RegisterRange term = builder.allocateVgpr(1);
    builder.andi32(addresses, dims.subgroupLanes - 1, klv);
    builder.shrui32(addresses, addresses, 1);
    builder.muli32(addresses, addresses, 4 * dims.roleSlots);
    builder.shrui32(term.ref(), klv, 3);
    builder.muli32(term.ref(), term.ref(), 32 * dims.roleSlots);
    builder.addi32(addresses, addresses, term.ref());
    builder.shrui32(term.ref(), tid, 6);
    builder.shli32(term.ref(), term.ref(), 2);
    builder.addi32(addresses, addresses, term.ref());
    return;
  }
  // The byte address of the lanes that hold scale dwords 0 and 1 of this lane:
  // 4 * (j * CG + cg) with the slot j(kl, d).
  RegisterRef a0 = addresses.slice(0);
  RegisterRef a1 = addresses.slice(1);
  RegisterRange term = builder.allocateVgpr(1);
  // The block pair of the lane in its subgroup, times CG.
  builder.andi32(a0, dims.subgroupLanes - 1, klv);
  builder.shrui32(a0, a0, 1);
  builder.shli32(a0, a0, dims.log2ColumnGroups);
  if (dims.lanesK > 8) {
    builder.shrui32(term.ref(), klv, 3);
    builder.shli32(term.ref(), term.ref(),
                   dims.log2Subgroup + dims.log2ColumnGroups);
    builder.addi32(a0, a0, term.ref());
  }
  if (dims.lanesK < 64) {
    builder.shrui32(term.ref(), lane, dims.log2LanesK);
    builder.addi32(a0, a0, term.ref());
  }
  builder.shli32(a0, a0, 2);
  builder.addi32(a1, a0, dims.halfSubgroup * dims.columnGroups * 4);
}

// Subgroup j of S lanes owns the slab j of 64 * S bytes, and the lane reads 16
// bytes at its place in the subgroup. A function as the tail step recomputes it
// after the lane values died.
static RegisterRange computeLaneBytes(IRBuilder &builder,
                                      const SkinnyDims &dims, RegisterRef klv) {
  RegisterRange laneBytes = builder.allocateVgpr(1);
  if (dims.lanesK == dims.subgroupLanes) {
    builder.shli32(laneBytes.ref(), klv, 4);
    return laneBytes;
  }
  RegisterRange inSubgroup = builder.allocateVgpr(1);
  builder.shrui32(laneBytes.ref(), klv, dims.log2Subgroup);
  builder.shli32(laneBytes.ref(), laneBytes.ref(), dims.log2Subgroup + 6);
  builder.andi32(inSubgroup.ref(), dims.subgroupLanes - 1, klv);
  builder.shli32(inSubgroup.ref(), inSubgroup.ref(), 4);
  builder.addi32(laneBytes.ref(), laneBytes.ref(), inSubgroup.ref());
  return laneBytes;
}

// The byte offset in an iteration of the scale slot roleJ that the lane loads:
// the slab roleJ / S of 64 * S bytes, the block pair roleJ mod (S / 2) of 64
// bytes and the dword step (roleJ mod S) / (S / 2) of 32 * S bytes. roleJ is
// the thread's in shared mode and the lane's cluster otherwise. A function as
// its derivation is independent of the other lane values.
static RegisterRange
computeScaleOffset(IRBuilder &builder, const SkinnyDims &dims, RegisterRef lane,
                   const std::optional<RegisterRange> &sharedRoleJ) {
  std::optional<RegisterRange> roleStorage;
  RegisterRef roleJ = lane;
  if (sharedRoleJ)
    roleJ = sharedRoleJ->ref();
  else if (dims.columnGroups > 1) {
    roleStorage = builder.allocateVgpr(1);
    builder.shrui32(roleStorage->ref(), lane, dims.log2ColumnGroups);
    roleJ = roleStorage->ref();
  }
  RegisterRange sclOff = builder.allocateVgpr(1);
  RegisterRange term = builder.allocateVgpr(1);
  builder.andi32(sclOff.ref(), dims.halfSubgroup - 1, roleJ);
  builder.shli32(sclOff.ref(), sclOff.ref(), 6);
  if (dims.lanesK > dims.subgroupLanes) {
    builder.shrui32(term.ref(), roleJ, dims.log2Subgroup);
    builder.shli32(term.ref(), term.ref(), dims.log2Subgroup + 6);
    builder.addi32(sclOff.ref(), sclOff.ref(), term.ref());
  }
  builder.andi32(term.ref(), dims.subgroupLanes - 1, roleJ);
  builder.shrui32(term.ref(), term.ref(), dims.log2Half);
  builder.shli32(term.ref(), term.ref(), dims.log2Subgroup + 5);
  builder.addi32(sclOff.ref(), sclOff.ref(), term.ref());
  return sclOff;
}

// slot = (wave * CG + lane mod CG) * TN, the lane part of w0Load = tx * BN +
// slot: the wide column that the lane loads scales for, which differs from the
// column it owns. In shared mode the slot is the thread's row slot tid mod R. A
// function as its derivation is independent of the other lane values.
static void
computeWideLoadSlot(IRBuilder &builder, const SkinnyDims &dims,
                    RegisterRef slot, RegisterRef tid, RegisterRef lane,
                    const std::optional<RegisterRange> &sharedRoleJ) {
  if (sharedRoleJ) {
    // slot = tid mod R given roleJ = tid / R.
    remainderByLiteral(builder, slot, tid, sharedRoleJ->ref(), dims.roleSlots);
    builder.muli32(slot, slot, dims.tn);
    return;
  }
  if (dims.waves > 1) {
    builder.shrui32(slot, tid, 6);
    builder.shli32(slot, slot, dims.log2ColumnGroups);
  }
  if (dims.columnGroups > 1) {
    RegisterRange inDomain = builder.allocateVgpr(1);
    builder.andi32(inDomain.ref(), dims.columnGroups - 1, lane);
    if (dims.waves > 1)
      builder.addi32(slot, slot, inDomain.ref());
    else
      builder.movb32(slot, inDomain.ref());
  }
  builder.muli32(slot, slot, dims.tn);
}

// The selector of the column group of a row: all rows of a 16-row half share
// one when BM or TN divides 16, since s0 and w0 are multiples of them. A
// function as the scale permutes of the K loop read the same registers.
static RegisterRef selector(const SkinnyDims &dims, RegisterRef selectors,
                            int32_t column) {
  bool skinny = column < dims.bm;
  bool skinnyShared = 16 % dims.bm == 0;
  bool wideShared = 16 % dims.tn == 0;
  int32_t skinnyCount = skinnyShared ? 1 : dims.bm;
  if (skinny)
    return selectors.slice(skinnyShared ? 0 : column);
  return selectors.slice(skinnyCount + (wideShared ? 0 : column - dims.bm));
}

// dst = base + ((col >> 4) & 1) * 0x01010101: the byte selector of v_perm_b32
// that packs the two scale dwords of a column whose row is in the lower or
// upper 16 rows of its 32-row group. A function for the name of the formula.
static void selectorValue(IRBuilder &builder, const SkinnyDims &dims,
                          RegisterRef dst, RegisterRef col) {
  int32_t selectorBase = dims.subgroupLanes == 8 ? 0x06040200 : 0x06020400;
  // A scalar column is computed in an SGPR and then copied to dst.
  std::optional<RegisterRange> value;
  RegisterRef work = dst;
  if (isScalar(col)) {
    value = builder.allocateSgpr(1);
    work = value->ref();
  }
  builder.shrui32(work, col, 4);
  builder.andi32(work, 1, work);
  builder.muli32(work, work, selectorMaskStep);
  builder.addi32(work, work, selectorBase);
  if (value)
    builder.movb32(dst, work);
}

// The 3-D launch allocates the ring and the tile's registers first, as the
// setup writes into them, and the setup values die when the tile's bases are
// done. The persistent launch (GRID_3D = 0) keeps the setup values for
// the tile loop and allocates the registers of the tile in every iteration:
// block b runs the tiles b, b + B, ... of the flattened tile space, in the
// order of the XCD remap and the grouped decode, every one from zeroed
// accumulators. As elsewhere, every operand is non-negative, so each quotient
// is one division and each remainder is a - q * d.
void SkinnyGEMMEmitter::emitValu() {
  if (dims.grid3d) {
    TileIds ids{kernel.workgroupIdX().ref(), kernel.workgroupIdY().ref(),
                kernel.workgroupIdZ().ref()};
    Ring ring(builder, dims);
    TileState tile(builder, dims);
    ScaleGather gather(builder, dims);
    tile.allocateOptionalVectors(builder, dims, cDwords(dims));
    SetupScalars scalars(builder, /*grid3d=*/true);
    LaneValues lane = emitSetup(scalars, gather, tile.w0.ref());
    emitTile(ids, scalars, /*loop=*/nullptr, lane, gather, std::move(ring),
             tile);
    builder.createEndProgram();
    return;
  }
  SetupScalars scalars(builder, /*grid3d=*/false);
  ScaleGather gather(builder, dims);
  LaneValues lane = emitSetup(scalars, gather, /*w0Target=*/std::nullopt);
  // The values that every tile shares. The rotation keys on the block id in the
  // plain order, so it is loop invariant, and on the wide slab of the tile when
  // reordering, which needs NI at every tile.
  TileLoop loop{builder.allocateSgpr(1),
                builder.allocateSgpr(1),
                builder.allocateSgpr(1),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt};
  {
    RegisterRange tilesY = builder.allocateSgpr(1);
    ceilDivByLiteral(builder, loop.tilesX.ref(), wideCols.ref(), dims.tileWide);
    ceilDivByLiteral(builder, tilesY.ref(), skinnyCols.ref(), dims.bm);
    builder.muli32(loop.plane.ref(), loop.tilesX.ref(), tilesY.ref());
    builder.muli32(loop.numTiles.ref(), loop.plane.ref(),
                   scalars.splits->ref());
    // The split count only sizes the workspace and the tile space, and the
    // loop holds the registers of every other setup scalar.
    scalars.splits->release();
    if (dims.reordered && dims.upIters != 1) {
      // The wide slab rotation of the reordered tiles divides by the dynamic
      // NI.
      loop.ni = builder.allocateSgpr(1);
      ceilDivByLiteral(builder, loop.ni->ref(), spk.ref(), dims.ki);
    } else if (!dims.reordered && dims.upIters != 1) {
      // The plain order only needs NI for the block's rotation.
      std::optional<RegisterRange> ni;
      if (dims.upIters == 0) {
        ni = builder.allocateSgpr(1);
        ceilDivByLiteral(builder, ni->ref(), spk.ref(), dims.ki);
      }
      loop.blockRot = builder.allocateSgpr(1);
      blockRotation(builder, dims, loop.blockRot->ref(),
                    kernel.workgroupIdX().ref(),
                    ni ? std::optional<RegisterRef>(ni->ref()) : std::nullopt);
    }
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
  }
  RegisterRef blockId = kernel.workgroupIdX().ref();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("tile_loop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("tile_loop_end");
  builder.create<rocisa::SCmpLtI32>(blockId.container(),
                                    loop.numTiles.container());
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  if (dims.scalesShared) {
    // The previous tile's scales must be read from LDS before this tile
    // overwrites them. gpu.barrier waits for every outstanding memory operation
    // first.
    builder.createWaitCnt(WaitCounts{.vectorLoads = 0, .lds = 0});
    builder.create<rocisa::SBarrier>();
  }
  {
    // Everything the tile writes is rewritten before it is read; only the lane
    // values and the loop invariants carry over.
    RegisterRange tx = builder.allocateSgpr(1);
    RegisterRange ty = builder.allocateSgpr(1);
    RegisterRange kz = builder.allocateSgpr(1);
    TileIds ids{tx.ref(), ty.ref(), kz.ref()};
    Ring ring(builder, dims);
    TileState tile(builder, dims);
    tile.allocateOptionalVectors(builder, dims, cDwords(dims));
    decodeTile(builder, dims, blockId, ids, loop);
    emitTile(ids, scalars, &loop, lane, gather, std::move(ring), tile);
  }
  builder.addi32(blockId, blockId, scalars.blocksX.ref());
  builder.create<rocisa::SCmpLtI32>(blockId.container(),
                                    loop.numTiles.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
  builder.createEndProgram();
}

// Loads the kernel arguments, derives the buffer sizes and makes the
// descriptors, selects the operands of the skinny and the wide dimension, and
// computes the per-lane values, which only depend on the thread id: the part
// of the wide columns without the tile's tx * BN (absent when the wide column
// is uniform), the byte offsets of the lane and the scale gather addresses. The
// 3-D launch passes the tile's w0 as w0Target to receive the lane part, which
// keeps the lane values one register smaller while they are all live.
SkinnyGEMMEmitter::LaneValues
SkinnyGEMMEmitter::emitSetup(SetupScalars &scalars, ScaleGather &gather,
                             std::optional<RegisterRef> w0Target) {
  emitArguments(scalars);

  // Lane values. The block is (64 * waves, 1, 1), so the packed y and z fields
  // of v0 are zero and no mask is needed.
  RegisterRef tid = kernel.workitemId().ref();
  std::optional<RegisterRange> laneStorage;
  RegisterRef lane = tid;
  if (dims.waves > 1) {
    laneStorage = builder.allocateVgpr(1);
    builder.andi32(laneStorage->ref(), 63, tid);
    lane = laneStorage->ref();
  }
  std::optional<RegisterRange> w0Lane;
  std::optional<RegisterRange> laneBytes;
  {
    // The lane's index in its cluster only serves the gather addresses and the
    // lane's bytes.
    std::optional<RegisterRange> klStorage;
    RegisterRef klv = lane;
    if (dims.lanesK < 64) {
      klStorage = builder.allocateVgpr(1);
      builder.andi32(klStorage->ref(), dims.lanesK - 1, tid);
      klv = klStorage->ref();
    }
    if (!dims.uniformWide) {
      // w0Lane = (wave * CG + cg) * TN, the lane part of w0 = tx * BN + (wave *
      // CG + cg) * TN. The slot wave * CG + cg is the thread id divided by
      // LANES_K.
      if (!w0Target)
        w0Lane = builder.allocateVgpr(1);
      RegisterRef target = w0Target ? *w0Target : w0Lane->ref();
      builder.shrui32(target, tid, dims.log2LanesK);
      builder.muli32(target, target, dims.tn);
    }
    computeAddresses(builder, dims, gather.addresses.ref(), klv, lane, tid);
    laneBytes = computeLaneBytes(builder, dims, klv);
  }
  std::optional<RegisterRange> roleJ;
  if (dims.scalesShared) {
    // The slot roleJ = id / R of the thread's scale load role, where R =
    // roleSlots is not a power of two for some block sizes, so the division is
    // the builder's division by a constant.
    roleJ = builder.allocateVgpr(1);
    divideByLiteral(builder, roleJ->ref(), tid, dims.roleSlots);
    builder.shli32(gather.ldsStage->ref(), tid, 2);
  }
  RegisterRange sclOff = computeScaleOffset(builder, dims, lane, roleJ);
  std::optional<RegisterRange> w0Load;
  if (!dims.uniformWide) {
    w0Load = builder.allocateVgpr(1);
    computeWideLoadSlot(builder, dims, w0Load->ref(), tid, lane, roleJ);
  }
  // v0 keeps kl for the epilogue; every use of the thread id is done.
  if (dims.waves > 1 || dims.lanesK < 64)
    builder.andi32(tid, dims.lanesK - 1, tid);
  return LaneValues{std::move(*laneBytes), std::move(sclOff), std::move(w0Load),
                    std::move(w0Lane)};
}

// The tile of the workgroup ids, or of the decoded block id, up to its
// epilogue: the K range, the bases, the selectors and the accumulators, then
// the K phase. The ring and the up-front scales die with the K phase, so the
// epilogue reuses their registers; the 3-D launch also frees the setup values
// there, the persistent launch keeps them for the next tile (loop is null in
// the 3-D launch).
void SkinnyGEMMEmitter::emitTile(const TileIds &ids, SetupScalars &scalars,
                                 const TileLoop *loop, LaneValues &lane,
                                 ScaleGather &gather, Ring ring,
                                 TileState &tile) {
  // chunkBase[0] is the split start, so it needs no register of its own.
  builder.muli32(tile.chunkBase.regs.ref(0), ids.kz, spk.ref());
  tile.chunkBase.ready[0] = true;
  // The extent of the split, as only the last split can be shorter than spk.
  builder.subi32(tile.nIterFull.ref(), k.ref(), tile.chunkBase.regs.ref(0));
  builder.minsi32(tile.nIterFull.ref(), tile.nIterFull.ref(), spk.ref());
  if (tile.tailBytes)
    builder.andi32(tile.tailBytes->ref(), dims.ki - 1, tile.nIterFull.ref());
  builder.shrsi32(tile.nIterFull.ref(), tile.nIterFull.ref(), dims.log2Ki);
  if (tile.fitRot) {
    std::optional<RegisterRange> rotStorage;
    if (!loop || !loop->blockRot) {
      // rot = (key + key / 16) mod NI with NI = ceilDiv(spk, KI), or the
      // literal UP_ITERS with up-front scales, which bounds the iterations of
      // every split. The ring-scale remainder uses the exact signed division
      // instead of a signed remainder expansion. The 3-D launch keys it on the
      // linear block id; the reordered persistent tiles key it on the wide slab
      // kz * tilesX + tx, so that the tiles that share a wide column read the
      // same K offset together.
      rotStorage = builder.allocateSgpr(1);
      RegisterRange key = builder.allocateSgpr(1);
      std::optional<RegisterRange> ni;
      std::optional<RegisterRef> niRef;
      if (loop) {
        builder.muli32(key.ref(), ids.kz, loop->tilesX.ref());
        builder.addi32(key.ref(), key.ref(), ids.tx);
        niRef = loop->ni->ref();
      } else {
        builder.muli32(key.ref(), scalars.blocksY->ref(), ids.kz);
        builder.addi32(key.ref(), key.ref(), ids.ty);
        builder.muli32(key.ref(), key.ref(), scalars.blocksX.ref());
        builder.addi32(key.ref(), key.ref(), ids.tx);
        if (dims.upIters == 0) {
          ni = builder.allocateSgpr(1);
          ceilDivByLiteral(builder, ni->ref(), spk.ref(), dims.ki);
          niRef = ni->ref();
        }
      }
      blockRotation(builder, dims, rotStorage->ref(), key.ref(), niRef);
    }
    RegisterRef rotation =
        rotStorage ? rotStorage->ref() : loop->blockRot->ref();
    // The rotation falls back to 0 when it is not below the iteration count,
    // so the physical iteration wraps once without a division.
    builder.create<rocisa::SCmpLtU32>(rotation.container(),
                                      tile.nIterFull.container());
    builder.selectb32(tile.fitRot->ref(), rotation, 0);
  }
  builder.muli32(tile.s0.ref(), ids.ty, dims.bm);
  if (!loop && dims.xcdRun > 1)
    groupTilesPerXcd(builder, dims, kernel.target().xcdCount(), ids.tx,
                     scalars.blocksX.ref());
  {
    // The tile adds its part to the lane part of the wide columns: w0 = w0Lane
    // + tx * BN, with w0 already holding the lane part in the 3-D launch, and
    // for the scale loads w0Load = lane.w0Load + tx * BN, in place in the 3-D
    // launch and absent when the wide column is uniform and w0 itself serves.
    std::optional<RegisterRange> w0LoadTile;
    RegisterRef w0Load = tile.w0.ref();
    if (dims.uniformWide)
      builder.muli32(tile.w0.ref(), ids.tx, dims.tileWide);
    else {
      RegisterRange tileBase = builder.allocateSgpr(1);
      builder.muli32(tileBase.ref(), ids.tx, dims.tileWide);
      RegisterRef w0Lane = lane.w0Lane ? lane.w0Lane->ref() : tile.w0.ref();
      builder.addi32(tile.w0.ref(), w0Lane, tileBase.ref());
      if (loop) {
        w0LoadTile = builder.allocateVgpr(1);
        builder.addi32(w0LoadTile->ref(), lane.w0Load->ref(), tileBase.ref());
        w0Load = w0LoadTile->ref();
      } else {
        builder.addi32(lane.w0Load->ref(), lane.w0Load->ref(), tileBase.ref());
        w0Load = lane.w0Load->ref();
      }
    }
    // dBase[r] = (s0 + r < skinnyCols) ? (s0 + r) * ldS + laneBytes : -1, and
    // likewise for the wide columns; the validity masks stay alive for the
    // stores.
    for (int32_t r = 0; r < dims.bm; ++r) {
      RegisterRange col = builder.allocateSgpr(1);
      builder.addi32(col.ref(), tile.s0.ref(), r);
      builder.cmpi32(Predicate::ult, tile.sValid.ref(2 * r, 2), col.ref(),
                     skinnyCols.ref());
      builder.muli32(col.ref(), col.ref(), scalars.ldS.ref());
      builder.addi32(tile.dBase.ref(r), col.ref(), lane.laneBytes.ref());
      builder.selectb32(tile.dBase.ref(r), tile.sValid.ref(2 * r, 2),
                        tile.dBase.ref(r), outOfRange);
    }
    for (int32_t t = 0; t < dims.tn; ++t) {
      RegisterRef base = tile.dBase.ref(dims.bm + t);
      RegisterRef valid = tile.wValid.ref(2 * t, 2);
      RegisterRange col =
          dims.uniformWide ? builder.allocateSgpr(1) : builder.allocateVgpr(1);
      builder.addi32(col.ref(), tile.w0.ref(), t);
      builder.cmpi32(Predicate::ult, valid, col.ref(), wideCols.ref());
      builder.muli32(col.ref(), col.ref(), scalars.ldW.ref());
      builder.addi32(base, col.ref(), lane.laneBytes.ref());
      builder.selectb32(base, valid, base, outOfRange);
    }
    // sBase[c] is the voffset of the scale dword that the lane loads for column
    // c: the load column of the wide operand differs from the lane's own
    // column.
    if (tile.scaleGroup)
      builder.andi32(tile.scaleGroup->ref(), -256, lane.sclOff.ref());
    for (int32_t r = 0; r < dims.bm; ++r) {
      RegisterRange col = builder.allocateSgpr(1);
      RegisterRange row = builder.allocateSgpr(1);
      builder.addi32(col.ref(), tile.s0.ref(), r);
      scaleRowOffset(builder, row.ref(), col.ref(), scalars.rowGroup.ref());
      builder.addi32(tile.sBase.ref(r), row.ref(), lane.sclOff.ref());
      builder.selectb32(tile.sBase.ref(r), tile.sValid.ref(2 * r, 2),
                        tile.sBase.ref(r), outOfRange);
    }
    for (int32_t t = 0; t < dims.tn; ++t) {
      RegisterRef base = tile.sBase.ref(dims.bm + t);
      RegisterRange valid = builder.allocateSgpr(2, 2);
      RegisterRange col =
          dims.uniformWide ? builder.allocateSgpr(1) : builder.allocateVgpr(1);
      RegisterRange row =
          dims.uniformWide ? builder.allocateSgpr(1) : builder.allocateVgpr(1);
      builder.addi32(col.ref(), w0Load, t);
      builder.cmpi32(Predicate::ult, valid.ref(), col.ref(), wideCols.ref());
      scaleRowOffset(builder, row.ref(), col.ref(), scalars.rowGroup.ref());
      builder.addi32(base, row.ref(), lane.sclOff.ref());
      builder.selectb32(base, valid.ref(), base, outOfRange);
    }
  }
  // Selectors.
  RegisterRef selectors = tile.selectors.ref();
  if (16 % dims.bm == 0)
    selectorValue(builder, dims, selector(dims, selectors, 0), tile.s0.ref());
  else {
    for (int32_t r = 0; r < dims.bm; ++r) {
      RegisterRange col = builder.allocateSgpr(1);
      builder.addi32(col.ref(), tile.s0.ref(), r);
      selectorValue(builder, dims, selector(dims, selectors, r), col.ref());
    }
  }
  if (16 % dims.tn == 0)
    selectorValue(builder, dims, selector(dims, selectors, dims.bm),
                  tile.w0.ref());
  else {
    for (int32_t t = 0; t < dims.tn; ++t) {
      RegisterRange col =
          dims.uniformWide ? builder.allocateSgpr(1) : builder.allocateVgpr(1);
      builder.addi32(col.ref(), tile.w0.ref(), t);
      selectorValue(builder, dims, selector(dims, selectors, dims.bm + t),
                    col.ref());
    }
  }
  for (int32_t p = 0; p < dims.accumulators; ++p)
    builder.movb32(tile.acc.ref(p), 0);
  if (!loop) {
    // The 3-D launch reads the lane values only up to here, and the K phase
    // needs their VGPRs.
    lane.sclOff.release();
    lane.laneBytes.release();
    lane.w0Load.reset();
  }

  // The K phase.
  if (tile.cData)
    emitCPrefetch(tile, ids);
  {
    Ring kRing = std::move(ring);
    UpFrontScales upFront;
    KLoopState state{builder,        dims,           sRsrc.ref(), wRsrc.ref(),
                     sSclRsrc.ref(), wSclRsrc.ref(), tile,        kRing,
                     gather,         upFront};
    if (dims.upIters > 0)
      emitUpFrontScales(state, UpFrontStep::issue);
    emitKLoop(state);
    // The steps run on ring slot 0 and read the up-front scales, so they
    // precede the release of both.
    if (dims.partialSteps == 1)
      emitTail(state);
  }
  emitEpilogue(tile, ids);
}

// Up-front scales.

namespace {

// A ds address register and the offset field that, together, name base +
// offset.
struct LdsAddress {
  RegisterRef base;
  int32_t offset;
};

} // namespace

// Names base + offset, adding an offset beyond the field to base in scratch,
// which the address of every ds instruction of the caller can share since a ds
// instruction reads its address when it issues.
static LdsAddress ldsAddress(IRBuilder &builder, RegisterRef base,
                             int32_t offset,
                             std::optional<RegisterRange> &scratch) {
  if (offset <= maxDsOffset)
    return {base, offset};
  if (!scratch)
    scratch = builder.allocateVgpr(1);
  builder.addi32(scratch->ref(), offset, base);
  return {scratch->ref(), 0};
}

// K phase helpers.

static constexpr int32_t exponentMask = 0x7F800000;

// Waits until no prologue load that writes one of the registers in written is
// in flight. A prologue load stays in flight when the split has fewer full
// iterations than PF and its slot is never consumed; loads complete in order,
// so the wait allows the later prologue loads and the issuedSince loads that
// every path issued after the prologue.
static void waitForPrologueLoads(IRBuilder &builder,
                                 const std::vector<RegisterRef> &record,
                                 const std::vector<RegisterRef> &written,
                                 int32_t issuedSince) {
  int32_t count = static_cast<int32_t>(record.size());
  for (int32_t newest = count - 1; newest >= 0; --newest) {
    bool conflict =
        std::any_of(written.begin(), written.end(), [&](RegisterRef reg) {
          RegisterRef load = record[newest];
          if (reg.kind != load.kind)
            return false;
          return reg.start < load.start + load.count &&
                 load.start < reg.start + reg.count;
        });
    if (!conflict)
      continue;
    waitForVectorLoads(builder, count - 1 - newest + issuedSince);
    return;
  }
}

static RegisterRef dataChunk(const KLoopState &state, int32_t slot,
                             int32_t chunk, int32_t column) {
  const SkinnyDims &dims = state.dims;
  int32_t index =
      ((slot * SkinnyDims::chunks + chunk) * dims.columns + column) * 4;
  return state.ring.data.ref(index, 4);
}

static RegisterRef rawScales(const KLoopState &state, int32_t slot) {
  const SkinnyDims &dims = state.dims;
  return state.ring.raw->ref(slot * dims.columns, dims.columns);
}

static RegisterRef dataRsrc(const KLoopState &state, int32_t column) {
  return column < state.dims.bm ? state.sRsrc : state.wRsrc;
}

// chunkBase[u] = ks + u * 16 * S, the part of the chunk soffsets that does not
// depend on the iteration; chunkBase[0] = ks is set by emitTile.
static void ensureChunkBase(const KLoopState &state, int32_t chunk) {
  ChunkBase &chunkBase = state.tile.chunkBase;
  if (chunk == 0 || chunkBase.ready[chunk])
    return;
  state.builder.addi32(chunkBase.regs.ref(chunk), chunkBase.regs.ref(0),
                       chunk * state.dims.chunkStride);
  chunkBase.ready[chunk] = true;
}

// Issues the scale dword load of every column of the iteration at soffset into
// the NC registers of raw. With neg (0 or -1 per lane) the voffset is ORed with
// it first, see emitSlotLoads. The destinations are appended to record, if any.
static void issueScales(const KLoopState &state, RegisterRef raw,
                        RegisterRef soffset,
                        std::optional<RegisterRef> neg = std::nullopt,
                        std::vector<RegisterRef> *record = nullptr) {
  std::vector<SlotLoad> loads;
  for (int32_t c = 0; c < state.dims.columns; ++c)
    loads.push_back(
        {.rsrc = c < state.dims.bm ? state.sSclRsrc : state.wSclRsrc,
         .voffset = state.tile.sBase.ref(c),
         .soffset = soffset,
         .width = MemoryWidth::b32,
         .dst = raw.slice(c),
         .steer = neg});
  emitSlotLoads(state.builder, loads, record);
}

// Loads the raw scale dword of every column of the iteration at soffset into
// raw, steering to -1 the lanes whose role group starts at or after k: past k a
// group belongs to the next row group of the scale matrix (other rows' scales,
// 255 decodes to +inf and 0 * inf is NaN). kAvail = k - iterOff, an SGPR, may
// be zero or negative for vector entries past k.
static void issueScalesWithinK(const KLoopState &state, RegisterRef raw,
                               RegisterRef soffset, RegisterRef kAvail) {
  IRBuilder &builder = state.builder;
  RegisterRange neg = builder.allocateVgpr(1);
  builder.subi32(neg.ref(), kAvail, state.tile.scaleGroup->ref());
  builder.addi32(neg.ref(), neg.ref(), -1);
  builder.shrsi32(neg.ref(), neg.ref(), 31);
  issueScales(state, raw, soffset, neg.ref());
}

// Loads chunk u of every column of the ring slot, steered by neg as the scales
// are.
static void issueChunk(const KLoopState &state, int32_t slot, int32_t chunk,
                       RegisterRef soffset,
                       std::optional<RegisterRef> neg = std::nullopt,
                       std::vector<RegisterRef> *record = nullptr) {
  std::vector<SlotLoad> loads;
  for (int32_t c = 0; c < state.dims.columns; ++c)
    loads.push_back({.rsrc = dataRsrc(state, c),
                     .voffset = state.tile.dBase.ref(c),
                     .soffset = soffset,
                     .width = MemoryWidth::b128,
                     .dst = dataChunk(state, slot, chunk, c),
                     .steer = neg});
  emitSlotLoads(state.builder, loads, record);
}

// Sets soffset = ks + (iteration << log2KI) for the refill iteration, which
// chunk 0 uses as its own, and issues the ring scales of the iteration;
// up-front scales are loaded once, before the loop.
static void startRefill(const KLoopState &state, const SlotContext &context,
                        RegisterRef soffset) {
  IRBuilder &builder = state.builder;
  builder.shli32(soffset, *context.refillIteration, state.dims.log2Ki);
  builder.addi32(soffset, soffset, state.tile.chunkBase.regs.ref(0));
  if (state.dims.upIters != 0)
    return;
  issueScales(state, rawScales(state, context.slot), soffset, std::nullopt,
              context.record);
}

// Chunk u >= 1 is at chunkBase[u] + (iteration << log2KI) in the loops, whose
// chunk bases are hoisted, and at the iteration's soffset plus u * 16 * S
// elsewhere. The shifted iteration replaces the unshifted one on first use.
static void refillChunk(const KLoopState &state, const SlotContext &context,
                        int32_t chunk, RegisterRef soffset) {
  IRBuilder &builder = state.builder;
  if (chunk == 0) {
    issueChunk(state, context.slot, 0, soffset, std::nullopt, context.record);
    return;
  }
  RegisterRange chunkOffset = builder.allocateSgpr(1);
  if (context.hoistedOffsets) {
    ensureChunkBase(state, chunk);
    RegisterRef iteration = *context.refillIteration;
    if (chunk == 1)
      builder.shli32(iteration, iteration, state.dims.log2Ki);
    builder.addi32(chunkOffset.ref(), state.tile.chunkBase.regs.ref(chunk),
                   iteration);
  } else
    builder.addi32(chunkOffset.ref(), soffset, chunk * state.dims.chunkStride);
  issueChunk(state, context.slot, chunk, chunkOffset.ref(), std::nullopt,
             context.record);
}

// Fetches the two scale dwords of every column from the lanes that hold them
// with ds_bpermute_b32: the first goes to lo, the second over the raw dword,
// which is dead once both permutes have issued. The raw dword of the first
// column has firstOutstanding vector loads issued after it.
static void gatherSlotScales(const KLoopState &state, RegisterRef raw,
                             RegisterRef lo, int32_t firstOutstanding) {
  IRBuilder &builder = state.builder;
  for (int32_t c = 0; c < state.dims.columns; ++c) {
    waitForVectorLoads(builder, firstOutstanding - c);
    RegisterRef dword = raw.slice(c);
    builder.create<rocisa::DSBPermuteB32>(lo.slice(c).container(),
                                          state.gather.addresses.sub(0),
                                          dword.container());
    builder.create<rocisa::DSBPermuteB32>(
        dword.container(), state.gather.addresses.sub(1), dword.container());
  }
}

// Packs the two dwords of every column into one whose byte u is the scale of
// chunk u. Permute c is the (NC - 1 - c)-th newest of the 2 * NC in flight.
static void permSlotScales(const KLoopState &state, RegisterRef selectors,
                           RegisterRef hi, RegisterRef lo, RegisterRef out) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
  for (int32_t c = 0; c < dims.columns; ++c) {
    int32_t outstanding = 2 * (dims.columns - 1 - c);
    builder.createWaitCnt(WaitCounts{
        .lds = std::min(outstanding, builder.target().maxLgkmcnt())});
    builder.create<rocisa::VPermB32>(
        out.slice(c).container(), hi.slice(c).container(),
        lo.slice(c).container(), selector(dims, selectors, c).container());
  }
}

// Decodes the E8M0 scale of chunk u to an f32: (dword << (23 - 8u)) &
// 0x7F800000, or a shift right by 1 for u = 3.
static void decodeScales(const KLoopState &state, RegisterRef dst,
                         RegisterRef packed, int32_t chunk) {
  IRBuilder &builder = state.builder;
  for (int32_t c = 0; c < state.dims.columns; ++c) {
    RegisterRef scale = dst.slice(c);
    if (chunk < SkinnyDims::chunks - 1)
      builder.shli32(scale, packed.slice(c), 23 - 8 * chunk);
    else
      builder.shrui32(scale, packed.slice(c), 1);
    builder.andi32(scale, exponentMask, scale);
  }
}

// Clears, in place, the decoded scales of the lanes whose chunk u starts at or
// after the end of the split: the data of the chunk is zero, but the block may
// hold scale padding, which decodes to infinity. The lanes that are inside or
// straddle the end start below it, so their block holds a real scale. The
// registers are dead after the chunk in the tail step, which runs once.
static void clearDeadChunkScales(const KLoopState &state, RegisterRef tailLive,
                                 int32_t chunk, RegisterRef scales) {
  IRBuilder &builder = state.builder;
  RegisterRange keep = builder.allocateVgpr(1);
  int32_t shift = 8 * (SkinnyDims::chunks - 1 - chunk);
  if (shift > 0) {
    builder.shli32(keep.ref(), tailLive, shift);
    builder.shrsi32(keep.ref(), keep.ref(), 24);
  } else
    builder.shrsi32(keep.ref(), tailLive, 24);
  for (int32_t c = 0; c < state.dims.columns; ++c)
    builder.andi32(scales.slice(c), scales.slice(c), keep.ref());
}

// Consumes chunk u of the slot: 4 dwords of 2 fp8 pairs per column, converted
// with the decoded scale and accumulated with v_dot2_f32_bf16.
static void computeChunk(const KLoopState &state, const SlotContext &context,
                         int32_t chunk, RegisterRef scales) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
  if (context.tailLive)
    clearDeadChunkScales(state, *context.tailLive, chunk, scales);
  RegisterRange pairs = builder.allocateVgpr(dims.columns);
  int32_t ownRefill =
      context.refillIteration ? dims.scaleLoads + chunk * dims.columns : 0;
  for (int32_t dword = 0; dword < 4; ++dword) {
    for (int32_t word = 0; word < 2; ++word) {
      std::optional<rocisa::VOP3PModifiers> highWord;
      if (word == 1)
        highWord = rocisa::VOP3PModifiers({1, 0, 0});
      for (int32_t c = 0; c < dims.columns; ++c) {
        // The first convert of column c of the chunk waits for its load, which
        // is complete once at most the loads issued after it are outstanding,
        // as vmcnt counts the loads of a wave in issue order. In a partial step
        // every chunk load may be skipped, so only the later columns of the
        // chunk itself are certain to be in flight; with ring scales the gather
        // already waited for every load, as the chunk loads follow the scale
        // loads.
        bool firstUse = dword == 0 && word == 0;
        if (firstUse && !context.partialStep)
          waitForVectorLoads(builder,
                             RingLoads{dims.loadsPerSlot}.issuedAfter(
                                 dims.scaleLoads + chunk * dims.columns + c,
                                 context.groupsAfter, ownRefill));
        else if (firstUse && dims.scaleLoads == 0)
          waitForVectorLoads(builder, dims.columns - 1 - c);
        builder.create<rocisa::VCvtScalePkFP8toBF16>(
            pairs.sub(c),
            dataChunk(state, context.slot, chunk, c).slice(dword).container(),
            scales.slice(c).container(), std::nullopt, highWord);
      }
      for (int32_t r = 0; r < dims.bm; ++r) {
        for (int32_t t = 0; t < dims.tn; ++t) {
          RegisterRef total = state.tile.acc.ref(r * dims.tn + t);
          builder.create<rocisa::VDot2F32BF16>(total.container(), pairs.sub(r),
                                               pairs.sub(dims.bm + t),
                                               total.container());
        }
      }
    }
  }
}

// Moves the packed dwords of the entry of the physical iteration into a new
// register per column. VGPRs cannot be indexed dynamically, so this is a chain
// of UP - 1 selects per column; the entry is the outer loop so that each mask
// dies after its sweep. A single entry needs no selection.
static std::optional<RegisterRange>
selectScales(const KLoopState &state, std::optional<RegisterRef> vectorIndex) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
  if (dims.upIters == 1)
    return std::nullopt;
  const RegisterRange &scaleVector = *state.upFront.scaleVector;
  RegisterRange packed = builder.allocateVgpr(dims.columns);
  for (int32_t entry = 1; entry < dims.upIters; ++entry) {
    builder.create<rocisa::SCmpEQI32>(vectorIndex->container(), entry);
    builder.create<rocisa::SCSelectB64>(std::make_shared<rocisa::VCC>(), -1, 0);
    for (int32_t c = 0; c < dims.columns; ++c) {
      // The first select reads entry 0 itself instead of a copy of it.
      RegisterRef current = entry == 1 ? scaleVector.ref(c) : packed.ref(c);
      builder.selectb32(packed.ref(c),
                        scaleVector.ref(entry * dims.columns + c), current);
    }
  }
  return packed;
}

// Decode-first schedule (PF >= 2): decode the four chunk scales before the
// scales are refilled, then per chunk compute and refill. The packed dwords
// die after the last decode. A partial step decodes chunk 3 in place over the
// packed dwords, to fit the register budget; the loop
// keeps its allocation.
static void computeDecodeFirst(const KLoopState &state,
                               const SlotContext &context,
                               RegisterRange &packed) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
  std::vector<RegisterRange> scales;
  std::vector<RegisterRef> chunkScales;
  int32_t separate =
      context.partialStep ? SkinnyDims::chunks - 1 : SkinnyDims::chunks;
  for (int32_t chunk = 0; chunk < separate; ++chunk) {
    scales.push_back(builder.allocateVgpr(dims.columns));
    chunkScales.push_back(scales.back().ref());
    decodeScales(state, chunkScales.back(), packed.ref(), chunk);
  }
  if (context.partialStep) {
    decodeScales(state, packed.ref(), packed.ref(), SkinnyDims::chunks - 1);
    chunkScales.push_back(packed.ref());
  } else
    packed.release();
  RegisterRange soffset = builder.allocateSgpr(1);
  if (context.refillIteration)
    startRefill(state, context, soffset.ref());
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    computeChunk(state, context, chunk, chunkScales[chunk]);
    // Each decoded scale dies after its chunk, so that the refill loads can
    // take its registers.
    if (chunk < separate)
      scales[chunk].release();
    else
      packed.release();
    if (context.refillIteration)
      refillChunk(state, context, chunk, soffset.ref());
  }
}

// Staged schedule (PF = 1 and the PF = 0 loop): refill the scales, then per
// chunk decode its scale, compute and refill the chunk, so that the new loads
// follow the consumption of the old ones.
static void computeStaged(const KLoopState &state, const SlotContext &context,
                          RegisterRange &packed) {
  IRBuilder &builder = state.builder;
  RegisterRange soffset = builder.allocateSgpr(1);
  if (context.refillIteration)
    startRefill(state, context, soffset.ref());
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    {
      RegisterRange scales = builder.allocateVgpr(state.dims.columns);
      decodeScales(state, scales.ref(), packed.ref(), chunk);
      computeChunk(state, context, chunk, scales.ref());
    }
    if (context.refillIteration)
      refillChunk(state, context, chunk, soffset.ref());
  }
}

// Up-front schedule: per chunk decode its scale, compute and refill the
// chunk; the scales are not refilled. The last chunk decodes over the packed
// dwords, which no chunk reads afterwards. Without packed dwords (one entry)
// the decoded scales were hoisted out of the loop.
static void computeUpFront(const KLoopState &state, const SlotContext &context,
                           std::optional<RegisterRange> &packed) {
  IRBuilder &builder = state.builder;
  RegisterRange soffset = builder.allocateSgpr(1);
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    std::optional<RegisterRange> decoded;
    RegisterRef scales;
    if (!packed)
      scales = state.upFront.hoistedScales[chunk].ref();
    else if (chunk == SkinnyDims::chunks - 1) {
      decodeScales(state, packed->ref(), packed->ref(), chunk);
      scales = packed->ref();
    } else {
      decoded = builder.allocateVgpr(state.dims.columns);
      decodeScales(state, decoded->ref(), packed->ref(), chunk);
      scales = decoded->ref();
    }
    computeChunk(state, context, chunk, scales);
    if (!context.refillIteration)
      continue;
    if (chunk == 0)
      startRefill(state, context, soffset.ref());
    refillChunk(state, context, chunk, soffset.ref());
  }
}

// The up-front scales have two moments in the K phase: the issue precedes every
// other load of the tile, and the pack follows the prologue, as the in-order
// vmcnt waits count the loads issued in between.
//
// Issue: loads every entry of the up-front vector, in order, before anything
// else that reads memory, so that the in-order vmcnt wait of the first use
// covers only the scales. Entry j is the iteration j of the split, at ks + j *
// KI, with ks = chunkBase[0]. With partial steps the entries are steered at k,
// as the vector may hold iterations that reach or pass it.
//
// Pack: packs the raw dwords of every entry into one dword per column whose
// byte u is the scale of chunk u, in place, through LDS or ds_bpermute, and for
// a single entry decodes the loop-invariant chunk scales.
void SkinnyGEMMEmitter::emitUpFrontScales(KLoopState &state, UpFrontStep step) {
  TileState &tile = state.tile;
  if (step == UpFrontStep::issue) {
    RegisterRef ks = tile.chunkBase.regs.ref(0);
    state.upFront.scaleVector =
        builder.allocateVgpr(dims.upIters * dims.columns);
    for (int32_t entry = 0; entry < dims.upIters; ++entry) {
      RegisterRef soffset = ks;
      std::optional<RegisterRange> entryOffset;
      if (entry > 0) {
        entryOffset = builder.allocateSgpr(1);
        soffset = entryOffset->ref();
        builder.addi32(soffset, ks, entry * dims.ki);
      }
      RegisterRef raw =
          state.upFront.scaleVector->ref(entry * dims.columns, dims.columns);
      if (dims.partialSteps == 0) {
        issueScales(state, raw, soffset);
        continue;
      }
      RegisterRange kAvail = builder.allocateSgpr(1);
      builder.subi32(kAvail.ref(), k.ref(), soffset);
      issueScalesWithinK(state, raw, soffset, kAvail.ref());
    }
    // No scale load remains, and only the ring-scale path reads these.
    tile.sBase.release();
    if (tile.scaleGroup)
      tile.scaleGroup->release();
    return;
  }
  ScaleGather &gather = state.gather;
  RegisterRange &scaleVector = *state.upFront.scaleVector;
  // Loads issued after the up-front load at index: the later up-front loads and
  // the PF slots of the prologue.
  auto loadsAfter = [&](int32_t index) {
    return RingLoads{dims.upIters * dims.columns}.issuedAfter(
        index, 0, dims.prefetch * dims.loadsPerSlot);
  };
  if (dims.scalesShared) {
    // Publishes the raw dwords through LDS: the dword (e * NC + c) of thread
    // tid goes to the LDS dword (e * NC + c) * WAVES * 64 + tid. Each write
    // waits only for its own load, as vmcnt counts in order, and the barrier
    // does not wait for vmcnt, unlike gpu.barrier, so the prologue data loads
    // stay in flight.
    {
      std::optional<RegisterRange> scratch;
      for (int32_t index = 0; index < dims.upIters * dims.columns; ++index) {
        builder.createWaitCnt(
            WaitCounts{.vectorLoads = std::min(loadsAfter(index),
                                               builder.target().maxVmcnt())});
        LdsAddress address = ldsAddress(builder, gather.ldsStage->ref(),
                                        index * dims.waves * 256, scratch);
        builder.storeDs(address.base, scaleVector.ref(index), MemoryWidth::b32,
                        address.offset);
      }
      builder.createWaitCnt(WaitCounts{.lds = 0});
      builder.create<rocisa::SBarrier>();
    }
    // The persistent launch reuses the staging address every tile.
    if (dims.grid3d)
      gather.ldsStage->release();
    // Every wave reads the two dwords of its entries: the lane reads the dwords
    // d = 0 and 1 of every column of the entry at the byte offsets of its
    // address, the second 4 * R * S / 2 bytes after the first.
    int32_t hiDelta = 4 * dims.roleSlots * dims.halfSubgroup;
    for (int32_t entry = 0; entry < dims.upIters; ++entry) {
      std::optional<RegisterRange> scratch;
      RegisterRef addresses = gather.addresses.ref();
      RegisterRange hi = builder.allocateVgpr(dims.columns);
      RegisterRef lo = scaleVector.ref(entry * dims.columns, dims.columns);
      for (int32_t c = 0; c < dims.columns; ++c) {
        int32_t offset = (entry * dims.columns + c) * dims.waves * 256;
        LdsAddress first = ldsAddress(builder, addresses, offset, scratch);
        builder.loadDs(lo.slice(c), first.base, MemoryWidth::b32, first.offset);
        LdsAddress second =
            ldsAddress(builder, addresses, offset + hiDelta, scratch);
        builder.loadDs(hi.ref(c), second.base, MemoryWidth::b32, second.offset);
      }
      permSlotScales(state, tile.selectors.ref(), hi.ref(), lo, lo);
    }
  } else {
    // An entry is packed in two phases, the 2 * NC fetches back to back and
    // then the permutes, and the entries stay in sequence: fetching all of them
    // first would keep two dwords per column and entry live.
    for (int32_t entry = 0; entry < dims.upIters; ++entry) {
      RegisterRange lo = builder.allocateVgpr(dims.columns);
      RegisterRef raw = scaleVector.ref(entry * dims.columns, dims.columns);
      gatherSlotScales(state, raw, lo.ref(), loadsAfter(entry * dims.columns));
      permSlotScales(state, tile.selectors.ref(), raw, lo.ref(), raw);
    }
  }
  // The persistent launch reuses the lane's addresses every tile.
  if (dims.grid3d) {
    gather.addresses.release();
    if (gather.ldsStage)
      gather.ldsStage->release();
  }
  // The selectors die with the pack.
  tile.selectors.release();
  if (dims.upIters != 1)
    return;
  // The single entry has a loop-invariant scale per chunk, which the loop reads
  // instead of decoding it every iteration.
  std::vector<RegisterRange> &hoisted = state.upFront.hoistedScales;
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    hoisted.push_back(builder.allocateVgpr(dims.columns));
    decodeScales(state, hoisted.back().ref(), scaleVector.ref(0, dims.columns),
                 chunk);
  }
  // The raw entry dies after the decode.
  scaleVector.release();
}

// The VALU half of the ring schedule over the full iterations of the split.
// The refill and the physical iteration of a slot are computed from the logical
// iteration with the K rotation folded in and wrapped past NI, so that the
// refills of the last PF iterations reload the first ones.
class SkinnyGEMMEmitter::ValuRingPath final : public RingPath {
public:
  ValuRingPath(SkinnyGEMMEmitter &emitter, KLoopState &state)
      : emitter(emitter), state(state), builder(emitter.builder),
        dims(emitter.dims), tile(state.tile),
        vectorized(emitter.dims.upIters >= 2) {}

  int32_t slots() const override { return dims.prefetch; }
  RegisterRef iterations() const override { return tile.nIterFull.ref(); }
  void fill(const RingStep &step) override;
  void afterPrologue() override;
  void beginLoop() override;
  void endLoop() override;
  void consume(const RingStep &step) override;
  void drain() override {}
  void endArm() override;

private:
  // dst = sum >= NIfull ? sum - NIfull : sum, the physical iteration of a
  // logical one, with the rotation added once.
  void wrapIteration(RegisterRef dst, RegisterRef sum);
  void hoistChunkBases();
  void fillSingleStep(const RingStep &step);
  void fillPrologueSlot(int32_t slot);
  void consumeInLoop(const RingStep &step);
  void consumeInArm(const RingStep &step);

  SkinnyGEMMEmitter &emitter;
  KLoopState &state;
  IRBuilder &builder;
  const SkinnyDims &dims;
  TileState &tile;
  // With several up-front entries, the slot selects its scales by the physical
  // iteration.
  bool vectorized;
  // The prologue's clamp of the last iteration, which only slots above 0 read.
  std::optional<RegisterRange> lastIteration;
  // The steady loop's bases that fold the rotation: refillBase[p] = fitRot + PF
  // + p and physBase[p] = fitRot + p. Without rotation they are the literals.
  std::optional<RegisterRange> refillBase, physBase;
  // The PF = 0 step, which keeps its SGPRs and the selected scales from the
  // loads to the end of the compute.
  std::optional<RegisterRange> stepPhysical, stepSoffset, stepPacked;
  // The arm: the logical iteration of its first step with the rotation, and the
  // refill iterations of the steps, which the vector index of the step PF later
  // is.
  std::optional<RegisterRange> armBase;
  std::vector<RegisterRange> keptRefills;
};

void SkinnyGEMMEmitter::ValuRingPath::wrapIteration(RegisterRef dst,
                                                    RegisterRef sum) {
  RegisterRange wrapped = builder.allocateSgpr(1);
  builder.subi32(wrapped.ref(), sum, tile.nIterFull.ref());
  builder.create<rocisa::SCmpGeU32>(sum.container(),
                                    tile.nIterFull.container());
  builder.selectb32(dst, wrapped.ref(), sum);
}

void SkinnyGEMMEmitter::ValuRingPath::hoistChunkBases() {
  for (int32_t chunk = 1; chunk < SkinnyDims::chunks; ++chunk)
    ensureChunkBase(state, chunk);
}

void SkinnyGEMMEmitter::ValuRingPath::fill(const RingStep &step) {
  if (dims.prefetch == 0)
    fillSingleStep(step);
  else
    fillPrologueSlot(step.slot);
}

// The select of the up-front scales precedes the loads so that its VALU work
// hides their latency.
void SkinnyGEMMEmitter::ValuRingPath::fillSingleStep(const RingStep &step) {
  // beginLoop released the raw scales, which live again from these loads to
  // the permutes of emitCompute.
  if (state.ring.raw)
    state.ring.raw = builder.allocateVgpr(dims.columns);
  stepPhysical = builder.allocateSgpr(1);
  RegisterRef physical = stepPhysical->ref();
  if (tile.fitRot) {
    builder.addi32(physical, step.base, tile.fitRot->ref());
    wrapIteration(physical, physical);
  } else
    wrapIteration(physical, step.base);
  if (dims.upIters > 0)
    stepPacked = selectScales(state, physical);
  stepSoffset = builder.allocateSgpr(1);
  SlotContext loads{0, 0, physical};
  startRefill(state, loads, stepSoffset->ref());
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    refillChunk(state, loads, chunk, stepSoffset->ref());
}

// Issues the iteration slot of the ring. A split with fewer full iterations
// reloads its last one through the clamp: a load dropped by the range check
// would still take a VMEM issue slot, and the extra slots are never consumed.
void SkinnyGEMMEmitter::ValuRingPath::fillPrologueSlot(int32_t slot) {
  int32_t pf = dims.prefetch;
  if (slot == 0 && pf > 1) {
    lastIteration = builder.allocateSgpr(1);
    builder.addi32(lastIteration->ref(), tile.nIterFull.ref(), -1);
    builder.maxsi32(lastIteration->ref(), lastIteration->ref(), 0);
  }
  RegisterRange physical = builder.allocateSgpr(1);
  if (slot > 0) {
    builder.minui32(physical.ref(), lastIteration->ref(), slot);
    if (tile.fitRot)
      builder.addi32(physical.ref(), physical.ref(), tile.fitRot->ref());
    wrapIteration(physical.ref(), physical.ref());
  } else if (tile.fitRot)
    wrapIteration(physical.ref(), tile.fitRot->ref());
  else {
    // The wrapped physical iteration of the logical iteration 0 without
    // rotation.
    RegisterRange wrapped = builder.allocateSgpr(1);
    builder.subi32(wrapped.ref(), 0, tile.nIterFull.ref());
    builder.create<rocisa::SCmpLeU32>(tile.nIterFull.container(), 0);
    builder.selectb32(physical.ref(), wrapped.ref(), 0);
  }
  // The chunk bases are hoisted out of the prologue from PF = 4 on.
  SlotContext context{slot, 0, physical.ref(), pf >= 4};
  context.record = &tile.prologueLoads;
  RegisterRange soffset = builder.allocateSgpr(1);
  startRefill(state, context, soffset.ref());
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    refillChunk(state, context, chunk, soffset.ref());
  if (slot == pf - 1)
    lastIteration.reset();
}

// The pack follows the prologue loads that its waits count.
void SkinnyGEMMEmitter::ValuRingPath::afterPrologue() {
  if (dims.upIters > 0)
    emitter.emitUpFrontScales(state, UpFrontStep::pack);
}

void SkinnyGEMMEmitter::ValuRingPath::beginLoop() {
  int32_t pf = dims.prefetch;
  if (pf == 0) {
    hoistChunkBases();
    // Nothing of the ring is loop carried, so the raw scales live only from the
    // loads of an iteration to its permutes.
    if (state.ring.raw)
      state.ring.raw->release();
    return;
  }
  if (tile.fitRot) {
    refillBase = builder.allocateSgpr(pf);
    builder.addi32(refillBase->ref(0), tile.fitRot->ref(), pf);
  }
  hoistChunkBases();
  if (vectorized)
    physBase = builder.allocateSgpr(pf);
  for (int32_t slot = 1; tile.fitRot && slot < pf; ++slot) {
    builder.addi32(refillBase->ref(slot), tile.fitRot->ref(), pf + slot);
    if (vectorized)
      builder.addi32(physBase->ref(slot), tile.fitRot->ref(), slot);
  }
}

void SkinnyGEMMEmitter::ValuRingPath::endLoop() {
  physBase.reset();
  refillBase.reset();
}

void SkinnyGEMMEmitter::ValuRingPath::consume(const RingStep &step) {
  if (dims.prefetch == 0) {
    SlotContext context{0, 0, std::nullopt};
    if (dims.upIters > 0)
      computeUpFront(state, context, stepPacked);
    else
      emitter.emitCompute(state, context);
    stepPacked.reset();
    stepSoffset.reset();
    stepPhysical.reset();
    return;
  }
  if (step.inLoop)
    consumeInLoop(step);
  else
    consumeInArm(step);
}

void SkinnyGEMMEmitter::ValuRingPath::consumeInLoop(const RingStep &step) {
  int32_t pf = dims.prefetch;
  RegisterRange refill = builder.allocateSgpr(1);
  if (refillBase)
    builder.addi32(refill.ref(), step.base, refillBase->ref(step.offset));
  else
    builder.addi32(refill.ref(), step.base, pf + step.offset);
  wrapIteration(refill.ref(), refill.ref());
  SlotContext context{step.slot, step.groupsAfter, refill.ref()};
  std::optional<RegisterRange> physical;
  if (vectorized) {
    physical = builder.allocateSgpr(1);
    if (step.offset == 0)
      builder.addi32(physical->ref(), tile.fitRot->ref(), step.base);
    else
      builder.addi32(physical->ref(), step.base, physBase->ref(step.offset));
    wrapIteration(physical->ref(), physical->ref());
    context.vectorIndex = physical->ref();
  }
  emitter.emitCompute(state, context);
}

// With several up-front entries the step also selects the scales of its
// physical iteration, which is the refill iteration of the step PF earlier from
// i = PF on.
void SkinnyGEMMEmitter::ValuRingPath::consumeInArm(const RingStep &step) {
  int32_t pf = dims.prefetch;
  int32_t i = step.offset;
  SlotContext context{step.slot, step.groupsAfter, std::nullopt, false};
  RegisterRange refill = builder.allocateSgpr(1);
  // The logical iteration of a refill is fitRot + steadyEnd + i + PF.
  if (!armBase && tile.fitRot && (vectorized || step.refill)) {
    armBase = builder.allocateSgpr(1);
    builder.addi32(armBase->ref(), tile.fitRot->ref(), step.base);
  }
  RegisterRef base = armBase ? armBase->ref() : step.base;
  if (step.refill) {
    builder.addi32(refill.ref(), base, i + pf);
    wrapIteration(refill.ref(), refill.ref());
    context.refillIteration = refill.ref();
  }
  std::optional<RegisterRange> physical;
  if (vectorized && i >= pf)
    context.vectorIndex = keptRefills[i - pf].ref();
  else if (vectorized) {
    physical = builder.allocateSgpr(1);
    if (i > 0)
      builder.addi32(physical->ref(), base, i);
    wrapIteration(physical->ref(), i > 0 ? physical->ref() : base);
    context.vectorIndex = physical->ref();
  }
  emitter.emitCompute(state, context);
  if (vectorized && step.refill)
    keptRefills.push_back(std::move(refill));
}

void SkinnyGEMMEmitter::ValuRingPath::endArm() {
  keptRefills.clear();
  armBase.reset();
}

void SkinnyGEMMEmitter::emitKLoop(KLoopState &state) {
  ValuRingPath path(*this, state);
  emitRingLoop(builder, path);
}

// One ring slot: the scales of the slot are gathered (ring scales) or selected
// (up-front scales) and decoded, its four chunks consumed and, with a refill
// iteration, the slot refilled. The schedules differ in where the refill sits
// relative to the decode and the compute, and are separate functions.
void SkinnyGEMMEmitter::emitCompute(KLoopState &state,
                                    const SlotContext &context) {
  if (dims.upIters > 0) {
    std::optional<RegisterRange> packed =
        selectScales(state, context.vectorIndex);
    computeUpFront(state, context, packed);
    return;
  }
  RegisterRange packed = builder.allocateVgpr(dims.columns);
  // A step issues only its own scale loads on every path: its chunk loads may
  // all be skipped.
  int32_t firstOutstanding =
      context.partialStep
          ? dims.columns - 1
          : RingLoads{dims.loadsPerSlot}.issuedAfter(0, context.groupsAfter);
  RegisterRef raw = rawScales(state, context.slot);
  gatherSlotScales(state, raw, packed.ref(), firstOutstanding);
  permSlotScales(state, state.tile.selectors.ref(), raw, packed.ref(),
                 packed.ref());
  // The PF = 0 loop frees the raw scales again, see ValuRingPath::beginLoop.
  if (dims.prefetch == 0)
    state.ring.raw->release();
  // Every depth but 1 decodes first; the PF = 0 loop keeps the
  // staged schedule, which its steps do not use.
  bool decodeFirst =
      dims.prefetch >= 2 || (dims.prefetch == 0 && context.partialStep);
  if (decodeFirst)
    computeDecodeFirst(state, context, packed);
  else
    computeStaged(state, context, packed);
}

// Partial K steps.

namespace {

// The VALU half of the tail iteration: the chunks of the tail slot, slot 0, are
// the ring's, at the iteration that starts at the soffset iterOff.
class ValuTailPath final : public TailPath {
public:
  ValuTailPath(const KLoopState &state, const TailLanes &laneValues,
               RegisterRef iterOff)
      : state(state), laneValues(laneValues), iterOff(iterOff) {}

  TailLanes lanes() const override { return laneValues; }
  std::vector<RegisterRef> chunkData(int32_t chunk) const override {
    std::vector<RegisterRef> data;
    for (int32_t column = 0; column < state.dims.columns; ++column)
      data.push_back(dataChunk(state, 0, chunk, column));
    return data;
  }
  void loadChunk(int32_t chunk, std::optional<RegisterRef> steer) override {
    if (chunk == 0) {
      issueChunk(state, 0, chunk, iterOff, steer);
      return;
    }
    RegisterRange soffset = state.builder.allocateSgpr(1);
    state.builder.addi32(soffset.ref(), iterOff,
                         chunk * state.dims.chunkStride);
    issueChunk(state, 0, chunk, soffset.ref(), steer);
  }
  // The slot's registers may still have a prologue load in flight when the
  // split has no full iteration; the step's scale loads are issued on every
  // path and complete after it.
  void beforeZero(int32_t chunk) override {
    const SkinnyDims &dims = state.dims;
    waitForPrologueLoads(
        state.builder, state.tile.prologueLoads,
        {state.ring.data.ref(chunk * dims.columns * 4, dims.columns * 4)},
        dims.scaleLoads);
  }

private:
  const KLoopState &state;
  TailLanes laneValues;
  RegisterRef iterOff;
};

} // namespace

// The tail step of PARTIAL_STEPS = 1: the iteration at ks + nIterFull * KI
// keeps the bytes [0, tailBytes), and the step is skipped when there are none.
// It runs on ring slot 0, which the K loop left free. Ring scales load as in
// the loop with the groups at or after k steered to -1, and the chunks are
// loaded, zeroed or masked by the shared three-arm tail (emitTailChunks), so
// that the slot holds exact zeros outside [0, tailBytes). The decoded scales of
// the lane chunks that start at or after tailBytes are cleared by tailLive, as
// their block may hold scale padding. The compute is the one of a slot without
// refill.
void SkinnyGEMMEmitter::emitTail(KLoopState &state) {
  TileState &tile = state.tile;
  RegisterRef hi = tile.tailBytes->ref();
  emitTailGuard(builder, hi, [&] {
    RegisterRange iterOff = builder.allocateSgpr(1);
    builder.shli32(iterOff.ref(), tile.nIterFull.ref(), dims.log2Ki);
    builder.addi32(iterOff.ref(), iterOff.ref(), tile.chunkBase.regs.ref(0));
    std::optional<RegisterRange> kAvail;
    if (dims.upIters == 0) {
      kAvail = builder.allocateSgpr(1);
      builder.subi32(kAvail->ref(), k.ref(), iterOff.ref());
    }
    // The up-front vector starts at the split's first iteration, so the entry
    // of the tail iteration is the number of full iterations.
    std::optional<RegisterRef> vectorIndex;
    if (dims.upIters >= 2)
      vectorIndex = tile.nIterFull.ref();
    // The PF = 0 loop released its raw scales, and the step takes their
    // registers back before anything else, so that a smaller value does not
    // split the hole they leave.
    if (dims.upIters == 0 && dims.prefetch == 0)
      state.ring.raw = builder.allocateVgpr(dims.columns);
    std::optional<RegisterRange> tailLive;
    {
      // v0 holds the lane's index in its cluster after the setup.
      RegisterRange laneBytes =
          computeLaneBytes(builder, dims, kernel.workitemId().ref());
      if (dims.upIters == 0)
        issueScalesWithinK(state, rawScales(state, 0), iterOff.ref(),
                           kAvail->ref());
      // The subgroups of S lanes start 64 * S bytes apart, so the last lane
      // starts at (LANES_K / S - 1) * 64 * S + 16 * (S - 1).
      int32_t subgroupLanes = dims.subgroupLanes;
      int32_t laneSpan =
          (dims.lanesK / subgroupLanes - 1) * 64 * subgroupLanes +
          16 * subgroupLanes;
      TailLanes lanes{hi, laneBytes.ref(), dims.chunkStride, laneSpan};
      ValuTailPath path(state, lanes, iterOff.ref());
      emitTailChunks(builder, path);
      tailLive = builder.allocateVgpr(1);
      emitChunkLiveBytes(builder, tailLive->ref(), lanes);
      // The only step runs under the runtime branch and nothing after it, the
      // epilogue included, reads the load state, so its registers are free for
      // the compute to fit the register budget. They belong to the tile, so no
      // scope can free them. Up-front scales released the scale bases and the
      // scale group already; ring scales still hold them here.
      if (tile.scaleGroup)
        tile.scaleGroup->release();
      tile.sBase.release();
      tile.dBase.release();
    }
    SlotContext context{0,
                        0,
                        std::nullopt,
                        true,
                        vectorIndex,
                        /*partialStep=*/true};
    context.tailLive = tailLive->ref();
    emitCompute(state, context);
  });
}

// C prefetch and epilogue.

namespace {

// The control of one DPP add of the reduction.
struct DppStep {
  int32_t rowBcast;
  std::vector<int32_t> quadPerm;
  bool rowMirror;
  bool rowHalfMirror;
  int32_t rowMask;
  int32_t boundCtrl;
};

// The VALU path's view of the output after the reduction: the BM x TN elements
// of the lane sit in the registers in the layout of the orientation, and only
// the leader lane of a cluster stores.
class ValuOutputMap final : public OutputMap {
public:
  ValuOutputMap(IRBuilder &builder, const SkinnyDims &dims,
                const TileState &tile, bool skinnyContiguous,
                RegisterRef values, RegisterRef off,
                std::optional<RegisterRef> wBase, RegisterRef leader)
      : builder(builder), dims(dims), tile(tile),
        layout(redLayout(dims, skinnyContiguous)), off(off), wBase(wBase),
        leader(leader) {
    this->skinnyContiguous = skinnyContiguous;
    this->values = values;
    skinnyCount = dims.bm;
    wideCount = dims.tn;
    runLength = layout.runs ? outputRunLength(dims, skinnyContiguous) : 1;
    segmentCount = layout.runs ? dims.accumulators / runLength : 1;
    segmentLength = layout.runs ? runLength : dims.accumulators;
    segmentStride = layout.runs ? layout.stride : dims.accumulators;
  }

  int32_t valueIndex(int32_t i, int32_t j) const override {
    return redIndex(dims, layout, i, j);
  }

  std::optional<RegisterRef> storeLanes() const override { return leader; }

  // The scalars of a store path, in bytes: the byte offset of the tile's first
  // element without the per-lane part of the wide column, which goes to wBase
  // (or into the scalar offset when w0 is uniform).
  void prepare(const StoreTarget &target) override {
    prepared.reset();
    prepared.emplace(Prepared{target, builder.allocateSgpr(1),
                              builder.allocateSgpr(1),
                              builder.allocateSgpr(1)});
    Prepared &scalars = *prepared;
    builder.muli32(scalars.base.ref(), tile.s0.ref(), target.strideS);
    if (target.sliceBase)
      builder.addi32(scalars.base.ref(), scalars.base.ref(), *target.sliceBase);
    if (wBase) {
      builder.muli32(*wBase, target.strideW, tile.w0.ref());
      return;
    }
    builder.muli32(scalars.scratch.ref(), tile.w0.ref(), target.strideW);
    builder.addi32(scalars.base.ref(), scalars.base.ref(),
                   scalars.scratch.ref());
  }

  // off = (s0 + i) * strideS + (w0 + j) * strideW + sliceBase, with all
  // strides in bytes.
  RegisterRef offset(int32_t i, int32_t j) override {
    Prepared &scalars = *prepared;
    RegisterRef uniform = scalars.base.ref();
    if (i != 0 || j != 0) {
      uniform = scalars.offset.ref();
      builder.movb32(uniform, scalars.base.ref());
      if (i != 0) {
        builder.muli32(scalars.scratch.ref(), scalars.target.strideS, i);
        builder.addi32(uniform, uniform, scalars.scratch.ref());
      }
      if (j != 0) {
        builder.muli32(scalars.scratch.ref(), scalars.target.strideW, j);
        builder.addi32(uniform, uniform, scalars.scratch.ref());
      }
    }
    if (wBase)
      builder.addi32(off, uniform, *wBase);
    else
      builder.movb32(off, uniform);
    return off;
  }

  void accessLanes(RegisterRef mask, int32_t i, int32_t j) override {
    builder.andi64(mask, tile.sValid.ref(2 * i, 2), tile.wValid.ref(2 * j, 2));
    builder.andi64(mask, mask, leader);
  }

private:
  struct Prepared {
    StoreTarget target;
    RegisterRange base, scratch, offset;
  };

  IRBuilder &builder;
  const SkinnyDims &dims;
  const TileState &tile;
  RedLayout layout;
  RegisterRef off;
  std::optional<RegisterRef> wBase;
  RegisterRef leader;
  std::optional<Prepared> prepared;
};

} // namespace

// Wait states between a v_dot2 write and a DPP read of its result (LLVM's
// DotWriteDifferentVALURead) and between a VALU write and a DPP read of it.
static constexpr int32_t dotToDppWaitStates = 3;

static constexpr int32_t valuToDppWaitStates = 2;

// Wait states of v_readlane_b32 after a VALU wrote its source, and of a VALU
// read of the SGPR that v_readlane_b32 wrote (gfx940, per LLVM's hazard
// recognizer).
static constexpr int32_t valuToReadlaneWaitStates = 1;

static constexpr int32_t readlaneToValuWaitStates = 2;

static constexpr int32_t bankMaskAll = 0xf;

// Reduces every accumulator over its LANES_K lanes into the red registers:
// DPP adds that leave the cluster sum in every lane, which a v_readlane of
// lane 63 broadcasts for 64 lanes. It is a function for the independence of
// the algorithm, which keeps emitEpilogue short.
static void reduceAccumulators(IRBuilder &builder, const SkinnyDims &dims,
                               const TileState &tile, RedLayout layout,
                               RegisterRef red) {
  auto emitNops = [&](int32_t count) {
    if (count > 0)
      builder.create<rocisa::VNop>(count);
  };
  auto emitDpp = [&](RegisterRef dst, RegisterRef src, const DppStep &step) {
    rocisa::DPPModifiers dpp(-1, step.rowBcast, step.boundCtrl, step.quadPerm,
                             -1, step.rowMirror, step.rowHalfMirror,
                             step.rowMask, bankMaskAll);
    builder.create<rocisa::VAddF32>(dst.container(), src.container(),
                                    src.container(), dpp);
  };
  // The DPP adds that follow the first one for a cluster of LANES_K lanes: the
  // quad permutes, the half-row and row mirrors, and for 64 lanes the row
  // broadcasts.
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
  int32_t valuSinceReduce = 0;
  for (int32_t p = 0; p < dims.accumulators; ++p) {
    // The last chunk ends with the dots of all accumulators in order, so p has
    // P - 1 - p VALU instructions after its last dot.
    int32_t sinceDot = dims.accumulators - 1 - p + valuSinceReduce;
    emitNops(dotToDppWaitStates - sinceDot);
    valuSinceReduce += std::max(0, dotToDppWaitStates - sinceDot);
    RegisterRef dst =
        red.slice(redIndex(dims, layout, p / dims.tn, p % dims.tn));
    emitDpp(dst, tile.acc.ref(p),
            DppStep{-1, {1, 0, 3, 2}, false, false, 0xf, 1});
    valuSinceReduce += 1;
    for (const DppStep &step : steps) {
      emitNops(valuToDppWaitStates);
      emitDpp(dst, dst, step);
      valuSinceReduce += valuToDppWaitStates + 1;
    }
    if (dims.lanesK < 64)
      continue;
    // Lane 63 holds the total; every lane takes it.
    RegisterRange total = builder.allocateSgpr(1);
    emitNops(valuToReadlaneWaitStates);
    builder.create<rocisa::VReadlaneB32>(total.container(), dst.container(),
                                         63);
    emitNops(readlaneToValuWaitStates);
    builder.movb32(dst, total.ref());
    valuSinceReduce +=
        valuToReadlaneWaitStates + 1 + readlaneToValuWaitStates + 1;
  }
}

// Whether either orientation has runs for the vector stores.
static bool anyVectorStores(const SkinnyDims &dims) {
  return hasVector(dims, true, false) || hasVector(dims, false, false);
}

// Prefetches the tile of C into cData, for the epilogue's beta * C, when the
// tile stores D directly (k <= spk) and beta is not zero; else it only costs
// the scalar tests. The loads are issued before every scale and ring load of
// the tile and after everything older, such as the stores of the previous tile
// of a persistent block, and the temporaries are dead before the ring loads.
//
// vmcnt counts the vector memory operations of a wave in issue order. Every
// wait either allows the N loads issued after its target or drains everything
// (the single vmcnt(0) wait of emitTailChunks and the persistent shared-scales
// drain), so the older C loads are complete whenever its target is, and both
// forms are safe with them.
// The epilogue consumes C after an explicit vmcnt(0). C may alias D: each lane
// loads its elements here and stores them in the epilogue, and the tiles are
// disjoint, so the next tile of a persistent block loads after this tile's
// epilogue consumed cData.
void SkinnyGEMMEmitter::emitCPrefetch(const TileState &tile,
                                      const TileIds &ids) {
  OutputContext ctx = outputContext();
  std::shared_ptr<rocisa::Label> skip = builder.createLabel("c_prefetch_skip");
  branchUnlessCIsUsed(ctx, skip);
  RegisterRange leader = builder.allocateSgpr(2, 2);
  builder.create<rocisa::VCmpEQI32>(leader.container(), 0,
                                    kernel.workitemId().container());
  RegisterRange off = builder.allocateVgpr(1);
  std::optional<RegisterRange> wBase;
  if (!dims.uniformWide)
    wBase = builder.allocateVgpr(1);
  forEachOrientation(ctx, anyVectorStores(dims), [&](bool skinnyContiguous) {
    ValuOutputMap map(
        builder, dims, tile, skinnyContiguous, tile.cData->ref(), off.ref(),
        wBase ? std::optional<RegisterRef>(wBase->ref()) : std::nullopt,
        leader.ref());
    loadCTile(ctx, map, ids);
  });
  builder.insert(skip);
}

// Reduces the accumulators over the lanes of a cluster, then the shared
// epilogue scales them by alpha, adds beta * C, and stores the tile: bf16
// straight to D for a single split, else the f32 partial slice into the
// workspace.
void SkinnyGEMMEmitter::emitEpilogue(const TileState &tile,
                                     const TileIds &ids) {
  OutputContext ctx = outputContext();
  forEachOrientation(ctx, anyVectorStores(dims), [&](bool skinnyContiguous) {
    RedLayout layout = redLayout(dims, skinnyContiguous);
    RegisterRef kl = kernel.workitemId().ref();
    // The registers are allocated up front so that one wait covers every
    // prologue load whose destination they reuse.
    RegisterRange red = builder.allocateVgpr(layout.total, layout.runs ? 2 : 1);
    RegisterRange off = builder.allocateVgpr(1);
    std::optional<RegisterRange> wBase;
    if (!dims.uniformWide)
      wBase = builder.allocateVgpr(1);
    RegisterRange leader = builder.allocateSgpr(2, 2);
    ValuOutputMap map(
        builder, dims, tile, skinnyContiguous, red.ref(), off.ref(),
        wBase ? std::optional<RegisterRef>(wBase->ref()) : std::nullopt,
        leader.ref());
    // The epilogue overwrites the ring registers, and a prologue load that was
    // never consumed (a split with fewer than PF iterations) may still be in
    // flight.
    std::vector<RegisterRef> written = {red.ref(), off.ref()};
    if (wBase)
      written.push_back(wBase->ref());
    waitForPrologueLoads(builder, tile.prologueLoads, written,
                         /*issuedSince=*/0);
    reduceAccumulators(builder, dims, tile, layout, red.ref());
    builder.create<rocisa::VCmpEQI32>(leader.container(), 0, kl.container());
    emitOutput(ctx, map, ids,
               tile.cData ? std::optional<RegisterRef>(tile.cData->ref())
                          : std::nullopt);
  });
}
