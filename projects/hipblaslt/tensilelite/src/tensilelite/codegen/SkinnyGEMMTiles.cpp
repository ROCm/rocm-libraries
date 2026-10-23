// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The persistent 1-D launch (GRID_3D = 0 of @gemm): block b runs the tiles b, b
// + B, ... of the flattened tile space, in the order of @remap_xcd and the
// grouped decode. The per-lane values and the tile-independent scalars are
// computed once before the loop; every tile runs @run_tile from zeroed
// accumulators.
//
// The reference lowers the floor divisions and moduli of its affine maps with
// sign corrections and recomputes the remainders. Every operand here is
// non-negative, so each quotient is one division and each remainder is a - q *
// d.

#include "SkinnyGEMMEmitter.hpp"

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <memory>
#include <optional>

namespace rocm::tensilelite::codegen::detail {

namespace {

// quotient = src / divisor for a non-negative src. quotient must not alias src.
void divideByLiteral(IRBuilder &builder, RegisterRef quotient, RegisterRef src,
                     int divisor) {
  if (isPowerOfTwo(divisor)) {
    shiftRight(builder, quotient, src, log2Exact(divisor));
    return;
  }
  divideByConstant(builder, quotient, src, divisor);
}

// dst = src - quotient * divisor, the remainder of a non-negative src.
void remainderByLiteral(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                        RegisterRef quotient, int divisor) {
  if (isPowerOfTwo(divisor)) {
    bitAnd(builder, dst, src, divisor - 1);
    return;
  }
  multiplyBy(builder, dst, quotient, divisor);
  builder.subi32(dst, src, dst);
}

// dst = src mod divisor for a non-negative src. dst must not alias src.
void moduloByLiteral(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                     int divisor) {
  if (isPowerOfTwo(divisor)) {
    bitAnd(builder, dst, src, divisor - 1);
    return;
  }
  RegisterRange quotient = builder.allocateSgpr(1);
  divideByConstant(builder, quotient.ref(), src, divisor);
  remainderByLiteral(builder, dst, src, quotient.ref(), divisor);
}

// dst = ceilDiv(src, divisor) for a non-negative src. dst must not alias src.
void ceilDivByLiteral(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                      int divisor) {
  if (isPowerOfTwo(divisor)) {
    builder.addi32(dst, src, divisor - 1);
    shiftRight(builder, dst, dst, log2Exact(divisor));
    return;
  }
  RegisterRange sum = builder.allocateSgpr(1);
  builder.addi32(sum.ref(), src, divisor - 1);
  divideByConstant(builder, dst, sum.ref(), divisor);
}

} // namespace

void SkinnyGEMMEmitter::emitPersistent() {
  SetupScalars scalars(builder, /*grid3d=*/false);
  {
    ResourceScalars resources(builder, /*grid3d=*/false);
    loadArguments(scalars, resources);
    makeResources(scalars, resources);
  }
  TileLoop loop = computeTileLoop(scalars);
  LaneValues lane = computeLaneValues();
  emitTileLoop(scalars, lane, loop);
  builder.createEndProgram();
}

// The values that every tile shares. The rotation keys on the block id in the
// plain order, so it is loop invariant, and on the wide slab of the tile when
// reordering, which needs NI at every tile.
SkinnyGEMMEmitter::TileLoop
SkinnyGEMMEmitter::computeTileLoop(SetupScalars &scalars) {
  TileLoop loop{sgpr(),       sgpr(),       sgpr(),
                std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, std::nullopt};
  RegisterRange tilesY = sgpr();
  ceilDivByLiteral(builder, loop.tilesX.ref(), wideCols.ref(), dims.tileWide);
  ceilDivByLiteral(builder, tilesY.ref(), skinnyCols.ref(), dims.bm);
  builder.muli32(loop.plane.ref(), loop.tilesX.ref(), tilesY.ref());
  builder.muli32(loop.numTiles.ref(), loop.plane.ref(), scalars.splits->ref());
  scalars.splits.reset();
  // The wide slab rotation of the reordered tiles divides by the dynamic NI.
  bool needsNi = dims.reordered ? dims.upIters != 1 : dims.upIters == 0;
  if (needsNi) {
    RegisterRange ni = sgpr();
    builder.addi32(ni.ref(), spk.ref(), dims.ki - 1);
    shiftRight(builder, ni.ref(), ni.ref(), dims.log2Ki);
    loop.ni = std::move(ni);
  }
  if (!dims.reordered && dims.upIters != 1) {
    loop.blockRot = sgpr();
    blockRotation(loop.blockRot->ref(), kernel.workgroupIdX().ref(),
                  loop.ni ? std::optional<RegisterRef>(loop.ni->ref())
                          : std::nullopt);
    loop.ni.reset();
  }
  if (dims.numXcds != 1)
    computeRemapInvariants(loop);
  if (dims.group != 1) {
    loop.tilesY = std::move(tilesY);
    loop.bandSize = sgpr();
    multiplyBy(builder, loop.bandSize->ref(), loop.tilesX.ref(), dims.group);
  }
  return loop;
}

// perXcd = ceilDiv(numTiles, X) and tall = (numTiles - 1) mod X + 1, the XCDs
// that hold perXcd tiles, the others hold one less.
void SkinnyGEMMEmitter::computeRemapInvariants(TileLoop &loop) {
  loop.perXcd = sgpr();
  ceilDivByLiteral(builder, loop.perXcd->ref(), loop.numTiles.ref(),
                   dims.numXcds);
  loop.tall = sgpr();
  RegisterRange last = sgpr();
  builder.addi32(last.ref(), loop.numTiles.ref(), -1);
  moduloByLiteral(builder, loop.tall->ref(), last.ref(), dims.numXcds);
  builder.addi32(loop.tall->ref(), loop.tall->ref(), 1);
}

void SkinnyGEMMEmitter::emitTileLoop(const SetupScalars &scalars,
                                     LaneValues &lane, const TileLoop &loop) {
  RegisterRef blockId = kernel.workgroupIdX().ref();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("tile_loop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("tile_loop_end");
  builder.create<rocisa::SCmpLtI32>(blockId.container(),
                                    loop.numTiles.container());
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  runPersistentTile(scalars, lane, loop);
  builder.addi32(blockId, blockId, scalars.blocksX.ref());
  builder.create<rocisa::SCmpLtI32>(blockId.container(),
                                    loop.numTiles.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

// One iteration of the tile loop: the tile of the block id b, which the loop
// counter holds, runs @run_tile. Everything the tile writes is rewritten here
// before it is read: the bases, the selectors, the accumulators and the K
// state. Only the lane values and the loop invariants carry over.
void SkinnyGEMMEmitter::runPersistentTile(const SetupScalars &scalars,
                                          LaneValues &lane,
                                          const TileLoop &loop) {
  if (dims.scalesShared)
    tileBarrier();
  RegisterRange tx = sgpr();
  RegisterRange ty = sgpr();
  RegisterRange kz = sgpr();
  tile = TileIds{tx.ref(), ty.ref(), kz.ref()};
  RegisterRef blockId = kernel.workgroupIdX().ref();
  {
    std::optional<RegisterRange> remapped;
    RegisterRef logical = blockId;
    if (dims.numXcds != 1) {
      remapped = sgpr();
      logical = remapped->ref();
      remapXcd(loop, logical, blockId);
    }
    decodeTile(loop, logical);
  }
  {
    std::optional<RegisterRange> slabRot;
    std::optional<RegisterRef> rotation;
    if (dims.reordered && dims.upIters != 1) {
      slabRot = sgpr();
      slabRotation(loop, slabRot->ref());
      rotation = slabRot->ref();
    } else if (loop.blockRot) {
      rotation = loop.blockRot->ref();
    }
    computeTile(scalars, rotation);
  }
  std::optional<RegisterRange> w0Load = computeTileWideBases(lane);
  fillDataBases(scalars, lane);
  fillScaleBases(scalars, lane, w0Load ? w0Load->ref() : w0.ref());
  w0Load.reset();
  fillSelectors();
  zeroAccumulators();
  runTile();
  emitEpilogue();
}

// The previous tile's scales must be read from LDS before this tile overwrites
// them. gpu.barrier waits for every outstanding memory operation first.
void SkinnyGEMMEmitter::tileBarrier() {
  builder.createWaitCnt(WaitCounts{.vectorLoads = 0, .lds = 0});
  builder.create<rocisa::SBarrier>();
}

// @remap_xcd: tile t is the position t / X of the XCD t mod X, and the logical
// tile is the position in the contiguous range of that XCD; the first tall XCDs
// hold one more tile than the others.
void SkinnyGEMMEmitter::remapXcd(const TileLoop &loop, RegisterRef logical,
                                 RegisterRef tileIndex) {
  RegisterRange local = sgpr();
  RegisterRange xcd = sgpr();
  RegisterRange shortage = sgpr();
  divideByLiteral(builder, local.ref(), tileIndex, dims.numXcds);
  remainderByLiteral(builder, xcd.ref(), tileIndex, local.ref(), dims.numXcds);
  builder.subi32(shortage.ref(), xcd.ref(), loop.tall->ref());
  builder.create<rocisa::SMaxI32>(shortage.container(), shortage.container(),
                                  0);
  builder.muli32(logical, xcd.ref(), loop.perXcd->ref());
  builder.addi32(logical, logical, local.ref());
  builder.subi32(logical, logical, shortage.ref());
}

// Decodes the logical tile into (tx, ty, kz): kz is the plane of tiles, and
// within the plane the plain order has tx fastest while the grouped one walks
// bands of GROUP skinny tile rows ty fastest.
void SkinnyGEMMEmitter::decodeTile(const TileLoop &loop, RegisterRef logical) {
  RegisterRange pos = sgpr();
  builder.divsi32(tile.kz, logical, loop.plane.ref(), /*useIEEE=*/true);
  builder.muli32(pos.ref(), tile.kz, loop.plane.ref());
  builder.subi32(pos.ref(), logical, pos.ref());
  if (dims.group != 1) {
    decodeGroupedTile(loop, pos.ref());
    return;
  }
  builder.divsi32(tile.ty, pos.ref(), loop.tilesX.ref(), /*useIEEE=*/true);
  builder.muli32(tile.tx, tile.ty, loop.tilesX.ref());
  builder.subi32(tile.tx, pos.ref(), tile.tx);
}

// The band holds GROUP rows, fewer in the last one, so a tile at inBand is at
// column inBand / rows and row firstTy + inBand mod rows.
void SkinnyGEMMEmitter::decodeGroupedTile(const TileLoop &loop,
                                          RegisterRef pos) {
  RegisterRange band = sgpr();
  RegisterRange inBand = sgpr();
  RegisterRange firstTy = sgpr();
  RegisterRange rows = sgpr();
  builder.divsi32(band.ref(), pos, loop.bandSize->ref(), /*useIEEE=*/true);
  builder.muli32(inBand.ref(), band.ref(), loop.bandSize->ref());
  builder.subi32(inBand.ref(), pos, inBand.ref());
  multiplyBy(builder, firstTy.ref(), band.ref(), dims.group);
  builder.subi32(rows.ref(), loop.tilesY->ref(), firstTy.ref());
  builder.create<rocisa::SMinI32>(rows.container(), rows.container(),
                                  dims.group);
  builder.divsi32(tile.tx, inBand.ref(), rows.ref(), /*useIEEE=*/true);
  builder.muli32(tile.ty, tile.tx, rows.ref());
  builder.subi32(tile.ty, inBand.ref(), tile.ty);
  builder.addi32(tile.ty, tile.ty, firstTy.ref());
}

// The reordered tiles key the rotation on the wide slab kz * tilesX + tx, so
// that the tiles that share a wide column read the same K offset together.
void SkinnyGEMMEmitter::slabRotation(const TileLoop &loop, RegisterRef rot) {
  RegisterRange slab = sgpr();
  builder.muli32(slab.ref(), tile.kz, loop.tilesX.ref());
  builder.addi32(slab.ref(), slab.ref(), tile.tx);
  blockRotation(rot, slab.ref(), loop.ni->ref());
}

} // namespace rocm::tensilelite::codegen::detail
