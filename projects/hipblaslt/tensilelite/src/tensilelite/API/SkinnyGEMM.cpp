// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The skinny MXFP8 GEMM emitter, common part: the knob validation, the kernel
// names and arguments, emit, which selects the VALU path (SkinnyGEMMValu.cpp)
// or the MFMA path (SkinnyGEMMMfma.cpp), the setup of the arguments and
// descriptors that both paths share, the tile decode, the math helpers, and the
// split-K reduce kernel.

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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::skinny;

// Shared helpers.

// Every pointer is copied once into a longer-lived resource, so a preloaded
// range is dead right after the copy.
static void loadPointer(Kernel &kernel, RegisterRef dst, int32_t offset) {
  kernel.loadArgument(dst, offset);
  kernel.releasePreloadedArgument(offset);
}

// Returns whether value is a positive power of two.
bool skinny::isPowerOfTwo(int32_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// Returns log2 of a power of two.
int32_t skinny::log2Exact(int32_t value) {
  if (!isPowerOfTwo(value))
    throw std::logic_error("value is not a power of two");
  int32_t log = 0;
  while ((1 << log) < value)
    ++log;
  return log;
}

// The helpers below take non-negative operands, so each quotient is one
// division and each remainder is a - q * d, without the sign corrections that
// a floor division and a modulus need.

// dst = src / divisor.
void skinny::divideByLiteral(IRBuilder &builder, RegisterRef dst,
                             RegisterRef src, int32_t divisor) {
  if (isPowerOfTwo(divisor)) {
    builder.shrui32(dst, src, log2Exact(divisor));
    return;
  }
  builder.divsi32(dst, src, divisor, /*useIEEE=*/true);
}

// dst = src - quotient * divisor, the remainder given the quotient.
void skinny::remainderByLiteral(IRBuilder &builder, RegisterRef dst,
                                RegisterRef src, RegisterRef quotient,
                                int32_t divisor) {
  if (isPowerOfTwo(divisor)) {
    builder.andi32(dst, divisor - 1, src);
    return;
  }
  builder.muli32(dst, quotient, divisor);
  builder.subi32(dst, src, dst);
}

// dst = ceilDiv(src, divisor). dst must not alias src.
void skinny::ceilDivByLiteral(IRBuilder &builder, RegisterRef dst,
                              RegisterRef src, int32_t divisor) {
  if (isPowerOfTwo(divisor)) {
    builder.addi32(dst, src, divisor - 1);
    builder.shrui32(dst, dst, log2Exact(divisor));
    return;
  }
  RegisterRange sum = builder.allocateSgpr(1);
  builder.addi32(sum.ref(), src, divisor - 1);
  builder.divsi32(dst, sum.ref(), divisor, /*useIEEE=*/true);
}

// dst = src mod divisor, with a scratch quotient when divisor is not a power of
// two.
void skinny::moduloByLiteral(IRBuilder &builder, RegisterRef dst,
                             RegisterRef src, int32_t divisor) {
  if (isPowerOfTwo(divisor)) {
    builder.andi32(dst, divisor - 1, src);
    return;
  }
  RegisterRange quotient = builder.allocateSgpr(1);
  divideByLiteral(builder, quotient.ref(), src, divisor);
  remainderByLiteral(builder, dst, src, quotient.ref(), divisor);
}

// The quotient and the remainder of an exact division of non-negative
// operands.
void skinny::divMod(IRBuilder &builder, RegisterRef quotient,
                    RegisterRef remainder, RegisterRef dividend,
                    RegisterRef divisor) {
  builder.divsi32(quotient, dividend, divisor, /*useIEEE=*/true);
  builder.muli32(remainder, quotient, divisor);
  builder.subi32(remainder, dividend, remainder);
}

SkinnyDims::SkinnyDims(const SkinnyGEMMConfig &config)
    : bm(config.bm), tn(config.tn), lanesK(config.lanesK), waves(config.waves),
      prefetch(config.prefetch), grid3d(config.grid3d), group(config.group),
      numXcds(config.numXcds), useBeta(config.useBeta),
      useSplitK(config.useSplitK),
      reordered(config.group != 1 || config.numXcds != 1),
      upIters(config.upIters) {
  mfma = config.mfma;
  lds = config.lds;
  columns = bm + tn;
  accumulators = bm * tn;
  columnGroups = 64 / lanesK;
  uniformWide = waves == 1 && lanesK == 64;
  wideTiles = mfma ? tn : 0;
  skinnyTiles = mfma ? bm / 16 : 0;
  tileWide = mfma ? 16 * tn : waves * columnGroups * tn;
  xcdRun = isPowerOfTwo(tileWide) && tileWide < scaleRowColumns
               ? scaleRowColumns / tileWide
               : 1;
  ki = mfma ? 256 : 64 * lanesK;
  subgroupLanes = std::min(lanesK, 8);
  halfSubgroup = subgroupLanes / 2;
  log2LanesK = log2Exact(lanesK);
  log2ColumnGroups = log2Exact(columnGroups);
  log2Subgroup = log2Exact(subgroupLanes);
  log2Half = log2Exact(halfSubgroup);
  log2Ki = log2Exact(ki);
  slots = std::max(prefetch, 1);
  scalesShared = upIters != 0 && waves > 1 && columnGroups < 4;
  roleSlots = scalesShared ? waves * columnGroups : columnGroups;
  scaleLoads = upIters != 0 ? 0 : columns;
  loadsPerSlot = scaleLoads + chunks * columns;
  ringBytes = lds ? slots * (wideTiles + skinnyTiles) * stagedTileBytes : 0;
  reduceBytes = mfma ? (waves - 1) * tn * (bm / 16) * 1024 : 0;
  remainder = mfma && grid3d && isPowerOfTwo(tn);
  ldsBytes = mfma           ? waves * ringBytes + reduceBytes
             : scalesShared ? upIters * columns * waves * 256
                            : 0;
  chunkStride = 16 * subgroupLanes;
}

// Setup.

// Bytes of an operand of cols columns: (cols - 1) * ld + alignUp(k, 4), as
// the last column is padded to 4 bytes, not to ld.
static void operandBytes(IRBuilder &builder, RegisterRef dst, RegisterRef cols,
                         RegisterRef ld, RegisterRef alignedK) {
  builder.addi32(dst, cols, -1);
  builder.muli32(dst, dst, ld);
  builder.addi32(dst, dst, alignedK);
}

// Bytes of the scale matrix of cols rows: ceilDiv(cols, 32) row groups.
static void scaleBytes(IRBuilder &builder, RegisterRef dst, RegisterRef cols,
                       RegisterRef rowGroup) {
  builder.addi32(dst, cols, 31);
  builder.shrui32(dst, dst, 5);
  builder.muli32(dst, dst, rowGroup);
}

// Bytes of a column-major bf16 matrix of m rows and n columns whose column
// stride ldBytes is in bytes: (n - 1) * ldBytes + 2 * m.
static void bf16MatrixBytes(IRBuilder &builder, RegisterRef dst, RegisterRef n,
                            RegisterRef ldBytes, RegisterRef m,
                            RegisterRef scratch) {
  builder.addi32(dst, n, -1);
  builder.muli32(dst, dst, ldBytes);
  builder.shli32(scratch, m, 1);
  builder.addi32(dst, dst, scratch);
}

// Branches to label when beta is zero. The sign bit is masked so that -0.0
// counts as zero.
void skinny::branchIfBetaZero(IRBuilder &builder, RegisterRef beta,
                              RegisterRef scratch,
                              const std::shared_ptr<rocisa::Label> &label) {
  builder.andi32(scratch, beta, 0x7fffffff);
  builder.create<rocisa::SCmpEQU32>(scratch.container(), 0);
  builder.create<rocisa::SCBranchSCC1>(label->getLabelName());
}

SkinnyGEMMEmitter::SetupScalars::SetupScalars(IRBuilder &builder,
                                              const SkinnyDims &dims)
    : blocksX(dims.grid3d
                  ? std::nullopt
                  : std::optional<RegisterRange>(builder.allocateSgpr(1))),
      rowGroup(builder.allocateSgpr(1)), ldS(builder.allocateSgpr(1)),
      ldW(builder.allocateSgpr(1)),
      splits(!dims.grid3d && dims.useSplitK
                 ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                 : std::nullopt),
      wholeTiles(dims.remainder
                     ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                     : std::nullopt),
      pieces(dims.remainder
                 ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                 : std::nullopt) {}

// Allocates an SGPR range only when enabled, for the registers that a knob
// adds.
static std::optional<RegisterRange> allocateIf(IRBuilder &builder, bool enabled,
                                               int32_t count,
                                               int32_t alignment = 1) {
  if (!enabled)
    return std::nullopt;
  return builder.allocateSgpr(count, alignment);
}

SkinnyGEMMEmitter::SkinnyGEMMEmitter(Kernel &kernel,
                                     const SkinnyGEMMConfig &config,
                                     const SkinnyArguments &arguments)
    : kernel(kernel), builder(kernel.builder()), dims(config),
      arguments(arguments), sRsrc(builder.allocateSgpr(4, 4)),
      wRsrc(builder.allocateSgpr(4, 4)), sSclRsrc(builder.allocateSgpr(4, 4)),
      wSclRsrc(builder.allocateSgpr(4, 4)), dRsrc(builder.allocateSgpr(4, 4)),
      wsRsrc(allocateIf(builder, dims.useSplitK, 4, 4)),
      cRsrc(allocateIf(builder, dims.useBeta, 4, 4)),
      alpha(builder.allocateSgpr(2, 2)),
      beta(allocateIf(builder, dims.useBeta, 1)), k(builder.allocateSgpr(1)),
      spk(allocateIf(builder, dims.useSplitK, 1)),
      mn(allocateIf(builder, dims.useSplitK, 1)),
      dStrideS(builder.allocateSgpr(1)), dStrideW(builder.allocateSgpr(1)),
      wsStrideS(allocateIf(builder, dims.useSplitK, 1)),
      wsStrideW(allocateIf(builder, dims.useSplitK, 1)),
      cStrideS(allocateIf(builder, dims.useBeta, 1)),
      cStrideW(allocateIf(builder, dims.useBeta, 1)),
      skinnyCols(builder.allocateSgpr(1)), wideCols(builder.allocateSgpr(1)),
      contigS(builder.allocateSgpr(1)) {}

RegisterRef SkinnyGEMMEmitter::splitExtent() const {
  return spk ? spk->ref() : k.ref();
}

// The xcdRun tiles that share a scale line run on one XCD, so its L2 fetches
// the line once instead of every XCD fetching it. The dispatcher sends
// workgroup x to XCD x mod X, so within each full block of X * xcdRun ids the
// tile is (x mod X) * xcdRun + (x / X) mod xcdRun; the tail of the grid keeps
// the identity so that every tile still runs exactly once. A function for the
// name of the XCD remap formula.
void skinny::groupTilesPerXcd(IRBuilder &builder, const SkinnyDims &dims,
                              int32_t xcds, RegisterRef tx,
                              RegisterRef tilesX) {
  int32_t span = xcds * dims.xcdRun;
  RegisterRange logical = builder.allocateSgpr(1);
  RegisterRange term = builder.allocateSgpr(1);
  builder.shrui32(logical.ref(), tx, log2Exact(xcds));
  builder.andi32(logical.ref(), dims.xcdRun - 1, logical.ref());
  builder.andi32(term.ref(), xcds - 1, tx);
  builder.shli32(term.ref(), term.ref(), log2Exact(dims.xcdRun));
  builder.addi32(logical.ref(), logical.ref(), term.ref());
  builder.andi32(term.ref(), -span, tx);
  builder.addi32(logical.ref(), logical.ref(), term.ref());
  builder.andi32(term.ref(), -span, tilesX);
  // SCC must survive from the compare to the select.
  builder.create<rocisa::SCmpLtU32>(tx.container(), term.container());
  builder.selectb32(tx, logical.ref(), tx);
}

// Splits the flat tile index into the split kz, the plane of tiles, with
// useSplitK, and within the plane the plain order with tx fastest.
void skinny::decodePlainTile(IRBuilder &builder, RegisterRef tile,
                             const TileIds &ids,
                             std::optional<RegisterRef> plane,
                             RegisterRef tilesX) {
  if (!ids.kz) {
    divMod(builder, ids.ty, ids.tx, tile, tilesX);
    return;
  }
  RegisterRange pos = builder.allocateSgpr(1);
  divMod(builder, *ids.kz, pos.ref(), tile, *plane);
  divMod(builder, ids.ty, ids.tx, pos.ref(), tilesX);
}

// Decodes the tile of the block id into (tx, ty), and kz with useSplitK, after
// the remap of the XCDs: kz is the plane of tiles, and within the plane the
// plain order has tx fastest while the grouped one walks bands of GROUP skinny
// tile rows ty fastest. A function for the name of the decode formula.
void skinny::decodeTile(IRBuilder &builder, const SkinnyDims &dims,
                        RegisterRef blockId, const TileIds &ids,
                        const TileLoop &loop) {
  std::optional<RegisterRange> remapped;
  RegisterRef logical = blockId;
  if (dims.numXcds != 1) {
    // Tile t is the position t / X of the XCD t mod X, and the logical tile is
    // the position in the contiguous range of that XCD; the first tall XCDs
    // hold one more tile than the others.
    remapped = builder.allocateSgpr(1);
    logical = remapped->ref();
    RegisterRange local = builder.allocateSgpr(1);
    RegisterRange xcd = builder.allocateSgpr(1);
    RegisterRange shortage = builder.allocateSgpr(1);
    divideByLiteral(builder, local.ref(), blockId, dims.numXcds);
    remainderByLiteral(builder, xcd.ref(), blockId, local.ref(), dims.numXcds);
    builder.subi32(shortage.ref(), xcd.ref(), loop.tall->ref());
    builder.maxsi32(shortage.ref(), shortage.ref(), 0);
    builder.muli32(logical, xcd.ref(), loop.perXcd->ref());
    builder.addi32(logical, logical, local.ref());
    builder.subi32(logical, logical, shortage.ref());
  }
  if (dims.group == 1) {
    std::optional<RegisterRef> plane;
    if (loop.plane)
      plane = loop.plane->ref();
    decodePlainTile(builder, logical, ids, plane, loop.tilesX.ref());
    return;
  }
  // Without split-K the whole flat index is the position in the plane.
  RegisterRef pos = logical;
  std::optional<RegisterRange> planePos;
  if (ids.kz) {
    planePos = builder.allocateSgpr(1);
    pos = planePos->ref();
    divMod(builder, *ids.kz, pos, logical, loop.plane->ref());
  }
  // The band holds GROUP rows, fewer in the last one, so a tile at inBand is at
  // column inBand / rows and row firstTy + inBand mod rows.
  RegisterRange band = builder.allocateSgpr(1);
  RegisterRange inBand = builder.allocateSgpr(1);
  RegisterRange firstTy = builder.allocateSgpr(1);
  RegisterRange rows = builder.allocateSgpr(1);
  divMod(builder, band.ref(), inBand.ref(), pos, loop.bandSize->ref());
  builder.muli32(firstTy.ref(), band.ref(), dims.group);
  builder.subi32(rows.ref(), loop.tilesY->ref(), firstTy.ref());
  builder.minsi32(rows.ref(), rows.ref(), dims.group);
  divMod(builder, ids.tx, ids.ty, inBand.ref(), rows.ref());
  builder.addi32(ids.ty, ids.ty, firstTy.ref());
}

// A label of the 3-D prologue, named after what it skips. It does not take an
// id from the builder, so that the labels of the rest of the kernel keep their
// numbers; each name is used once per kernel.
static std::shared_ptr<rocisa::Label> prologueLabel(const std::string &name) {
  return std::make_shared<rocisa::Label>(name, "");
}

// The tile rows that the decode of a few rows takes by subtraction.
static constexpr int32_t maxSubtractedRows = 4;

// ty = tx / tilesX and tx = tx mod tilesX for tx < maxSubtractedRows * tilesX,
// by conditional subtractions of the tiles per row: exact, and without the
// round trip through the VALU of an exact division.
static void subtractRows(IRBuilder &builder, const TileIds &ids,
                         RegisterRef tilesX) {
  RegisterRange reduced = builder.allocateSgpr(1);
  RegisterRange next = builder.allocateSgpr(1);
  for (int32_t row = 1; row < maxSubtractedRows; ++row) {
    builder.subi32(reduced.ref(), ids.tx, tilesX);
    builder.addi32(next.ref(), ids.ty, 1);
    // SCC must survive from the compare to the selects.
    builder.create<rocisa::SCmpGeU32>(ids.tx.container(), tilesX.container());
    builder.selectb32(ids.tx, reduced.ref(), ids.tx);
    builder.selectb32(ids.ty, next.ref(), ids.ty);
  }
}

// The flat id of the 3-D launch is the linearized id of the old (tilesX,
// tilesY, splits) grid, so the tiles and their XCDs are those of that grid.
// The decode sits on the critical path before the first loads, so the common
// shapes skip their exact divisions with uniform branches: one split (k <=
// spk) needs none, and up to maxSubtractedRows tile rows only subtract. Without
// useSplitK the flat id has no split to decode.
void skinny::decodeFlatTile(IRBuilder &builder, const SkinnyDims &dims,
                            int32_t xcds, RegisterRef tile, const TileIds &ids,
                            RegisterRef wideCols, RegisterRef skinnyCols,
                            RegisterRef k, std::optional<RegisterRef> spk) {
  RegisterRange tilesX = builder.allocateSgpr(1);
  RegisterRange pos = builder.allocateSgpr(1);
  ceilDivByLiteral(builder, tilesX.ref(), wideCols, dims.tileWide);
  if (spk) {
    std::shared_ptr<rocisa::Label> splitDone = prologueLabel("flat_split_done");
    builder.movb32(*ids.kz, 0);
    builder.movb32(pos.ref(), tile);
    builder.create<rocisa::SCmpLeI32>(k.container(), spk->container());
    builder.create<rocisa::SCBranchSCC1>(splitDone->getLabelName());
    // ty only receives its value after the plane is consumed, so it holds the
    // plane meanwhile.
    RegisterRef plane = ids.ty;
    ceilDivByLiteral(builder, plane, skinnyCols, dims.bm);
    builder.muli32(plane, plane, tilesX.ref());
    divMod(builder, *ids.kz, pos.ref(), tile, plane);
    builder.insert(splitDone);
  } else {
    // pos stays apart from tile, which ids.tx may alias: divMod writes the
    // remainder before it reads the dividend.
    builder.movb32(pos.ref(), tile);
  }
  std::shared_ptr<rocisa::Label> rowDone = prologueLabel("flat_row_done");
  std::shared_ptr<rocisa::Label> manyRows = prologueLabel("flat_many_rows");
  builder.movb32(ids.ty, 0);
  builder.movb32(ids.tx, pos.ref());
  builder.create<rocisa::SCmpLeU32>(skinnyCols.container(), dims.bm);
  builder.create<rocisa::SCBranchSCC1>(rowDone->getLabelName());
  builder.create<rocisa::SCmpLeU32>(skinnyCols.container(),
                                    maxSubtractedRows * dims.bm);
  builder.create<rocisa::SCBranchSCC0>(manyRows->getLabelName());
  subtractRows(builder, ids, tilesX.ref());
  builder.create<rocisa::SBranch>(rowDone->getLabelName());
  builder.insert(manyRows);
  divMod(builder, ids.ty, ids.tx, pos.ref(), tilesX.ref());
  builder.insert(rowDone);
  if (dims.xcdRun > 1)
    groupTilesPerXcd(builder, dims, xcds, ids.tx, tilesX.ref());
}

// Emits the kernel of the path that the knobs select.
void SkinnyGEMMEmitter::emit() {
  if (dims.mfma)
    emitMfma();
  else
    emitValu();
}

// Loads the kernel arguments, derives the buffer sizes, makes the descriptors
// and selects the operands, strides and extents of the skinny and the wide
// dimension. independent runs while the loads are in flight.
void SkinnyGEMMEmitter::emitArguments(
    SetupScalars &scalars, const std::function<void()> &independent) {
  // The scalars that only the descriptors and the dispatch read.
  RegisterRange ldD = builder.allocateSgpr(1), ldA = builder.allocateSgpr(1),
                ldB = builder.allocateSgpr(1), m = builder.allocateSgpr(1),
                n = builder.allocateSgpr(1);
  RegisterRange aPtr = builder.allocateSgpr(2, 2),
                bPtr = builder.allocateSgpr(2, 2),
                sclAPtr = builder.allocateSgpr(2, 2),
                sclBPtr = builder.allocateSgpr(2, 2);
  RegisterRange aBytes = builder.allocateSgpr(1),
                bBytes = builder.allocateSgpr(1),
                sclABytes = builder.allocateSgpr(1),
                sclBBytes = builder.allocateSgpr(1);
  std::optional<RegisterRange> ldC;
  if (dims.useBeta)
    ldC = builder.allocateSgpr(1);
  // The pointers load straight into the words 0 and 1 of the resources of D
  // and the workspace; the others wait in pairs for the path dispatch.
  if (scalars.blocksX)
    kernel.loadArgument(scalars.blocksX->ref(), arguments.blocksX);
  if (scalars.pieces) {
    kernel.loadArgument(scalars.wholeTiles->ref(), arguments.wholeTiles);
    kernel.loadArgument(scalars.pieces->ref(), arguments.pieces);
  }
  if (spk)
    kernel.loadArgument(spk->ref(), arguments.spk);
  kernel.loadArgument(alpha.ref(0, 1), arguments.alpha);
  kernel.loadArgument(k.ref(), arguments.k);
  kernel.loadArgument(n.ref(), arguments.n);
  kernel.loadArgument(m.ref(), arguments.m);
  kernel.loadArgument(ldB.ref(), arguments.ldB);
  kernel.loadArgument(ldA.ref(), arguments.ldA);
  kernel.loadArgument(ldD.ref(), arguments.ldD);
  if (wsRsrc)
    loadPointer(kernel, wsRsrc->ref(0, 2), arguments.ws);
  else
    // ws is in the ABI but only split-K reads it, so its preloaded range is not
    // held.
    kernel.releasePreloadedArgument(arguments.ws);
  loadPointer(kernel, sclBPtr.ref(), arguments.scalesB);
  loadPointer(kernel, sclAPtr.ref(), arguments.scalesA);
  loadPointer(kernel, bPtr.ref(), arguments.b);
  loadPointer(kernel, aPtr.ref(), arguments.a);
  loadPointer(kernel, dRsrc.ref(0, 2), arguments.d);
  if (dims.useBeta) {
    kernel.loadArgument(beta->ref(), arguments.beta);
    kernel.loadArgument(ldC->ref(), arguments.ldC);
    loadPointer(kernel, cRsrc->ref(0, 2), arguments.c);
  } else {
    // C is in the ABI but never read, so its preloaded range is not held.
    kernel.releasePreloadedArgument(arguments.c);
  }
  if (independent)
    independent();
  // Scalar loads return out of order, so only lgkmcnt(0) guarantees any of
  // them.
  if (kernel.loadsArguments())
    builder.createWaitCnt(WaitCounts{.scalarMemory = 0});

  // Buffer sizes.
  {
    RegisterRange alignedK = builder.allocateSgpr(1);
    builder.addi32(alignedK.ref(), k, 3);
    builder.shrui32(alignedK.ref(), alignedK.ref(), 2);
    builder.shli32(alignedK.ref(), alignedK.ref(), 2);
    operandBytes(builder, aBytes.ref(), m.ref(), ldA.ref(), alignedK.ref());
    operandBytes(builder, bBytes.ref(), n.ref(), ldB.ref(), alignedK.ref());
    // A row group of the scales holds k rounded up to 256 bytes.
    builder.addi32(scalars.rowGroup.ref(), k, 255);
    builder.shrui32(scalars.rowGroup.ref(), scalars.rowGroup.ref(), 8);
    builder.shli32(scalars.rowGroup.ref(), scalars.rowGroup.ref(), 8);
    scaleBytes(builder, sclABytes.ref(), m.ref(), scalars.rowGroup.ref());
    scaleBytes(builder, sclBBytes.ref(), n.ref(), scalars.rowGroup.ref());
    bf16MatrixBytes(builder, dRsrc.ref(2), n.ref(), ldD.ref(), m.ref(),
                    alignedK.ref());
    if (dims.useBeta)
      bf16MatrixBytes(builder, cRsrc->ref(2), n.ref(), ldC->ref(), m.ref(),
                      alignedK.ref());
    // The workspace holds splits * m * n f32 elements. The persistent launch
    // sizes it here; the 3-D launch sets the end of each slice in the
    // epilogue.
    if (scalars.splits) {
      // The exact signed division, evaluated at run time.
      RegisterRef splits = scalars.splits->ref();
      builder.addi32(splits, k.ref(), spk->ref());
      builder.addi32(splits, splits, -1);
      builder.divsi32(splits, splits, spk->ref(), /*useIEEE=*/true);
      RegisterRef wsBytes = wsRsrc->ref(2);
      builder.muli32(wsBytes, m.ref(), splits);
      builder.muli32(wsBytes, wsBytes, n.ref());
      builder.shli32(wsBytes, wsBytes, 2);
    }
    if (mn)
      builder.muli32(mn->ref(), m.ref(), n.ref());
  }

  // Path dispatch: selects the operands, strides and extents of the skinny
  // and the wide dimension on m <= n. The s_cselects follow one s_cmp and no
  // SALU between them writes SCC. The f32 workspace column is 4 * m bytes,
  // computed before the s_cmp as the shift writes SCC.
  if (wsStrideW)
    builder.shli32(wsStrideW->ref(), m.ref(), 2);
  builder.create<rocisa::SCmpLeU32>(m.container(), n.container());
  builder.selectb64(sRsrc.ref(0, 2), aPtr.ref(), bPtr.ref());
  builder.selectb32(sRsrc.ref(2), aBytes.ref(), bBytes.ref());
  builder.selectb64(wRsrc.ref(0, 2), bPtr.ref(), aPtr.ref());
  builder.selectb32(wRsrc.ref(2), bBytes.ref(), aBytes.ref());
  builder.selectb64(sSclRsrc.ref(0, 2), sclAPtr.ref(), sclBPtr.ref());
  builder.selectb32(sSclRsrc.ref(2), sclABytes.ref(), sclBBytes.ref());
  builder.selectb64(wSclRsrc.ref(0, 2), sclBPtr.ref(), sclAPtr.ref());
  builder.selectb32(wSclRsrc.ref(2), sclBBytes.ref(), sclABytes.ref());
  builder.selectb32(scalars.ldS.ref(), ldA.ref(), ldB.ref());
  builder.selectb32(scalars.ldW.ref(), ldB.ref(), ldA.ref());
  // The output strides are in bytes: the unit stride is one bf16 (2) or f32
  // (4) element.
  builder.selectb32(dStrideS.ref(), 2, ldD.ref());
  builder.selectb32(dStrideW.ref(), ldD.ref(), 2);
  if (dims.useBeta) {
    builder.selectb32(cStrideS->ref(), 2, ldC->ref());
    builder.selectb32(cStrideW->ref(), ldC->ref(), 2);
  }
  if (wsStrideW) {
    builder.selectb32(wsStrideS->ref(), 4, wsStrideW->ref());
    builder.selectb32(wsStrideW->ref(), wsStrideW->ref(), 4);
  }
  builder.selectb32(contigS.ref(), 1, 0);
  builder.minui32(skinnyCols.ref(), m.ref(), n.ref());
  builder.maxui32(wideCols.ref(), m.ref(), n.ref());
  for (RegisterRange *rsrc : {&sRsrc, &wRsrc, &sSclRsrc, &wSclRsrc})
    builder.makeRsrc(rsrc->ref(), rsrc->ref(0, 2), rsrc->ref(2));
  std::vector<RegisterRange *> rsrcs = {&dRsrc};
  if (wsRsrc)
    rsrcs.push_back(&*wsRsrc);
  if (dims.useBeta)
    rsrcs.push_back(&*cRsrc);
  for (RegisterRange *rsrc : rsrcs)
    builder.makeRsrc(rsrc->ref(), rsrc->ref(0, 2), rsrc->ref(2));
}

// Split-K reduce.

// Threads per block of the reduce kernel.
static constexpr int32_t reduceBlock = 256;
// Splits whose ws loads the reduce kernel keeps in flight together.
static constexpr int32_t reduceUnroll = 4;

// off = j * ldBytes + 2 * i for the thread's element, from the thread id.
static void columnMajorOffset(Kernel &kernel, RegisterRef off, RegisterRef ld,
                              RegisterRef tmp) {
  IRBuilder &builder = kernel.builder();
  builder.shli32(off, kernel.workitemId().ref(), 1);
  builder.shli32(tmp, kernel.workgroupIdX().ref(), 9);
  builder.addi32(off, tmp, off);
  builder.muli32(tmp, kernel.workgroupIdY(), ld);
  builder.addi32(off, tmp, off);
}

namespace {

struct ReduceArguments {
  int32_t d;
  int32_t ws;
  int32_t c;
  int32_t ldD;
  int32_t m;
  int32_t n;
  int32_t splits;
  int32_t ldC;
  int32_t alpha;
  int32_t beta;
};

// The registers of the split loop, which only the descriptor sizes and the loop
// read.
struct SplitLoop {
  RegisterRange splits, wsRsrc, rowBytes, sliceBytes;
};

// Emits the split-K reduce kernel: emit loads the arguments and
// builds the descriptors, then runs the split sum and the store. The members
// live from the argument loads to the final store; the values that only travel
// between the phases are locals of emit.
class SplitKReduceEmitter {
public:
  SplitKReduceEmitter(Kernel &kernel, const ReduceArguments &offsets)
      : kernel(kernel), builder(kernel.builder()), offsets(offsets),
        ldD(builder.allocateSgpr(1)), ldC(builder.allocateSgpr(1)),
        alpha(builder.allocateSgpr(1)), beta(builder.allocateSgpr(1)),
        dRsrc(builder.allocateSgpr(4, 4)), cRsrc(builder.allocateSgpr(4, 4)) {}

  void emit() {
    // The scratch of every part.
    RegisterRange tmp = builder.allocateSgpr(1);
    SplitLoop loop{builder.allocateSgpr(1), builder.allocateSgpr(4, 4),
                   builder.allocateSgpr(1), builder.allocateSgpr(1)};
    // The row i, then the sum over the splits.
    RegisterRange row = builder.allocateVgpr(1);
    // The load and store address of the thread's element.
    RegisterRange value = builder.allocateVgpr(1);
    RegisterRange n = builder.allocateSgpr(1);
    RegisterRange m = builder.allocateSgpr(1);
    // The pointers load straight into words 0 and 1 of their descriptors.
    kernel.loadArgument(loop.splits.ref(), offsets.splits);
    kernel.loadArgument(n.ref(), offsets.n);
    kernel.loadArgument(m.ref(), offsets.m);
    kernel.loadArgument(ldD.ref(), offsets.ldD);
    kernel.loadArgument(ldC.ref(), offsets.ldC);
    kernel.loadArgument(alpha.ref(), offsets.alpha);
    kernel.loadArgument(beta.ref(), offsets.beta);
    loadPointer(kernel, loop.wsRsrc.ref(0, 2), offsets.ws);
    loadPointer(kernel, dRsrc.ref(0, 2), offsets.d);
    loadPointer(kernel, cRsrc.ref(0, 2), offsets.c);
    // The block is (256, 1, 1), so the packed y and z fields of v0 are zero and
    // no mask is needed.
    builder.shli32(tmp.ref(), kernel.workgroupIdX().ref(), 8);
    builder.addi32(row.ref(), tmp.ref(), kernel.workitemId().ref());
    // Scalar loads return out of order, so only lgkmcnt(0) guarantees any of
    // the arguments.
    if (kernel.loadsArguments())
      builder.createWaitCnt(WaitCounts{.scalarMemory = 0});
    // The row i and inRange (in vcc); iLoad then overwrites i.
    builder.create<rocisa::VCmpGtU32>(std::make_shared<rocisa::VCC>(),
                                      m.container(), row.container());
    builder.subi32(tmp.ref(), m.ref(), 1);
    // Both operands are below 2^31 because m <= 2^20 and i < m + 256, so the
    // signed min equals the unsigned one; rocisa has no v_min_u32.
    builder.minsi32(row.ref(), tmp.ref(), row.ref());
    // Descriptors of ws, D and C, which bound the accesses.
    builder.shli32(loop.rowBytes.ref(), m.ref(), 2);
    builder.muli32(loop.sliceBytes.ref(), n.ref(), loop.rowBytes.ref());
    builder.muli32(loop.wsRsrc.ref(2), loop.splits.ref(),
                   loop.sliceBytes.ref());
    bf16MatrixBytes(builder, dRsrc.ref(2), n.ref(), ldD.ref(), m.ref(),
                    tmp.ref());
    bf16MatrixBytes(builder, cRsrc.ref(2), n.ref(), ldC.ref(), m.ref(),
                    tmp.ref());
    for (RegisterRange *rsrc : {&loop.wsRsrc, &dRsrc, &cRsrc})
      builder.makeRsrc(rsrc->ref(), rsrc->ref(0, 2), rsrc->ref(2));
    RegisterRange c = builder.allocateVgpr(1);
    prefetchC(tmp.ref(), c.ref());
    sumSplits(loop, tmp.ref(), row.ref(), value.ref());
    storeBf16(tmp.ref(), row.ref(), value.ref(), c.ref());
  }

private:
  // Loads the thread's C element into c unless beta is zero, before the split
  // loop, so that its latency overlaps the sum. Every lane loads, as exec is
  // only narrowed for the store; a lane outside the matrix loads a value it
  // never uses. The load precedes every ws load, so the in-order vmcnt wait of
  // any ws load, and splits >= 1 makes one, also completes it.
  void prefetchC(RegisterRef tmp, RegisterRef c) {
    std::shared_ptr<rocisa::Label> noBeta = builder.createLabel("no_c");
    branchIfBetaZero(builder, beta.ref(), tmp, noBeta);
    // The load overwrites its address register.
    columnMajorOffset(kernel, c, ldC.ref(), tmp);
    builder.loadBuffer(c, cRsrc.ref(), c, 0, MemoryWidth::b16);
    builder.insert(noBeta);
  }

  // sum = the f32 sum over the splits, added in split order. Each lane loads
  // its element of ws at (z * m * n + j * m + iLoad) * 4, so splits >= 1 is a
  // precondition. Groups of reduceUnroll splits keep their loads in flight
  // together; a loop of single splits takes the rest.
  void sumSplits(const SplitLoop &loop, RegisterRef tmp, RegisterRef row,
                 RegisterRef value) {
    RegisterRange base = builder.allocateVgpr(1);
    RegisterRange split = builder.allocateSgpr(1);
    RegisterRange groupEnd = builder.allocateSgpr(1);
    RegisterRange more = builder.allocateVgpr(reduceUnroll - 1);
    std::vector<RegisterRef> values = {value};
    for (int32_t i = 0; i + 1 < reduceUnroll; ++i)
      values.push_back(more.ref(i));
    // The row register is dead after base, so it holds the sum.
    RegisterRef sum = row;
    builder.shli32(row, row, 2);
    builder.muli32(tmp, kernel.workgroupIdY(), loop.rowBytes.ref());
    builder.addi32(base.ref(), tmp, row);
    builder.movb32(split.ref(), 0);
    builder.movb32(sum, 0);
    // The groups start below splits - (reduceUnroll - 1).
    builder.addi32(groupEnd.ref(), loop.splits.ref(), 1 - reduceUnroll);
    std::shared_ptr<rocisa::Label> group = builder.createLabel("group_loop");
    std::shared_ptr<rocisa::Label> groupExit =
        builder.createLabel("group_loop_end");
    builder.create<rocisa::SCmpLtI32>(split.container(), groupEnd.container());
    builder.create<rocisa::SCBranchSCC0>(groupExit->getLabelName());
    builder.insert(group);
    builder.muli32(tmp, split.ref(), loop.sliceBytes.ref());
    for (int32_t i = 0; i < reduceUnroll; ++i) {
      if (i > 0)
        builder.addi32(tmp, tmp, loop.sliceBytes.ref());
      builder.addi32(values[i], tmp, base.ref());
      builder.loadBuffer(values[i], loop.wsRsrc.ref(), values[i], 0,
                         MemoryWidth::b32);
    }
    // The loads return in order, so the load i is complete once the later
    // ones are the only outstanding loads.
    for (int32_t i = 0; i < reduceUnroll; ++i) {
      builder.createWaitCnt(WaitCounts{.vectorLoads = reduceUnroll - 1 - i});
      builder.create<rocisa::VAddF32>(sum.container(), sum.container(),
                                      values[i].container());
    }
    builder.addi32(split.ref(), split.ref(), reduceUnroll);
    builder.create<rocisa::SCmpLtI32>(split.container(), groupEnd.container());
    builder.create<rocisa::SCBranchSCC1>(group->getLabelName());
    builder.insert(groupExit);
    std::shared_ptr<rocisa::Label> body = builder.createLabel("loop");
    std::shared_ptr<rocisa::Label> exit = builder.createLabel("loop_end");
    builder.create<rocisa::SCmpLtI32>(split.container(),
                                      loop.splits.container());
    builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
    builder.insert(body);
    builder.muli32(tmp, split.ref(), loop.sliceBytes.ref());
    builder.addi32(value, tmp, base.ref());
    builder.loadBuffer(value, loop.wsRsrc.ref(), value, 0, MemoryWidth::b32);
    // The load is the newest vector memory operation, so nothing is issued
    // after it and vmcnt waits down to 0.
    builder.createWaitCnt(WaitCounts{.vectorLoads = 0});
    builder.create<rocisa::VAddF32>(sum.container(), sum.container(),
                                    value.container());
    builder.addi32(split.ref(), split.ref(), 1);
    builder.create<rocisa::SCmpLtI32>(split.container(),
                                      loop.splits.container());
    builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
    builder.insert(exit);
  }

  // Computes D = bf16(fma(beta, C, alpha * sum)) and stores it where inRange,
  // with c from prefetchC.
  void storeBf16(RegisterRef tmp, RegisterRef row, RegisterRef value,
                 RegisterRef c) {
    RegisterRange savedExec = builder.allocateSgpr(2, 2);
    RegisterRef sum = row;
    builder.create<rocisa::VMulF32>(sum.container(), alpha.container(),
                                    sum.container());
    std::shared_ptr<rocisa::Label> skip = builder.createLabel("skip");
    builder.create<rocisa::SAndSaveExecB64>(savedExec.container(),
                                            std::make_shared<rocisa::VCC>());
    builder.create<rocisa::SCBranchVCCZ>(skip->getLabelName());
    // The sum becomes fma(beta, C, sum), skipped, as C was not loaded, when
    // beta is zero. The sign bit is masked so that -0.0 counts as zero.
    std::shared_ptr<rocisa::Label> noBeta = builder.createLabel("no_beta");
    branchIfBetaZero(builder, beta.ref(), tmp, noBeta);
    // bf16 widens to f32 by a left shift, exactly.
    builder.shli32(c, c, 16);
    builder.create<rocisa::VMacF32>(sum.container(), beta.container(),
                                    c.container());
    builder.insert(noBeta);
    // Rounds to nearest even; the store keeps the low half.
    builder.create<rocisa::VCvtPkF32toBF16>(sum.container(), sum.container(),
                                            sum.container());
    // The load address register is dead.
    RegisterRef dOff = value;
    columnMajorOffset(kernel, dOff, ldD.ref(), tmp);
    builder.storeBuffer(sum, dRsrc.ref(), dOff, 0, MemoryWidth::b16);
    builder.insert(skip);
    builder.create<rocisa::SMovB64>(std::make_shared<rocisa::EXEC>(),
                                    savedExec.container());
    builder.createEndProgram();
  }

  Kernel &kernel;
  IRBuilder &builder;
  ReduceArguments offsets;
  RegisterRange ldD, ldC, alpha, beta, dRsrc, cRsrc;
};

} // namespace

// Public entry points.

static void require(bool condition, const char *knob, int32_t value,
                    const char *rule) {
  if (condition)
    return;
  throw std::invalid_argument(std::string(knob) + " must be " + rule +
                              ", got " + std::to_string(value));
}

static void validate(const SkinnyGEMMConfig &config) {
  require(config.lanesK == 4 || config.lanesK == 8 || config.lanesK == 16 ||
              config.lanesK == 64,
          "lanesK", config.lanesK, "4, 8, 16 or 64");
  require(config.bm >= 1, "bm", config.bm, "at least 1");
  require(config.tn >= 1, "tn", config.tn, "at least 1");
  // A block of 64 * waves lanes cannot exceed 1024 lanes.
  require(config.waves >= 1 && config.waves <= 16, "waves", config.waves,
          "between 1 and 16");
  require(config.prefetch >= 0, "prefetch", config.prefetch, "at least 0");
  require(config.upIters >= 0, "upIters", config.upIters, "at least 0");
  require(config.group >= 1, "group", config.group, "at least 1");
  require(config.numXcds >= 1, "numXcds", config.numXcds, "at least 1");
  require(config.mfma || !config.lds, "lds without mfma", config.lds, "false");
  if (!config.mfma)
    return;
  require(config.bm == 16 || config.bm == 32, "bm with mfma", config.bm,
          "16 or 32");
  require(config.lanesK == 64, "lanesK with mfma", config.lanesK, "64");
  require(config.upIters == 0, "upIters with mfma", config.upIters, "0");
  require(config.prefetch >= 1, "prefetch with mfma", config.prefetch,
          "at least 1");
}

// The kernarg layout: the pointers, the i32 and f32 scalars,
// then the host-filled workgroup count and the remainder narrowing arguments;
// no hidden arguments, so it works at
// every code object version.
static SkinnyArguments declareArguments(KernelSignature &signature) {
  SkinnyArguments arguments;
  arguments.d = signature.addPointer("D");
  arguments.a = signature.addPointer("A");
  arguments.b = signature.addPointer("B");
  arguments.scalesA = signature.addPointer("scalesA");
  arguments.scalesB = signature.addPointer("scalesB");
  arguments.ws = signature.addPointer("ws");
  arguments.c = signature.addPointer("C");
  arguments.ldD = signature.addScalar("ldD", 4);
  arguments.ldA = signature.addScalar("ldA", 4);
  arguments.ldB = signature.addScalar("ldB", 4);
  arguments.m = signature.addScalar("m", 4);
  arguments.n = signature.addScalar("n", 4);
  arguments.k = signature.addScalar("k", 4);
  arguments.spk = signature.addScalar("spk", 4);
  arguments.ldC = signature.addScalar("ldC", 4);
  // The f32 scalars only need their size: the metadata has no value type.
  arguments.alpha = signature.addScalar("alpha", 4);
  arguments.beta = signature.addScalar("beta", 4);
  // The host fills numWorkGroupsX with the launched workgroup count; only the
  // persistent launch reads it.
  arguments.blocksX = signature.addScalar("numWorkGroupsX", 4);
  // Zero disables the remainder narrowing; only the MFMA 3-D kernel reads
  // these.
  arguments.wholeTiles = signature.addScalar("wholeTiles", 4);
  arguments.pieces = signature.addScalar("pieces", 4);
  return arguments;
}

std::string
rocm::tensilelite::skinnyGEMMKernelName(const SkinnyGEMMConfig &config) {
  validate(config);
  return "gemm_BM" + std::to_string(config.bm) + "_TN" +
         std::to_string(config.tn) + "_LANES_K" +
         std::to_string(config.lanesK) + "_WAVES" +
         std::to_string(config.waves) + "_UP_ITERS" +
         std::to_string(config.upIters) + "_PF" +
         std::to_string(config.prefetch) + "_GROUP" +
         std::to_string(config.group) + "_NUM_XCDS" +
         std::to_string(config.numXcds) + "_GRID_3D" +
         std::to_string(config.grid3d ? 1 : 0) + "_USE_BETA" +
         std::to_string(config.useBeta ? 1 : 0) +
         (config.mfma ? "_MFMA1" : "") + (config.lds ? "_LDS1" : "") +
         (config.useSplitK ? "_USE_SPLIT_K1" : "");
}

std::string rocm::tensilelite::emitSkinnyGEMM(const Target &target,
                                              const SkinnyGEMMConfig &config,
                                              const KernelAbi &abi) {
  validate(config);
  SkinnyDims dims(config);
  if (dims.ldsBytes > target.maxLdsSize())
    throw std::invalid_argument(
        "config exceeds the LDS budget: " + std::to_string(dims.ldsBytes) +
        " bytes of " + std::to_string(target.maxLdsSize()));
  // The reduction addresses its slices with 16-bit ds offsets.
  if (dims.reduceBytes > maxDsOffset + 1)
    throw std::invalid_argument(
        "config exceeds the LDS budget: " + std::to_string(dims.reduceBytes) +
        " bytes of " + std::to_string(maxDsOffset + 1) + " addressable");
  KernelSignature signature;
  SkinnyArguments arguments = declareArguments(signature);
  // Both launches only use the workgroup id x; the 3-D launch decodes its tile
  // and split from it.
  KernelOptions options;
  options.abi = abi;
  Kernel kernel(target, skinnyGEMMKernelName(config), 64 * config.waves,
                std::move(signature), options);
  if (dims.ldsBytes > 0)
    kernel.setLdsSize(dims.ldsBytes);
  SkinnyGEMMEmitter(kernel, config, arguments).emit();
  try {
    kernel.builder().allocator().checkLimits();
  } catch (const std::runtime_error &error) {
    throw std::invalid_argument(std::string("config exceeds the register "
                                            "budget: ") +
                                error.what());
  }
  return kernel.toAssembly();
}

std::string rocm::tensilelite::emitSplitKReduce(const Target &target,
                                                const KernelAbi &abi) {
  KernelSignature signature;
  ReduceArguments offsets;
  offsets.d = signature.addPointer("D");
  offsets.ws = signature.addPointer("ws");
  offsets.c = signature.addPointer("C");
  offsets.ldD = signature.addScalar("ldD", 4);
  offsets.m = signature.addScalar("m", 4);
  offsets.n = signature.addScalar("n", 4);
  offsets.splits = signature.addScalar("splits", 4);
  offsets.ldC = signature.addScalar("ldC", 4);
  offsets.alpha = signature.addScalar("alpha", 4);
  offsets.beta = signature.addScalar("beta", 4);
  // The workgroup id y is the column j.
  KernelOptions options;
  options.abi = abi;
  options.workgroupIdY = true;
  Kernel kernel(target, "splitk_reduce", reduceBlock, std::move(signature),
                options);
  SplitKReduceEmitter(kernel, offsets).emit();
  return kernel.toAssembly();
}
