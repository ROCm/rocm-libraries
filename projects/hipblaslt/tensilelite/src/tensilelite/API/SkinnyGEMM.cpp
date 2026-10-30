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

#include <algorithm>
#include <cstdint>
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
      reordered(config.group != 1 || config.numXcds != 1),
      upIters(config.upIters), partialSteps(config.partialSteps) {
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
  ringBytes = lds ? (slots + partialSteps) * (wideTiles + skinnyTiles) *
                        stagedTileBytes
                  : 0;
  reduceBytes = mfma ? (waves - 1) * tn * (bm / 16) * 1024 : 0;
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

SkinnyGEMMEmitter::SetupScalars::SetupScalars(IRBuilder &builder, bool grid3d)
    : blocksX(builder.allocateSgpr(1)),
      blocksY(grid3d ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                     : std::nullopt),
      rowGroup(builder.allocateSgpr(1)), ldS(builder.allocateSgpr(1)),
      ldW(builder.allocateSgpr(1)),
      splits(grid3d ? std::nullopt
                    : std::optional<RegisterRange>(builder.allocateSgpr(1))) {}

SkinnyGEMMEmitter::SkinnyGEMMEmitter(Kernel &kernel,
                                     const SkinnyGEMMConfig &config,
                                     const SkinnyArguments &arguments)
    : kernel(kernel), builder(kernel.builder()), dims(config),
      arguments(arguments), sRsrc(builder.allocateSgpr(4, 4)),
      wRsrc(builder.allocateSgpr(4, 4)), sSclRsrc(builder.allocateSgpr(4, 4)),
      wSclRsrc(builder.allocateSgpr(4, 4)), dRsrc(builder.allocateSgpr(4, 4)),
      wsRsrc(builder.allocateSgpr(4, 4)),
      cRsrc(dims.useBeta
                ? std::optional<RegisterRange>(builder.allocateSgpr(4, 4))
                : std::nullopt),
      alpha(builder.allocateSgpr(2, 2)),
      beta(dims.useBeta ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                        : std::nullopt),
      k(builder.allocateSgpr(1)), spk(builder.allocateSgpr(1)),
      mn(builder.allocateSgpr(1)), dStrideS(builder.allocateSgpr(1)),
      dStrideW(builder.allocateSgpr(1)), wsStrideS(builder.allocateSgpr(1)),
      wsStrideW(builder.allocateSgpr(1)),
      cStrideS(dims.useBeta
                   ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                   : std::nullopt),
      cStrideW(dims.useBeta
                   ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                   : std::nullopt),
      skinnyCols(builder.allocateSgpr(1)), wideCols(builder.allocateSgpr(1)),
      contigS(builder.allocateSgpr(1)) {}

// The xcdRun tiles that share a scale line run on one XCD, so its L2 fetches
// the line once instead of every XCD fetching it. The dispatcher sends
// workgroup x to XCD x mod X, so within each full block of X * xcdRun ids the
// tile is (x mod X) * xcdRun + (x / X) mod xcdRun; the tail of the grid keeps
// the identity so that every tile still runs exactly once. A function for the
// name of the XCD remap formula.
void skinny::groupTilesPerXcd(IRBuilder &builder, const SkinnyDims &dims,
                              int32_t xcds, RegisterRef tx,
                              RegisterRef blocksX) {
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
  builder.andi32(term.ref(), -span, blocksX);
  // SCC must survive from the compare to the select.
  builder.create<rocisa::SCmpLtU32>(tx.container(), term.container());
  builder.selectb32(tx, logical.ref(), tx);
}

// Decodes the tile of the block id into (tx, ty, kz), after the remap of
// the XCDs: kz is the plane of tiles, and within the plane the plain order has
// tx fastest while the grouped one walks bands of GROUP skinny tile rows ty
// fastest. A function for the name of the decode formula.
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
  RegisterRange pos = builder.allocateSgpr(1);
  divMod(builder, ids.kz, pos.ref(), logical, loop.plane.ref());
  if (dims.group != 1) {
    // The band holds GROUP rows, fewer in the last one, so a tile at inBand is
    // at column inBand / rows and row firstTy + inBand mod rows.
    RegisterRange band = builder.allocateSgpr(1);
    RegisterRange inBand = builder.allocateSgpr(1);
    RegisterRange firstTy = builder.allocateSgpr(1);
    RegisterRange rows = builder.allocateSgpr(1);
    divMod(builder, band.ref(), inBand.ref(), pos.ref(), loop.bandSize->ref());
    builder.muli32(firstTy.ref(), band.ref(), dims.group);
    builder.subi32(rows.ref(), loop.tilesY->ref(), firstTy.ref());
    builder.minsi32(rows.ref(), rows.ref(), dims.group);
    divMod(builder, ids.tx, ids.ty, inBand.ref(), rows.ref());
    builder.addi32(ids.ty, ids.ty, firstTy.ref());
    return;
  }
  divMod(builder, ids.ty, ids.tx, pos.ref(), loop.tilesX.ref());
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
// dimension.
void SkinnyGEMMEmitter::emitArguments(SetupScalars &scalars) {
  // The scalars that only the descriptors and the dispatch read; blocksZ, the
  // split count, only exists in the 3-D launch.
  RegisterRange ldD = builder.allocateSgpr(1), ldA = builder.allocateSgpr(1),
                ldB = builder.allocateSgpr(1), m = builder.allocateSgpr(1),
                n = builder.allocateSgpr(1);
  std::optional<RegisterRange> blocksZ;
  if (dims.grid3d)
    blocksZ = builder.allocateSgpr(1);
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
  if (blocksZ)
    kernel.loadArgument(blocksZ->ref(), arguments.blocksZ);
  kernel.loadArgument(scalars.blocksX.ref(), arguments.blocksX);
  if (scalars.blocksY)
    kernel.loadArgument(scalars.blocksY->ref(), arguments.blocksY);
  kernel.loadArgument(spk.ref(), arguments.spk);
  kernel.loadArgument(alpha.ref(0, 1), arguments.alpha);
  kernel.loadArgument(k.ref(), arguments.k);
  kernel.loadArgument(n.ref(), arguments.n);
  kernel.loadArgument(m.ref(), arguments.m);
  kernel.loadArgument(ldB.ref(), arguments.ldB);
  kernel.loadArgument(ldA.ref(), arguments.ldA);
  kernel.loadArgument(ldD.ref(), arguments.ldD);
  loadPointer(kernel, wsRsrc.ref(0, 2), arguments.ws);
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
  // Scalar loads return out of order, so only lgkmcnt(0) guarantees any of
  // them.
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
    // The workspace holds splits * m * n f32 elements; the splits are the
    // blocks of the grid's z dimension, or ceilDiv(k, spk) in the 1-D launch.
    RegisterRef splits = blocksZ ? blocksZ->ref() : scalars.splits->ref();
    if (!blocksZ) {
      // The exact signed division, evaluated at run time.
      builder.addi32(splits, k.ref(), spk.ref());
      builder.addi32(splits, splits, -1);
      builder.divsi32(splits, splits, spk.ref(), /*useIEEE=*/true);
    }
    RegisterRef wsBytes = wsRsrc.ref(2);
    builder.muli32(wsBytes, m.ref(), splits);
    builder.muli32(wsBytes, wsBytes, n.ref());
    builder.shli32(wsBytes, wsBytes, 2);
    builder.muli32(mn.ref(), m.ref(), n.ref());
  }

  // Path dispatch: selects the operands, strides and extents of the skinny
  // and the wide dimension on m <= n. The s_cselects follow one s_cmp and no
  // SALU between them writes SCC. The f32 workspace column is 4 * m bytes,
  // computed before the s_cmp as the shift writes SCC.
  builder.shli32(wsStrideW.ref(), m.ref(), 2);
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
  builder.selectb32(wsStrideS.ref(), 4, wsStrideW.ref());
  builder.selectb32(wsStrideW.ref(), wsStrideW.ref(), 4);
  builder.selectb32(contigS.ref(), 1, 0);
  builder.minui32(skinnyCols.ref(), m.ref(), n.ref());
  builder.maxui32(wideCols.ref(), m.ref(), n.ref());
  for (RegisterRange *rsrc : {&sRsrc, &wRsrc, &sSclRsrc, &wSclRsrc})
    builder.makeRsrc(rsrc->ref(), rsrc->ref(0, 2), rsrc->ref(2));
  std::vector<RegisterRange *> rsrcs = {&dRsrc, &wsRsrc};
  if (dims.useBeta)
    rsrcs.push_back(&*cRsrc);
  for (RegisterRange *rsrc : rsrcs)
    builder.makeRsrc(rsrc->ref(), rsrc->ref(0, 2), rsrc->ref(2));
}

// Split-K reduce.

// Threads per block of the reduce kernel.
static constexpr int32_t reduceBlock = 256;

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
    sumSplits(loop, tmp.ref(), row.ref(), value.ref());
    storeBf16(tmp.ref(), row.ref(), value.ref());
  }

private:
  // sum = the f32 sum over the splits. Each lane loads its element of ws at
  // (z * m * n + j * m + iLoad) * 4, so splits >= 1 is a precondition.
  void sumSplits(const SplitLoop &loop, RegisterRef tmp, RegisterRef row,
                 RegisterRef value) {
    RegisterRange base = builder.allocateVgpr(1);
    RegisterRange split = builder.allocateSgpr(1);
    // The row register is dead after base, so it holds the sum.
    RegisterRef sum = row;
    builder.shli32(row, row, 2);
    builder.muli32(tmp, kernel.workgroupIdY(), loop.rowBytes.ref());
    builder.addi32(base.ref(), tmp, row);
    builder.movb32(split.ref(), 0);
    builder.movb32(sum, 0);
    std::shared_ptr<rocisa::Label> body = builder.createLabel("loop");
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
  }

  // Computes D = bf16(fma(beta, C, alpha * sum)) and stores it where inRange.
  void storeBf16(RegisterRef tmp, RegisterRef row, RegisterRef value) {
    RegisterRange savedExec = builder.allocateSgpr(2, 2);
    RegisterRef sum = row;
    builder.create<rocisa::VMulF32>(sum.container(), alpha.container(),
                                    sum.container());
    std::shared_ptr<rocisa::Label> skip = builder.createLabel("skip");
    builder.create<rocisa::SAndSaveExecB64>(savedExec.container(),
                                            std::make_shared<rocisa::VCC>());
    builder.create<rocisa::SCBranchVCCZ>(skip->getLabelName());
    // The sum becomes fma(beta, C, sum), skipped, without loading C, when beta
    // is zero. The sign bit is masked so that -0.0 counts as zero.
    std::shared_ptr<rocisa::Label> noBeta = builder.createLabel("no_beta");
    branchIfBetaZero(builder, beta.ref(), tmp, noBeta);
    // The load overwrites its address register.
    RegisterRef c = value;
    columnMajorOffset(kernel, c, ldC.ref(), tmp);
    builder.loadBuffer(c, cRsrc.ref(), c, 0, MemoryWidth::b16);
    builder.createWaitCnt(WaitCounts{.vectorLoads = 0});
    // bf16 widens to f32 by a left shift, exactly.
    builder.shli32(c, c, 16);
    builder.create<rocisa::VMacF32>(sum.container(), beta.container(),
                                    c.container());
    builder.insert(noBeta);
    // truncf keeps the top 16 bits instead of rounding to nearest.
    builder.shrui32(sum, sum, 16);
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
  require(config.partialSteps >= 0 && config.partialSteps <= 1, "partialSteps",
          config.partialSteps, "0 or 1");
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
// then the host-filled workgroup counts; no hidden arguments, so it works at
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
  // The host fills the grid dimensions with the launched workgroup counts.
  arguments.blocksX = signature.addScalar("numWorkGroupsX", 4);
  arguments.blocksY = signature.addScalar("numWorkGroupsY", 4);
  arguments.blocksZ = signature.addScalar("numWorkGroupsZ", 4);
  return arguments;
}

std::string
rocm::tensilelite::skinnyGEMMKernelName(const SkinnyGEMMConfig &config) {
  validate(config);
  return "gemm_BM" + std::to_string(config.bm) + "_TN" +
         std::to_string(config.tn) + "_LANES_K" +
         std::to_string(config.lanesK) + "_WAVES" +
         std::to_string(config.waves) + "_UP_ITERS" +
         std::to_string(config.upIters) + "_PARTIAL_STEPS" +
         std::to_string(config.partialSteps) + "_PF" +
         std::to_string(config.prefetch) + "_GROUP" +
         std::to_string(config.group) + "_NUM_XCDS" +
         std::to_string(config.numXcds) + "_GRID_3D" +
         std::to_string(config.grid3d ? 1 : 0) + "_USE_BETA" +
         std::to_string(config.useBeta ? 1 : 0) +
         (config.mfma ? "_MFMA1" : "") + (config.lds ? "_LDS1" : "");
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
  // The workgroup ids are the tile (tx, ty) and the split kz; the persistent
  // launch only has the block id x.
  KernelOptions options;
  options.abi = abi;
  options.workgroupIdY = config.grid3d;
  options.workgroupIdZ = config.grid3d;
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
