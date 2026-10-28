// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The skinny MXFP8 GEMM emitter, which implements the kernel of
// gemm_mxfp8_skinny.mlir for both launch forms (GRID_3D = 1 and the persistent
// GRID_3D = 0), any PARTIAL_STEPS, ring scales (UP_ITERS = 0) or up-front
// scales, plus @splitk_reduce. emit drives emitSetup and emitTile, which runs
// emitCPrefetch, emitUpFrontScales, emitKLoop (with emitCompute), emitTail and
// emitEpilogue. They cover the parts of @gemm and @run_tile, not 1:1 by name.

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
#include <cassert>
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

static constexpr int32_t outOfRange = -1;

static bool isScalar(RegisterRef ref) { return ref.kind == RegisterKind::sgpr; }

// Every pointer is copied once into a longer-lived resource, so a preloaded
// range is dead right after the copy.
static void loadPointer(Kernel &kernel, RegisterRef dst, int32_t offset) {
  kernel.loadArgument(dst, offset);
  kernel.releasePreloadedArgument(offset);
}

// Returns whether value is a positive power of two.
static bool isPowerOfTwo(int32_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// Returns log2 of a power of two.
static int32_t log2Exact(int32_t value) {
  if (!isPowerOfTwo(value))
    throw std::logic_error("value is not a power of two");
  int32_t log = 0;
  while ((1 << log) < value)
    ++log;
  return log;
}

// The helpers below take non-negative operands, so each quotient is one
// division and each remainder is a - q * d, without the sign corrections of the
// reference lowering of floor divisions and moduli.

// dst = src / divisor.
static void divideByLiteral(IRBuilder &builder, RegisterRef dst,
                            RegisterRef src, int32_t divisor) {
  if (isPowerOfTwo(divisor)) {
    builder.shrui32(dst, src, log2Exact(divisor));
    return;
  }
  builder.divsi32(dst, src, divisor, /*useIEEE=*/true);
}

// dst = src - quotient * divisor, the remainder given the quotient.
static void remainderByLiteral(IRBuilder &builder, RegisterRef dst,
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
static void ceilDivByLiteral(IRBuilder &builder, RegisterRef dst,
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
static void moduloByLiteral(IRBuilder &builder, RegisterRef dst,
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
// operands, as in the reference.
static void divMod(IRBuilder &builder, RegisterRef quotient,
                   RegisterRef remainder, RegisterRef dividend,
                   RegisterRef divisor) {
  builder.divsi32(quotient, dividend, divisor, /*useIEEE=*/true);
  builder.muli32(remainder, quotient, divisor);
  builder.subi32(remainder, dividend, remainder);
}

namespace {

// Loads that stay in flight at a point of the schedule, counted in issue
// order so that an in-order vmcnt wait can name them.
struct SlotLoadPosition {
  // Position of the load among the loads of its slot.
  int32_t position;
  // Full ring slots issued after the slot of the load.
  int32_t groupsAfter;
  // Loads of the slot's own refill that were issued before the consumer.
  int32_t ownRefill;
};

} // namespace

// Number of vector memory loads issued after the load at where; the count a
// vmcnt wait must allow so that it is complete.
static int32_t loadsIssuedAfter(const SkinnyDims &dims,
                                SlotLoadPosition where) {
  int32_t sameSlot = dims.loadsPerSlot - 1 - where.position;
  return sameSlot + where.groupsAfter * dims.loadsPerSlot + where.ownRefill;
}

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
static int32_t runLength(const SkinnyDims &dims, bool skinnyContiguous) {
  return skinnyContiguous ? dims.bm : dims.tn;
}

// A chunk must start dword aligned: an f32 run of at least 2 elements or an
// even bf16 run.
static bool hasVector(const SkinnyDims &dims, bool skinnyContiguous,
                      bool bf16) {
  int32_t length = runLength(dims, skinnyContiguous);
  return bf16 ? length % 2 == 0 : length >= 2;
}

static RedLayout redLayout(const SkinnyDims &dims, bool skinnyContiguous) {
  int32_t length = runLength(dims, skinnyContiguous);
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

SkinnyDims::SkinnyDims(const SkinnyGEMMConfig &config)
    : bm(config.bm), tn(config.tn), lanesK(config.lanesK), waves(config.waves),
      prefetch(config.prefetch), grid3d(config.grid3d), group(config.group),
      numXcds(config.numXcds), useBeta(config.useBeta),
      reordered(config.group != 1 || config.numXcds != 1),
      upIters(config.upIters), partialSteps(config.partialSteps) {
  columns = bm + tn;
  accumulators = bm * tn;
  columnGroups = 64 / lanesK;
  uniformWide = waves == 1 && lanesK == 64;
  tileWide = waves * columnGroups * tn;
  xcdRun = isPowerOfTwo(tileWide) && tileWide < scaleRowColumns
               ? scaleRowColumns / tileWide
               : 1;
  ki = 64 * lanesK;
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
  ldsBytes = scalesShared ? upIters * columns * waves * 256 : 0;
  chunkStride = 16 * subgroupLanes;
}

// Setup.

static constexpr int32_t selectorMaskStep = 0x01010101;

// Aligns a hot loop header to 16 bytes: a header at 4 mod 8 made the loop 5 to
// 7 percent slower on some points, and one at 8 mod 16 made another 4 percent
// slower, as measured by the perf A/B harness.
static void alignLoopHeader(IRBuilder &builder) {
  builder.create<rocisa::TextBlock>(".p2align 4\n");
}

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
static void branchIfBetaZero(IRBuilder &builder, RegisterRef beta,
                             RegisterRef scratch,
                             const std::shared_ptr<rocisa::Label> &label) {
  builder.andi32(scratch, beta, 0x7fffffff);
  builder.create<rocisa::SCmpEQU32>(scratch.container(), 0);
  builder.create<rocisa::SCBranchSCC1>(label->getLabelName());
}

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

SkinnyGEMMEmitter::SetupScalars::SetupScalars(IRBuilder &builder, bool grid3d)
    : blocksX(builder.allocateSgpr(1)),
      blocksY(grid3d ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                     : std::nullopt),
      rowGroup(builder.allocateSgpr(1)), ldS(builder.allocateSgpr(1)),
      ldW(builder.allocateSgpr(1)),
      splits(grid3d ? std::nullopt
                    : std::optional<RegisterRange>(builder.allocateSgpr(1))) {}

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
// name of the @remap_xcd formula.
static void groupTilesPerXcd(IRBuilder &builder, const SkinnyDims &dims,
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
  // 4 * (j * CG + cg) with the slot j(kl, d) of the MLIR's @scl_slot.
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

// Decodes the tile of the block id into (tx, ty, kz), after the @remap_xcd of
// the XCDs: kz is the plane of tiles, and within the plane the plain order has
// tx fastest while the grouped one walks bands of GROUP skinny tile rows ty
// fastest. A function for the name of the decode formula.
static void decodeTile(IRBuilder &builder, const SkinnyDims &dims,
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

// The 3-D launch allocates the ring and the tile's registers first, as the
// setup writes into them, and the setup values die when the tile's bases are
// done. The persistent launch (GRID_3D = 0 of @gemm) keeps the setup values for
// the tile loop and allocates the registers of the tile in every iteration:
// block b runs the tiles b, b + B, ... of the flattened tile space, in the
// order of @remap_xcd and the grouped decode, every one from zeroed
// accumulators. As elsewhere, every operand is non-negative, so each quotient
// is one division and each remainder is a - q * d.
void SkinnyGEMMEmitter::emit() {
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
  {
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
        // The exact signed division, as the reference divides at run time.
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
      // instead of LLVM's srem expansion. The 3-D launch keys it on the linear
      // block id; the reordered persistent tiles key it on the wide slab kz *
      // tilesX + tx, so that the tiles that share a wide column read the same K
      // offset together.
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

// The largest value of the 16-bit offset field of a ds instruction.
static constexpr int32_t maxDsOffset = 65535;

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

// K phase helpers shared by several phases.

static constexpr int32_t exponentMask = 0x7F800000;

// Waits until at most outstanding vector memory loads are in flight, capped at
// the counter's range.
static void waitForVectorLoads(IRBuilder &builder, int32_t outstanding) {
  builder.createWaitCnt(WaitCounts{
      .vectorLoads = std::min(outstanding, builder.target().maxVmcnt())});
}

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

// Issues one load per column of the iteration at soffset. With neg (0 or -1
// per lane) the voffset is ORed with it first, into one temporary that every
// column reuses since a load reads its voffset when it issues. The temporary
// never goes into the destination registers, which may still have a prologue
// load in flight. The destinations are appended to record, if any.
static void
issueColumnLoads(const KLoopState &state,
                 const std::function<RegisterRef(int32_t)> &destination,
                 const std::function<RegisterRef(int32_t)> &rsrc,
                 const RegisterRange &voffsetBase, MemoryWidth width,
                 RegisterRef soffset, std::optional<RegisterRef> neg,
                 std::vector<RegisterRef> *record) {
  IRBuilder &builder = state.builder;
  std::optional<RegisterRange> steered;
  if (neg)
    steered = builder.allocateVgpr(1);
  for (int32_t c = 0; c < state.dims.columns; ++c) {
    RegisterRef dst = destination(c);
    RegisterRef voffset = voffsetBase.ref(c);
    if (neg) {
      builder.ori32(steered->ref(), voffset, *neg);
      voffset = steered->ref();
    }
    builder.loadBuffer(dst, rsrc(c), voffset, soffset, width);
    if (record)
      record->push_back(dst);
  }
}

// Loads the scale dword of every column of the iteration at soffset into the
// NC registers of raw.
static void issueScales(const KLoopState &state, RegisterRef raw,
                        RegisterRef soffset,
                        std::optional<RegisterRef> neg = std::nullopt,
                        std::vector<RegisterRef> *record = nullptr) {
  issueColumnLoads(
      state, [&](int32_t c) { return raw.slice(c); },
      [&](int32_t c) {
        return c < state.dims.bm ? state.sSclRsrc : state.wSclRsrc;
      },
      state.tile.sBase, MemoryWidth::b32, soffset, neg, record);
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
  issueColumnLoads(
      state, [&](int32_t c) { return dataChunk(state, slot, chunk, c); },
      [&](int32_t c) { return dataRsrc(state, c); }, state.tile.dBase,
      MemoryWidth::b128, soffset, neg, record);
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

// Consumes chunk u of the slot: 4 dwords of 2 fp8 pairs per column, converted
// with the decoded scale and accumulated with v_dot2_f32_bf16.
static void computeChunk(const KLoopState &state, const SlotContext &context,
                         int32_t chunk, RegisterRef scales) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
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
          waitForVectorLoads(
              builder, loadsIssuedAfter(
                           dims, {dims.scaleLoads + chunk * dims.columns + c,
                                  context.groupsAfter, ownRefill}));
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
// packed dwords, as the reference does, to fit the register budget; the loop
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
    return dims.upIters * dims.columns - 1 - index +
           dims.prefetch * dims.loadsPerSlot;
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

// The K loop over the full iterations of the split, in one of two forms. PF = 0
// issues the chunks of an iteration, and its scales with ring scales, and then
// consumes them. PF > 0 fills the ring with the prologue, runs the steady loop
// unrolled over the PF slots and finishes the 1 to 2 * PF - 1 remaining
// iterations in straight-line arms.
void SkinnyGEMMEmitter::emitKLoop(KLoopState &state) {
  TileState &tile = state.tile;
  int32_t pf = dims.prefetch;
  // dst = sum >= NIfull ? sum - NIfull : sum, the physical iteration of a
  // logical one, with the rotation added once.
  auto wrapIteration = [&](RegisterRef dst, RegisterRef sum) {
    RegisterRange wrapped = builder.allocateSgpr(1);
    builder.subi32(wrapped.ref(), sum, tile.nIterFull.ref());
    builder.create<rocisa::SCmpGeU32>(sum.container(),
                                      tile.nIterFull.container());
    builder.selectb32(dst, wrapped.ref(), sum);
  };
  auto hoistChunkBases = [&] {
    for (int32_t chunk = 1; chunk < SkinnyDims::chunks; ++chunk)
      ensureChunkBase(state, chunk);
  };

  // PF = 0.
  if (pf == 0) {
    if (dims.upIters > 0)
      emitUpFrontScales(state, UpFrontStep::pack);
    hoistChunkBases();
    // Nothing of the ring is loop carried, so the raw scales live only from the
    // loads of an iteration to its permutes: the hole they leave is taken back
    // inside the loop body and by the tail step.
    if (state.ring.raw)
      state.ring.raw->release();
    RegisterRange iteration = builder.allocateSgpr(1);
    std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
    std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
    builder.create<rocisa::SCmpGtI32>(tile.nIterFull.container(), 0);
    builder.movb32(iteration.ref(), 0);
    builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
    alignLoopHeader(builder);
    builder.insert(body);
    if (state.ring.raw)
      state.ring.raw = builder.allocateVgpr(dims.columns);
    RegisterRange physical = builder.allocateSgpr(1);
    if (tile.fitRot) {
      builder.addi32(physical.ref(), iteration.ref(), tile.fitRot->ref());
      wrapIteration(physical.ref(), physical.ref());
    } else
      wrapIteration(physical.ref(), iteration.ref());
    // The select precedes the loads so that its VALU work hides their latency.
    std::optional<RegisterRange> packed;
    if (dims.upIters > 0)
      packed = selectScales(state, physical.ref());
    RegisterRange soffset = builder.allocateSgpr(1);
    SlotContext loads{0, 0, physical.ref()};
    startRefill(state, loads, soffset.ref());
    for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
      refillChunk(state, loads, chunk, soffset.ref());
    SlotContext consume{0, 0, std::nullopt};
    if (dims.upIters > 0)
      computeUpFront(state, consume, packed);
    else
      emitCompute(state, consume);
    builder.addi32(iteration.ref(), iteration.ref(), 1);
    builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                      tile.nIterFull.container());
    builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
    builder.insert(exit);
    return;
  }

  // PF > 0, prologue: issues the iterations [0, PF) into the ring. A split with
  // fewer full iterations reloads its last one through the clamp: a load
  // dropped by the range check would still take a VMEM issue slot, and the
  // extra slots are never consumed.
  {
    std::optional<RegisterRange> lastIteration;
    if (pf > 1) {
      lastIteration = builder.allocateSgpr(1);
      builder.addi32(lastIteration->ref(), tile.nIterFull.ref(), -1);
      builder.maxsi32(lastIteration->ref(), lastIteration->ref(), 0);
    }
    // LLVM hoists the chunk bases out of the prologue from PF = 4 on.
    bool hoisted = pf >= 4;
    for (int32_t slot = 0; slot < pf; ++slot) {
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
      SlotContext context{slot, 0, physical.ref(), hoisted};
      context.record = &tile.prologueLoads;
      RegisterRange soffset = builder.allocateSgpr(1);
      startRefill(state, context, soffset.ref());
      for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
        refillChunk(state, context, chunk, soffset.ref());
    }
  }
  // steadyEnd = max((NIfull / PF - 1) * PF, 0), so every refill of the steady
  // loop targets an iteration below NIfull.
  RegisterRange steadyEnd = builder.allocateSgpr(1);
  if (pf == 1)
    builder.addi32(steadyEnd.ref(), tile.nIterFull.ref(), -1);
  else if (isPowerOfTwo(pf)) {
    int32_t log2Pf = log2Exact(pf);
    builder.shrsi32(steadyEnd.ref(), tile.nIterFull.ref(), log2Pf);
    builder.shli32(steadyEnd.ref(), steadyEnd.ref(), log2Pf);
    builder.addi32(steadyEnd.ref(), steadyEnd.ref(), -pf);
  } else {
    builder.divsi32(steadyEnd.ref(), tile.nIterFull.ref(), pf,
                    /*useIEEE=*/true);
    builder.muli32(steadyEnd.ref(), steadyEnd.ref(), pf);
    builder.addi32(steadyEnd.ref(), steadyEnd.ref(), -pf);
  }
  builder.maxsi32(steadyEnd.ref(), steadyEnd.ref(), 0);
  // The pack follows the prologue loads that its waits count.
  if (dims.upIters > 0)
    emitUpFrontScales(state, UpFrontStep::pack);

  // Steady loop: unrolled over the slots, so every slot index is a constant.
  // Slot p refills with the iteration it0 + PF + p and, with several up-front
  // entries, selects the scales of the physical iteration it0 + p. The bases
  // fold the rotation: refillBase[p] = fitRot + PF + p and physBase[p] = fitRot
  // + p. Without rotation they are the literals.
  bool vectorized = dims.upIters >= 2;
  {
    std::optional<RegisterRange> refillBase;
    std::optional<RegisterRange> physBase;
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
    RegisterRange iteration = builder.allocateSgpr(1);
    std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
    std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
    builder.create<rocisa::SCmpGtI32>(steadyEnd.container(), 0);
    builder.movb32(iteration.ref(), 0);
    builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
    alignLoopHeader(builder);
    builder.insert(body);
    for (int32_t slot = 0; slot < pf; ++slot) {
      RegisterRange refill = builder.allocateSgpr(1);
      if (refillBase)
        builder.addi32(refill.ref(), iteration.ref(), refillBase->ref(slot));
      else
        builder.addi32(refill.ref(), iteration.ref(), pf + slot);
      wrapIteration(refill.ref(), refill.ref());
      SlotContext context{slot, pf - 1, refill.ref()};
      std::optional<RegisterRange> physical;
      if (vectorized) {
        physical = builder.allocateSgpr(1);
        if (slot == 0)
          builder.addi32(physical->ref(), tile.fitRot->ref(), iteration.ref());
        else
          builder.addi32(physical->ref(), iteration.ref(), physBase->ref(slot));
        wrapIteration(physical->ref(), physical->ref());
        context.vectorIndex = physical->ref();
      }
      emitCompute(state, context);
    }
    builder.addi32(iteration.ref(), iteration.ref(), pf);
    builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                      steadyEnd.container());
    builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
    builder.insert(exit);
  }

  // Remainder arms: the left = NIfull - steadyEnd (1 to 2 * PF - 1) remaining
  // iterations run in the arm of a chain of independent ifs on left. Iteration
  // i of the arm uses ring slot i mod PF and refills it only when iteration i +
  // PF exists. Of the loads issued after the slot's group, the PF - 1 - i
  // initial groups and the refills of the earlier iterations remain. With
  // several up-front entries it also selects the scales of its physical
  // iteration, which is the refill iteration of i - PF from i = PF on.
  RegisterRange left = builder.allocateSgpr(1);
  builder.subi32(left.ref(), tile.nIterFull.ref(), steadyEnd.ref());
  for (int32_t remaining = 1; remaining < 2 * pf; ++remaining) {
    std::shared_ptr<rocisa::Label> next = builder.createLabel("arm_end");
    builder.create<rocisa::SCmpEQI32>(left.container(), remaining);
    builder.create<rocisa::SCBranchSCC0>(next->getLabelName());
    // The logical iteration of a refill is fitRot + steadyEnd + i + PF.
    std::optional<RegisterRange> base;
    std::vector<RegisterRange> keptRefills;
    for (int32_t i = 0; i < remaining; ++i) {
      int32_t refilled = std::clamp(remaining - pf, 0, i);
      SlotContext context{i % pf, pf - 1 + refilled - i, std::nullopt, false};
      RegisterRange refill = builder.allocateSgpr(1);
      bool hasRefill = i + pf < remaining;
      if (!base && tile.fitRot && (vectorized || hasRefill)) {
        base = builder.allocateSgpr(1);
        builder.addi32(base->ref(), tile.fitRot->ref(), steadyEnd.ref());
      }
      RegisterRef baseRef = base ? base->ref() : steadyEnd.ref();
      if (hasRefill) {
        builder.addi32(refill.ref(), baseRef, i + pf);
        wrapIteration(refill.ref(), refill.ref());
        context.refillIteration = refill.ref();
      }
      std::optional<RegisterRange> physical;
      if (vectorized && i >= pf)
        context.vectorIndex = keptRefills[i - pf].ref();
      else if (vectorized) {
        physical = builder.allocateSgpr(1);
        if (i > 0)
          builder.addi32(physical->ref(), baseRef, i);
        wrapIteration(physical->ref(), i > 0 ? physical->ref() : baseRef);
        context.vectorIndex = physical->ref();
      }
      emitCompute(state, context);
      if (vectorized && hasRefill)
        keptRefills.push_back(std::move(refill));
    }
    builder.insert(next);
  }
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
          : loadsIssuedAfter(dims, {0, context.groupsAfter, /*ownRefill=*/0});
  RegisterRef raw = rawScales(state, context.slot);
  gatherSlotScales(state, raw, packed.ref(), firstOutstanding);
  permSlotScales(state, state.tile.selectors.ref(), raw, packed.ref(),
                 packed.ref());
  // The PF = 0 loop frees the raw scales again, see emitKLoop.
  if (dims.prefetch == 0)
    state.ring.raw->release();
  // The MLIR decodes first for every depth but 1; the PF = 0 loop keeps the
  // staged schedule, which its steps do not use.
  bool decodeFirst =
      dims.prefetch >= 2 || (dims.prefetch == 0 && context.partialStep);
  if (decodeFirst)
    computeDecodeFirst(state, context, packed);
  else
    computeStaged(state, context, packed);
}

// Partial K steps.

static constexpr int32_t dwordBytes = 4;
static constexpr int32_t dwordsPerChunk = 4;
static constexpr int32_t partBytes = dwordBytes * dwordsPerChunk;
// The largest byte shift of a dword mask: a 32-bit shift is poison in the MLIR,
// and a dword that needs a shift of 4 bytes is steered anyway.
static constexpr int32_t maxMaskBytes = 3;

// dst = -1 when src + delta is negative, else 0.
static void signOfSum(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                      int32_t delta) {
  builder.addi32(dst, src, delta);
  builder.shrsi32(dst, dst, 31);
}

// dst = 8 * clamp(src, 0, 3), the shift in bits of a mask of whole bytes.
static void clampedBitShift(IRBuilder &builder, RegisterRef dst,
                            RegisterRef src) {
  builder.maxsi32(dst, 0, src);
  builder.minsi32(dst, maxMaskBytes, dst);
  builder.shli32(dst, dst, 3);
}

// Loads dword d in [0, 4) of chunk u for every column. A lane reads dword d
// only when 4d < laneRem, else its voffset is the out-of-range -1. The dword
// that straddles hi is loaded at its aligned address and its bytes at or after
// hi cleared with an AND mask. Those bytes lie inside the column's storage,
// since every A and B column is padded to 4 bytes, but belong to another split
// or to the padding. The steering, mask and soffset of a dword do not depend on
// the column, so the loop over d is outermost and only one steering and mask
// pair is live. This is a function because inlining it would nest the loops of
// emitTail five deep.
static void loadTailDwords(const KLoopState &state, int32_t chunk,
                           RegisterRef soffset, RegisterRef laneRem) {
  IRBuilder &builder = state.builder;
  RegisterRange voffset = builder.allocateVgpr(1);
  for (int32_t d = 0; d < dwordsPerChunk; ++d) {
    int32_t start = d * dwordBytes;
    RegisterRange steer = builder.allocateVgpr(1);
    RegisterRange mask = builder.allocateVgpr(1);
    // Steering: 0 when the dword starts below hi, else -1.
    signOfSum(builder, steer.ref(), laneRem, -(start + 1));
    // Keep the bytes of the dword below hi.
    builder.subi32(mask.ref(), start + dwordBytes, laneRem);
    clampedBitShift(builder, mask.ref(), mask.ref());
    builder.shrui32(mask.ref(), -1, mask.ref());
    std::optional<RegisterRange> dwordOffset;
    RegisterRef dwordSoffset = soffset;
    if (d > 0) {
      dwordOffset = builder.allocateSgpr(1);
      dwordSoffset = dwordOffset->ref();
      builder.addi32(dwordSoffset, soffset, start);
    }
    for (int32_t column = 0; column < state.dims.columns; ++column) {
      // Loads one dword of the column, steered, and clears the bytes at or
      // after hi. The AND consumes the load just issued, so nothing may stay in
      // flight.
      RegisterRef dst = dataChunk(state, 0, chunk, column).slice(d);
      builder.ori32(voffset.ref(), state.tile.dBase.ref(column), steer.ref());
      builder.loadBuffer(dst, dataRsrc(state, column), voffset.ref(),
                         dwordSoffset, MemoryWidth::b32);
      waitForVectorLoads(builder, 0);
      builder.andi32(dst, dst, mask.ref());
    }
  }
}

// The tail step of PARTIAL_STEPS = 1: the iteration at ks + nIterFull * KI
// keeps the bytes [0, tailBytes), and the step is skipped when there are none.
// It runs on ring slot 0, which the K loop left free. Ring scales load as in
// the loop with the groups at or after k steered to -1, and every chunk loads
// behind a dynamic branch, so that the slot holds exact zeros outside [0,
// tailBytes). The compute is @process_slot without refill.
void SkinnyGEMMEmitter::emitTail(KLoopState &state) {
  TileState &tile = state.tile;
  RegisterRef hi = tile.tailBytes->ref();
  std::shared_ptr<rocisa::Label> stepEnd = builder.createLabel("step_end");
  builder.create<rocisa::SCmpEQU32>(tile.tailBytes->container(), 0);
  builder.create<rocisa::SCBranchSCC1>(stepEnd->getLabelName());
  RegisterRange iterOff = builder.allocateSgpr(1);
  builder.shli32(iterOff.ref(), tile.nIterFull.ref(), dims.log2Ki);
  builder.addi32(iterOff.ref(), iterOff.ref(), tile.chunkBase.regs.ref(0));
  std::optional<RegisterRange> kAvail;
  if (dims.upIters == 0) {
    kAvail = builder.allocateSgpr(1);
    builder.subi32(kAvail->ref(), k.ref(), iterOff.ref());
  }
  // The up-front vector starts at the split's first iteration, so the entry of
  // the tail iteration is the number of full iterations.
  std::optional<RegisterRef> vectorIndex;
  if (dims.upIters >= 2)
    vectorIndex = tile.nIterFull.ref();
  // The PF = 0 loop released its raw scales, and the step takes their registers
  // back before anything else, so that a smaller value does not split the hole
  // they leave.
  if (dims.upIters == 0 && dims.prefetch == 0)
    state.ring.raw = builder.allocateVgpr(dims.columns);
  {
    // v0 holds the lane's index in its cluster after the setup.
    RegisterRange laneBytes =
        computeLaneBytes(builder, dims, kernel.workitemId().ref());
    if (dims.upIters == 0)
      issueScalesWithinK(state, rawScales(state, 0), iterOff.ref(),
                         kAvail->ref());
    // Bit u of partialChunks is @straddles_in_chunk(hi, u): the bound is not a
    // multiple of 16 and lies in the chunk (hi / (16 S)) mod 4, whose part of
    // one lane per column holds it.
    RegisterRange partialChunks = builder.allocateSgpr(1);
    {
      RegisterRange misaligned = builder.allocateSgpr(1);
      builder.shrui32(partialChunks.ref(), hi, log2Exact(dims.chunkStride));
      builder.andi32(partialChunks.ref(), SkinnyDims::chunks - 1,
                     partialChunks.ref());
      builder.shli32(partialChunks.ref(), 1, partialChunks.ref());
      builder.andi32(misaligned.ref(), partBytes - 1, hi);
      builder.selectb32(partialChunks.ref(), partialChunks.ref(), 0);
    }
    // Loads chunk u of the step for every column, in one of three arms. At most
    // one lane per bound straddles it, in the chunk of its bound: that chunk
    // loads dwords (@load_tail_dwords). In every other chunk each lane lies
    // fully inside or outside [0, hi), so steering the voffset of the outside
    // lanes to -1 suffices, and a chunk with no live lane is zeroed.
    for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
      int32_t offset = chunk * dims.chunkStride;
      // All SALU first: s_add writes SCC, which the branches read.
      std::optional<RegisterRange> soffsetStorage;
      std::optional<RegisterRange> hiStorage;
      RegisterRef soffset = iterOff.ref();
      RegisterRef chunkHi = hi;
      if (chunk > 0) {
        soffsetStorage = builder.allocateSgpr(1);
        soffset = soffsetStorage->ref();
        builder.addi32(soffset, iterOff.ref(), offset);
        hiStorage = builder.allocateSgpr(1);
        chunkHi = hiStorage->ref();
        builder.addi32(chunkHi, hi, -offset);
      }
      // The bytes of the lane's 16 below hi.
      RegisterRange laneRem = builder.allocateVgpr(1);
      builder.subi32(laneRem.ref(), chunkHi, laneBytes.ref());
      std::shared_ptr<rocisa::Label> notPartial =
          builder.createLabel("step_load");
      std::shared_ptr<rocisa::Label> zero = builder.createLabel("step_zero");
      std::shared_ptr<rocisa::Label> chunkEnd =
          builder.createLabel("step_chunk_end");
      builder.create<rocisa::SBitcmp1B32>(partialChunks.container(), chunk);
      builder.create<rocisa::SCBranchSCC0>(notPartial->getLabelName());
      loadTailDwords(state, chunk, soffset, laneRem.ref());
      builder.create<rocisa::SBranch>(chunkEnd->getLabelName());
      builder.insert(notPartial);
      // A necessary condition for any lane of the chunk to lie inside [0, hi);
      // when it holds but no lane qualifies, the steered loads read 0.
      builder.create<rocisa::SCmpGeI32>(chunkHi.container(), partBytes);
      builder.create<rocisa::SCBranchSCC0>(zero->getLabelName());
      {
        RegisterRange neg = builder.allocateVgpr(1);
        signOfSum(builder, neg.ref(), laneRem.ref(), -partBytes);
        issueChunk(state, 0, chunk, soffset, neg.ref());
      }
      builder.create<rocisa::SBranch>(chunkEnd->getLabelName());
      builder.insert(zero);
      // The slot's registers may still have a prologue load in flight when the
      // split has no full iteration; the step's scale loads are issued on every
      // path and complete after it.
      waitForPrologueLoads(
          builder, tile.prologueLoads,
          {state.ring.data.ref(chunk * dims.columns * dwordsPerChunk,
                               dims.columns * dwordsPerChunk)},
          dims.scaleLoads);
      for (int32_t column = 0; column < dims.columns; ++column)
        for (int32_t dword = 0; dword < dwordsPerChunk; ++dword)
          builder.movb32(dataChunk(state, 0, chunk, column).slice(dword), 0);
      builder.insert(chunkEnd);
    }
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
  emitCompute(state, SlotContext{0, 0, std::nullopt, true, vectorIndex,
                                 /*partialStep=*/true});
  builder.insert(stepEnd);
}

// C prefetch and epilogue.

namespace {

// Which variant of a vector-or-masked access follows the branch.
enum class FallThrough { masked, vector };

// Where one store path writes.
struct StoreTarget {
  RegisterRef rsrc;
  RegisterRef strideS;
  RegisterRef strideW;
  std::optional<RegisterRef> sliceBase;
  bool bf16;
};

// The scalars of a store path, in bytes.
struct StoreScalars {
  RegisterRange base, scratch, offset;
};

// Registers and flags of one epilogue variant. The C prefetch also builds one,
// where red names the C data registers and off, wBase and leader are its
// temporaries.
struct EpilogueState {
  IRBuilder &builder;
  const SkinnyDims &dims;
  const TileState &tile;
  RedLayout layout;
  RegisterRef red;
  RegisterRef off;
  std::optional<RegisterRef> wBase;
  RegisterRef leader;
  std::optional<RegisterRef> tileFull;
};

// The control of one DPP add of the reduction.
struct DppStep {
  int32_t rowBcast;
  std::vector<int32_t> quadPerm;
  bool rowMirror;
  bool rowHalfMirror;
  int32_t rowMask;
  int32_t boundCtrl;
};

} // namespace

// Elements of a bf16 run in the C registers pack in pairs, each run starting
// at an even register so that every dwordx2 and dwordx4 load is aligned.
static int32_t cRunStride(int32_t length) { return (length / 2 + 1) & ~1; }

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

// Calls emit(r, t, length) for every run of the orientation, the runs lying
// along the skinny dimension when skinnyContiguous and along the wide one else.
static void forEachRun(
    const SkinnyDims &dims, bool skinnyContiguous,
    const std::function<void(int32_t r, int32_t t, int32_t length)> &emit) {
  int32_t runs = skinnyContiguous ? dims.tn : dims.bm;
  for (int32_t run = 0; run < runs; ++run) {
    if (skinnyContiguous)
      emit(0, run, dims.bm);
    else
      emit(run, 0, dims.tn);
  }
}

// Emits the code of the epilogue and of the C prefetch for the orientation of
// the output, which contigS selects at run time.
static void forEachOrientation(IRBuilder &builder, const SkinnyDims &dims,
                               RegisterRef contigS,
                               const std::function<void(bool)> &emitVariant) {
  bool vectorA = runLength(dims, true) >= 2;
  bool vectorB = runLength(dims, false) >= 2;
  // Without vector stores the two orientations are the same code.
  if (!vectorA && !vectorB) {
    emitVariant(true);
    return;
  }
  std::shared_ptr<rocisa::Label> orientationB =
      builder.createLabel("orientation_b");
  std::shared_ptr<rocisa::Label> end = builder.createLabel("epilogue_end");
  builder.create<rocisa::SCmpEQU32>(contigS.container(), 1);
  builder.create<rocisa::SCBranchSCC0>(orientationB->getLabelName());
  emitVariant(true);
  builder.create<rocisa::SBranch>(end->getLabelName());
  builder.insert(orientationB);
  emitVariant(false);
  builder.insert(end);
}

// Emits the masked or the vector variant of a tile access. The vector one runs
// when the tile lies inside the output; fallThrough names the variant that
// follows the branch, the other sits behind targetLabel.
static void emitVectorOrMasked(const EpilogueState &state, bool vectorSupported,
                               FallThrough fallThrough, const char *targetLabel,
                               const char *doneLabel,
                               const std::function<void()> &emitMasked,
                               const std::function<void()> &emitVector) {
  if (!state.tileFull || !vectorSupported) {
    emitMasked();
    return;
  }
  IRBuilder &builder = state.builder;
  std::shared_ptr<rocisa::Label> target = builder.createLabel(targetLabel);
  std::shared_ptr<rocisa::Label> done = builder.createLabel(doneLabel);
  builder.create<rocisa::SCmpEQU32>(state.tileFull->container(), 1);
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
static void computeTileFull(IRBuilder &builder, const SkinnyDims &dims,
                            const TileIds &ids, RegisterRef skinnyCols,
                            RegisterRef wideCols, RegisterRef tileFull) {
  RegisterRange skinnyEnd = builder.allocateSgpr(1);
  RegisterRange wideEnd = builder.allocateSgpr(1);
  RegisterRange skinnyIn = builder.allocateSgpr(1);
  RegisterRange wideIn = builder.allocateSgpr(1);
  builder.addi32(skinnyEnd.ref(), ids.ty, 1);
  builder.muli32(skinnyEnd.ref(), skinnyEnd.ref(), dims.bm);
  builder.addi32(wideEnd.ref(), ids.tx, 1);
  builder.muli32(wideEnd.ref(), wideEnd.ref(), dims.tileWide);
  builder.cmpi32(Predicate::ule, skinnyIn.ref(), skinnyEnd.ref(), skinnyCols);
  builder.cmpi32(Predicate::ule, wideIn.ref(), wideEnd.ref(), wideCols);
  builder.andi32(tileFull, skinnyIn.ref(), wideIn.ref());
}

// The scalars of one store path, in bytes: the byte offset of the tile's first
// element without the per-lane part of the wide column, which goes to the
// wBase register of the state (or into the scalar offset when w0 is uniform).
static StoreScalars makeStoreScalars(const EpilogueState &state,
                                     const StoreTarget &target) {
  IRBuilder &builder = state.builder;
  StoreScalars scalars{builder.allocateSgpr(1), builder.allocateSgpr(1),
                       builder.allocateSgpr(1)};
  const TileState &tile = state.tile;
  builder.muli32(scalars.base.ref(), tile.s0.ref(), target.strideS);
  if (target.sliceBase)
    builder.addi32(scalars.base.ref(), scalars.base.ref(), *target.sliceBase);
  if (state.wBase) {
    builder.muli32(*state.wBase, target.strideW, tile.w0.ref());
    return scalars;
  }
  builder.muli32(scalars.scratch.ref(), tile.w0.ref(), target.strideW);
  builder.addi32(scalars.base.ref(), scalars.base.ref(), scalars.scratch.ref());
  return scalars;
}

// off = the byte offset of tile element (r, t) for this lane, (s0 + r) *
// strideS + (w0 + t) * strideW + sliceBase, with all strides in bytes.
static void computeOffset(const EpilogueState &state, const StoreTarget &target,
                          const StoreScalars &scalars, int32_t r, int32_t t) {
  IRBuilder &builder = state.builder;
  RegisterRef uniform = scalars.base.ref();
  if (r != 0 || t != 0) {
    uniform = scalars.offset.ref();
    builder.movb32(uniform, scalars.base.ref());
    if (r != 0) {
      builder.muli32(scalars.scratch.ref(), target.strideS, r);
      builder.addi32(uniform, uniform, scalars.scratch.ref());
    }
    if (t != 0) {
      builder.muli32(scalars.scratch.ref(), target.strideW, t);
      builder.addi32(uniform, uniform, scalars.scratch.ref());
    }
  }
  if (state.wBase) {
    builder.addi32(state.off, uniform, *state.wBase);
    return;
  }
  builder.movb32(state.off, uniform);
}

// Sets off of the element (r, t) to out of range unless the element is inside
// the output and the lane is a leader.
static void maskedOffset(const EpilogueState &state, RegisterRef mask,
                         int32_t r, int32_t t) {
  IRBuilder &builder = state.builder;
  builder.andi64(mask, state.tile.sValid.ref(2 * r, 2),
                 state.tile.wValid.ref(2 * t, 2));
  builder.andi64(mask, mask, state.leader);
  builder.selectb32(state.off, mask, state.off, outOfRange);
}

// Loads the C tile with the addressing of the D store, through the vector
// loads when the tile lies inside the output and the orientation has even
// runs, else per element.
static void loadCTile(const EpilogueState &state, const StoreTarget &target) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
  StoreScalars scalars = makeStoreScalars(state, target);
  bool skinnyContiguous = state.layout.skinnyContiguous;
  // One zero-extending 16-bit load per element, into dword redIndex of the C
  // registers; the lanes that are not leaders and the elements outside the
  // output load out of range, so they cause no traffic.
  auto loadMasked = [&] {
    RegisterRange mask = builder.allocateSgpr(2, 2);
    for (int32_t r = 0; r < dims.bm; ++r) {
      for (int32_t t = 0; t < dims.tn; ++t) {
        computeOffset(state, target, scalars, r, t);
        maskedOffset(state, mask.ref(), r, t);
        builder.loadBuffer(state.red.slice(redIndex(dims, state.layout, r, t)),
                           target.rsrc, state.off, 0, MemoryWidth::b16);
      }
    }
  };
  // The runs load with the widest chunks first, as the D store does: dwordx4,
  // dwordx2, dword. Chunk first / 2 lands at that dword of the run's
  // registers.
  auto loadVector = [&] {
    forEachRun(
        dims, skinnyContiguous, [&](int32_t r, int32_t t, int32_t length) {
          int32_t run = skinnyContiguous ? t : r;
          forEachRunChunk(length, 8, [&](int32_t first, int32_t chunk) {
            computeOffset(state, target, scalars,
                          skinnyContiguous ? r + first : r,
                          skinnyContiguous ? t : t + first);
            builder.selectb32(state.off, state.leader, state.off, outOfRange);
            int32_t dwords = chunk / 2;
            RegisterRef data =
                state.red.slice(run * cRunStride(length) + first / 2, dwords);
            MemoryWidth width = dwords == 1   ? MemoryWidth::b32
                                : dwords == 2 ? MemoryWidth::b64
                                              : MemoryWidth::b128;
            builder.loadBuffer(data, target.rsrc, state.off, 0, width);
          });
        });
  };
  emitVectorOrMasked(state, hasVector(dims, skinnyContiguous, true),
                     FallThrough::masked, "c_vector", "c_loaded", loadMasked,
                     loadVector);
}

// Stores the tile into target: through the vector stores when the tile lies
// inside the output and the orientation has them, else per element with the
// out-of-range offset masking the tail columns and the non-leader lanes.
static void storeTile(const EpilogueState &state, const StoreTarget &target) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
  bool skinnyContiguous = state.layout.skinnyContiguous;
  // One store per element: bf16 is the truncation, the top 16 bits of the f32.
  auto storeMasked = [&] {
    StoreScalars scalars = makeStoreScalars(state, target);
    RegisterRange mask = builder.allocateSgpr(2, 2);
    for (int32_t r = 0; r < dims.bm; ++r) {
      for (int32_t t = 0; t < dims.tn; ++t) {
        RegisterRef value = state.red.slice(redIndex(dims, state.layout, r, t));
        if (target.bf16)
          builder.shrui32(value, value, 16);
        computeOffset(state, target, scalars, r, t);
        maskedOffset(state, mask.ref(), r, t);
        if (target.bf16)
          builder.storeBuffer(value, target.rsrc, state.off, 0,
                              MemoryWidth::b16);
        else
          builder.storeBuffer(value, target.rsrc, state.off, 0,
                              MemoryWidth::b32);
      }
    }
  };
  // Stores the elements [first, first + elements) of the run that starts at
  // (r, t) with the widest chunks first: dwordx4, dwordx2, dword. A bf16 run
  // has an even length.
  auto storeChunk = [&](const StoreScalars &scalars, int32_t r, int32_t t,
                        int32_t first, int32_t elements) {
    int32_t rc = skinnyContiguous ? r + first : r;
    int32_t tc = skinnyContiguous ? t : t + first;
    computeOffset(state, target, scalars, rc, tc);
    builder.selectb32(state.off, state.leader, state.off, outOfRange);
    int32_t start = redIndex(dims, state.layout, rc, tc);
    int32_t dwords = elements;
    if (target.bf16) {
      // Pack pairs of truncated values in place; dword k reads the registers
      // 2k and 2k + 1, which no earlier dword has overwritten.
      dwords = elements / 2;
      for (int32_t j = 0; j < elements; ++j) {
        RegisterRef value = state.red.slice(start + j);
        builder.shrui32(value, value, 16);
      }
      for (int32_t k = 0; k < dwords; ++k)
        builder.create<rocisa::VPackF16toB32>(
            state.red.slice(start + k).container(),
            state.red.slice(start + 2 * k).container(),
            state.red.slice(start + 2 * k + 1).container());
    }
    RegisterRef data = state.red.slice(start, dwords);
    MemoryWidth width = dwords == 1   ? MemoryWidth::b32
                        : dwords == 2 ? MemoryWidth::b64
                                      : MemoryWidth::b128;
    builder.storeBuffer(data, target.rsrc, state.off, 0, width);
  };
  auto storeVector = [&] {
    StoreScalars scalars = makeStoreScalars(state, target);
    forEachRun(dims, skinnyContiguous,
               [&](int32_t r, int32_t t, int32_t length) {
                 forEachRunChunk(length, target.bf16 ? 8 : 4,
                                 [&](int32_t first, int32_t chunk) {
                                   storeChunk(scalars, r, t, first, chunk);
                                 });
               });
  };
  emitVectorOrMasked(state, hasVector(dims, skinnyContiguous, target.bf16),
                     FallThrough::masked, "vector", "tile_stored", storeMasked,
                     storeVector);
}

// Wait states between a v_dot2 write and a DPP read of its result (LLVM's
// DotWriteDifferentVALURead) and between a VALU write and a DPP read of it.
static constexpr int32_t dotToDppWaitStates = 3;
static constexpr int32_t valuToDppWaitStates = 2;
// Wait states of v_readlane_b32 after a VALU wrote its source, and of a VALU
// read of the SGPR that v_readlane_b32 wrote (gfx940, per LLVM's hazard
// recognizer; the reference omits the second).
static constexpr int32_t valuToReadlaneWaitStates = 1;
static constexpr int32_t readlaneToValuWaitStates = 2;
static constexpr int32_t bankMaskAll = 0xf;

// Reduces every accumulator over its LANES_K lanes into the red registers:
// DPP adds that leave the cluster sum in every lane, which a v_readlane of
// lane 63 broadcasts for 64 lanes. It is a function for the independence of
// the algorithm, which keeps emitEpilogue short.
static void reduceAccumulators(const EpilogueState &state) {
  IRBuilder &builder = state.builder;
  const SkinnyDims &dims = state.dims;
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
        state.red.slice(redIndex(dims, state.layout, p / dims.tn, p % dims.tn));
    emitDpp(dst, state.tile.acc.ref(p),
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

// Prefetches the tile of C into cData, for the epilogue's beta * C, when the
// tile stores D directly (k <= spk) and beta is not zero; else it only costs
// the scalar tests. The loads are issued before every scale and ring load of
// the tile and after everything older, such as the stores of the previous tile
// of a persistent block, and the temporaries are dead before the ring loads.
//
// vmcnt counts the vector memory operations of a wave in issue order. Every
// wait either allows the N loads issued after its target or drains everything
// (the vmcnt(0) waits in loadTailDwords and the persistent shared-scales
// drain), so the older C loads are complete whenever its target is, and both
// forms are safe with them.
// The epilogue consumes C after an explicit vmcnt(0). C may alias D: each lane
// loads its elements here and stores them in the epilogue, and the tiles are
// disjoint, so the next tile of a persistent block loads after this tile's
// epilogue consumed cData.
void SkinnyGEMMEmitter::emitCPrefetch(const TileState &tile,
                                      const TileIds &ids) {
  std::shared_ptr<rocisa::Label> skip = builder.createLabel("c_prefetch_skip");
  builder.create<rocisa::SCmpLeU32>(k.container(), spk.container());
  builder.create<rocisa::SCBranchSCC0>(skip->getLabelName());
  {
    RegisterRange magnitude = builder.allocateSgpr(1);
    branchIfBetaZero(builder, beta->ref(), magnitude.ref(), skip);
  }
  RegisterRange leader = builder.allocateSgpr(2, 2);
  builder.create<rocisa::VCmpEQI32>(leader.container(), 0,
                                    kernel.workitemId().container());
  RegisterRange off = builder.allocateVgpr(1);
  std::optional<RegisterRange> wBase;
  if (!dims.uniformWide)
    wBase = builder.allocateVgpr(1);
  forEachOrientation(builder, dims, contigS.ref(), [&](bool skinnyContiguous) {
    RegisterRange tileFull = builder.allocateSgpr(1);
    bool vector = hasVector(dims, skinnyContiguous, true);
    if (vector)
      computeTileFull(builder, dims, ids, skinnyCols.ref(), wideCols.ref(),
                      tileFull.ref());
    EpilogueState state{
        builder,
        dims,
        tile,
        redLayout(dims, skinnyContiguous),
        tile.cData->ref(),
        off.ref(),
        wBase ? std::optional<RegisterRef>(wBase->ref()) : std::nullopt,
        leader.ref(),
        vector ? std::optional<RegisterRef>(tileFull.ref()) : std::nullopt};
    loadCTile(state, StoreTarget{cRsrc->ref(), cStrideS->ref(), cStrideW->ref(),
                                 std::nullopt, true});
  });
  builder.insert(skip);
}

// Reduces the accumulators over the lanes of a cluster, scales them by alpha,
// adds beta * C, and stores the tile: bf16 straight to D for a single split,
// else the f32 partial slice into the workspace.
void SkinnyGEMMEmitter::emitEpilogue(const TileState &tile,
                                     const TileIds &ids) {
  forEachOrientation(builder, dims, contigS.ref(), [&](bool skinnyContiguous) {
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
    RegisterRange tileFull = builder.allocateSgpr(1);
    bool anyVector = hasVector(dims, skinnyContiguous, true) ||
                     hasVector(dims, skinnyContiguous, false);
    EpilogueState state{
        builder,
        dims,
        tile,
        layout,
        red.ref(),
        off.ref(),
        wBase ? std::optional<RegisterRef>(wBase->ref()) : std::nullopt,
        leader.ref(),
        anyVector ? std::optional<RegisterRef>(tileFull.ref()) : std::nullopt};
    // The epilogue overwrites the ring registers, and a prologue load that was
    // never consumed (a split with fewer than PF iterations) may still be in
    // flight.
    std::vector<RegisterRef> written = {state.red, state.off};
    if (state.wBase)
      written.push_back(*state.wBase);
    waitForPrologueLoads(builder, tile.prologueLoads, written,
                         /*issuedSince=*/0);
    reduceAccumulators(state);
    if (anyVector)
      computeTileFull(builder, dims, ids, skinnyCols.ref(), wideCols.ref(),
                      tileFull.ref());
    builder.create<rocisa::VCmpEQI32>(leader.container(), 0, kl.container());
    // A single split stores bf16 straight to D, else each split stores its f32
    // partial slice into the workspace.
    std::shared_ptr<rocisa::Label> workspace = builder.createLabel("workspace");
    std::shared_ptr<rocisa::Label> done = builder.createLabel("stored");
    builder.create<rocisa::SCmpLeU32>(k.container(), spk.container());
    builder.create<rocisa::SCBranchSCC0>(workspace->getLabelName());
    // Multiplies the reduced values by alpha, in f32 and before the bf16
    // truncation, two at a time with v_pk_mul_f32 where the pair is even
    // aligned. The pair reads the low dword of alpha for both halves
    // (op_sel_hi).
    int32_t length =
        layout.runs ? runLength(dims, skinnyContiguous) : dims.accumulators;
    int32_t segments = dims.accumulators / length;
    int32_t segmentStride = layout.runs ? layout.stride : length;
    for (int32_t segment = 0; segment < segments; ++segment) {
      for (int32_t i = 0; i < length;) {
        RegisterRef values = state.red.slice(segment * segmentStride + i);
        bool pair = i + 1 < length && values.start % 2 == 0;
        if (!pair) {
          builder.create<rocisa::VMulF32>(values.container(),
                                          alpha.ref(0, 1).container(),
                                          values.container());
          ++i;
          continue;
        }
        values = state.red.slice(segment * segmentStride + i, 2);
        builder.create<rocisa::VMulPKF32>(values.container(), alpha.container(),
                                          values.container(),
                                          rocisa::VOP3PModifiers({}, {0, 1}));
        i += 2;
      }
    }
    if (tile.cData) {
      // red = fma(beta, C, red) for the prefetched C tile, skipped when beta is
      // zero as the prefetch is. The C loads are older than every other load of
      // the tile, so one vmcnt(0) completes them; the temporary is allocated
      // after it because it may reuse a ring register that an unconsumed
      // prologue load writes.
      std::shared_ptr<rocisa::Label> noBeta = builder.createLabel("no_beta");
      {
        RegisterRange magnitude = builder.allocateSgpr(1);
        branchIfBetaZero(builder, beta->ref(), magnitude.ref(), noBeta);
      }
      waitForVectorLoads(builder, 0);
      RegisterRange c = builder.allocateVgpr(1);
      int32_t stride = cRunStride(runLength(dims, skinnyContiguous));
      // The vector path loaded runs of dword pairs: element j of run q is the
      // half j % 2 of dword q * stride + j / 2. The masked path loaded one
      // zero-extended element per dword.
      auto addC = [&](bool vector) {
        for (int32_t r = 0; r < dims.bm; ++r) {
          for (int32_t t = 0; t < dims.tn; ++t) {
            int32_t index = redIndex(dims, layout, r, t);
            int32_t run = skinnyContiguous ? t : r;
            int32_t position = skinnyContiguous ? r : t;
            RegisterRef src =
                vector ? tile.cData->ref(run * stride + position / 2, 1)
                       : tile.cData->ref(index, 1);
            unpackBf16(builder, c.ref(), src, vector && position % 2 == 1);
            builder.create<rocisa::VMacF32>(state.red.slice(index).container(),
                                            beta->container(), c.container());
          }
        }
      };
      emitVectorOrMasked(
          state, hasVector(dims, skinnyContiguous, true), FallThrough::vector,
          "c_masked", "c_added", [&] { addC(false); }, [&] { addC(true); });
      builder.insert(noBeta);
    }
    storeTile(state, StoreTarget{dRsrc.ref(), dStrideS.ref(), dStrideW.ref(),
                                 std::nullopt, true});
    builder.create<rocisa::SBranch>(done->getLabelName());
    builder.insert(workspace);
    RegisterRange slice = builder.allocateSgpr(1);
    builder.muli32(slice.ref(), ids.kz, mn.ref());
    builder.shli32(slice.ref(), slice.ref(), 2);
    storeTile(state, StoreTarget{wsRsrc.ref(), wsStrideS.ref(), wsStrideW.ref(),
                                 slice.ref(), false});
    builder.insert(done);
  });
}

// Split-K reduce.

// Matches #red_row and the gpu.block_dims of the MLIR.
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

// Emits @splitk_reduce of the MLIR reference: emit loads the arguments and
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
}

// The kernarg layout of the reference: the pointers, the i32 and f32 scalars,
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
         std::to_string(config.useBeta ? 1 : 0);
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
  if (dims.scalesShared)
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
