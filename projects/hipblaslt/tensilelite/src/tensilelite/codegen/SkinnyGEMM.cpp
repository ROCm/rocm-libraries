// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The skinny MXFP8 GEMM emitter, which implements @gemm, @run_tile and their
// helpers of gemm_mxfp8_skinny.mlir for both launch forms (GRID_3D = 1 and the
// persistent GRID_3D = 0), any PARTIAL_STEPS, ring scales (UP_ITERS = 0) or
// up-front scales, plus @splitk_reduce. The method names follow the MLIR
// functions.

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Kernel.hpp>
#include <tensilelite/codegen/Registers.hpp>
#include <tensilelite/codegen/SkinnyGEMM.hpp>

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/cvt.hpp"
#include "instruction/mem.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rocm::tensilelite::codegen;

// Types.

namespace {

/// Kernarg offsets of the GEMM kernel's arguments.
struct SkinnyArguments {
  int d, a, b, scalesA, scalesB, ws;
  int ldD, ldA, ldB, m, n, k, spk;
  int blocksX, blocksY, blocksZ;
};

/// A ds address register and the offset field that, together, name base +
/// offset.
struct LdsAddress {
  RegisterRef base;
  int offset;
};

/// Constants derived from the knobs, named after the MLIR values.
struct SkinnyDims {
  explicit SkinnyDims(const SkinnyGEMMConfig &config);

  /// Chunks per iteration; every lane loads 4 dwordx4 per column.
  static constexpr int chunks = 4;

  int bm, tn, lanesK, waves, prefetch;
  /// GRID_3D; group and numXcds only apply to the persistent launch.
  bool grid3d;
  int group, numXcds;
  /// Whether the persistent launch reorders the tiles (GROUP or NUM_XCDS not
  /// 1), which changes the key of the K rotation.
  bool reordered;
  /// UP_ITERS up-front scale entries per split, 0 for ring scales.
  int upIters;
  /// PARTIAL_STEPS: 0, 1 (tail step only) or 2 (head and tail steps).
  int partialSteps;
  /// NC = BM + TN columns per lane.
  int columns;
  /// P = BM * TN accumulators per lane.
  int accumulators;
  /// CG = 64 / LANES_K column groups per wave.
  int columnGroups;
  /// BN = WAVES * CG * TN wide columns per tile.
  int tileWide;
  /// G: consecutive wide tiles that share their scale lines and so run on one
  /// XCD in the 3-D launch; 1 when BN is not a power of two below 32.
  int xcdRun;
  /// Wide columns of a scale row group, which share every 128-byte scale line.
  static constexpr int scaleRowColumns = 32;
  /// KI = 64 * LANES_K elements per iteration.
  int ki;
  /// S = min(LANES_K, 8) lanes per subgroup.
  int subgroupLanes;
  int halfSubgroup;
  int log2LanesK, log2ColumnGroups, log2Subgroup, log2Half, log2Ki;
  /// max(PF, 1) ring slots.
  int slots;
  /// Whether the up-front scales go through LDS (@scales_shared): several
  /// waves and CG < 4.
  bool scalesShared;
  /// R row slots of the scale load role domain: the block when shared, else
  /// the wave.
  int roleSlots;
  /// Ring scale dwords per slot, NC with ring scales and 0 with up-front
  /// scales.
  int scaleLoads;
  /// L loads per ring slot: the scale dwords of a ring-scale slot (scaleLoads
  /// of them, NC), then 4 chunks of NC.
  int loadsPerSlot;
  /// Bytes of LDS holding the shared up-front scales, UP * NC * WAVES * 256.
  int ldsBytes;
  /// Bytes between the chunks of a lane, 16 * S.
  int chunkStride;
};

/// Multiplier and shift of LLVM's signed division by a positive constant.
struct SignedMagic {
  int32_t multiplier;
  int shift;
};

/// Loads that stay in flight at a point of the schedule, counted in issue
/// order so that an in-order vmcnt wait can name them.
struct SlotLoadPosition {
  /// Position of the load among the loads of its slot.
  int position;
  /// Full ring slots issued after the slot of the load.
  int groupsAfter;
  /// Loads of the slot's own refill that were issued before the consumer.
  int ownRefill;
};

/// A slot of the prefetch ring being consumed.
struct SlotContext {
  int slot;
  /// Full ring slots issued after this slot's group of loads.
  int groupsAfter;
  /// Physical iteration to refill the slot with, an SGPR; empty without
  /// refill.
  std::optional<RegisterRef> refillIteration;
  /// Whether the chunk soffsets of the refill use the hoisted chunk bases, as
  /// the loops do, or add the chunk offset to the soffset of the iteration.
  bool hoistedOffsets = true;
  /// Physical iteration of the slot, an SGPR: the entry of the up-front scale
  /// vector, as the vector starts at the split's first iteration. Only set
  /// with at least 2 up-front entries.
  std::optional<RegisterRef> vectorIndex;
  /// Whether the slot is a partial step's, whose chunk loads sit behind dynamic
  /// branches that may load nothing.
  bool partialStep = false;
};

/// Emits the GEMM kernel body.
class SkinnyGEMMEmitter {
public:
  SkinnyGEMMEmitter(Kernel &kernel, const SkinnyGEMMConfig &config,
                    const SkinnyArguments &arguments);

  void emit();

private:
  // The tile of the workgroup as SGPRs: the workgroup ids in the 3-D launch,
  // the decoded tile of the current iteration in the persistent launch.
  struct TileIds {
    RegisterRef tx, ty, kz;
  };

  // SGPRs that the setup reads after the resources are made. blocksX is the
  // block count x, the number of persistent blocks in the 1-D launch, which has
  // no blocksY and keeps the split count of the workspace size in splits.
  struct SetupScalars {
    SetupScalars(IRBuilder &builder, bool grid3d);
    RegisterRange blocksX;
    std::optional<RegisterRange> blocksY;
    RegisterRange rowGroup, ldS, ldW;
    std::optional<RegisterRange> splits;
  };

  // SGPRs that only the resources and the path dispatch read; blocksZ, the
  // split count, only exists in the 3-D launch.
  struct ResourceScalars {
    ResourceScalars(IRBuilder &builder, bool grid3d);
    RegisterRange ldD, ldA, ldB, m, n;
    std::optional<RegisterRange> blocksZ;
    RegisterRange aPtr, bPtr, sclAPtr, sclBPtr;
    RegisterRange aBytes, bBytes, sclABytes, sclBBytes;
  };

  // Per-lane values, which only depend on the thread id. In the 3-D launch
  // w0Load is the wide column the lane loads scales for and the values die with
  // the setup. In the persistent launch they live through the tile loop, w0Lane
  // and w0Load are the lane part of the wide columns without the tile's tx * BN
  // and both are absent when the wide column is uniform.
  struct LaneValues {
    RegisterRange laneBytes, sclOff;
    std::optional<RegisterRange> w0Load, w0Lane;
  };

  // The values of the persistent tile loop that do not depend on the tile.
  struct TileLoop {
    // The number of tiles and the tiles per wide and skinny row, and tilesX *
    // tilesY.
    RegisterRange numTiles, tilesX, plane;
    // Only the grouped decode needs tilesY.
    std::optional<RegisterRange> tilesY;
    // NI, the iterations of a split, and the rotation of the block id.
    std::optional<RegisterRange> ni, blockRot;
    // The remap of the XCDs: tiles per XCD and the XCDs with one more.
    std::optional<RegisterRange> perXcd, tall;
    // GROUP * tilesX, the tiles of a band.
    std::optional<RegisterRange> bandSize;
  };

  // Layout of the reduced accumulators: the natural p order, or runs of
  // consecutive output elements, each starting at an even register, for the
  // vector stores of one orientation.
  struct RedLayout {
    bool runs;
    bool skinnyContiguous;
    int stride;
    int total;
  };

  // The control of one DPP add of the reduction.
  struct DppStep {
    int rowBcast;
    std::vector<int> quadPerm;
    bool rowMirror;
    bool rowHalfMirror;
    int rowMask;
    int boundCtrl;
  };

  // Registers and flags of one epilogue variant.
  struct EpilogueState {
    RedLayout layout;
    RegisterRef red;
    RegisterRef off;
    std::optional<RegisterRef> wBase;
    RegisterRef leader;
    std::optional<RegisterRef> tileFull;
  };

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

  // Setup.
  void setupResources(SetupScalars &scalars);
  void makeBufferSizes(const SetupScalars &scalars,
                       const ResourceScalars &resources);
  void dispatchPaths(const SetupScalars &scalars,
                     const ResourceScalars &resources);
  void computeTile(const SetupScalars &scalars,
                   std::optional<RegisterRef> rotation = std::nullopt);
  void computeRotation(const SetupScalars &scalars, RegisterRef rot);
  void groupTilesPerXcd(const SetupScalars &scalars);
  void blockRotation(RegisterRef rot, RegisterRef key,
                     std::optional<RegisterRef> ni);
  LaneValues computeLaneValues();
  void computeLaneWideBase(RegisterRef dst);
  void addTileBase(RegisterRef dst);
  RegisterRange computeLaneBytes(RegisterRef klv);
  RegisterRange computeRoleJ(RegisterRef tid);
  RegisterRange computeScaleOffset(RegisterRef lane,
                                   const std::optional<RegisterRange> &roleJ);
  void computeWideLoadSlot(RegisterRef slot, RegisterRef tid, RegisterRef lane,
                           const std::optional<RegisterRange> &roleJ);
  void computeAddresses(RegisterRef klv, RegisterRef lane, RegisterRef tid);
  void fillDataBases(const SetupScalars &scalars, LaneValues &lane);
  void fillScaleBases(const SetupScalars &scalars, const LaneValues &lane,
                      RegisterRef w0Load);
  void fillSelectors();
  void zeroAccumulators();
  RegisterRef selector(int column) const;
  void selectorValue(RegisterRef dst, RegisterRef col);

  // Persistent launch.
  TileLoop computeTileLoop(SetupScalars &scalars);
  void runPersistentTile(const SetupScalars &scalars, LaneValues &lane,
                         const TileLoop &loop);
  void decodeTile(const TileLoop &loop);

  // K loop.
  void runTile();
  void runNoPrefetchLoop();
  void runPrefetchedLoop();
  void emitPrologue();
  void emitSteadyLoop(RegisterRef steadyEnd);
  void emitRemainderArm(RegisterRef left, RegisterRef steadyEnd, int remaining);
  void hoistChunkBases();
  void ensureChunkBase(int chunk);
  void wrapIteration(RegisterRef dst, RegisterRef sum);
  void issueScales(RegisterRef raw, RegisterRef soffset,
                   std::optional<RegisterRef> neg = std::nullopt);
  void issueChunk(int slot, int chunk, RegisterRef soffset,
                  std::optional<RegisterRef> neg = std::nullopt);
  void refillOffset(const SlotContext &context, RegisterRef soffset);
  void startRefill(const SlotContext &context, RegisterRef soffset);
  void refillChunk(const SlotContext &context, int chunk, RegisterRef soffset);
  void waitForVectorLoads(int outstanding);
  void gatherSlotScales(RegisterRef raw, RegisterRef lo, int firstOutstanding);
  void permSlotScales(RegisterRef hi, RegisterRef lo, RegisterRef out);
  void decodeScales(RegisterRef dst, RegisterRef packed, int chunk);
  void waitBeforeFirstUse(const SlotContext &context, int chunk, int column,
                          int ownRefill);
  void computeChunk(const SlotContext &context, int chunk, RegisterRef scales);
  void processSlot(const SlotContext &context);
  void processDecodeFirst(const SlotContext &context, RegisterRange &packed);
  void processUpFront(const SlotContext &context,
                      std::optional<RegisterRange> &packed);
  RegisterRef dataChunk(int slot, int chunk, int column) const;
  RegisterRef rawScales(int slot) const;
  RegisterRef dataRsrc(int column) const;

  // Up-front scales.
  void issueUpFrontScales();
  void packScaleVectors();
  void stageScalesInLds();
  void readSharedEntry(int entry);
  std::optional<RegisterRange> selectScales(const SlotContext &context);
  void decodeHoistedScales();
  int upFrontLoadsAfter(int index) const;
  RegisterRef vectorEntry(RegisterRef physical,
                          std::optional<RegisterRange> &storage);

  // Partial steps.
  //
  // The bounds of a partial step (@run_partial_step), SGPRs: the K offset of
  // its iteration, the bytes [lo, hi) of it that the split keeps (lo is absent
  // when it is the literal 0 of a tail-only step), the K extent from the
  // iteration on (ring scales) and the up-front vector entry (two entries or
  // more), with laneBytes the VGPR of the lane's byte offset in a chunk 0.
  struct PartialStep {
    RegisterRef iterOff;
    std::optional<RegisterRef> lo;
    RegisterRef hi;
    std::optional<RegisterRef> kAvail;
    std::optional<RegisterRef> vectorIndex;
    RegisterRef laneBytes;
  };

  void allocateStepRawScales();
  void runTailStep();
  void runHeadAndTailSteps();
  void loadPartialStep(const PartialStep &step);
  void computePartialStep(const PartialStep &step);
  void issueScalesWithinK(RegisterRef raw, RegisterRef soffset,
                          RegisterRef kAvail);
  void straddlesInChunk(RegisterRef dst, RegisterRef bound);
  void loadStepChunk(const PartialStep &step, int chunk,
                     RegisterRef partialChunks);
  void loadTailDwords(int chunk, RegisterRef soffset, RegisterRef laneRem,
                      std::optional<RegisterRef> laneLo);
  void loadMaskedDword(int chunk, int column, int dword, RegisterRef voffset,
                       RegisterRef steer, RegisterRef soffset,
                       RegisterRef mask);

  // Epilogue.
  void emitEpilogue();
  void emitEpilogueVariant(bool skinnyContiguous);
  RedLayout redLayout(bool skinnyContiguous) const;
  int runLength(bool skinnyContiguous) const;
  int redIndex(const RedLayout &layout, int r, int t) const;
  void waitForPrologueLoads(const std::vector<RegisterRef> &written,
                            int issuedSince);
  void reduceAccumulators(const EpilogueState &state);
  void emitDpp(RegisterRef dst, RegisterRef src, const DppStep &step);
  void emitNops(int count);
  void computeTileFull(RegisterRef tileFull);
  void storeTile(const EpilogueState &state, const StoreTarget &target);
  StoreScalars makeStoreScalars(const EpilogueState &state,
                                const StoreTarget &target);
  void storeMasked(const EpilogueState &state, const StoreTarget &target);
  void storeRun(const EpilogueState &state, const StoreTarget &target,
                const StoreScalars &scalars, int r, int t, int length);
  void storeChunk(const EpilogueState &state, const StoreTarget &target,
                  const StoreScalars &scalars, int r, int t, int first,
                  int elements);
  void computeOffset(const EpilogueState &state, const StoreTarget &target,
                     const StoreScalars &scalars, int r, int t);
  bool hasVector(bool skinnyContiguous, bool bf16) const;

  RegisterRange sgpr(int count = 1, int alignment = 1);
  RegisterRange vgpr(int count = 1, int alignment = 1);
  bool uniformWide() const;

  Kernel &kernel;
  IRBuilder &builder;
  SkinnyDims dims;
  SkinnyArguments arguments;
  TileIds tile{};

  // VGPRs, in allocation order: the ring first so that its tuples are
  // aligned, then the values that live through the K loop.
  RegisterRange ringData;
  std::optional<RegisterRange> ringRaw;
  RegisterRange w0;
  RegisterRange acc;
  RegisterRange dBase;
  RegisterRange sBase;
  RegisterRange selectors;
  RegisterRange addresses;
  // Up-front scales: the thread's LDS staging address (shared only), the raw
  // entries that pack in place, and for one entry the decoded chunk scales.
  std::optional<RegisterRange> ldsStage;
  // With partial steps, the K offset of the 8-block scale group of the lane's
  // load role, sclOff & -256 (@scl_group).
  std::optional<RegisterRange> scaleGroup;
  std::optional<RegisterRange> scaleVector;
  std::vector<RegisterRange> hoistedScales;

  // SGPRs that live through the K loop and the epilogue.
  RegisterRange sRsrc, wRsrc, sSclRsrc, wSclRsrc, dRsrc, wsRsrc;
  RegisterRange k, spk, mn;
  RegisterRange dStrideS, dStrideW, wsStrideS, wsStrideW;
  RegisterRange skinnyCols, wideCols, contigS;
  RegisterRange s0;
  RegisterRange nIterFull;
  // The rotation, absent with one up-front entry, where it is the constant 0.
  std::optional<RegisterRange> fitRot;
  RegisterRange kBaseIter;
  // The split's K range [ks, ke) with partial steps: ks when the split can
  // start inside an iteration (PARTIAL_STEPS = 2), ke whenever it can end
  // inside one, and startIter = floor(ks / KI).
  std::optional<RegisterRange> splitStart;
  std::optional<RegisterRange> splitEnd;
  std::optional<RegisterRange> startIter;
  // The first full iteration's up-front vector entry: 1 when the split starts
  // inside an iteration, whose entry 0 belongs to the head step, else 0.
  std::optional<RegisterRange> vecOff;
  // chunkBase[0] = kBaseIter << log2KI and chunkBase[u] = chunkBase[0] + u * 16
  // * S, emitted when first needed.
  RegisterRange chunkBase;
  std::vector<bool> chunkBaseReady;
  RegisterRange sValid, wValid;

  // Destinations of the prologue loads in issue order, for the epilogue's
  // write-after-write wait.
  std::vector<RegisterRef> prologueLoads;
  bool recordLoads = false;
};

} // namespace

// Shared helpers.

static constexpr int outOfRange = -1;

// The modifiers of a buffer load or store addressed with a voffset.
static rocisa::MUBUFModifiers offen() {
  return rocisa::MUBUFModifiers(true, 0);
}

static bool isScalar(RegisterRef ref) { return ref.kind == RegisterKind::sgpr; }

// Returns whether value is a positive power of two.
static bool isPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// Returns log2 of a power of two.
static int log2Exact(int value) {
  if (!isPowerOfTwo(value))
    throw std::logic_error("value is not a power of two");
  int log = 0;
  while ((1 << log) < value)
    ++log;
  return log;
}

// Returns the magic number of the signed division by divisor >= 2, after
// Hacker's Delight.
static SignedMagic signedMagic(int32_t d) {
  constexpr uint32_t two31 = 0x80000000u;
  uint32_t ad = static_cast<uint32_t>(d);
  uint32_t t = two31;
  uint32_t anc = t - 1 - t % ad;
  int p = 31;
  uint32_t q1 = two31 / anc;
  uint32_t r1 = two31 - q1 * anc;
  uint32_t q2 = two31 / ad;
  uint32_t r2 = two31 - q2 * ad;
  uint32_t delta = 0;
  do {
    ++p;
    q1 *= 2;
    r1 *= 2;
    if (r1 >= anc) {
      ++q1;
      r1 -= anc;
    }
    q2 *= 2;
    r2 *= 2;
    if (r2 >= ad) {
      ++q2;
      r2 -= ad;
    }
    delta = ad - r2;
  } while (q1 < delta || (q1 == delta && r1 == 0));
  return {static_cast<int32_t>(q2 + 1), p - 32};
}

// 32-bit ALU shorthands that pick the SALU or VALU instruction from the kind
// of dst, which must not be an AGPR. A VALU src must be a VGPR.
static void shiftLeft(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                      int amount) {
  if (amount == 0) {
    if (dst != src)
      builder.movb32(dst, src);
    return;
  }
  if (isScalar(dst))
    builder.create<rocisa::SLShiftLeftB32>(dst.container(), amount,
                                           src.container());
  else
    builder.create<rocisa::VLShiftLeftB32>(dst.container(), amount,
                                           src.container());
}

static void shiftRight(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                       int amount) {
  if (amount == 0) {
    if (dst != src)
      builder.movb32(dst, src);
    return;
  }
  if (isScalar(dst))
    builder.create<rocisa::SLShiftRightB32>(dst.container(), amount,
                                            src.container());
  else
    builder.create<rocisa::VLShiftRightB32>(dst.container(), amount,
                                            src.container());
}

static void bitAnd(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                   int mask) {
  if (isScalar(dst))
    builder.create<rocisa::SAndB32>(dst.container(), mask, src.container());
  else
    builder.create<rocisa::VAndB32>(dst.container(), mask, src.container());
}

// dst = src * factor, a shift when factor is a power of two.
static void multiplyBy(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                       int factor) {
  if (isPowerOfTwo(factor)) {
    shiftLeft(builder, dst, src, log2Exact(factor));
    return;
  }
  builder.muli32(dst, src, factor);
}

// Emits quotient = dividend / divisor for a non-negative SGPR dividend and a
// divisor >= 2 that is not a power of two, as LLVM expands the division.
static void divideByConstant(IRBuilder &builder, RegisterRef quotient,
                             RegisterRef dividend, int divisor) {
  SignedMagic magic = signedMagic(divisor);
  RegisterRange sign = builder.allocateSgpr(1);
  builder.create<rocisa::SMulHII32>(quotient.container(), dividend.container(),
                                    magic.multiplier);
  if (magic.multiplier < 0)
    builder.addi32(quotient, quotient, dividend);
  if (magic.shift > 0)
    builder.create<rocisa::SAShiftRightI32>(quotient.container(), magic.shift,
                                            quotient.container());
  shiftRight(builder, sign.ref(), quotient, 31);
  builder.addi32(quotient, quotient, sign.ref());
}

// Number of vector memory loads issued after the load at where; the count a
// vmcnt wait must allow so that it is complete.
static int loadsIssuedAfter(const SkinnyDims &dims, SlotLoadPosition where) {
  int sameSlot = dims.loadsPerSlot - 1 - where.position;
  return sameSlot + where.groupsAfter * dims.loadsPerSlot + where.ownRefill;
}

SkinnyDims::SkinnyDims(const SkinnyGEMMConfig &config)
    : bm(config.bm), tn(config.tn), lanesK(config.lanesK), waves(config.waves),
      prefetch(config.prefetch), grid3d(config.grid3d), group(config.group),
      numXcds(config.numXcds),
      reordered(config.group != 1 || config.numXcds != 1),
      upIters(config.upIters), partialSteps(config.partialSteps) {
  columns = bm + tn;
  accumulators = bm * tn;
  columnGroups = 64 / lanesK;
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

static constexpr int selectorMaskStep = 0x01010101;

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
  shiftRight(builder, dst, dst, 5);
  builder.muli32(dst, dst, rowGroup);
}

// The scale base of a column of a row group matrix: (col / 32) * rowGroup +
// (col mod 16) * 4, before the lane's role offset. Overwrites col.
static void scaleRowOffset(IRBuilder &builder, RegisterRef dst, RegisterRef col,
                           RegisterRef rowGroup) {
  shiftRight(builder, dst, col, 5);
  builder.muli32(dst, dst, rowGroup);
  bitAnd(builder, col, col, 15);
  shiftLeft(builder, col, col, 2);
  builder.addi32(dst, dst, col);
}

// dst = SCC ? ifSet : otherwise for one register or an aligned pair.
static void selectOnScc(IRBuilder &builder, RegisterRef dst, RegisterRef ifSet,
                        RegisterRef otherwise) {
  if (dst.count == 2)
    builder.create<rocisa::SCSelectB64>(dst.container(), ifSet.container(),
                                        otherwise.container());
  else
    builder.create<rocisa::SCSelectB32>(dst.container(), ifSet.container(),
                                        otherwise.container());
}

SkinnyGEMMEmitter::SetupScalars::SetupScalars(IRBuilder &builder, bool grid3d)
    : blocksX(builder.allocateSgpr(1)),
      blocksY(grid3d ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                     : std::nullopt),
      rowGroup(builder.allocateSgpr(1)), ldS(builder.allocateSgpr(1)),
      ldW(builder.allocateSgpr(1)),
      splits(grid3d ? std::nullopt
                    : std::optional<RegisterRange>(builder.allocateSgpr(1))) {}

SkinnyGEMMEmitter::ResourceScalars::ResourceScalars(IRBuilder &builder,
                                                    bool grid3d)
    : ldD(builder.allocateSgpr(1)), ldA(builder.allocateSgpr(1)),
      ldB(builder.allocateSgpr(1)), m(builder.allocateSgpr(1)),
      n(builder.allocateSgpr(1)),
      blocksZ(grid3d ? std::optional<RegisterRange>(builder.allocateSgpr(1))
                     : std::nullopt),
      aPtr(builder.allocateSgpr(2, 2)), bPtr(builder.allocateSgpr(2, 2)),
      sclAPtr(builder.allocateSgpr(2, 2)), sclBPtr(builder.allocateSgpr(2, 2)),
      aBytes(builder.allocateSgpr(1)), bBytes(builder.allocateSgpr(1)),
      sclABytes(builder.allocateSgpr(1)), sclBBytes(builder.allocateSgpr(1)) {}

SkinnyGEMMEmitter::SkinnyGEMMEmitter(Kernel &kernel,
                                     const SkinnyGEMMConfig &config,
                                     const SkinnyArguments &arguments)
    : kernel(kernel), builder(kernel.builder()), dims(config),
      arguments(arguments),
      ringData(vgpr(dims.slots * SkinnyDims::chunks * dims.columns * 4, 2)),
      ringRaw(dims.upIters == 0 ? std::optional<RegisterRange>(
                                      vgpr(dims.slots * dims.columns))
                                : std::nullopt),
      w0(uniformWide() ? sgpr() : vgpr()), acc(vgpr(dims.accumulators)),
      dBase(vgpr(dims.columns)), sBase(vgpr(dims.columns)),
      selectors(vgpr((16 % dims.bm == 0 ? 1 : dims.bm) +
                     (16 % dims.tn == 0 ? 1 : dims.tn))),
      addresses(vgpr(dims.scalesShared ? 1 : 2)),
      ldsStage(dims.scalesShared ? std::optional<RegisterRange>(vgpr())
                                 : std::nullopt),
      scaleGroup(dims.partialSteps != 0 ? std::optional<RegisterRange>(vgpr())
                                        : std::nullopt),
      sRsrc(sgpr(4, 4)), wRsrc(sgpr(4, 4)), sSclRsrc(sgpr(4, 4)),
      wSclRsrc(sgpr(4, 4)), dRsrc(sgpr(4, 4)), wsRsrc(sgpr(4, 4)), k(sgpr()),
      spk(sgpr()), mn(sgpr()), dStrideS(sgpr()), dStrideW(sgpr()),
      wsStrideS(sgpr()), wsStrideW(sgpr()), skinnyCols(sgpr()),
      wideCols(sgpr()), contigS(sgpr()), s0(sgpr()), nIterFull(sgpr()),
      fitRot(dims.upIters != 1 ? std::optional<RegisterRange>(sgpr())
                               : std::nullopt),
      kBaseIter(sgpr()),
      splitStart(dims.partialSteps == 2 ? std::optional<RegisterRange>(sgpr())
                                        : std::nullopt),
      splitEnd(dims.partialSteps != 0 ? std::optional<RegisterRange>(sgpr())
                                      : std::nullopt),
      startIter(dims.partialSteps == 2 ? std::optional<RegisterRange>(sgpr())
                                       : std::nullopt),
      vecOff(dims.partialSteps == 2 && dims.upIters >= 2
                 ? std::optional<RegisterRange>(sgpr())
                 : std::nullopt),
      chunkBase(sgpr(4)), chunkBaseReady(4, false),
      sValid(sgpr(2 * dims.bm, 2)), wValid(sgpr(2 * dims.tn, 2)) {}

RegisterRange SkinnyGEMMEmitter::sgpr(int count, int alignment) {
  return builder.allocateSgpr(count, alignment);
}

RegisterRange SkinnyGEMMEmitter::vgpr(int count, int alignment) {
  return builder.allocateVgpr(count, alignment);
}

// With one wave of 64 lanes in one cluster the wide column of every lane is
// the same, so it stays in an SGPR.
bool SkinnyGEMMEmitter::uniformWide() const {
  return dims.waves == 1 && dims.lanesK == 64;
}

void SkinnyGEMMEmitter::emit() {
  if (!dims.grid3d) {
    SetupScalars scalars(builder, /*grid3d=*/false);
    setupResources(scalars);
    TileLoop loop = computeTileLoop(scalars);
    LaneValues lane = computeLaneValues();
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
    builder.createEndProgram();
    return;
  }
  tile = TileIds{kernel.workgroupIdX().ref(), kernel.workgroupIdY().ref(),
                 kernel.workgroupIdZ().ref()};
  {
    SetupScalars scalars(builder, /*grid3d=*/true);
    setupResources(scalars);
    computeTile(scalars);
    LaneValues lane = computeLaneValues();
    fillDataBases(scalars, lane);
    fillScaleBases(scalars, lane, lane.w0Load->ref());
    fillSelectors();
    zeroAccumulators();
  }
  runTile();
  emitEpilogue();
  builder.createEndProgram();
}

// The pointers load straight into the words 0 and 1 of the resources of D and
// the workspace; the others wait in pairs for the path dispatch.
void SkinnyGEMMEmitter::setupResources(SetupScalars &scalars) {
  ResourceScalars resources(builder, dims.grid3d);
  if (resources.blocksZ)
    kernel.loadArgument(resources.blocksZ->ref(), arguments.blocksZ);
  kernel.loadArgument(scalars.blocksX.ref(), arguments.blocksX);
  if (scalars.blocksY)
    kernel.loadArgument(scalars.blocksY->ref(), arguments.blocksY);
  kernel.loadArgument(spk.ref(), arguments.spk);
  kernel.loadArgument(k.ref(), arguments.k);
  kernel.loadArgument(resources.n.ref(), arguments.n);
  kernel.loadArgument(resources.m.ref(), arguments.m);
  kernel.loadArgument(resources.ldB.ref(), arguments.ldB);
  kernel.loadArgument(resources.ldA.ref(), arguments.ldA);
  kernel.loadArgument(resources.ldD.ref(), arguments.ldD);
  kernel.loadArgument(wsRsrc.ref(0, 2), arguments.ws);
  kernel.loadArgument(resources.sclBPtr.ref(), arguments.scalesB);
  kernel.loadArgument(resources.sclAPtr.ref(), arguments.scalesA);
  kernel.loadArgument(resources.bPtr.ref(), arguments.b);
  kernel.loadArgument(resources.aPtr.ref(), arguments.a);
  kernel.loadArgument(dRsrc.ref(0, 2), arguments.d);
  // Scalar loads return out of order, so only lgkmcnt(0) guarantees any of
  // them.
  builder.createWaitCnt(WaitCounts{.scalarMemory = 0});
  makeBufferSizes(scalars, resources);
  dispatchPaths(scalars, resources);
  builder.makeRsrc(dRsrc.ref(), dRsrc.ref(0, 2), dRsrc.ref(2));
  builder.makeRsrc(wsRsrc.ref(), wsRsrc.ref(0, 2), wsRsrc.ref(2));
}

void SkinnyGEMMEmitter::makeBufferSizes(const SetupScalars &scalars,
                                        const ResourceScalars &resources) {
  RegisterRange alignedK = sgpr();
  builder.addi32(alignedK.ref(), k, 3);
  shiftRight(builder, alignedK.ref(), alignedK.ref(), 2);
  shiftLeft(builder, alignedK.ref(), alignedK.ref(), 2);
  operandBytes(builder, resources.aBytes.ref(), resources.m.ref(),
               resources.ldA.ref(), alignedK.ref());
  operandBytes(builder, resources.bBytes.ref(), resources.n.ref(),
               resources.ldB.ref(), alignedK.ref());
  // A row group of the scales holds k rounded up to 256 bytes.
  builder.addi32(scalars.rowGroup.ref(), k, 255);
  shiftRight(builder, scalars.rowGroup.ref(), scalars.rowGroup.ref(), 8);
  shiftLeft(builder, scalars.rowGroup.ref(), scalars.rowGroup.ref(), 8);
  scaleBytes(builder, resources.sclABytes.ref(), resources.m.ref(),
             scalars.rowGroup.ref());
  scaleBytes(builder, resources.sclBBytes.ref(), resources.n.ref(),
             scalars.rowGroup.ref());
  // D holds ((n - 1) * ldD + m) bf16 elements, as (2n - 2) * ldD + 2m bytes.
  RegisterRef dBytes = dRsrc.ref(2);
  shiftLeft(builder, dBytes, resources.n.ref(), 1);
  builder.addi32(dBytes, dBytes, -2);
  builder.muli32(dBytes, dBytes, resources.ldD.ref());
  shiftLeft(builder, alignedK.ref(), resources.m.ref(), 1);
  builder.addi32(dBytes, dBytes, alignedK.ref());
  // The workspace holds splits * m * n f32 elements; the splits are the
  // blocks of the grid's z dimension, or ceilDiv(k, spk) in the 1-D launch.
  RegisterRef splits =
      resources.blocksZ ? resources.blocksZ->ref() : scalars.splits->ref();
  if (!resources.blocksZ) {
    // The exact signed division, as the reference divides at run time.
    builder.addi32(splits, k.ref(), spk.ref());
    builder.addi32(splits, splits, -1);
    builder.divsi32(splits, splits, spk.ref(), /*useIEEE=*/true);
  }
  RegisterRef wsBytes = wsRsrc.ref(2);
  builder.muli32(wsBytes, resources.m.ref(), splits);
  builder.muli32(wsBytes, wsBytes, resources.n.ref());
  shiftLeft(builder, wsBytes, wsBytes, 2);
  builder.muli32(mn.ref(), resources.m.ref(), resources.n.ref());
}

// Selects the operands, strides and extents of the skinny and the wide
// dimension on m <= n. The s_cselects follow one s_cmp and no SALU between
// them writes SCC.
void SkinnyGEMMEmitter::dispatchPaths(const SetupScalars &scalars,
                                      const ResourceScalars &resources) {
  builder.create<rocisa::SCmpLeU32>(resources.m.container(),
                                    resources.n.container());
  selectOnScc(builder, sRsrc.ref(0, 2), resources.aPtr.ref(),
              resources.bPtr.ref());
  selectOnScc(builder, sRsrc.ref(2), resources.aBytes.ref(),
              resources.bBytes.ref());
  selectOnScc(builder, wRsrc.ref(0, 2), resources.bPtr.ref(),
              resources.aPtr.ref());
  selectOnScc(builder, wRsrc.ref(2), resources.bBytes.ref(),
              resources.aBytes.ref());
  selectOnScc(builder, sSclRsrc.ref(0, 2), resources.sclAPtr.ref(),
              resources.sclBPtr.ref());
  selectOnScc(builder, sSclRsrc.ref(2), resources.sclABytes.ref(),
              resources.sclBBytes.ref());
  selectOnScc(builder, wSclRsrc.ref(0, 2), resources.sclBPtr.ref(),
              resources.sclAPtr.ref());
  selectOnScc(builder, wSclRsrc.ref(2), resources.sclBBytes.ref(),
              resources.sclABytes.ref());
  selectOnScc(builder, scalars.ldS.ref(), resources.ldA.ref(),
              resources.ldB.ref());
  selectOnScc(builder, scalars.ldW.ref(), resources.ldB.ref(),
              resources.ldA.ref());
  builder.create<rocisa::SCSelectB32>(dStrideS.container(), 1,
                                      resources.ldD.container());
  builder.create<rocisa::SCSelectB32>(dStrideW.container(),
                                      resources.ldD.container(), 1);
  builder.create<rocisa::SCSelectB32>(wsStrideS.container(), 1,
                                      resources.m.container());
  builder.create<rocisa::SCSelectB32>(wsStrideW.container(),
                                      resources.m.container(), 1);
  builder.create<rocisa::SCSelectB32>(contigS.container(), 1, 0);
  builder.create<rocisa::SMinU32>(
      skinnyCols.container(), resources.m.container(), resources.n.container());
  builder.create<rocisa::SMaxU32>(wideCols.container(), resources.m.container(),
                                  resources.n.container());
  for (RegisterRange *rsrc : {&sRsrc, &wRsrc, &sSclRsrc, &wSclRsrc})
    builder.makeRsrc(rsrc->ref(), rsrc->ref(0, 2), rsrc->ref(2));
}

// The persistent launch passes the rotation of the tile, the 3-D launch
// computes it from the workgroup ids and then regroups tx per XCD.
void SkinnyGEMMEmitter::computeTile(const SetupScalars &scalars,
                                    std::optional<RegisterRef> rotation) {
  RegisterRef ty = tile.ty;
  RegisterRef kz = tile.kz;
  // With partial steps the bounds outlive the setup, in members.
  std::optional<RegisterRange> startLocal;
  std::optional<RegisterRange> endLocal;
  if (!splitStart)
    startLocal = sgpr();
  if (!splitEnd)
    endLocal = sgpr();
  RegisterRef ks = splitStart ? splitStart->ref() : startLocal->ref();
  RegisterRef ke = splitEnd ? splitEnd->ref() : endLocal->ref();
  // Only the last split can be shorter than spk.
  builder.muli32(ks, kz, spk.ref());
  builder.addi32(ke, ks, spk.ref());
  builder.create<rocisa::SMinI32>(ke.container(), ke.container(),
                                  k.container());
  // The steps need ke itself, so its iteration index goes to a temporary.
  std::optional<RegisterRange> endIterLocal;
  RegisterRef keIter = ke;
  if (splitEnd) {
    endIterLocal = sgpr();
    keIter = endIterLocal->ref();
  }
  builder.create<rocisa::SAShiftRightI32>(keIter.container(), dims.log2Ki,
                                          ke.container());
  builder.addi32(kBaseIter.ref(), ks, dims.ki - 1);
  builder.create<rocisa::SAShiftRightI32>(kBaseIter.container(), dims.log2Ki,
                                          kBaseIter.container());
  builder.subi32(nIterFull.ref(), keIter, kBaseIter.ref());
  builder.create<rocisa::SMaxI32>(nIterFull.container(), nIterFull.container(),
                                  0);
  if (startIter)
    shiftRight(builder, startIter->ref(), ks, dims.log2Ki);
  if (vecOff) {
    // The head step owns entry 0 of the up-front vector when the split starts
    // inside an iteration.
    RegisterRange misaligned = sgpr();
    bitAnd(builder, misaligned.ref(), ks, dims.ki - 1);
    builder.create<rocisa::SCSelectB32>(vecOff->container(), 1, 0);
  }
  if (fitRot) {
    std::optional<RegisterRange> computed;
    if (!rotation) {
      computed = sgpr();
      computeRotation(scalars, computed->ref());
      rotation = computed->ref();
    }
    // The rotation falls back to 0 when it is not below the iteration count,
    // so the physical iteration wraps once without a division.
    builder.create<rocisa::SCmpLtU32>(rotation->container(),
                                      nIterFull.container());
    builder.create<rocisa::SCSelectB32>(fitRot->container(),
                                        rotation->container(), 0);
  }
  multiplyBy(builder, s0.ref(), ty, dims.bm);
  if (dims.grid3d && dims.xcdRun > 1)
    groupTilesPerXcd(scalars);
}

// The xcdRun tiles that share a scale line run on one XCD, so its L2 fetches
// the line once instead of every XCD fetching it. The dispatcher sends
// workgroup x to XCD x mod X, so within each full block of X * xcdRun ids the
// tile is (x mod X) * xcdRun + (x / X) mod xcdRun; the tail of the grid keeps
// the identity so that every tile still runs exactly once.
void SkinnyGEMMEmitter::groupTilesPerXcd(const SetupScalars &scalars) {
  int xcds = kernel.target().xcdCount();
  int span = xcds * dims.xcdRun;
  RegisterRange logical = sgpr();
  RegisterRange term = sgpr();
  shiftRight(builder, logical.ref(), tile.tx, log2Exact(xcds));
  bitAnd(builder, logical.ref(), logical.ref(), dims.xcdRun - 1);
  bitAnd(builder, term.ref(), tile.tx, xcds - 1);
  shiftLeft(builder, term.ref(), term.ref(), log2Exact(dims.xcdRun));
  builder.addi32(logical.ref(), logical.ref(), term.ref());
  bitAnd(builder, term.ref(), tile.tx, -span);
  builder.addi32(logical.ref(), logical.ref(), term.ref());
  bitAnd(builder, term.ref(), scalars.blocksX.ref(), -span);
  // SCC must survive from the compare to the select.
  builder.create<rocisa::SCmpLtU32>(tile.tx.container(), term.container());
  builder.create<rocisa::SCSelectB32>(tile.tx.container(), logical.container(),
                                      tile.tx.container());
}

// rot = (key + key / 16) mod NI with key the linear block id and NI =
// ceilDiv(spk, KI), or the literal UP_ITERS with up-front scales, which bounds
// the iterations of every split. The ring-scale remainder uses the exact
// signed division instead of LLVM's srem expansion.
void SkinnyGEMMEmitter::computeRotation(const SetupScalars &scalars,
                                        RegisterRef rot) {
  RegisterRef tx = tile.tx;
  RegisterRef ty = tile.ty;
  RegisterRef kz = tile.kz;
  RegisterRange key = sgpr();
  builder.muli32(key.ref(), scalars.blocksY->ref(), kz);
  builder.addi32(key.ref(), key.ref(), ty);
  builder.muli32(key.ref(), key.ref(), scalars.blocksX.ref());
  builder.addi32(key.ref(), key.ref(), tx);
  std::optional<RegisterRange> ni;
  if (dims.upIters == 0) {
    ni = sgpr();
    builder.addi32(ni->ref(), spk.ref(), dims.ki - 1);
    shiftRight(builder, ni->ref(), ni->ref(), dims.log2Ki);
  }
  blockRotation(rot, key.ref(),
                ni ? std::optional<RegisterRef>(ni->ref()) : std::nullopt);
}

// rot = (key + key / 16) mod NI, or mod the literal UP_ITERS without ni.
void SkinnyGEMMEmitter::blockRotation(RegisterRef rot, RegisterRef key,
                                      std::optional<RegisterRef> ni) {
  RegisterRange sum = sgpr();
  shiftRight(builder, sum.ref(), key, 4);
  builder.addi32(sum.ref(), sum.ref(), key);
  if (ni) {
    RegisterRange quotient = sgpr();
    builder.divsi32(quotient.ref(), sum.ref(), *ni, /*useIEEE=*/true);
    builder.muli32(quotient.ref(), quotient.ref(), *ni);
    builder.subi32(rot, sum.ref(), quotient.ref());
    return;
  }
  if (isPowerOfTwo(dims.upIters)) {
    builder.create<rocisa::SAndB32>(rot.container(), sum.container(),
                                    dims.upIters - 1);
    return;
  }
  RegisterRange quotient = sgpr();
  divideByConstant(builder, quotient.ref(), sum.ref(), dims.upIters);
  builder.muli32(quotient.ref(), quotient.ref(), dims.upIters);
  builder.subi32(rot, sum.ref(), quotient.ref());
}

// dst = (wave * CG + cg) * TN, the lane part of w0 = tx * BN + (wave * CG + cg)
// * TN. The slot wave * CG + cg is the thread id divided by LANES_K.
void SkinnyGEMMEmitter::computeLaneWideBase(RegisterRef dst) {
  RegisterRef tid = kernel.workitemId().ref();
  shiftRight(builder, dst, tid, dims.log2LanesK);
  multiplyBy(builder, dst, dst, dims.tn);
}

// dst += tx * BN.
void SkinnyGEMMEmitter::addTileBase(RegisterRef dst) {
  RegisterRange tileBase = sgpr();
  builder.muli32(tileBase.ref(), tile.tx, dims.tileWide);
  builder.addi32(dst, dst, tileBase.ref());
}

void SkinnyGEMMEmitter::computeAddresses(RegisterRef klv, RegisterRef lane,
                                         RegisterRef tid) {
  if (dims.scalesShared) {
    // Shared scales: the LDS byte offset 4 * (j * R + wave) of the lowest dword
    // of this lane, with the slot j(kl, 0) = (kl / 8) * 8 + (kl mod 8) / 2 of
    // LANES_K = 64. The other dword is a constant offset away, so one address
    // suffices.
    RegisterRef a0 = addresses.ref();
    RegisterRange term = vgpr();
    bitAnd(builder, a0, klv, dims.subgroupLanes - 1);
    shiftRight(builder, a0, a0, 1);
    multiplyBy(builder, a0, a0, 4 * dims.roleSlots);
    shiftRight(builder, term.ref(), klv, 3);
    multiplyBy(builder, term.ref(), term.ref(), 32 * dims.roleSlots);
    builder.addi32(a0, a0, term.ref());
    shiftRight(builder, term.ref(), tid, 6);
    shiftLeft(builder, term.ref(), term.ref(), 2);
    builder.addi32(a0, a0, term.ref());
    return;
  }
  // The byte address of the lanes that hold scale dwords 0 and 1 of this lane:
  // 4 * (j * CG + cg) with the slot j(kl, d) of the MLIR's @scl_slot.
  RegisterRef a0 = addresses.ref(0);
  RegisterRef a1 = addresses.ref(1);
  RegisterRange term = vgpr();
  // The block pair of the lane in its subgroup, times CG.
  bitAnd(builder, a0, klv, dims.subgroupLanes - 1);
  shiftRight(builder, a0, a0, 1);
  shiftLeft(builder, a0, a0, dims.log2ColumnGroups);
  if (dims.lanesK > 8) {
    shiftRight(builder, term.ref(), klv, 3);
    shiftLeft(builder, term.ref(), term.ref(),
              dims.log2Subgroup + dims.log2ColumnGroups);
    builder.addi32(a0, a0, term.ref());
  }
  if (dims.lanesK < 64) {
    shiftRight(builder, term.ref(), lane, dims.log2LanesK);
    builder.addi32(a0, a0, term.ref());
  }
  shiftLeft(builder, a0, a0, 2);
  builder.addi32(a1, a0, dims.halfSubgroup * dims.columnGroups * 4);
}

// Subgroup j of S lanes owns the slab j of 64 * S bytes, and the lane reads 16
// bytes at its place in the subgroup.
RegisterRange SkinnyGEMMEmitter::computeLaneBytes(RegisterRef klv) {
  RegisterRange laneBytes = vgpr();
  if (dims.lanesK == dims.subgroupLanes) {
    shiftLeft(builder, laneBytes.ref(), klv, 4);
    return laneBytes;
  }
  RegisterRange inSubgroup = vgpr();
  shiftRight(builder, laneBytes.ref(), klv, dims.log2Subgroup);
  shiftLeft(builder, laneBytes.ref(), laneBytes.ref(), dims.log2Subgroup + 6);
  bitAnd(builder, inSubgroup.ref(), klv, dims.subgroupLanes - 1);
  shiftLeft(builder, inSubgroup.ref(), inSubgroup.ref(), 4);
  builder.addi32(laneBytes.ref(), laneBytes.ref(), inSubgroup.ref());
  return laneBytes;
}

// The slot roleJ = id / R of the thread's scale load role, where R = roleSlots
// is not a power of two for some block sizes, so the division is LLVM's
// expansion of the signed division by a constant.
RegisterRange SkinnyGEMMEmitter::computeRoleJ(RegisterRef tid) {
  RegisterRange roleJ = vgpr();
  if (isPowerOfTwo(dims.roleSlots)) {
    shiftRight(builder, roleJ.ref(), tid, log2Exact(dims.roleSlots));
    return roleJ;
  }
  SignedMagic magic = signedMagic(dims.roleSlots);
  RegisterRange sign = vgpr();
  builder.movb32(roleJ.ref(), magic.multiplier);
  builder.create<rocisa::VMulHII32>(roleJ.container(), tid.container(),
                                    roleJ.container());
  if (magic.multiplier < 0)
    builder.addi32(roleJ.ref(), roleJ.ref(), tid);
  if (magic.shift > 0)
    builder.create<rocisa::VAShiftRightI32>(roleJ.container(), magic.shift,
                                            roleJ.container(), "");
  shiftRight(builder, sign.ref(), roleJ.ref(), 31);
  builder.addi32(roleJ.ref(), roleJ.ref(), sign.ref());
  return roleJ;
}

// The byte offset in an iteration of the scale slot roleJ that the lane loads:
// the slab roleJ / S of 64 * S bytes, the block pair roleJ mod (S / 2) of 64
// bytes and the dword step (roleJ mod S) / (S / 2) of 32 * S bytes. roleJ is
// the thread's in shared mode and the lane's cluster otherwise.
RegisterRange SkinnyGEMMEmitter::computeScaleOffset(
    RegisterRef lane, const std::optional<RegisterRange> &sharedRoleJ) {
  std::optional<RegisterRange> roleStorage;
  RegisterRef roleJ = lane;
  if (sharedRoleJ) {
    roleJ = sharedRoleJ->ref();
  } else if (dims.columnGroups > 1) {
    roleStorage = vgpr();
    shiftRight(builder, roleStorage->ref(), lane, dims.log2ColumnGroups);
    roleJ = roleStorage->ref();
  }
  RegisterRange sclOff = vgpr();
  RegisterRange term = vgpr();
  bitAnd(builder, sclOff.ref(), roleJ, dims.halfSubgroup - 1);
  shiftLeft(builder, sclOff.ref(), sclOff.ref(), 6);
  if (dims.lanesK > dims.subgroupLanes) {
    shiftRight(builder, term.ref(), roleJ, dims.log2Subgroup);
    shiftLeft(builder, term.ref(), term.ref(), dims.log2Subgroup + 6);
    builder.addi32(sclOff.ref(), sclOff.ref(), term.ref());
  }
  bitAnd(builder, term.ref(), roleJ, dims.subgroupLanes - 1);
  shiftRight(builder, term.ref(), term.ref(), dims.log2Half);
  shiftLeft(builder, term.ref(), term.ref(), dims.log2Subgroup + 5);
  builder.addi32(sclOff.ref(), sclOff.ref(), term.ref());
  return sclOff;
}

// slot = (wave * CG + lane mod CG) * TN, the lane part of w0Load = tx * BN +
// slot: the wide column that the lane loads scales for, which differs from the
// column it owns. In shared mode the slot is the thread's row slot tid mod R.
void SkinnyGEMMEmitter::computeWideLoadSlot(
    RegisterRef slot, RegisterRef tid, RegisterRef lane,
    const std::optional<RegisterRange> &sharedRoleJ) {
  if (sharedRoleJ) {
    // slot = tid mod R given roleJ = tid / R.
    if (isPowerOfTwo(dims.roleSlots)) {
      bitAnd(builder, slot, tid, dims.roleSlots - 1);
    } else {
      builder.muli32(slot, sharedRoleJ->ref(), dims.roleSlots);
      builder.subi32(slot, tid, slot);
    }
    multiplyBy(builder, slot, slot, dims.tn);
    return;
  }
  if (dims.waves > 1) {
    shiftRight(builder, slot, tid, 6);
    shiftLeft(builder, slot, slot, dims.log2ColumnGroups);
  }
  if (dims.columnGroups > 1) {
    RegisterRange inDomain = vgpr();
    bitAnd(builder, inDomain.ref(), lane, dims.columnGroups - 1);
    if (dims.waves > 1)
      builder.addi32(slot, slot, inDomain.ref());
    else
      builder.movb32(slot, inDomain.ref());
  }
  multiplyBy(builder, slot, slot, dims.tn);
}

SkinnyGEMMEmitter::LaneValues SkinnyGEMMEmitter::computeLaneValues() {
  RegisterRef tid = kernel.workitemId().ref();
  // The block is (64 * waves, 1, 1), so the packed y and z fields of v0 are
  // zero and no mask is needed.
  std::optional<RegisterRange> laneStorage;
  RegisterRef lane = tid;
  if (dims.waves > 1) {
    laneStorage = vgpr();
    bitAnd(builder, laneStorage->ref(), tid, 63);
    lane = laneStorage->ref();
  }
  std::optional<RegisterRange> klStorage;
  RegisterRef klv = lane;
  if (dims.lanesK < 64) {
    klStorage = vgpr();
    bitAnd(builder, klStorage->ref(), tid, dims.lanesK - 1);
    klv = klStorage->ref();
  }
  std::optional<RegisterRange> w0Lane;
  if (dims.grid3d && uniformWide()) {
    builder.muli32(w0.ref(), tile.tx, dims.tileWide);
  } else if (dims.grid3d) {
    computeLaneWideBase(w0.ref());
    addTileBase(w0.ref());
  } else if (!uniformWide()) {
    w0Lane = vgpr();
    computeLaneWideBase(w0Lane->ref());
  }
  computeAddresses(klv, lane, tid);
  RegisterRange laneBytes = computeLaneBytes(klv);
  if (klStorage)
    klStorage->release();
  std::optional<RegisterRange> roleJ;
  if (dims.scalesShared) {
    roleJ = computeRoleJ(tid);
    shiftLeft(builder, ldsStage->ref(), tid, 2);
  }
  RegisterRange sclOff = computeScaleOffset(lane, roleJ);
  std::optional<RegisterRange> w0Load;
  if (dims.grid3d && uniformWide()) {
    w0Load = sgpr();
    builder.muli32(w0Load->ref(), tile.tx, dims.tileWide);
  } else if (!uniformWide()) {
    w0Load = vgpr();
    computeWideLoadSlot(w0Load->ref(), tid, lane, roleJ);
    if (dims.grid3d)
      addTileBase(w0Load->ref());
  }
  // v0 keeps kl for the epilogue; every use of the thread id is done.
  if (dims.waves > 1 || dims.lanesK < 64)
    bitAnd(builder, tid, tid, dims.lanesK - 1);
  return LaneValues{std::move(laneBytes), std::move(sclOff), std::move(w0Load),
                    std::move(w0Lane)};
}

// dBase[r] = (s0 + r < skinnyCols) ? (s0 + r) * ldS + laneBytes : -1, and
// likewise for the wide columns; the validity masks stay alive for the
// stores.
void SkinnyGEMMEmitter::fillDataBases(const SetupScalars &scalars,
                                      LaneValues &lane) {
  for (int r = 0; r < dims.bm; ++r) {
    RegisterRange col = sgpr();
    builder.addi32(col.ref(), s0.ref(), r);
    builder.cmpi32(Predicate::ult, sValid.ref(2 * r, 2), col.ref(),
                   skinnyCols.ref());
    builder.muli32(col.ref(), col.ref(), scalars.ldS.ref());
    builder.addi32(dBase.ref(r), col.ref(), lane.laneBytes.ref());
    builder.selectb32(dBase.ref(r), sValid.ref(2 * r, 2), dBase.ref(r),
                      outOfRange);
  }
  for (int t = 0; t < dims.tn; ++t) {
    RegisterRef base = dBase.ref(dims.bm + t);
    RegisterRef valid = wValid.ref(2 * t, 2);
    RegisterRange col = uniformWide() ? sgpr() : vgpr();
    builder.addi32(col.ref(), w0.ref(), t);
    builder.cmpi32(Predicate::ult, valid, col.ref(), wideCols.ref());
    builder.muli32(col.ref(), col.ref(), scalars.ldW.ref());
    builder.addi32(base, col.ref(), lane.laneBytes.ref());
    builder.selectb32(base, valid, base, outOfRange);
  }
  // The persistent launch reuses the lane's bytes every tile.
  if (dims.grid3d)
    lane.laneBytes.release();
}

// sBase[c] is the voffset of the scale dword that the lane loads for column
// c: the load column of the wide operand differs from the lane's own column.
void SkinnyGEMMEmitter::fillScaleBases(const SetupScalars &scalars,
                                       const LaneValues &lane,
                                       RegisterRef w0Load) {
  if (scaleGroup)
    bitAnd(builder, scaleGroup->ref(), lane.sclOff.ref(), -256);
  for (int r = 0; r < dims.bm; ++r) {
    RegisterRange col = sgpr();
    RegisterRange row = sgpr();
    builder.addi32(col.ref(), s0.ref(), r);
    scaleRowOffset(builder, row.ref(), col.ref(), scalars.rowGroup.ref());
    builder.addi32(sBase.ref(r), row.ref(), lane.sclOff.ref());
    builder.selectb32(sBase.ref(r), sValid.ref(2 * r, 2), sBase.ref(r),
                      outOfRange);
  }
  for (int t = 0; t < dims.tn; ++t) {
    RegisterRef base = sBase.ref(dims.bm + t);
    RegisterRange valid = sgpr(2, 2);
    RegisterRange col = uniformWide() ? sgpr() : vgpr();
    RegisterRange row = uniformWide() ? sgpr() : vgpr();
    builder.addi32(col.ref(), w0Load, t);
    builder.cmpi32(Predicate::ult, valid.ref(), col.ref(), wideCols.ref());
    scaleRowOffset(builder, row.ref(), col.ref(), scalars.rowGroup.ref());
    builder.addi32(base, row.ref(), lane.sclOff.ref());
    builder.selectb32(base, valid.ref(), base, outOfRange);
  }
}

// The selector of the column group of a row: all rows of a 16-row half share
// one when BM or TN divides 16, since s0 and w0 are multiples of them.
RegisterRef SkinnyGEMMEmitter::selector(int column) const {
  bool skinny = column < dims.bm;
  bool skinnyShared = 16 % dims.bm == 0;
  bool wideShared = 16 % dims.tn == 0;
  int skinnyCount = skinnyShared ? 1 : dims.bm;
  if (skinny)
    return selectors.ref(skinnyShared ? 0 : column);
  return selectors.ref(skinnyCount + (wideShared ? 0 : column - dims.bm));
}

// dst = base + ((col >> 4) & 1) * 0x01010101: the byte selector of v_perm_b32
// that packs the two scale dwords of a column whose row is in the lower or
// upper 16 rows of its 32-row group.
void SkinnyGEMMEmitter::selectorValue(RegisterRef dst, RegisterRef col) {
  int selectorBase = dims.subgroupLanes == 8 ? 0x06040200 : 0x06020400;
  // A scalar column is computed in an SGPR and then copied to dst.
  std::optional<RegisterRange> value;
  RegisterRef work = dst;
  if (isScalar(col)) {
    value = sgpr();
    work = value->ref();
  }
  shiftRight(builder, work, col, 4);
  bitAnd(builder, work, work, 1);
  builder.muli32(work, work, selectorMaskStep);
  builder.addi32(work, work, selectorBase);
  if (value)
    builder.movb32(dst, work);
}

void SkinnyGEMMEmitter::fillSelectors() {
  if (16 % dims.bm == 0) {
    selectorValue(selector(0), s0.ref());
  } else {
    for (int r = 0; r < dims.bm; ++r) {
      RegisterRange col = sgpr();
      builder.addi32(col.ref(), s0.ref(), r);
      selectorValue(selector(r), col.ref());
    }
  }
  if (16 % dims.tn == 0) {
    selectorValue(selector(dims.bm), w0.ref());
    return;
  }
  for (int t = 0; t < dims.tn; ++t) {
    RegisterRange col = uniformWide() ? sgpr() : vgpr();
    builder.addi32(col.ref(), w0.ref(), t);
    selectorValue(selector(dims.bm + t), col.ref());
  }
}

void SkinnyGEMMEmitter::zeroAccumulators() {
  for (int p = 0; p < dims.accumulators; ++p)
    builder.movb32(acc.ref(p), 0);
}

// Up-front scales.

// The largest value of the 16-bit offset field of a ds instruction.
static constexpr int maxDsOffset = 65535;

// Names base + offset, adding an offset beyond the field to base in scratch,
// which the address of every ds instruction of the caller can share since a ds
// instruction reads its address when it issues.
static LdsAddress ldsAddress(IRBuilder &builder, RegisterRef base, int offset,
                             std::optional<RegisterRange> &scratch) {
  if (offset <= maxDsOffset)
    return {base, offset};
  if (!scratch)
    scratch = builder.allocateVgpr(1);
  builder.addi32(scratch->ref(), offset, base);
  return {scratch->ref(), 0};
}

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
  if (dims.scalesShared) {
    // Publishes the raw dwords through LDS, then every wave reads the two
    // dwords of its entries.
    stageScalesInLds();
    if (dims.grid3d)
      ldsStage.reset();
    for (int entry = 0; entry < dims.upIters; ++entry)
      readSharedEntry(entry);
  } else {
    // An entry is packed in two phases, the 2 * NC fetches back to back and
    // then the permutes, and the entries stay in sequence: fetching all of them
    // first would keep two dwords per column and entry live.
    for (int entry = 0; entry < dims.upIters; ++entry) {
      RegisterRange lo = vgpr(dims.columns);
      RegisterRef raw = scaleVector->ref(entry * dims.columns, dims.columns);
      gatherSlotScales(raw, lo.ref(), upFrontLoadsAfter(entry * dims.columns));
      permSlotScales(raw, lo.ref(), raw);
    }
  }
  selectors.release();
  // The persistent launch reuses the lane's addresses every tile.
  if (dims.grid3d)
    addresses.release();
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

// K loop.

static constexpr int exponentMask = 0x7F800000;

RegisterRef SkinnyGEMMEmitter::dataChunk(int slot, int chunk,
                                         int column) const {
  int index = ((slot * SkinnyDims::chunks + chunk) * dims.columns + column) * 4;
  return ringData.ref(index, 4);
}

RegisterRef SkinnyGEMMEmitter::rawScales(int slot) const {
  return ringRaw->ref(slot * dims.columns, dims.columns);
}

RegisterRef SkinnyGEMMEmitter::dataRsrc(int column) const {
  return column < dims.bm ? sRsrc.ref() : wRsrc.ref();
}

void SkinnyGEMMEmitter::runTile() {
  if (dims.upIters > 0)
    issueUpFrontScales();
  if (dims.prefetch == 0)
    runNoPrefetchLoop();
  else
    runPrefetchedLoop();
  // The steps run on ring slot 0 and read the up-front scales, so they precede
  // the release of both.
  if (dims.partialSteps == 1)
    runTailStep();
  else if (dims.partialSteps == 2)
    runHeadAndTailSteps();
  scaleGroup.reset();
  // The ring and the scale vector are dead, so the epilogue reuses their
  // registers.
  ringData.release();
  if (ringRaw)
    ringRaw->release();
  scaleVector.reset();
  hoistedScales.clear();
}

// dst = sum >= NIfull ? sum - NIfull : sum, the physical iteration of a
// logical one, with the rotation added once.
void SkinnyGEMMEmitter::wrapIteration(RegisterRef dst, RegisterRef sum) {
  RegisterRange wrapped = sgpr();
  builder.subi32(wrapped.ref(), sum, nIterFull.ref());
  builder.create<rocisa::SCmpGeU32>(sum.container(), nIterFull.container());
  builder.create<rocisa::SCSelectB32>(dst.container(), wrapped.container(),
                                      sum.container());
}

// chunkBase[u] = (kBaseIter << log2KI) + u * 16 * S, the part of the chunk
// soffsets that does not depend on the iteration.
void SkinnyGEMMEmitter::ensureChunkBase(int chunk) {
  if (!chunkBaseReady[0]) {
    shiftLeft(builder, chunkBase.ref(0), kBaseIter.ref(), dims.log2Ki);
    chunkBaseReady[0] = true;
  }
  if (chunk == 0 || chunkBaseReady[chunk])
    return;
  builder.addi32(chunkBase.ref(chunk), chunkBase.ref(0),
                 chunk * dims.chunkStride);
  chunkBaseReady[chunk] = true;
}

void SkinnyGEMMEmitter::hoistChunkBases() {
  for (int chunk = 1; chunk < SkinnyDims::chunks; ++chunk)
    ensureChunkBase(chunk);
}

// Loads the scale dword of every column of the iteration at soffset into the
// NC registers of raw. With neg (0 or -1 per lane) the voffset is ORed with it
// first, into one temporary that every column reuses since a load reads its
// voffset when it issues.
void SkinnyGEMMEmitter::issueScales(RegisterRef raw, RegisterRef soffset,
                                    std::optional<RegisterRef> neg) {
  std::optional<RegisterRange> steered;
  if (neg)
    steered = vgpr();
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef dst = raw.slice(c);
    std::shared_ptr<rocisa::RegisterContainer> voffset = sBase.sub(c);
    if (neg) {
      builder.create<rocisa::VOrB32>(steered->container(), voffset,
                                     neg->container());
      voffset = steered->container();
    }
    RegisterRef rsrc = c < dims.bm ? sSclRsrc.ref() : wSclRsrc.ref();
    builder.create<rocisa::BufferLoadB32>(dst.container(), voffset,
                                          rsrc.container(), soffset.container(),
                                          offen());
    if (recordLoads)
      prologueLoads.push_back(dst);
  }
}

// Loads chunk u of every column of the ring slot, steered by neg as the scales
// are. The steered voffset never goes into the destination registers, which
// may still have a prologue load in flight.
void SkinnyGEMMEmitter::issueChunk(int slot, int chunk, RegisterRef soffset,
                                   std::optional<RegisterRef> neg) {
  std::optional<RegisterRange> steered;
  if (neg)
    steered = vgpr();
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef dst = dataChunk(slot, chunk, c);
    std::shared_ptr<rocisa::RegisterContainer> voffset = dBase.sub(c);
    if (neg) {
      builder.create<rocisa::VOrB32>(steered->container(), voffset,
                                     neg->container());
      voffset = steered->container();
    }
    builder.create<rocisa::BufferLoadB128>(dst.container(), voffset,
                                           dataRsrc(c).container(),
                                           soffset.container(), offen());
    if (recordLoads)
      prologueLoads.push_back(dst);
  }
}

// soffset = (kBaseIter + iteration) << log2KI for the refill iteration, which
// chunk 0 uses as its own.
void SkinnyGEMMEmitter::refillOffset(const SlotContext &context,
                                     RegisterRef soffset) {
  builder.addi32(soffset, kBaseIter.ref(), *context.refillIteration);
  shiftLeft(builder, soffset, soffset, dims.log2Ki);
}

// Computes the offset of the refill iteration and issues its ring scales;
// up-front scales are loaded once, before the loop.
void SkinnyGEMMEmitter::startRefill(const SlotContext &context,
                                    RegisterRef soffset) {
  refillOffset(context, soffset);
  if (dims.upIters == 0)
    issueScales(rawScales(context.slot), soffset);
}

// Chunk u >= 1 is at chunkBase[u] + (iteration << log2KI) in the loops, whose
// chunk bases are hoisted, and at the iteration's soffset plus u * 16 * S
// elsewhere. The shifted iteration replaces the unshifted one on first use.
void SkinnyGEMMEmitter::refillChunk(const SlotContext &context, int chunk,
                                    RegisterRef soffset) {
  if (chunk == 0) {
    issueChunk(context.slot, 0, soffset);
    return;
  }
  RegisterRange chunkOffset = sgpr();
  if (context.hoistedOffsets) {
    ensureChunkBase(chunk);
    RegisterRef iteration = *context.refillIteration;
    if (chunk == 1)
      shiftLeft(builder, iteration, iteration, dims.log2Ki);
    builder.addi32(chunkOffset.ref(), chunkBase.ref(chunk), iteration);
  } else {
    builder.addi32(chunkOffset.ref(), soffset, chunk * dims.chunkStride);
  }
  issueChunk(context.slot, chunk, chunkOffset.ref());
}

// Waits until at most outstanding vector memory loads are in flight, capped at
// the counter's range.
void SkinnyGEMMEmitter::waitForVectorLoads(int outstanding) {
  builder.createWaitCnt(WaitCounts{
      .vectorLoads = std::min(outstanding, kernel.target().maxVmcnt())});
}

// Fetches the two scale dwords of every column from the lanes that hold them
// with ds_bpermute_b32: the first goes to lo, the second over the raw dword,
// which is dead once both permutes have issued. The raw dword of the first
// column has firstOutstanding vector loads issued after it.
void SkinnyGEMMEmitter::gatherSlotScales(RegisterRef raw, RegisterRef lo,
                                         int firstOutstanding) {
  for (int c = 0; c < dims.columns; ++c) {
    waitForVectorLoads(firstOutstanding - c);
    RegisterRef dword = raw.slice(c);
    builder.create<rocisa::DSBPermuteB32>(lo.slice(c).container(),
                                          addresses.sub(0), dword.container());
    builder.create<rocisa::DSBPermuteB32>(dword.container(), addresses.sub(1),
                                          dword.container());
  }
}

// Packs the two dwords of every column into one whose byte u is the scale of
// chunk u. Permute c is the (NC - 1 - c)-th newest of the 2 * NC in flight.
void SkinnyGEMMEmitter::permSlotScales(RegisterRef hi, RegisterRef lo,
                                       RegisterRef out) {
  for (int c = 0; c < dims.columns; ++c) {
    int outstanding = 2 * (dims.columns - 1 - c);
    builder.createWaitCnt(
        WaitCounts{.lds = std::min(outstanding, kernel.target().maxLgkmcnt())});
    builder.create<rocisa::VPermB32>(
        out.slice(c).container(), hi.slice(c).container(),
        lo.slice(c).container(), selector(c).container());
  }
}

// Decodes the E8M0 scale of chunk u to an f32: (dword << (23 - 8u)) &
// 0x7F800000, or a shift right by 1 for u = 3.
void SkinnyGEMMEmitter::decodeScales(RegisterRef dst, RegisterRef packed,
                                     int chunk) {
  for (int c = 0; c < dims.columns; ++c) {
    RegisterRef scale = dst.slice(c);
    if (chunk < SkinnyDims::chunks - 1)
      shiftLeft(builder, scale, packed.slice(c), 23 - 8 * chunk);
    else
      shiftRight(builder, scale, packed.slice(c), 1);
    bitAnd(builder, scale, scale, exponentMask);
  }
}

// The first convert of column c of the chunk waits for its load. In a partial
// step every chunk load may be skipped, so only the later columns of the chunk
// itself are certain to be in flight; with ring scales the gather already
// waited for every load, as the chunk loads follow the scale loads.
void SkinnyGEMMEmitter::waitBeforeFirstUse(const SlotContext &context,
                                           int chunk, int column,
                                           int ownRefill) {
  if (!context.partialStep) {
    // The load is complete once at most the loads issued after it are
    // outstanding, as vmcnt counts the loads of a wave in issue order.
    waitForVectorLoads(
        loadsIssuedAfter(dims, {dims.scaleLoads + chunk * dims.columns + column,
                                context.groupsAfter, ownRefill}));
    return;
  }
  if (dims.scaleLoads == 0)
    waitForVectorLoads(dims.columns - 1 - column);
}

// Consumes chunk u of the slot: 4 dwords of 2 fp8 pairs per column, converted
// with the decoded scale and accumulated with v_dot2_f32_bf16.
void SkinnyGEMMEmitter::computeChunk(const SlotContext &context, int chunk,
                                     RegisterRef scales) {
  RegisterRange pairs = vgpr(dims.columns);
  int ownRefill =
      context.refillIteration ? dims.scaleLoads + chunk * dims.columns : 0;
  for (int dword = 0; dword < 4; ++dword) {
    for (int word = 0; word < 2; ++word) {
      std::optional<rocisa::VOP3PModifiers> highWord;
      if (word == 1)
        highWord = rocisa::VOP3PModifiers({1, 0, 0});
      for (int c = 0; c < dims.columns; ++c) {
        if (dword == 0 && word == 0)
          waitBeforeFirstUse(context, chunk, c, ownRefill);
        builder.create<rocisa::VCvtScalePkFP8toBF16>(
            pairs.sub(c),
            dataChunk(context.slot, chunk, c).slice(dword).container(),
            scales.slice(c).container(), std::nullopt, highWord);
      }
      for (int r = 0; r < dims.bm; ++r) {
        for (int t = 0; t < dims.tn; ++t) {
          RegisterRef total = acc.ref(r * dims.tn + t);
          builder.create<rocisa::VDot2F32BF16>(total.container(), pairs.sub(r),
                                               pairs.sub(dims.bm + t),
                                               total.container());
        }
      }
    }
  }
}

// Decode-first schedule (PF >= 2): decode the four chunk scales before the
// scales are refilled, then per chunk compute and refill. The packed dwords
// die after the last decode. A partial step decodes chunk 3 in place over the
// packed dwords, as the reference does, to fit the register budget; the loop
// keeps its allocation.
void SkinnyGEMMEmitter::processDecodeFirst(const SlotContext &context,
                                           RegisterRange &packed) {
  std::vector<RegisterRange> scales;
  std::vector<RegisterRef> chunkScales;
  int separate =
      context.partialStep ? SkinnyDims::chunks - 1 : SkinnyDims::chunks;
  for (int chunk = 0; chunk < separate; ++chunk) {
    scales.push_back(vgpr(dims.columns));
    chunkScales.push_back(scales.back().ref());
    decodeScales(chunkScales.back(), packed.ref(), chunk);
  }
  if (context.partialStep) {
    decodeScales(packed.ref(), packed.ref(), SkinnyDims::chunks - 1);
    chunkScales.push_back(packed.ref());
  } else {
    packed.release();
  }
  RegisterRange soffset = sgpr();
  if (context.refillIteration)
    startRefill(context, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    computeChunk(context, chunk, chunkScales[chunk]);
    if (chunk < separate)
      scales[chunk].release();
    else
      packed.release();
    if (context.refillIteration)
      refillChunk(context, chunk, soffset.ref());
  }
}

// Up-front schedule: per chunk decode its scale, compute and refill the
// chunk; the scales are not refilled. The last chunk decodes over the packed
// dwords, which no chunk reads afterwards. Without packed dwords (one entry)
// the decoded scales were hoisted out of the loop.
void SkinnyGEMMEmitter::processUpFront(const SlotContext &context,
                                       std::optional<RegisterRange> &packed) {
  RegisterRange soffset = sgpr();
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    std::optional<RegisterRange> decoded;
    RegisterRef scales;
    if (!packed) {
      scales = hoistedScales[chunk].ref();
    } else if (chunk == SkinnyDims::chunks - 1) {
      decodeScales(packed->ref(), packed->ref(), chunk);
      scales = packed->ref();
    } else {
      decoded = vgpr(dims.columns);
      decodeScales(decoded->ref(), packed->ref(), chunk);
      scales = decoded->ref();
    }
    computeChunk(context, chunk, scales);
    if (!context.refillIteration)
      continue;
    if (chunk == 0)
      refillOffset(context, soffset.ref());
    refillChunk(context, chunk, soffset.ref());
  }
}

void SkinnyGEMMEmitter::processSlot(const SlotContext &context) {
  if (dims.upIters > 0) {
    std::optional<RegisterRange> packed = selectScales(context);
    processUpFront(context, packed);
    return;
  }
  RegisterRange packed = vgpr(dims.columns);
  // A step issues only its own scale loads on every path: its chunk loads may
  // all be skipped.
  int firstOutstanding =
      context.partialStep
          ? dims.columns - 1
          : loadsIssuedAfter(dims, {0, context.groupsAfter, /*ownRefill=*/0});
  gatherSlotScales(rawScales(context.slot), packed.ref(), firstOutstanding);
  permSlotScales(rawScales(context.slot), packed.ref(), packed.ref());
  if (dims.prefetch == 0)
    ringRaw->release();
  // The MLIR decodes first for every depth but 1; the PF = 0 loop keeps the
  // staged schedule, which its steps do not use.
  bool decodeFirst =
      dims.prefetch >= 2 || (dims.prefetch == 0 && context.partialStep);
  if (decodeFirst) {
    processDecodeFirst(context, packed);
    return;
  }
  // Staged schedule (PF = 1 and the PF = 0 loop): refill the scales, then per
  // chunk decode its scale, compute and refill the chunk, so that the new loads
  // follow the consumption of the old ones.
  RegisterRange soffset = sgpr();
  if (context.refillIteration)
    startRefill(context, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    RegisterRange scales = vgpr(dims.columns);
    decodeScales(scales.ref(), packed.ref(), chunk);
    computeChunk(context, chunk, scales.ref());
    scales.release();
    if (context.refillIteration)
      refillChunk(context, chunk, soffset.ref());
  }
}

// PF = 0: each iteration issues its chunks, and its scales with ring scales,
// then consumes them.
void SkinnyGEMMEmitter::runNoPrefetchLoop() {
  if (dims.upIters > 0)
    packScaleVectors();
  if (dims.upIters == 1)
    decodeHoistedScales();
  hoistChunkBases();
  // Nothing of the ring is loop carried, so the raw scales live only from the
  // loads of an iteration to its permutes.
  if (ringRaw)
    ringRaw->release();
  RegisterRange iteration = sgpr();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(nIterFull.container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  if (ringRaw)
    ringRaw = vgpr(dims.columns);
  RegisterRange physical = sgpr();
  if (fitRot) {
    builder.addi32(physical.ref(), iteration.ref(), fitRot->ref());
    wrapIteration(physical.ref(), physical.ref());
  } else {
    wrapIteration(physical.ref(), iteration.ref());
  }
  // The select precedes the loads so that its VALU work hides their latency.
  std::optional<RegisterRange> packed;
  std::optional<RegisterRange> entry;
  if (dims.upIters > 0)
    packed = selectScales(SlotContext{0, 0, std::nullopt, true,
                                      vectorEntry(physical.ref(), entry)});
  RegisterRange soffset = sgpr();
  SlotContext loads{0, 0, physical.ref()};
  startRefill(loads, soffset.ref());
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    refillChunk(loads, chunk, soffset.ref());
  SlotContext consume{0, 0, std::nullopt};
  if (dims.upIters > 0)
    processUpFront(consume, packed);
  else
    processSlot(consume);
  builder.addi32(iteration.ref(), iteration.ref(), 1);
  builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                    nIterFull.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

void SkinnyGEMMEmitter::runPrefetchedLoop() {
  emitPrologue();
  if (dims.upIters > 0)
    packScaleVectors();
  // steadyEnd = max((NIfull / PF - 1) * PF, 0), so every refill of the steady
  // loop targets an iteration below NIfull.
  RegisterRange steadyEnd = sgpr();
  int pf = dims.prefetch;
  if (pf == 1) {
    builder.addi32(steadyEnd.ref(), nIterFull.ref(), -1);
  } else if (isPowerOfTwo(pf)) {
    int log2Pf = log2Exact(pf);
    builder.create<rocisa::SAShiftRightI32>(steadyEnd.container(), log2Pf,
                                            nIterFull.container());
    shiftLeft(builder, steadyEnd.ref(), steadyEnd.ref(), log2Pf);
    builder.addi32(steadyEnd.ref(), steadyEnd.ref(), -pf);
  } else {
    divideByConstant(builder, steadyEnd.ref(), nIterFull.ref(), pf);
    builder.muli32(steadyEnd.ref(), steadyEnd.ref(), pf);
    builder.addi32(steadyEnd.ref(), steadyEnd.ref(), -pf);
  }
  builder.create<rocisa::SMaxI32>(steadyEnd.container(), steadyEnd.container(),
                                  0);
  if (dims.upIters == 1)
    decodeHoistedScales();
  emitSteadyLoop(steadyEnd.ref());
  // The left = NIfull - steadyEnd (1 to 2 * PF - 1) remaining iterations run in
  // the arm of a chain of independent ifs on left.
  RegisterRange left = sgpr();
  builder.subi32(left.ref(), nIterFull.ref(), steadyEnd.ref());
  for (int remaining = 1; remaining < 2 * pf; ++remaining)
    emitRemainderArm(left.ref(), steadyEnd.ref(), remaining);
}

// Issues the iterations [0, PF) into the ring. A split with fewer full
// iterations reloads its last one through the clamp: a load dropped by the
// range check would still take a VMEM issue slot, and the extra slots are
// never consumed.
void SkinnyGEMMEmitter::emitPrologue() {
  recordLoads = true;
  std::optional<RegisterRange> lastIteration;
  if (dims.prefetch > 1) {
    lastIteration = sgpr();
    builder.addi32(lastIteration->ref(), nIterFull.ref(), -1);
    builder.create<rocisa::SMaxI32>(lastIteration->container(),
                                    lastIteration->container(), 0);
  }
  // LLVM hoists the chunk bases out of the prologue from PF = 4 on.
  bool hoisted = dims.prefetch >= 4;
  for (int slot = 0; slot < dims.prefetch; ++slot) {
    RegisterRange physical = sgpr();
    if (slot > 0) {
      builder.create<rocisa::SMinU32>(physical.container(),
                                      lastIteration->container(), slot);
      if (fitRot)
        builder.addi32(physical.ref(), physical.ref(), fitRot->ref());
      wrapIteration(physical.ref(), physical.ref());
    } else if (fitRot) {
      wrapIteration(physical.ref(), fitRot->ref());
    } else {
      // The wrapped physical iteration of the logical iteration 0 without
      // rotation.
      RegisterRange wrapped = sgpr();
      builder.subi32(wrapped.ref(), 0, nIterFull.ref());
      builder.create<rocisa::SCmpLeU32>(nIterFull.container(), 0);
      builder.create<rocisa::SCSelectB32>(physical.container(),
                                          wrapped.container(), 0);
    }
    SlotContext context{slot, 0, physical.ref(), hoisted};
    RegisterRange soffset = sgpr();
    startRefill(context, soffset.ref());
    for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
      refillChunk(context, chunk, soffset.ref());
  }
  recordLoads = false;
}

// The loop body is unrolled over the slots, so every slot index is a
// constant. Slot p refills with the iteration it0 + PF + p and, with several
// up-front entries, selects the scales of the physical iteration it0 + p. The
// bases fold the rotation: refillBase[p] = fitRot + PF + p and physBase[p] =
// fitRot + p. Without rotation they are the literals.
void SkinnyGEMMEmitter::emitSteadyLoop(RegisterRef steadyEnd) {
  int pf = dims.prefetch;
  bool vectorized = dims.upIters >= 2;
  std::optional<RegisterRange> refillBase;
  std::optional<RegisterRange> physBase;
  if (fitRot) {
    refillBase = sgpr(pf);
    builder.addi32(refillBase->ref(0), fitRot->ref(), pf);
  }
  hoistChunkBases();
  if (vectorized)
    physBase = sgpr(pf);
  for (int slot = 1; fitRot && slot < pf; ++slot) {
    builder.addi32(refillBase->ref(slot), fitRot->ref(), pf + slot);
    if (vectorized)
      builder.addi32(physBase->ref(slot), fitRot->ref(), slot);
  }
  RegisterRange iteration = sgpr();
  std::shared_ptr<rocisa::Label> body = builder.createLabel("kloop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("kloop_end");
  builder.create<rocisa::SCmpGtI32>(steadyEnd.container(), 0);
  builder.movb32(iteration.ref(), 0);
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  for (int slot = 0; slot < pf; ++slot) {
    RegisterRange refill = sgpr();
    if (refillBase)
      builder.addi32(refill.ref(), iteration.ref(), refillBase->ref(slot));
    else
      builder.addi32(refill.ref(), iteration.ref(), pf + slot);
    wrapIteration(refill.ref(), refill.ref());
    SlotContext context{slot, pf - 1, refill.ref()};
    std::optional<RegisterRange> physical;
    std::optional<RegisterRange> entry;
    if (vectorized) {
      physical = sgpr();
      if (slot == 0)
        builder.addi32(physical->ref(), fitRot->ref(), iteration.ref());
      else
        builder.addi32(physical->ref(), iteration.ref(), physBase->ref(slot));
      wrapIteration(physical->ref(), physical->ref());
      context.vectorIndex = vectorEntry(physical->ref(), entry);
    }
    processSlot(context);
  }
  builder.addi32(iteration.ref(), iteration.ref(), pf);
  builder.create<rocisa::SCmpLtI32>(iteration.container(),
                                    steadyEnd.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

// Iteration i of the arm uses ring slot i mod PF and refills it only when
// iteration i + PF exists. Of the loads issued after the slot's group, the
// PF - 1 - i initial groups and the refills of the earlier iterations remain.
// With several up-front entries it also selects the scales of its physical
// iteration, which is the refill iteration of i - PF from i = PF on.
void SkinnyGEMMEmitter::emitRemainderArm(RegisterRef left,
                                         RegisterRef steadyEnd, int remaining) {
  int pf = dims.prefetch;
  bool vectorized = dims.upIters >= 2;
  std::shared_ptr<rocisa::Label> next = builder.createLabel("arm_end");
  builder.create<rocisa::SCmpEQI32>(left.container(), remaining);
  builder.create<rocisa::SCBranchSCC0>(next->getLabelName());
  // The logical iteration of a refill is fitRot + steadyEnd + i + PF.
  std::optional<RegisterRange> base;
  std::vector<RegisterRange> keptRefills;
  for (int i = 0; i < remaining; ++i) {
    int refilled = std::clamp(remaining - pf, 0, i);
    SlotContext context{i % pf, pf - 1 + refilled - i, std::nullopt, false};
    RegisterRange refill = sgpr();
    bool hasRefill = i + pf < remaining;
    if (!base && fitRot && (vectorized || hasRefill)) {
      base = sgpr();
      builder.addi32(base->ref(), fitRot->ref(), steadyEnd);
    }
    RegisterRef baseRef = base ? base->ref() : steadyEnd;
    if (hasRefill) {
      builder.addi32(refill.ref(), baseRef, i + pf);
      wrapIteration(refill.ref(), refill.ref());
      context.refillIteration = refill.ref();
    }
    std::optional<RegisterRange> physical;
    std::optional<RegisterRange> entry;
    if (vectorized && i >= pf) {
      context.vectorIndex = vectorEntry(keptRefills[i - pf].ref(), entry);
    } else if (vectorized) {
      physical = sgpr();
      if (i > 0)
        builder.addi32(physical->ref(), baseRef, i);
      wrapIteration(physical->ref(), i > 0 ? physical->ref() : baseRef);
      context.vectorIndex = vectorEntry(physical->ref(), entry);
    }
    processSlot(context);
    if (vectorized && hasRefill)
      keptRefills.push_back(std::move(refill));
  }
  builder.insert(next);
}

// Partial K steps.

static constexpr int dwordBytes = 4;
static constexpr int dwordsPerChunk = 4;
static constexpr int partBytes = dwordBytes * dwordsPerChunk;
// The largest byte shift of a dword mask: a 32-bit shift is poison in the MLIR,
// and a dword that needs a shift of 4 bytes is steered anyway.
static constexpr int maxMaskBytes = 3;

// The register of an optional range, if any.
static std::optional<RegisterRef>
optionalRef(const std::optional<RegisterRange> &range) {
  if (!range)
    return std::nullopt;
  return range->ref();
}

// dst = -1 when src + delta is negative, else 0.
static void signOfSum(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                      int delta) {
  builder.addi32(dst, src, delta);
  builder.create<rocisa::VAShiftRightI32>(dst.container(), 31, dst.container(),
                                          "");
}

// dst = -1 when delta - src is negative, else 0.
static void signOfDifference(IRBuilder &builder, RegisterRef dst, int delta,
                             RegisterRef src) {
  builder.subi32(dst, delta, src);
  builder.create<rocisa::VAShiftRightI32>(dst.container(), 31, dst.container(),
                                          "");
}

// dst = 8 * clamp(src, 0, 3), the shift in bits of a mask of whole bytes.
static void clampedBitShift(IRBuilder &builder, RegisterRef dst,
                            RegisterRef src) {
  builder.create<rocisa::VMaxI32>(dst.container(), 0, src.container());
  builder.create<rocisa::VMinI32>(dst.container(), maxMaskBytes,
                                  dst.container());
  shiftLeft(builder, dst, dst, 3);
}

// Loads the raw scale dword of every column of the iteration at soffset into
// raw, steering to -1 the lanes whose role group starts at or after k: past k a
// group belongs to the next row group of the scale matrix (other rows' scales,
// 255 decodes to +inf and 0 * inf is NaN). kAvail = k - iterOff, an SGPR, may
// be zero or negative for vector entries past k.
void SkinnyGEMMEmitter::issueScalesWithinK(RegisterRef raw, RegisterRef soffset,
                                           RegisterRef kAvail) {
  RegisterRange neg = vgpr();
  builder.subi32(neg.ref(), kAvail, scaleGroup->ref());
  builder.addi32(neg.ref(), neg.ref(), -1);
  builder.create<rocisa::VAShiftRightI32>(neg.container(), 31, neg.container(),
                                          "");
  issueScales(raw, soffset, neg.ref());
}

// Bit u of dst is @straddles_in_chunk(bound, u): the bound is not a multiple of
// 16 and lies in the chunk (bound / (16 S)) mod 4, whose part of one lane per
// column holds it.
void SkinnyGEMMEmitter::straddlesInChunk(RegisterRef dst, RegisterRef bound) {
  RegisterRange misaligned = sgpr();
  shiftRight(builder, dst, bound, log2Exact(dims.chunkStride));
  bitAnd(builder, dst, dst, SkinnyDims::chunks - 1);
  builder.create<rocisa::SLShiftLeftB32>(dst.container(), dst.container(), 1);
  bitAnd(builder, misaligned.ref(), bound, partBytes - 1);
  builder.create<rocisa::SCSelectB32>(dst.container(), dst.container(), 0);
}

// The PF = 0 loop released its raw scales, and the step takes their registers
// back before anything else, so that a smaller value does not split the hole
// they leave.
void SkinnyGEMMEmitter::allocateStepRawScales() {
  if (dims.upIters == 0 && dims.prefetch == 0)
    ringRaw = vgpr(dims.columns);
}

// The tail step of PARTIAL_STEPS = 1: ks is a multiple of KI, so the split has
// no head and the tail keeps the bytes [0, ke mod KI) of its iteration.
void SkinnyGEMMEmitter::runTailStep() {
  std::shared_ptr<rocisa::Label> stepEnd = builder.createLabel("step_end");
  RegisterRange hi = sgpr();
  bitAnd(builder, hi.ref(), splitEnd->ref(), dims.ki - 1);
  builder.create<rocisa::SCBranchSCC0>(stepEnd->getLabelName());
  RegisterRange iterOff = sgpr();
  builder.subi32(iterOff.ref(), splitEnd->ref(), hi.ref());
  std::optional<RegisterRange> kAvail;
  if (dims.upIters == 0) {
    kAvail = sgpr();
    builder.subi32(kAvail->ref(), k.ref(), iterOff.ref());
  }
  // The up-front vector starts at the split's first iteration, kBaseIter.
  std::optional<RegisterRange> vectorIndex;
  if (dims.upIters >= 2) {
    vectorIndex = sgpr();
    shiftRight(builder, vectorIndex->ref(), iterOff.ref(), dims.log2Ki);
    builder.subi32(vectorIndex->ref(), vectorIndex->ref(), kBaseIter.ref());
  }
  // v0 holds the lane's index in its cluster after the setup.
  allocateStepRawScales();
  RegisterRange laneBytes = computeLaneBytes(kernel.workitemId().ref());
  PartialStep step{iterOff.ref(),
                   std::nullopt,
                   hi.ref(),
                   optionalRef(kAvail),
                   optionalRef(vectorIndex),
                   laneBytes.ref()};
  loadPartialStep(step);
  // The only step runs under the runtime branch and nothing after it, the
  // epilogue included, reads the load state, so its registers are free for the
  // compute to fit the register budget.
  dBase.release();
  sBase.release();
  scaleGroup.reset();
  laneBytes.release();
  computePartialStep(step);
  builder.insert(stepEnd);
}

// The head and tail steps of PARTIAL_STEPS = 2 in one dynamic loop over j in
// [first, end), so that the large step body exists once: j = 0 is the head,
// which holds ks, and j = 1 the tail, which holds ke. The tail is skipped when
// the head covers both bounds.
void SkinnyGEMMEmitter::runHeadAndTailSteps() {
  RegisterRange endIter = sgpr();
  RegisterRange j = sgpr();
  RegisterRange end = sgpr();
  RegisterRange coverage = sgpr();
  shiftRight(builder, endIter.ref(), splitEnd->ref(), dims.log2Ki);
  // j starts at the head, or past it when the split starts on an iteration.
  bitAnd(builder, coverage.ref(), splitStart->ref(), dims.ki - 1);
  builder.create<rocisa::SCSelectB32>(j.container(), 0, 1);
  bitAnd(builder, coverage.ref(), splitEnd->ref(), dims.ki - 1);
  builder.create<rocisa::SCSelectB32>(end.container(), 2, 1);
  // coverage is 0 exactly when there is a head and it holds ke too.
  builder.create<rocisa::SCmpEQU32>(startIter->container(),
                                    endIter.container());
  builder.create<rocisa::SCSelectB32>(coverage.container(), j.container(), 1);
  builder.create<rocisa::SCmpEQU32>(coverage.container(), 0);
  builder.create<rocisa::SCSelectB32>(end.container(), 1, end.container());
  coverage.release();
  allocateStepRawScales();
  RegisterRange laneBytes = computeLaneBytes(kernel.workitemId().ref());

  std::shared_ptr<rocisa::Label> body = builder.createLabel("step_loop");
  std::shared_ptr<rocisa::Label> exit = builder.createLabel("step_loop_end");
  builder.create<rocisa::SCmpLtI32>(j.container(), end.container());
  builder.create<rocisa::SCBranchSCC0>(exit->getLabelName());
  builder.insert(body);
  {
    RegisterRange iter = sgpr();
    RegisterRange iterOff = sgpr();
    RegisterRange lo = sgpr();
    RegisterRange hi = sgpr();
    builder.create<rocisa::SCmpEQU32>(j.container(), 0);
    builder.create<rocisa::SCSelectB32>(
        iter.container(), startIter->container(), endIter.container());
    shiftLeft(builder, iterOff.ref(), iter.ref(), dims.log2Ki);
    builder.subi32(lo.ref(), splitStart->ref(), iterOff.ref());
    builder.create<rocisa::SMaxI32>(lo.container(), lo.container(), 0);
    builder.subi32(hi.ref(), splitEnd->ref(), iterOff.ref());
    builder.create<rocisa::SMinI32>(hi.container(), hi.container(), dims.ki);
    std::optional<RegisterRange> kAvail;
    if (dims.upIters == 0) {
      kAvail = sgpr();
      builder.subi32(kAvail->ref(), k.ref(), iterOff.ref());
    }
    std::optional<RegisterRange> vectorIndex;
    if (dims.upIters >= 2) {
      vectorIndex = sgpr();
      builder.subi32(vectorIndex->ref(), iter.ref(), startIter->ref());
    }
    PartialStep step{iterOff.ref(),
                     lo.ref(),
                     hi.ref(),
                     optionalRef(kAvail),
                     optionalRef(vectorIndex),
                     laneBytes.ref()};
    loadPartialStep(step);
    computePartialStep(step);
  }
  builder.addi32(j.ref(), j.ref(), 1);
  builder.create<rocisa::SCmpLtI32>(j.container(), end.container());
  builder.create<rocisa::SCBranchSCC1>(body->getLabelName());
  builder.insert(exit);
}

// A partial step: the iteration at K offset iterOff, of which the split keeps
// only the bytes [lo, hi), run on ring slot 0, which the K loop left free. Ring
// scales load as in the loop with the groups at or after k steered to -1, and
// every chunk loads behind a dynamic branch, so that the slot holds exact
// zeros outside [lo, hi). The compute is @process_slot without refill. This
// loads the scales and the chunks of the step.
void SkinnyGEMMEmitter::loadPartialStep(const PartialStep &step) {
  if (dims.upIters == 0)
    issueScalesWithinK(rawScales(0), step.iterOff, *step.kAvail);
  RegisterRange partialChunks = sgpr();
  straddlesInChunk(partialChunks.ref(), step.hi);
  if (step.lo) {
    RegisterRange loChunks = sgpr();
    straddlesInChunk(loChunks.ref(), *step.lo);
    builder.create<rocisa::SOrB32>(partialChunks.container(),
                                   partialChunks.container(),
                                   loChunks.container());
  }
  for (int chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    loadStepChunk(step, chunk, partialChunks.ref());
  partialChunks.release();
}

void SkinnyGEMMEmitter::computePartialStep(const PartialStep &step) {
  processSlot(SlotContext{0, 0, std::nullopt, true, step.vectorIndex,
                          /*partialStep=*/true});
}

// Loads chunk u of the step for every column, in one of three arms. At most one
// lane per bound straddles it, in the chunk of its bound: that chunk loads
// dwords (@load_tail_dwords). In every other chunk each lane lies fully inside
// or outside [lo, hi), so steering the voffset of the outside lanes to -1
// suffices, and a chunk with no live lane is zeroed.
void SkinnyGEMMEmitter::loadStepChunk(const PartialStep &step, int chunk,
                                      RegisterRef partialChunks) {
  int offset = chunk * dims.chunkStride;
  // All SALU first: s_add writes SCC, which the branches read.
  std::optional<RegisterRange> soffsetStorage;
  std::optional<RegisterRange> hiStorage;
  std::optional<RegisterRange> loStorage;
  RegisterRef soffset = step.iterOff;
  RegisterRef chunkHi = step.hi;
  std::optional<RegisterRef> chunkLo = step.lo;
  if (chunk > 0) {
    soffsetStorage = sgpr();
    soffset = soffsetStorage->ref();
    builder.addi32(soffset, step.iterOff, offset);
    hiStorage = sgpr();
    chunkHi = hiStorage->ref();
    builder.addi32(chunkHi, step.hi, -offset);
    if (step.lo) {
      loStorage = sgpr();
      chunkLo = loStorage->ref();
      builder.addi32(*chunkLo, *step.lo, -offset);
    }
  }
  // The bytes of the lane's 16 below hi and below lo.
  RegisterRange laneRem = vgpr();
  builder.subi32(laneRem.ref(), chunkHi, step.laneBytes);
  std::optional<RegisterRange> laneLo;
  if (chunkLo) {
    laneLo = vgpr();
    builder.subi32(laneLo->ref(), *chunkLo, step.laneBytes);
  }
  std::shared_ptr<rocisa::Label> notPartial = builder.createLabel("step_load");
  std::shared_ptr<rocisa::Label> zero = builder.createLabel("step_zero");
  std::shared_ptr<rocisa::Label> chunkEnd =
      builder.createLabel("step_chunk_end");
  builder.create<rocisa::SBitcmp1B32>(partialChunks.container(), chunk);
  builder.create<rocisa::SCBranchSCC0>(notPartial->getLabelName());
  loadTailDwords(chunk, soffset, laneRem.ref(),
                 laneLo ? std::optional<RegisterRef>(laneLo->ref())
                        : std::nullopt);
  builder.create<rocisa::SBranch>(chunkEnd->getLabelName());
  builder.insert(notPartial);
  // A necessary condition for any lane of the chunk to lie inside [lo, hi);
  // when it holds but no lane qualifies, the steered loads read 0.
  builder.create<rocisa::SCmpGeI32>(chunkHi.container(), partBytes);
  builder.create<rocisa::SCBranchSCC0>(zero->getLabelName());
  if (step.lo) {
    int chunkLast = dims.ki - 3 * dims.chunkStride + offset - partBytes;
    builder.create<rocisa::SCmpLeI32>(step.lo->container(), chunkLast);
    builder.create<rocisa::SCBranchSCC0>(zero->getLabelName());
  }
  {
    RegisterRange neg = vgpr();
    signOfSum(builder, neg.ref(), laneRem.ref(), -partBytes);
    if (laneLo) {
      RegisterRange belowLo = vgpr();
      signOfDifference(builder, belowLo.ref(), partBytes - 1, laneLo->ref());
      builder.create<rocisa::VOrB32>(neg.container(), neg.container(),
                                     belowLo.container());
    }
    issueChunk(0, chunk, soffset, neg.ref());
  }
  builder.create<rocisa::SBranch>(chunkEnd->getLabelName());
  builder.insert(zero);
  // The slot's registers may still have a prologue load in flight when the
  // split has no full iteration; the step's scale loads are issued on every
  // path and complete after it.
  waitForPrologueLoads({ringData.ref(chunk * dims.columns * dwordsPerChunk,
                                     dims.columns * dwordsPerChunk)},
                       dims.scaleLoads);
  for (int column = 0; column < dims.columns; ++column)
    for (int dword = 0; dword < dwordsPerChunk; ++dword)
      builder.movb32(dataChunk(0, chunk, column).slice(dword), 0);
  builder.insert(chunkEnd);
}

// Loads dword d in [0, 4) of chunk u for every column. A lane reads dword d
// only when it overlaps [lo, hi) (4d < laneRem and 4d + 4 > laneLo), else its
// voffset is the out-of-range -1. The dwords that straddle lo or hi are loaded
// at their aligned addresses and their bytes outside [lo, hi) cleared with an
// AND mask (both masks when lo and hi share a dword). Those bytes lie inside
// the column's storage, since every A and B column is padded to 4 bytes, but
// belong to another split or to the padding. The steering, mask and soffset of
// a dword do not depend on the column, so the loop over d is outermost and
// only one steering and mask pair is live.
void SkinnyGEMMEmitter::loadTailDwords(int chunk, RegisterRef soffset,
                                       RegisterRef laneRem,
                                       std::optional<RegisterRef> laneLo) {
  RegisterRange voffset = vgpr();
  for (int d = 0; d < dwordsPerChunk; ++d) {
    int start = d * dwordBytes;
    RegisterRange steer = vgpr();
    RegisterRange mask = vgpr();
    // Only a step with a lower bound needs a second temporary.
    std::optional<RegisterRange> hiTemp;
    if (laneLo)
      hiTemp = vgpr();
    // Steering: 0 when the dword overlaps [lo, hi), else -1. The mask register
    // holds the lo side until the final AND.
    signOfSum(builder, steer.ref(), laneRem, -(start + 1));
    if (laneLo) {
      signOfDifference(builder, mask.ref(), start + dwordBytes - 1, *laneLo);
      builder.create<rocisa::VOrB32>(steer.container(), steer.container(),
                                     mask.container());
    }
    // Clear the bytes of the dword below lo and at or after hi.
    if (laneLo) {
      RegisterRef below = *laneLo;
      if (start != 0) {
        builder.addi32(mask.ref(), *laneLo, -start);
        below = mask.ref();
      }
      clampedBitShift(builder, mask.ref(), below);
      builder.create<rocisa::VLShiftLeftB32>(mask.container(), mask.container(),
                                             -1);
    }
    RegisterRef hiMask = laneLo ? hiTemp->ref() : mask.ref();
    builder.subi32(hiMask, start + dwordBytes, laneRem);
    clampedBitShift(builder, hiMask, hiMask);
    builder.create<rocisa::VLShiftRightB32>(hiMask.container(),
                                            hiMask.container(), -1);
    if (laneLo)
      builder.create<rocisa::VAndB32>(mask.container(), mask.container(),
                                      hiMask.container());
    std::optional<RegisterRange> dwordOffset;
    RegisterRef dwordSoffset = soffset;
    if (d > 0) {
      dwordOffset = sgpr();
      dwordSoffset = dwordOffset->ref();
      builder.addi32(dwordSoffset, soffset, start);
    }
    for (int column = 0; column < dims.columns; ++column)
      loadMaskedDword(chunk, column, d, voffset.ref(), steer.ref(),
                      dwordSoffset, mask.ref());
  }
}

// Loads one dword of a column, steered, and clears the bytes outside [lo, hi).
// The AND consumes the load just issued, so nothing may stay in flight.
void SkinnyGEMMEmitter::loadMaskedDword(int chunk, int column, int dword,
                                        RegisterRef voffset, RegisterRef steer,
                                        RegisterRef soffset, RegisterRef mask) {
  RegisterRef dst = dataChunk(0, chunk, column).slice(dword);
  builder.create<rocisa::VOrB32>(voffset.container(), dBase.sub(column),
                                 steer.container());
  builder.create<rocisa::BufferLoadB32>(dst.container(), voffset.container(),
                                        dataRsrc(column).container(),
                                        soffset.container(), offen());
  waitForVectorLoads(0);
  builder.create<rocisa::VAndB32>(dst.container(), dst.container(),
                                  mask.container());
}

// Persistent tile loop.
//
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

// dst = src - quotient * divisor, the remainder of a non-negative src.
static void remainderByLiteral(IRBuilder &builder, RegisterRef dst,
                               RegisterRef src, RegisterRef quotient,
                               int divisor) {
  if (isPowerOfTwo(divisor)) {
    bitAnd(builder, dst, src, divisor - 1);
    return;
  }
  multiplyBy(builder, dst, quotient, divisor);
  builder.subi32(dst, src, dst);
}

// dst = ceilDiv(src, divisor) for a non-negative src. dst must not alias src.
static void ceilDivByLiteral(IRBuilder &builder, RegisterRef dst,
                             RegisterRef src, int divisor) {
  if (isPowerOfTwo(divisor)) {
    builder.addi32(dst, src, divisor - 1);
    shiftRight(builder, dst, dst, log2Exact(divisor));
    return;
  }
  RegisterRange sum = builder.allocateSgpr(1);
  builder.addi32(sum.ref(), src, divisor - 1);
  divideByConstant(builder, dst, sum.ref(), divisor);
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
  if (dims.numXcds != 1) {
    // perXcd = ceilDiv(numTiles, X) and tall = (numTiles - 1) mod X + 1, the
    // XCDs that hold perXcd tiles, the others hold one less.
    loop.perXcd = sgpr();
    ceilDivByLiteral(builder, loop.perXcd->ref(), loop.numTiles.ref(),
                     dims.numXcds);
    loop.tall = sgpr();
    RegisterRange last = sgpr();
    builder.addi32(last.ref(), loop.numTiles.ref(), -1);
    if (isPowerOfTwo(dims.numXcds)) {
      bitAnd(builder, loop.tall->ref(), last.ref(), dims.numXcds - 1);
    } else {
      RegisterRange quotient = sgpr();
      divideByConstant(builder, quotient.ref(), last.ref(), dims.numXcds);
      remainderByLiteral(builder, loop.tall->ref(), last.ref(), quotient.ref(),
                         dims.numXcds);
    }
    builder.addi32(loop.tall->ref(), loop.tall->ref(), 1);
  }
  if (dims.group != 1) {
    loop.tilesY = std::move(tilesY);
    loop.bandSize = sgpr();
    multiplyBy(builder, loop.bandSize->ref(), loop.tilesX.ref(), dims.group);
  }
  return loop;
}

// One iteration of the tile loop: the tile of the block id b, which the loop
// counter holds, runs @run_tile. Everything the tile writes is rewritten here
// before it is read: the bases, the selectors, the accumulators and the K
// state. Only the lane values and the loop invariants carry over.
void SkinnyGEMMEmitter::runPersistentTile(const SetupScalars &scalars,
                                          LaneValues &lane,
                                          const TileLoop &loop) {
  if (dims.scalesShared) {
    // The previous tile's scales must be read from LDS before this tile
    // overwrites them. gpu.barrier waits for every outstanding memory operation
    // first.
    builder.createWaitCnt(WaitCounts{.vectorLoads = 0, .lds = 0});
    builder.create<rocisa::SBarrier>();
  }
  RegisterRange tx = sgpr();
  RegisterRange ty = sgpr();
  RegisterRange kz = sgpr();
  tile = TileIds{tx.ref(), ty.ref(), kz.ref()};
  decodeTile(loop);
  {
    std::optional<RegisterRange> slabRot;
    std::optional<RegisterRef> rotation;
    if (dims.reordered && dims.upIters != 1) {
      // The reordered tiles key the rotation on the wide slab kz * tilesX + tx,
      // so that the tiles that share a wide column read the same K offset
      // together.
      slabRot = sgpr();
      RegisterRange slab = sgpr();
      builder.muli32(slab.ref(), tile.kz, loop.tilesX.ref());
      builder.addi32(slab.ref(), slab.ref(), tile.tx);
      blockRotation(slabRot->ref(), slab.ref(), loop.ni->ref());
      rotation = slabRot->ref();
    } else if (loop.blockRot) {
      rotation = loop.blockRot->ref();
    }
    computeTile(scalars, rotation);
  }
  // The persistent launch adds the tile's part to the lane part of the wide
  // columns every tile: w0 = w0Lane + tx * BN and, for the scale loads, a
  // temporary w0Load = lane.w0Load + tx * BN, absent when the wide column is
  // uniform and w0 itself serves.
  std::optional<RegisterRange> w0Load;
  if (uniformWide()) {
    builder.muli32(w0.ref(), tile.tx, dims.tileWide);
  } else {
    RegisterRange tileBase = sgpr();
    builder.muli32(tileBase.ref(), tile.tx, dims.tileWide);
    builder.addi32(w0.ref(), lane.w0Lane->ref(), tileBase.ref());
    w0Load = vgpr();
    builder.addi32(w0Load->ref(), lane.w0Load->ref(), tileBase.ref());
  }
  fillDataBases(scalars, lane);
  fillScaleBases(scalars, lane, w0Load ? w0Load->ref() : w0.ref());
  w0Load.reset();
  fillSelectors();
  zeroAccumulators();
  runTile();
  emitEpilogue();
}

// Decodes the tile of the block id into (tx, ty, kz), after the @remap_xcd of
// the XCDs: kz is the plane of tiles, and within the plane the plain order has
// tx fastest while the grouped one walks bands of GROUP skinny tile rows ty
// fastest.
void SkinnyGEMMEmitter::decodeTile(const TileLoop &loop) {
  RegisterRef blockId = kernel.workgroupIdX().ref();
  std::optional<RegisterRange> remapped;
  RegisterRef logical = blockId;
  if (dims.numXcds != 1) {
    // Tile t is the position t / X of the XCD t mod X, and the logical tile is
    // the position in the contiguous range of that XCD; the first tall XCDs
    // hold one more tile than the others.
    remapped = sgpr();
    logical = remapped->ref();
    RegisterRange local = sgpr();
    RegisterRange xcd = sgpr();
    RegisterRange shortage = sgpr();
    if (isPowerOfTwo(dims.numXcds))
      shiftRight(builder, local.ref(), blockId, log2Exact(dims.numXcds));
    else
      divideByConstant(builder, local.ref(), blockId, dims.numXcds);
    remainderByLiteral(builder, xcd.ref(), blockId, local.ref(), dims.numXcds);
    builder.subi32(shortage.ref(), xcd.ref(), loop.tall->ref());
    builder.create<rocisa::SMaxI32>(shortage.container(), shortage.container(),
                                    0);
    builder.muli32(logical, xcd.ref(), loop.perXcd->ref());
    builder.addi32(logical, logical, local.ref());
    builder.subi32(logical, logical, shortage.ref());
  }
  RegisterRange pos = sgpr();
  builder.divsi32(tile.kz, logical, loop.plane.ref(), /*useIEEE=*/true);
  builder.muli32(pos.ref(), tile.kz, loop.plane.ref());
  builder.subi32(pos.ref(), logical, pos.ref());
  if (dims.group != 1) {
    // The band holds GROUP rows, fewer in the last one, so a tile at inBand is
    // at column inBand / rows and row firstTy + inBand mod rows.
    RegisterRange band = sgpr();
    RegisterRange inBand = sgpr();
    RegisterRange firstTy = sgpr();
    RegisterRange rows = sgpr();
    builder.divsi32(band.ref(), pos.ref(), loop.bandSize->ref(),
                    /*useIEEE=*/true);
    builder.muli32(inBand.ref(), band.ref(), loop.bandSize->ref());
    builder.subi32(inBand.ref(), pos.ref(), inBand.ref());
    multiplyBy(builder, firstTy.ref(), band.ref(), dims.group);
    builder.subi32(rows.ref(), loop.tilesY->ref(), firstTy.ref());
    builder.create<rocisa::SMinI32>(rows.container(), rows.container(),
                                    dims.group);
    builder.divsi32(tile.tx, inBand.ref(), rows.ref(), /*useIEEE=*/true);
    builder.muli32(tile.ty, tile.tx, rows.ref());
    builder.subi32(tile.ty, inBand.ref(), tile.ty);
    builder.addi32(tile.ty, tile.ty, firstTy.ref());
    return;
  }
  builder.divsi32(tile.ty, pos.ref(), loop.tilesX.ref(), /*useIEEE=*/true);
  builder.muli32(tile.tx, tile.ty, loop.tilesX.ref());
  builder.subi32(tile.tx, pos.ref(), tile.tx);
}

// Epilogue.

// Wait states between a v_dot2 write and a DPP read of its result (LLVM's
// DotWriteDifferentVALURead) and between a VALU write and a DPP read of it.
static constexpr int dotToDppWaitStates = 3;
static constexpr int valuToDppWaitStates = 2;
// Wait states of v_readlane_b32 after a VALU wrote its source, and of a VALU
// read of the SGPR that v_readlane_b32 wrote (gfx940, per LLVM's hazard
// recognizer; the reference omits the second).
static constexpr int valuToReadlaneWaitStates = 1;
static constexpr int readlaneToValuWaitStates = 2;
static constexpr int bankMaskAll = 0xf;

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
  // The epilogue overwrites the ring registers, and a prologue load that was
  // never consumed (a split with fewer than PF iterations) may still be in
  // flight.
  std::vector<RegisterRef> written = {state.red, state.off};
  if (state.wBase)
    written.push_back(*state.wBase);
  waitForPrologueLoads(written, /*issuedSince=*/0);
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
          RegisterRef load = prologueLoads[newest];
          if (reg.kind != load.kind)
            return false;
          return reg.start < load.start + load.count &&
                 load.start < reg.start + reg.count;
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

// Reduces every accumulator over its LANES_K lanes into the red registers:
// DPP adds that leave the cluster sum in every lane, which a v_readlane of
// lane 63 broadcasts for 64 lanes.
void SkinnyGEMMEmitter::reduceAccumulators(const EpilogueState &state) {
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
  int valuSinceReduce = 0;
  for (int p = 0; p < dims.accumulators; ++p) {
    // The last chunk ends with the dots of all accumulators in order, so p has
    // P - 1 - p VALU instructions after its last dot.
    int sinceDot = dims.accumulators - 1 - p + valuSinceReduce;
    emitNops(dotToDppWaitStates - sinceDot);
    valuSinceReduce += std::max(0, dotToDppWaitStates - sinceDot);
    RegisterRef dst =
        state.red.slice(redIndex(state.layout, p / dims.tn, p % dims.tn));
    emitDpp(dst, acc.ref(p), DppStep{-1, {1, 0, 3, 2}, false, false, 0xf, 1});
    valuSinceReduce += 1;
    for (const DppStep &step : steps) {
      emitNops(valuToDppWaitStates);
      emitDpp(dst, dst, step);
      valuSinceReduce += valuToDppWaitStates + 1;
    }
    if (dims.lanesK < 64)
      continue;
    // Lane 63 holds the total; every lane takes it.
    RegisterRange total = sgpr();
    emitNops(valuToReadlaneWaitStates);
    builder.create<rocisa::VReadlaneB32>(total.container(), dst.container(),
                                         63);
    emitNops(readlaneToValuWaitStates);
    builder.movb32(dst, total.ref());
    valuSinceReduce +=
        valuToReadlaneWaitStates + 1 + readlaneToValuWaitStates + 1;
  }
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
  StoreScalars scalars = makeStoreScalars(state, target);
  bool skinnyContiguous = state.layout.skinnyContiguous;
  int runs = skinnyContiguous ? dims.tn : dims.bm;
  for (int run = 0; run < runs; ++run) {
    if (skinnyContiguous)
      storeRun(state, target, scalars, 0, run, dims.bm);
    else
      storeRun(state, target, scalars, run, 0, dims.tn);
  }
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
            0, offen());
      else
        builder.create<rocisa::BufferStoreB32>(
            value.container(), state.off.container(), target.rsrc.container(),
            0, offen());
    }
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
  RegisterRef off = state.off;
  switch (dwords) {
  case 1:
    builder.create<rocisa::BufferStoreB32>(data, off.container(),
                                           target.rsrc.container(), 0, offen());
    return;
  case 2:
    builder.create<rocisa::BufferStoreB64>(data, off.container(),
                                           target.rsrc.container(), 0, offen());
    return;
  default:
    builder.create<rocisa::BufferStoreB128>(
        data, off.container(), target.rsrc.container(), 0, offen());
  }
}

// Split-K reduce.

// Matches #red_row and the gpu.block_dims of the MLIR.
static constexpr int reduceBlock = 256;

namespace {

struct ReduceArguments {
  int d;
  int ws;
  int ldD;
  int m;
  int n;
  int splits;
};

// Emits @splitk_reduce of the MLIR reference, one method per part of it.
class SplitKReduceEmitter {
public:
  SplitKReduceEmitter(Kernel &kernel, const ReduceArguments &offsets)
      : kernel(kernel), builder(kernel.builder()), offsets(offsets),
        splits(builder.allocateSgpr(1)), n(builder.allocateSgpr(1)),
        m(builder.allocateSgpr(1)), ldD(builder.allocateSgpr(1)),
        wsRsrc(builder.allocateSgpr(4, 4)), dRsrc(builder.allocateSgpr(4, 4)),
        tmp(builder.allocateSgpr(1)), rowBytes(builder.allocateSgpr(1)),
        sliceBytes(builder.allocateSgpr(1)), split(builder.allocateSgpr(1)),
        savedExec(builder.allocateSgpr(2, 2)), row(builder.allocateVgpr(1)),
        base(builder.allocateVgpr(1)), value(builder.allocateVgpr(1)) {}

  void emit() {
    // The pointers load straight into words 0 and 1 of their descriptors.
    kernel.loadArgument(splits.ref(), offsets.splits);
    kernel.loadArgument(n.ref(), offsets.n);
    kernel.loadArgument(m.ref(), offsets.m);
    kernel.loadArgument(ldD.ref(), offsets.ldD);
    kernel.loadArgument(wsRsrc.ref(0, 2), offsets.ws);
    kernel.loadArgument(dRsrc.ref(0, 2), offsets.d);
    computeRows();
    makeResources();
    sumSplits();
    storeBf16();
  }

private:
  // Computes i, inRange (in vcc) and iLoad, which overwrites i.
  void computeRows() {
    // The block is (256, 1, 1), so the packed y and z fields of v0 are zero and
    // no mask is needed.
    RegisterRef tid = kernel.workitemId().ref();
    shiftLeft(builder, tmp.ref(), kernel.workgroupIdX().ref(), 8);
    builder.addi32(row.ref(), tmp, tid);
    // Scalar loads return out of order, so only lgkmcnt(0) guarantees any of
    // the six arguments.
    builder.createWaitCnt(WaitCounts{.scalarMemory = 0});
    builder.create<rocisa::VCmpGtU32>(std::make_shared<rocisa::VCC>(),
                                      m.container(), row.container());
    builder.subi32(tmp.ref(), m, 1);
    // Both operands are below 2^31 because m <= 2^20 and i < m + 256, so the
    // signed min equals the unsigned one; rocisa has no v_min_u32.
    builder.create<rocisa::VMinI32>(row.container(), tmp.container(),
                                    row.container());
  }

  // Descriptors of ws (wsBytes) and D (dBytes), which bound the accesses.
  void makeResources() {
    shiftLeft(builder, rowBytes.ref(), m.ref(), 2);
    builder.muli32(sliceBytes.ref(), n, rowBytes);
    builder.muli32(wsRsrc.ref(2), splits, sliceBytes);
    // dBytes = ((n - 1) * ldD + m) * 2, as (2n - 2) * ldD + 2m.
    RegisterRef dBytes = dRsrc.ref(2);
    shiftLeft(builder, dBytes, n.ref(), 1);
    builder.addi32(dBytes, dBytes, -2);
    builder.muli32(dBytes, dBytes, ldD);
    shiftLeft(builder, tmp.ref(), m.ref(), 1);
    builder.addi32(dBytes, dBytes, tmp);
    builder.makeRsrc(wsRsrc.ref(), wsRsrc.ref(0, 2), wsRsrc.ref(2));
    builder.makeRsrc(dRsrc.ref(), dRsrc.ref(0, 2), dBytes);
  }

  // sum = the f32 sum over the splits. Each lane loads its element of ws at
  // (z * m * n + j * m + iLoad) * 4, so splits >= 1 is a precondition.
  void sumSplits() {
    // The row register is dead after base, so it holds the sum.
    RegisterRef sum = row.ref();
    shiftLeft(builder, row.ref(), row.ref(), 2);
    builder.muli32(tmp.ref(), kernel.workgroupIdY(), rowBytes);
    builder.addi32(base.ref(), tmp, row);
    builder.movb32(split.ref(), 0);
    builder.movb32(sum, 0);
    std::shared_ptr<rocisa::Label> loop = builder.createLabel("loop");
    builder.insert(loop);
    builder.muli32(tmp.ref(), split, sliceBytes);
    builder.addi32(value.ref(), tmp, base);
    builder.create<rocisa::BufferLoadB32>(value.container(), value.container(),
                                          wsRsrc.container(), 0, offen());
    // The load is the newest vector memory operation, so nothing is issued
    // after it and vmcnt waits down to 0.
    builder.createWaitCnt(WaitCounts{.vectorLoads = 0});
    builder.create<rocisa::VAddF32>(sum.container(), sum.container(),
                                    value.container());
    builder.addi32(split.ref(), split, 1);
    builder.create<rocisa::SCmpLtI32>(split.container(), splits.container());
    builder.create<rocisa::SCBranchSCC1>(loop->getLabelName());
  }

  // Truncates the sum to bf16 and stores it where inRange.
  void storeBf16() {
    RegisterRef sum = row.ref();
    RegisterRef tid = kernel.workitemId().ref();
    RegisterRef blockX = kernel.workgroupIdX().ref();
    // truncf keeps the top 16 bits instead of rounding to nearest.
    shiftRight(builder, sum, sum, 16);
    std::shared_ptr<rocisa::Label> skip = builder.createLabel("skip");
    builder.create<rocisa::SAndSaveExecB64>(savedExec.container(),
                                            std::make_shared<rocisa::VCC>());
    builder.create<rocisa::SCBranchVCCZ>(skip->getLabelName());
    // dOff = (j * ldD + i) * 2, recomputed from the thread id; the load
    // address register is dead.
    RegisterRef dOff = value.ref();
    shiftLeft(builder, dOff, tid, 1);
    shiftLeft(builder, tmp.ref(), blockX, 9);
    builder.addi32(dOff, tmp, dOff);
    builder.muli32(tmp.ref(), kernel.workgroupIdY(), ldD);
    shiftLeft(builder, tmp.ref(), tmp.ref(), 1);
    builder.addi32(dOff, tmp, dOff);
    builder.create<rocisa::BufferStoreB16>(sum.container(), dOff.container(),
                                           dRsrc.container(), 0, offen());
    builder.insert(skip);
    builder.create<rocisa::SMovB64>(std::make_shared<rocisa::EXEC>(),
                                    savedExec.container());
    builder.createEndProgram();
  }

  Kernel &kernel;
  IRBuilder &builder;
  ReduceArguments offsets;
  RegisterRange splits, n, m, ldD, wsRsrc, dRsrc, tmp, rowBytes, sliceBytes,
      split, savedExec, row, base, value;
};

} // namespace

// Public entry points.

static void require(bool condition, const char *knob, int value,
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
  require(config.partialSteps >= 0 && config.partialSteps <= 2, "partialSteps",
          config.partialSteps, "0, 1 or 2");
  require(config.group >= 1, "group", config.group, "at least 1");
  require(config.numXcds >= 1, "numXcds", config.numXcds, "at least 1");
}

// The kernarg layout of the reference: the pointers, the i32 scalars, then
// the hidden group sizes and block counts.
static SkinnyArguments declareArguments(KernelSignature &signature) {
  SkinnyArguments arguments;
  arguments.d = signature.addPointer("D");
  arguments.a = signature.addPointer("A");
  arguments.b = signature.addPointer("B");
  arguments.scalesA = signature.addPointer("scalesA");
  arguments.scalesB = signature.addPointer("scalesB");
  arguments.ws = signature.addPointer("ws");
  arguments.ldD = signature.addScalar("ldD", 4);
  arguments.ldA = signature.addScalar("ldA", 4);
  arguments.ldB = signature.addScalar("ldB", 4);
  arguments.m = signature.addScalar("m", 4);
  arguments.n = signature.addScalar("n", 4);
  arguments.k = signature.addScalar("k", 4);
  arguments.spk = signature.addScalar("spk", 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeY, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeZ, 4);
  arguments.blocksX =
      signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  arguments.blocksY =
      signature.addHidden(KernelArgumentKind::hiddenBlockCountY, 4);
  arguments.blocksZ =
      signature.addHidden(KernelArgumentKind::hiddenBlockCountZ, 4);
  return arguments;
}

std::string rocm::tensilelite::codegen::skinnyGEMMKernelName(
    const SkinnyGEMMConfig &config) {
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
         std::to_string(config.grid3d ? 1 : 0);
}

std::string
rocm::tensilelite::codegen::emitSkinnyGEMM(const Target &target,
                                           const SkinnyGEMMConfig &config) {
  validate(config);
  SkinnyDims dims(config);
  if (dims.ldsBytes > target.maxLdsSize())
    throw std::invalid_argument(
        "config exceeds the LDS budget: " + std::to_string(dims.ldsBytes) +
        " bytes of " + std::to_string(target.maxLdsSize()));
  Kernel kernel(target, skinnyGEMMKernelName(config), 64 * config.waves);
  if (dims.scalesShared)
    kernel.setLdsSize(dims.ldsBytes);
  // The workgroup ids are the tile (tx, ty) and the split kz; the persistent
  // launch only has the block id x.
  if (config.grid3d)
    kernel.enableWorkgroupIds(true, true);
  SkinnyArguments arguments = declareArguments(kernel.signature());
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

std::string rocm::tensilelite::codegen::emitSplitKReduce(const Target &target) {
  Kernel kernel(target, "splitk_reduce", reduceBlock);
  // The workgroup id y is the column j.
  kernel.enableWorkgroupIds(true, false);
  KernelSignature &signature = kernel.signature();
  ReduceArguments offsets;
  offsets.d = signature.addPointer("D");
  offsets.ws = signature.addPointer("ws");
  offsets.ldD = signature.addScalar("ldD", 4);
  offsets.m = signature.addScalar("m", 4);
  offsets.n = signature.addScalar("n", 4);
  offsets.splits = signature.addScalar("splits", 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeY, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeZ, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountY, 4);
  signature.addHidden(KernelArgumentKind::hiddenBlockCountZ, 4);
  SplitKReduceEmitter(kernel, offsets).emit();
  return kernel.toAssembly();
}
