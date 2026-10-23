// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Private header of the skinny MXFP8 GEMM emitter, which implements @gemm,
// @run_tile and their helpers of gemm_mxfp8_skinny.mlir for GRID_3D = 1 and any
// PARTIAL_STEPS, with ring scales (UP_ITERS = 0) or up-front scales. The method
// names follow the MLIR functions: the setup is in SkinnyGEMMSetup.cpp, the K
// loop in SkinnyGEMMLoop.cpp, the up-front scales in SkinnyGEMMScales.cpp, the
// partial K steps in SkinnyGEMMSteps.cpp and the epilogue in
// SkinnyGEMMEpilogue.cpp.

#ifndef ROCM_TENSILELITE_CODEGEN_SKINNYGEMMEMITTER_HPP
#define ROCM_TENSILELITE_CODEGEN_SKINNYGEMMEMITTER_HPP

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Kernel.hpp>
#include <tensilelite/codegen/Registers.hpp>
#include <tensilelite/codegen/SkinnyGEMM.hpp>

#include "instruction/mem.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace rocm::tensilelite::codegen::detail {

/// Kernarg offsets of the GEMM kernel's arguments.
struct SkinnyArguments {
  int d, a, b, scalesA, scalesB, ws;
  int ldD, ldA, ldB, m, n, k, spk;
  int blocksX, blocksY, blocksZ;
};

/// Constants derived from the knobs, named after the MLIR values.
struct SkinnyDims {
  explicit SkinnyDims(const SkinnyGEMMConfig &config);

  /// Chunks per iteration; every lane loads 4 dwordx4 per column.
  static constexpr int chunks = 4;

  int bm, tn, lanesK, waves, prefetch;
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

/// The modifiers of a buffer load or store addressed with a voffset.
rocisa::MUBUFModifiers offen();

/// Returns log2 of a power of two.
int log2Exact(int value);

/// Multiplier and shift of LLVM's signed division by a positive constant.
struct SignedMagic {
  int32_t multiplier;
  int shift;
};

/// Returns the magic number of the signed division by divisor >= 2.
SignedMagic signedMagic(int32_t divisor);

/// Emits quotient = dividend / divisor for a non-negative SGPR dividend and a
/// divisor >= 2 that is not a power of two, as LLVM expands the division.
void divideByConstant(IRBuilder &builder, RegisterRef quotient,
                      RegisterRef dividend, int divisor);

// 32-bit ALU shorthands that pick the SALU or VALU instruction from the kind
// of dst, which must not be an AGPR. A VALU src must be a VGPR.
void shiftLeft(IRBuilder &builder, RegisterRef dst, RegisterRef src,
               int amount);
void shiftRight(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                int amount);
void bitAnd(IRBuilder &builder, RegisterRef dst, RegisterRef src, int mask);
/// dst = src * factor, a shift when factor is a power of two.
void multiplyBy(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                int factor);

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

/// Number of vector memory loads issued after the load at where; the count a
/// vmcnt wait must allow so that it is complete.
int loadsIssuedAfter(const SkinnyDims &dims, SlotLoadPosition where);

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
  // SGPRs that the setup reads after the resources are made.
  struct SetupScalars {
    explicit SetupScalars(IRBuilder &builder);
    RegisterRange blocksX, blocksY, rowGroup, ldS, ldW;
  };

  // SGPRs that only the resources and the path dispatch read.
  struct ResourceScalars {
    explicit ResourceScalars(IRBuilder &builder);
    RegisterRange ldD, ldA, ldB, m, n, blocksZ;
    RegisterRange aPtr, bPtr, sclAPtr, sclBPtr;
    RegisterRange aBytes, bBytes, sclABytes, sclBBytes;
  };

  // Per-lane values that only the setup reads.
  struct LaneValues {
    RegisterRange laneBytes, sclOff, w0Load;
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

  // Setup (SkinnyGEMMSetup.cpp).
  void loadArguments(SetupScalars &scalars, ResourceScalars &resources);
  void makeResources(const SetupScalars &scalars,
                     const ResourceScalars &resources);
  void makeBufferSizes(const SetupScalars &scalars,
                       const ResourceScalars &resources);
  void dispatchPaths(const SetupScalars &scalars,
                     const ResourceScalars &resources);
  void computeTile(const SetupScalars &scalars);
  void computeRotation(const SetupScalars &scalars, RegisterRef rot);
  LaneValues computeLaneValues();
  void computeWideBase();
  RegisterRange computeLaneBytes(RegisterRef klv);
  RegisterRange computeRoleJ(RegisterRef tid);
  void threadInDomain(RegisterRef dst, RegisterRef tid, RegisterRef roleJ);
  RegisterRange computeScaleOffset(RegisterRef lane,
                                   const std::optional<RegisterRange> &roleJ);
  RegisterRange computeWideLoadBase(RegisterRef tid, RegisterRef lane,
                                    const std::optional<RegisterRange> &roleJ);
  void computeAddresses(RegisterRef klv, RegisterRef lane);
  void computeSharedAddress(RegisterRef klv, RegisterRef tid);
  void fillDataBases(const SetupScalars &scalars, LaneValues &lane);
  void fillScaleBases(const SetupScalars &scalars, const LaneValues &lane);
  void fillSelectors();
  void zeroAccumulators();
  RegisterRef selector(int column) const;
  void selectorValue(RegisterRef dst, RegisterRef col);

  // K loop (SkinnyGEMMLoop.cpp).
  void runTile();
  void runNoPrefetchLoop();
  void runPrefetchedLoop();
  void emitPrologue();
  void computeSteadyEnd(RegisterRef steadyEnd);
  void emitSteadyLoop(RegisterRef steadyEnd);
  void emitRemainderArms(RegisterRef steadyEnd);
  void emitRemainderArm(RegisterRef left, RegisterRef steadyEnd, int remaining);
  void hoistChunkBases();
  void ensureChunkBase(int chunk);
  void wrapIteration(RegisterRef dst, RegisterRef sum);
  void wrapZero(RegisterRef dst);
  void issueScales(RegisterRef raw, RegisterRef soffset,
                   std::optional<RegisterRef> neg = std::nullopt);
  void issueChunk(int slot, int chunk, RegisterRef soffset,
                  std::optional<RegisterRef> neg = std::nullopt);
  void refillOffset(const SlotContext &context, RegisterRef soffset);
  void startRefill(const SlotContext &context, RegisterRef soffset);
  void refillChunk(const SlotContext &context, int chunk, RegisterRef soffset);
  void waitForLoad(SlotLoadPosition where);
  void waitForVectorLoads(int outstanding);
  RegisterRange gatherSlotScales(const SlotContext &context);
  void gatherSlotScales(RegisterRef raw, RegisterRef lo, int firstOutstanding);
  void permSlotScales(RegisterRef hi, RegisterRef lo, RegisterRef out);
  void decodeScales(RegisterRef dst, RegisterRef packed, int chunk);
  void waitBeforeFirstUse(const SlotContext &context, int chunk, int column,
                          int ownRefill);
  void computeChunk(const SlotContext &context, int chunk, RegisterRef scales);
  void processSlot(const SlotContext &context);
  void processStaged(const SlotContext &context, RegisterRef packed);
  void processDecodeFirst(const SlotContext &context, RegisterRange &packed);
  void processUpFront(const SlotContext &context,
                      std::optional<RegisterRange> &packed);
  RegisterRef dataChunk(int slot, int chunk, int column) const;
  RegisterRef rawScales(int slot) const;
  RegisterRef dataRsrc(int column) const;
  RegisterRef scaleRsrc(int column) const;

  // Up-front scales (SkinnyGEMMScales.cpp).
  void issueUpFrontScales();
  void packScaleVectors();
  void packPerWave();
  void packShared();
  void stageScalesInLds();
  void readSharedEntry(int entry);
  std::optional<RegisterRange> selectScales(const SlotContext &context);
  void decodeHoistedScales();
  int upFrontLoadsAfter(int index) const;
  RegisterRef vectorEntry(RegisterRef physical,
                          std::optional<RegisterRange> &storage);

  // Partial steps (SkinnyGEMMSteps.cpp).
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

  // The steering, mask and soffset of one dword of a tail chunk.
  struct TailDword {
    RegisterRef steer;
    RegisterRef soffset;
    RegisterRef mask;
  };

  void allocateStepRawScales();
  void runTailStep();
  void runHeadAndTailSteps();
  void runPartialStep(const PartialStep &step);
  void loadPartialStep(const PartialStep &step);
  void computePartialStep(const PartialStep &step);
  void issueScalesWithinK(RegisterRef raw, RegisterRef soffset,
                          RegisterRef kAvail);
  void straddlesInChunk(RegisterRef dst, RegisterRef bound);
  void loadStepChunk(const PartialStep &step, int chunk,
                     RegisterRef partialChunks);
  void loadTailDwords(const PartialStep &step, int chunk, RegisterRef soffset,
                      RegisterRef laneRem, std::optional<RegisterRef> laneLo);
  void loadMaskedDword(int chunk, int column, int dword, RegisterRef voffset,
                       const TailDword &tail);

  // Epilogue (SkinnyGEMMEpilogue.cpp).
  void emitEpilogue();
  void emitEpilogueVariant(bool skinnyContiguous);
  RedLayout redLayout(bool skinnyContiguous) const;
  int runLength(bool skinnyContiguous) const;
  int redIndex(const RedLayout &layout, int r, int t) const;
  void waitForEpilogueRegisters(const EpilogueState &state);
  void waitForPrologueLoads(const std::vector<RegisterRef> &written,
                            int issuedSince);
  void reduceAccumulators(const EpilogueState &state);
  int reduceAccumulator(const EpilogueState &state, int r, int t);
  void emitDpp(RegisterRef dst, RegisterRef src, const DppStep &step);
  std::vector<DppStep> reductionSteps() const;
  void emitNops(int count);
  void computeTileFull(RegisterRef tileFull);
  void storeTile(const EpilogueState &state, const StoreTarget &target);
  StoreScalars makeStoreScalars(const EpilogueState &state,
                                const StoreTarget &target);
  void storeMasked(const EpilogueState &state, const StoreTarget &target);
  void storeVector(const EpilogueState &state, const StoreTarget &target);
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

} // namespace rocm::tensilelite::codegen::detail

#endif // ROCM_TENSILELITE_CODEGEN_SKINNYGEMMEMITTER_HPP
