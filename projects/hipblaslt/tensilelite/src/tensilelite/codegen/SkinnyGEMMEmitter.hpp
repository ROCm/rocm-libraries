// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Private header of the skinny MXFP8 GEMM emitter, which implements @gemm,
// @run_tile and their helpers of gemm_mxfp8_skinny.mlir for GRID_3D = 1,
// UP_ITERS = 0 and PARTIAL_STEPS = 0. The method names follow the MLIR
// functions: the setup is in SkinnyGEMMSetup.cpp, the K loop in
// SkinnyGEMMLoop.cpp and the epilogue in SkinnyGEMMEpilogue.cpp.

#ifndef ROCM_TENSILELITE_CODEGEN_SKINNYGEMMEMITTER_HPP
#define ROCM_TENSILELITE_CODEGEN_SKINNYGEMMEMITTER_HPP

#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Kernel.hpp>
#include <tensilelite/codegen/Registers.hpp>
#include <tensilelite/codegen/SkinnyGEMM.hpp>

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
  /// L = 5 * NC loads per ring slot: NC scale dwords, then 4 chunks of NC.
  int loadsPerSlot;
  /// Bytes between the chunks of a lane, 16 * S.
  int chunkStride;
};

/// Returns log2 of a power of two.
int log2Exact(int value);

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
  LaneValues computeLaneValues(const SetupScalars &scalars);
  void computeWideBase();
  RegisterRange computeLaneBytes(RegisterRef klv);
  RegisterRange computeScaleOffset(RegisterRef lane);
  RegisterRange computeWideLoadBase(RegisterRef tid, RegisterRef lane);
  void computeAddresses(RegisterRef klv, RegisterRef lane);
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
  void issueScales(int slot, RegisterRef soffset);
  void issueChunk(int slot, int chunk, RegisterRef soffset);
  void startRefill(const SlotContext &context, RegisterRef soffset);
  void refillChunk(const SlotContext &context, int chunk, RegisterRef soffset);
  void waitForLoad(SlotLoadPosition where);
  RegisterRange gatherSlotScales(const SlotContext &context);
  void permSlotScales(int slot, RegisterRef packed);
  void decodeScales(RegisterRef dst, RegisterRef packed, int chunk);
  void computeChunk(const SlotContext &context, int chunk, RegisterRef scales);
  void processSlot(const SlotContext &context);
  void processStaged(const SlotContext &context, RegisterRef packed);
  void processDecodeFirst(const SlotContext &context, RegisterRange &packed);
  RegisterRef dataChunk(int slot, int chunk, int column) const;
  RegisterRef rawScale(int slot, int column) const;
  RegisterRef dataRsrc(int column) const;
  RegisterRef scaleRsrc(int column) const;

  // Epilogue (SkinnyGEMMEpilogue.cpp).
  void emitEpilogue();
  void emitEpilogueVariant(bool skinnyContiguous);
  RedLayout redLayout(bool skinnyContiguous) const;
  int runLength(bool skinnyContiguous) const;
  int redIndex(const RedLayout &layout, int r, int t) const;
  void waitForEpilogueRegisters(const EpilogueState &state);
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
  RegisterRange ringRaw;
  RegisterRange w0;
  RegisterRange acc;
  RegisterRange dBase;
  RegisterRange sBase;
  RegisterRange selectors;
  RegisterRange addresses;

  // SGPRs that live through the K loop and the epilogue.
  RegisterRange sRsrc, wRsrc, sSclRsrc, wSclRsrc, dRsrc, wsRsrc;
  RegisterRange k, spk, mn;
  RegisterRange dStrideS, dStrideW, wsStrideS, wsStrideW;
  RegisterRange skinnyCols, wideCols, contigS;
  RegisterRange s0;
  RegisterRange nIterFull, fitRot, kBaseIter;
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
