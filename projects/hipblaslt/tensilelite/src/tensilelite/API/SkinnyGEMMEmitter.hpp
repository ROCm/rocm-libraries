// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Declarations of the skinny MXFP8 GEMM emitter and its helper types.

#ifndef ROCM_TENSILELITE_API_SKINNYGEMMEMITTER_HPP
#define ROCM_TENSILELITE_API_SKINNYGEMMEMITTER_HPP

#include <tensilelite/API/SkinnyGEMM.hpp>
#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Kernel.hpp>
#include <tensilelite/Codegen/Registers.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace rocm::tensilelite::skinny {

using codegen::IRBuilder;
using codegen::Kernel;
using codegen::MemoryWidth;
using codegen::RegisterRange;
using codegen::RegisterRef;

/// Kernarg offsets of the GEMM kernel's arguments.
struct SkinnyArguments {
  int32_t d, a, b, scalesA, scalesB, ws, c;
  int32_t ldD, ldA, ldB, m, n, k, spk, ldC, alpha, beta;
  int32_t blocksX, blocksY, blocksZ;
};

/// Constants derived from the knobs, named after the quantities they stand for.
struct SkinnyDims {
  explicit SkinnyDims(const SkinnyGEMMConfig &config);

  /// Chunks per iteration; every lane loads 4 dwordx4 per column.
  static constexpr int32_t chunks = 4;

  int32_t bm, tn, lanesK, waves, prefetch;
  /// GRID_3D; group and numXcds only apply to the persistent launch.
  bool grid3d;
  int32_t group, numXcds;
  /// USE_BETA: whether the kernel prefetches C and adds beta * C.
  bool useBeta;
  /// Whether the persistent launch reorders the tiles (GROUP or NUM_XCDS not
  /// 1), which changes the key of the K rotation.
  bool reordered;
  /// UP_ITERS up-front scale entries per split, 0 for ring scales.
  int32_t upIters;
  /// PARTIAL_STEPS: 0 or 1 (tail step).
  int32_t partialSteps;
  /// NC = BM + TN columns per lane.
  int32_t columns;
  /// P = BM * TN accumulators per lane.
  int32_t accumulators;
  /// CG = 64 / LANES_K column groups per wave.
  int32_t columnGroups;
  /// With one wave of 64 lanes in one cluster the wide column of every lane is
  /// the same, so it stays in an SGPR.
  bool uniformWide;
  /// BN = WAVES * CG * TN wide columns per tile.
  int32_t tileWide;
  /// G: consecutive wide tiles that share their scale lines and so run on one
  /// XCD in the 3-D launch; 1 when BN is not a power of two below 32.
  int32_t xcdRun;
  /// Wide columns of a scale row group, which share every 128-byte scale line.
  static constexpr int32_t scaleRowColumns = 32;
  /// KI = 64 * LANES_K elements per iteration.
  int32_t ki;
  /// S = min(LANES_K, 8) lanes per subgroup.
  int32_t subgroupLanes;
  int32_t halfSubgroup;
  int32_t log2LanesK, log2ColumnGroups, log2Subgroup, log2Half, log2Ki;
  /// max(PF, 1) ring slots.
  int32_t slots;
  /// Whether the up-front scales go through LDS: several
  /// waves and CG < 4.
  bool scalesShared;
  /// R row slots of the scale load role domain: the block when shared, else
  /// the wave.
  int32_t roleSlots;
  /// Ring scale dwords per slot, NC with ring scales and 0 with up-front
  /// scales.
  int32_t scaleLoads;
  /// L loads per ring slot: the scale dwords of a ring-scale slot (scaleLoads
  /// of them, NC), then 4 chunks of NC.
  int32_t loadsPerSlot;
  /// Bytes of LDS: the shared up-front scales, UP * NC * WAVES * 256, or with
  /// mfma the rings of the waves (with lds) and the K reduction.
  int32_t ldsBytes;
  /// Bytes between the chunks of a lane, 16 * S.
  int32_t chunkStride;
  /// MFMA: whether the scaled-MFMA compute path runs. Then tileWide is 16 * TN,
  /// KI is 256 and ldsBytes holds the intra-block K reduction.
  bool mfma;
  /// MFMA: the 16-column wide tiles (TN) and skinny tiles (BM / 16) per wave,
  /// 0 without mfma.
  int32_t wideTiles, skinnyTiles;
  /// MFMA: whether the operand tiles are staged through a private LDS ring per
  /// wave.
  bool lds;
  /// LDS: bytes of one 16-column tile of one iteration in a ring slot (16
  /// columns x 256 K bytes).
  static constexpr int32_t stagedTileBytes = 4096;
  /// LDS: bytes of the ring of one wave, PF slots of every wide and skinny
  /// tile, plus one slot for the tail iteration with partial steps; 0 without
  /// lds.
  int32_t ringBytes;
  /// MFMA: bytes of the K reduction area, (WAVES - 1) * TN * (BM / 16) * 1024,
  /// placed after the rings; 0 without mfma.
  int32_t reduceBytes;
};

/// A slot of the prefetch ring being consumed.
struct SlotContext {
  int32_t slot;
  /// Full ring slots issued after this slot's group of loads.
  int32_t groupsAfter;
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
  /// Where the loads of the slot record their destinations, null when they are
  /// not recorded. Only the prologue records.
  std::vector<RegisterRef> *record = nullptr;
  /// VALU tail step only: a VGPR whose byte u is 0xFF when the lane's chunk u
  /// starts below the end of the split and 0 else; the decoded scale of a lane
  /// chunk that starts at or after it is cleared, as its block may hold
  /// padding.
  std::optional<RegisterRef> tailLive;
};

/// The tile of the workgroup as SGPRs: the workgroup ids in the 3-D launch,
/// the decoded tile of the current iteration in the persistent launch.
struct TileIds {
  RegisterRef tx, ty, kz;
};

/// The values of the persistent tile loop that do not depend on the tile.
struct TileLoop {
  /// The number of tiles and the tiles per wide and skinny row, and tilesX *
  /// tilesY.
  RegisterRange numTiles, tilesX, plane;
  /// Only the grouped decode needs tilesY.
  std::optional<RegisterRange> tilesY;
  /// NI, the iterations of a split, and the rotation of the block id.
  std::optional<RegisterRange> ni, blockRot;
  /// The remap of the XCDs: tiles per XCD and the XCDs with one more.
  std::optional<RegisterRange> perXcd, tall;
  /// GROUP * tilesX, the tiles of a band.
  std::optional<RegisterRange> bandSize;
};

/// Remaps the workgroup id tx so that the tiles that share scale lines run on
/// one XCD.
void groupTilesPerXcd(IRBuilder &builder, const SkinnyDims &dims, int32_t xcds,
                      RegisterRef tx, RegisterRef blocksX);

/// Decodes the tile of the block id into (tx, ty, kz) in the persistent launch.
void decodeTile(IRBuilder &builder, const SkinnyDims &dims, RegisterRef blockId,
                const TileIds &ids, const TileLoop &loop);

/// Invalid offset of a buffer access, whose range check then zero-fills the
/// load and drops the store.
inline constexpr int32_t outOfRange = -1;

/// The largest value of the 16-bit offset field of a ds instruction.
inline constexpr int32_t maxDsOffset = 65535;

/// Returns whether value is a positive power of two.
bool isPowerOfTwo(int32_t value);

/// Returns log2 of a power of two.
int32_t log2Exact(int32_t value);

/// dst = src / divisor, for non-negative operands.
void divideByLiteral(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                     int32_t divisor);

/// dst = src - quotient * divisor, the remainder given the quotient.
void remainderByLiteral(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                        RegisterRef quotient, int32_t divisor);

/// dst = src mod divisor, for non-negative operands.
void moduloByLiteral(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                     int32_t divisor);

/// dst = ceil(src / divisor).
void ceilDivByLiteral(IRBuilder &builder, RegisterRef dst, RegisterRef src,
                      int32_t divisor);

/// The quotient and the remainder of an exact division of non-negative
/// operands.
void divMod(IRBuilder &builder, RegisterRef quotient, RegisterRef remainder,
            RegisterRef dividend, RegisterRef divisor);

/// Aligns a hot loop header to 16 bytes.
void alignLoopHeader(IRBuilder &builder);

/// Branches to label when beta is zero, -0.0 included.
void branchIfBetaZero(IRBuilder &builder, RegisterRef beta, RegisterRef scratch,
                      const std::shared_ptr<rocisa::Label> &label);

/// Waits until at most outstanding vector memory loads are in flight.
void waitForVectorLoads(IRBuilder &builder, int32_t outstanding);

/// The loads of one ring slot, in issue order, and the in-order vmcnt
/// arithmetic over them: a vmcnt wait that must see the load at a position of a
/// slot complete allows exactly the loads issued after it.
struct RingLoads {
  /// Loads in one ring slot.
  int32_t perSlot;
  /// Loads issued after the load at position of a slot: the rest of its slot,
  /// groupsAfter full slots, and extra more loads.
  int32_t issuedAfter(int32_t position, int32_t groupsAfter,
                      int32_t extra = 0) const;
};

/// The byte offset in a ring of LDS that a buffer load writes to, relative to
/// the SGPR base of the ring.
struct LdsTarget {
  RegisterRef base;
  int32_t offset;
};

/// One buffer load of a ring slot. The destination is a VGPR range (dst) or an
/// LDS ring (lds), never both.
struct SlotLoad {
  RegisterRef rsrc, voffset, soffset;
  /// The buffer access offset.
  int32_t immediate = 0;
  MemoryWidth width;
  std::optional<RegisterRef> dst;
  std::optional<LdsTarget> lds;
  /// A per-lane mask ORed into the voffset first, which steers the lanes whose
  /// mask is -1 out of range.
  std::optional<RegisterRef> steer;
};

/// Emits the loads in order. A steered voffset goes through one
/// temporary, which every load reuses since a load reads its voffset when it
/// issues; it never aliases a destination, which may still have a load in
/// flight. The VGPR destinations are appended to record, if any.
void emitSlotLoads(IRBuilder &builder, const std::vector<SlotLoad> &loads,
                   std::vector<RegisterRef> *record = nullptr);

/// One step of the ring schedule: the consumption of the slot of one iteration
/// and, with refill, its refill by the iteration PF later.
struct RingStep {
  int32_t slot;
  /// Full ring slots issued after this slot's loads when it is consumed.
  int32_t groupsAfter;
  /// Whether the step refills the slot, and whether a later slot follows.
  bool refill, hasNext;
  /// The logical iteration of the slot is base + offset, base an SGPR (the loop
  /// counter, or the first iteration of a remainder arm) and offset a literal;
  /// the refill iteration is that plus PF. A prologue fill has the slot as
  /// offset and no meaningful base. Straight-line steps (the arms and the
  /// prologue) set inLoop to false.
  RegisterRef base;
  int32_t offset;
  bool inLoop;
};

/// The per-path half of the ring schedule that emitRingLoop drives: the layout
/// of the slots and the loads and compute of one step.
class RingPath {
public:
  virtual ~RingPath() = default;
  /// PF; 0 selects the form that issues and consumes one iteration per round.
  virtual int32_t slots() const = 0;
  /// The SGPR with the number of iterations of the split.
  virtual RegisterRef iterations() const = 0;
  /// Issues the loads of the iteration of the step: the prologue slot, or with
  /// PF = 0 the iteration of the loop body.
  virtual void fill(const RingStep &step) = 0;
  /// Called once the prologue and the steady-end computation are emitted.
  virtual void afterPrologue() = 0;
  /// Called before the loop header, to hoist loop invariants.
  virtual void beginLoop() = 0;
  /// Called after the loop, to release what beginLoop hoisted.
  virtual void endLoop() = 0;
  /// Waits for the slot, computes, and with step.refill refills it, in the
  /// placement that the path chooses.
  virtual void consume(const RingStep &step) = 0;
  /// Called after the last arm, once every step is emitted.
  virtual void drain() = 0;
  /// Called after the last step of a remainder arm.
  virtual void endArm() = 0;
};

/// Emits the K loop over the iterations of the path: the PF = 0 loop, or the
/// prologue that fills the ring, the steady loop unrolled over the PF slots and
/// the straight-line arms for the 1 to 2 * PF - 1 remaining iterations.
void emitRingLoop(IRBuilder &builder, RingPath &path);

/// The lanes of a tail iteration, the last iteration of a split when it keeps
/// only some of its K bytes. The iteration has SkinnyDims::chunks chunks, and
/// the chunk u of a lane covers the 16 K bytes [laneBytes + u * chunkStride,
/// +16).
struct TailLanes {
  /// An SGPR with the bytes of the iteration that the split keeps, in [1,
  /// laneSpan + (chunks - 1) * chunkStride).
  RegisterRef end;
  /// A VGPR with the K byte offset of the lane's piece of chunk 0, a multiple
  /// of 16 in [0, laneSpan - 16].
  RegisterRef laneBytes;
  /// The bytes between the chunks of a lane.
  int32_t chunkStride;
  /// The largest laneBytes plus 16: the span of the pieces of all lanes in a
  /// chunk.
  int32_t laneSpan;
};

/// The per-path half of the tail iteration that emitTailChunks drives: where
/// the operand data of the tail slot lives and how it is loaded.
///
/// With chunkHi = end - u * chunkStride, the chunk u is decided by uniform
/// scalar compares: FULL when chunkHi >= laneSpan, as every lane then ends at
/// or below chunkHi; DEAD when chunkHi <= 0, as every lane starts at or after
/// 0; else EDGE. A lane of an EDGE chunk has rem = chunkHi - laneBytes bytes
/// below the end: it lies inside when rem >= 16, straddles when 0 < rem < 16
/// and is dead when rem <= 0. A FULL chunk needs nothing, a DEAD chunk is
/// zeroed, and an EDGE chunk has the dead lanes steered out of range, so that
/// they load zeros, and its bytes at or after the end cleared with a mask, as
/// the bytes of the straddling lane belong to the next split or to the padding
/// and may be NaNs.
///
/// Neutralising the scales of the K blocks at or after the end is the duty of
/// the path, and differs by design: the VALU path holds one decoded f32 scale
/// per lane chunk, which it ANDs with a per-chunk mask that is zero for the
/// chunks that start at or after the end, while the MFMA path holds the raw
/// E8M0 bytes of packed per-tile dwords, whose bytes are 32-byte blocks, so it
/// overrides the bytes of the blocks past the end to 127.
class TailPath {
public:
  virtual ~TailPath() = default;
  virtual TailLanes lanes() const = 0;
  /// The 4-dword VGPR range of chunk u of the tail slot of every operand
  /// column.
  virtual std::vector<RegisterRef> chunkData(int32_t chunk) const = 0;
  /// Whether the chunks are loaded by loadChunk; false when the data is already
  /// in the registers, with the bytes of the whole iteration.
  virtual bool loadsChunks() const { return true; }
  /// Issues the loads of chunk u into chunkData(u). The lanes whose steer, a
  /// VGPR of 0 or -1, is -1 load out of range.
  virtual void loadChunk(int32_t chunk, std::optional<RegisterRef> steer) = 0;
  /// Called before the registers of a dead chunk are zeroed, for loads that may
  /// still write them.
  virtual void beforeZero(int32_t chunk) { (void)chunk; }
};

/// Skips body when tailBytes, an SGPR, is zero.
void emitTailGuard(IRBuilder &builder, RegisterRef tailBytes,
                   const std::function<void()> &body);

/// Loads, zeroes or masks the chunks of the tail iteration by the three arms of
/// TailPath, then waits for the loads.
void emitTailChunks(IRBuilder &builder, TailPath &path);

/// dst, a VGPR, = the bitmask whose byte u is 0xFF when the chunk u of the lane
/// starts below the end, else 0.
void emitChunkLiveBytes(IRBuilder &builder, RegisterRef dst,
                        const TailLanes &lanes);

// The VGPR ring of the K phase: 4 chunks of every column per slot and, with
// ring scales, the raw scale dwords. It is allocated before every other VGPR
// for the alignment of its tuples and moved into emitTile's K phase, so it
// dies before the epilogue allocates.
struct Ring {
  Ring(IRBuilder &builder, const SkinnyDims &dims);
  RegisterRange data;
  std::optional<RegisterRange> raw;
};

// chunkBase[0] = ks, the split's start, and chunkBase[u] = chunkBase[0] + u *
// 16 * S, emitted when first needed.
struct ChunkBase {
  RegisterRange regs;
  std::vector<bool> ready;
};

// The registers of one tile, which the tile driver owns from the setup to the
// end of the epilogue.
struct TileState {
  TileState(IRBuilder &builder, const SkinnyDims &dims);
  // Allocates scaleGroup and cData, the VGPRs that the knobs make optional.
  // The 3-D launch calls it after the scale gather, which keeps their place
  // in the register layout.
  void allocateOptionalVectors(IRBuilder &builder, const SkinnyDims &dims,
                               int32_t cDwords);
  // The first wide column of the lane, an SGPR when it is uniform.
  RegisterRange w0;
  RegisterRange acc;
  // The voffsets of the data and the scale dword of every column.
  RegisterRange dBase;
  RegisterRange sBase;
  // The byte selectors of the scale permutes.
  RegisterRange selectors;
  RegisterRange s0;
  RegisterRange nIterFull;
  // The rotation, absent with one up-front entry, where it is the constant 0.
  std::optional<RegisterRange> fitRot;
  // With partial steps, the bytes of the tail iteration that the split keeps,
  // 0 when it has no tail.
  std::optional<RegisterRange> tailBytes;
  ChunkBase chunkBase;
  RegisterRange sValid, wValid;
  // With partial steps, the K offset of the 8-block scale group of the lane's
  // load role, sclOff & -256.
  std::optional<RegisterRange> scaleGroup;
  // Only with useBeta: the VGPRs that the tile-start prefetch loads C into.
  std::optional<RegisterRange> cData;
  // Destinations of the prologue loads in issue order, for the epilogue's
  // write-after-write wait. It records indices only and is never an operand
  // after the ring dies: the epilogue compares the indices of the freed ring
  // registers it reuses with it to find the loads that nothing consumed.
  std::vector<RegisterRef> prologueLoads;
};

// The lane addresses of the scale gather (ds_bpermute or the LDS read) and,
// with shared up-front scales, the thread's LDS staging address. The ring
// scales read them every iteration; the 3-D launch frees them after the
// up-front pack, the persistent launch reuses them every tile.
struct ScaleGather {
  ScaleGather(IRBuilder &builder, const SkinnyDims &dims);
  RegisterRange addresses;
  std::optional<RegisterRange> ldsStage;
};

// The up-front scales of the tile: the raw entries that pack in place, and
// for one entry the decoded chunk scales. They die with the K phase.
struct UpFrontScales {
  std::optional<RegisterRange> scaleVector;
  std::vector<RegisterRange> hoistedScales;
};

// What the K-loop phases read: the builder, the knob constants, the descriptors
// of the data and the scales of the skinny and the wide operand, and the tile's
// registers.
struct KLoopState {
  IRBuilder &builder;
  const SkinnyDims &dims;
  RegisterRef sRsrc, wRsrc, sSclRsrc, wSclRsrc;
  TileState &tile;
  Ring &ring;
  ScaleGather &gather;
  UpFrontScales &upFront;
};

// The two moments of the up-front scales in the K phase: the loads precede the
// prologue, and the pack follows it, as the in-order vmcnt waits count the
// loads issued in between.
enum class UpFrontStep { issue, pack };

/// Where one store or load path accesses memory.
struct StoreTarget {
  /// The buffer descriptor and the byte strides along the skinny and the wide
  /// dimension.
  RegisterRef rsrc, strideS, strideW;
  /// The workspace slice offset in bytes, folded into the offsets.
  std::optional<RegisterRef> sliceBase;
  /// bf16 (D, C) or raw f32 (workspace).
  bool bf16;
};

/// What the epilogue and the C prefetch read, in both compute paths.
struct OutputContext {
  IRBuilder &builder;
  const SkinnyDims &dims;
  /// The SGPR pair with alpha in the low dword, the operand of v_pk_mul_f32.
  RegisterRef alpha;
  std::optional<RegisterRef> beta;
  RegisterRef k, spk, mn, skinnyCols, wideCols, contigS;
  /// workspace.sliceBase is unset here; the epilogue sets it.
  StoreTarget d, workspace;
  /// Only with useBeta.
  std::optional<StoreTarget> c;
};

/// The final values of a tile in one lane after the reduction of the compute
/// path, as the shared epilogue sees them, for one orientation of the output.
/// The lane owns a grid of skinnyCount x wideCount output elements (i, j); the
/// path computes their addresses itself.
class OutputMap {
public:
  virtual ~OutputMap() = default;
  int32_t skinnyCount = 0, wideCount = 0;
  bool skinnyContiguous = false;
  /// Consecutive output elements along the contiguous dimension that sit in
  /// consecutive registers starting at an even one, so that one wide access
  /// covers them; 1 when there are none. Runs start at multiples of runLength
  /// along the contiguous dimension.
  int32_t runLength = 1;
  /// The VGPRs that hold the values, or for the C prefetch the C data, in the
  /// same layout.
  RegisterRef values;
  /// Contiguous register intervals of values, for the alpha multiply:
  /// segmentCount intervals of segmentLength registers, segmentStride apart.
  int32_t segmentCount = 1, segmentLength = 0, segmentStride = 0;
  /// Index in values of the register of element (i, j), which is also the
  /// dword of the masked per-element C data.
  virtual int32_t valueIndex(int32_t i, int32_t j) const = 0;
  /// The lanes that store when the whole tile is inside the output; nullopt
  /// when all lanes store.
  virtual std::optional<RegisterRef> storeLanes() const = 0;
  /// Emits the code that all offsets of a target share; called before its
  /// elements.
  virtual void prepare(const StoreTarget &target) = 0;
  /// Returns the VGPR that holds the byte offset of element (i, j) in the
  /// prepared target, after emitting its computation. The next call
  /// overwrites it.
  virtual RegisterRef offset(int32_t i, int32_t j) = 0;
  /// mask, an SGPR pair, = the lanes that access element (i, j): inside the
  /// output and a storing lane.
  virtual void accessLanes(RegisterRef mask, int32_t i, int32_t j) = 0;
};

/// Emits emitVariant(skinnyContiguous) for both orientations of the output,
/// selected at run time by contigS, or once when vectorStores is false, as the
/// orientations are then the same code.
void forEachOrientation(
    const OutputContext &ctx, bool vectorStores,
    const std::function<void(bool skinnyContiguous)> &emitVariant);

/// Branches to skip unless the tile's C is read: k <= spk, so that D is stored
/// directly, and beta != 0.
void branchUnlessCIsUsed(const OutputContext &ctx,
                         const std::shared_ptr<rocisa::Label> &skip);

/// The C prefetch of one orientation: loads the tile of C into map.values with
/// the addressing of the D store, through vector loads when the tile lies
/// inside the output and the orientation has runs, else per element.
void loadCTile(const OutputContext &ctx, OutputMap &map, const TileIds &ids);

/// The epilogue after the reduction of the compute path. A single split
/// (k <= spk) scales the values by alpha, adds beta * C from cData, the
/// prefetched C registers in the layout loadCTile wrote, and stores bf16 to D;
/// else the raw f32 values go to the slice kz * mn * 4 bytes of the workspace.
void emitOutput(const OutputContext &ctx, OutputMap &map, const TileIds &ids,
                std::optional<RegisterRef> cData);

/// Emits the GEMM kernel body.
class SkinnyGEMMEmitter {
public:
  SkinnyGEMMEmitter(Kernel &kernel, const SkinnyGEMMConfig &config,
                    const SkinnyArguments &arguments);

  void emit();

private:
  // SGPRs that the setup leaves for the tiles. blocksX is the block count x,
  // the number of persistent blocks in the 1-D launch, which has no blocksY and
  // keeps the split count in splits.
  struct SetupScalars {
    SetupScalars(IRBuilder &builder, bool grid3d);
    RegisterRange blocksX;
    std::optional<RegisterRange> blocksY;
    RegisterRange rowGroup, ldS, ldW;
    std::optional<RegisterRange> splits;
  };

  // Per-lane values, which only depend on the thread id: the lane part of the
  // wide columns the lane owns (w0Lane) and loads scales for (w0Load), without
  // the tile's tx * BN, both absent when the wide column is uniform. They die
  // with the tile setup in the 3-D launch, which writes w0Lane into the tile's
  // w0 instead, and live through the tile loop in the persistent launch.
  struct LaneValues {
    RegisterRange laneBytes, sclOff;
    std::optional<RegisterRange> w0Load, w0Lane;
  };

  // Setup and the tile loop. emitSetup returns the lane values; emitTile runs
  // one tile up to and including its epilogue, with loop null in the 3-D
  // launch.
  LaneValues emitSetup(SetupScalars &scalars, ScaleGather &gather,
                       std::optional<RegisterRef> w0Target);
  void emitTile(const TileIds &ids, SetupScalars &scalars, const TileLoop *loop,
                LaneValues &lane, ScaleGather &gather, Ring ring,
                TileState &tile);

  // K phase: the up-front scale loads and packing, the K loop (emitRingLoop
  // over a ValuRingPath), one ring slot and the tail step.
  class ValuRingPath;
  void emitUpFrontScales(KLoopState &state, UpFrontStep step);
  void emitKLoop(KLoopState &state);
  void emitCompute(KLoopState &state, const SlotContext &context);
  void emitTail(KLoopState &state);

  // The loads of the kernel arguments, the buffer sizes, the path dispatch and
  // the descriptors, the first part of emitSetup, which the MFMA path shares.
  void emitArguments(SetupScalars &scalars);

  // The whole kernel body of the VALU path, in SkinnyGEMMValu.cpp, and of the
  // scaled-MFMA path, in SkinnyGEMMMfma.cpp.
  void emitValu();
  void emitMfma();

  // The registers and descriptors that the epilogue and the C prefetch read.
  OutputContext outputContext();

  // The C prefetch of the tile and its epilogue, which run before and after
  // the K phase.
  void emitCPrefetch(const TileState &tile, const TileIds &ids);
  void emitEpilogue(const TileState &tile, const TileIds &ids);

  Kernel &kernel;
  IRBuilder &builder;
  SkinnyDims dims;
  SkinnyArguments arguments;

  // The SGPRs that live from the setup to the end of the last epilogue in both
  // launch forms.
  RegisterRange sRsrc, wRsrc, sSclRsrc, wSclRsrc, dRsrc, wsRsrc;
  // Only with useBeta: the descriptor of C.
  std::optional<RegisterRange> cRsrc;
  // The pair holds alpha in its low dword; the high dword is unused and only
  // forms the 2-SGPR operand pair that v_pk_mul_f32 needs.
  RegisterRange alpha;
  // Only with useBeta.
  std::optional<RegisterRange> beta;
  RegisterRange k, spk, mn;
  RegisterRange dStrideS, dStrideW, wsStrideS, wsStrideW;
  // The strides of C along the skinny and the wide dimension, only with
  // useBeta.
  std::optional<RegisterRange> cStrideS, cStrideW;
  RegisterRange skinnyCols, wideCols, contigS;
};

} // namespace rocm::tensilelite::skinny

#endif // ROCM_TENSILELITE_API_SKINNYGEMMEMITTER_HPP
