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
#include <optional>
#include <vector>

namespace rocm::tensilelite::skinny {

using codegen::IRBuilder;
using codegen::Kernel;
using codegen::RegisterRange;
using codegen::RegisterRef;

/// Kernarg offsets of the GEMM kernel's arguments.
struct SkinnyArguments {
  int32_t d, a, b, scalesA, scalesB, ws, c;
  int32_t ldD, ldA, ldB, m, n, k, spk, ldC, alpha, beta;
  int32_t blocksX, blocksY, blocksZ;
};

/// Constants derived from the knobs, named after the MLIR values.
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
  /// Whether the up-front scales go through LDS (@scales_shared): several
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
  /// Bytes of LDS holding the shared up-front scales, UP * NC * WAVES * 256.
  int32_t ldsBytes;
  /// Bytes between the chunks of a lane, 16 * S.
  int32_t chunkStride;
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
  // load role, sclOff & -256 (@scl_group).
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

  // K phase: the up-front scale loads and packing, the K loop, one ring slot
  // and the tail step.
  void emitUpFrontScales(KLoopState &state, UpFrontStep step);
  void emitKLoop(KLoopState &state);
  void emitCompute(KLoopState &state, const SlotContext &context);
  void emitTail(KLoopState &state);

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
