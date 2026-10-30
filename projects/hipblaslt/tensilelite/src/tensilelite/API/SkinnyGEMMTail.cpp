// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The part of the tail iteration that the VALU and the MFMA paths share: the
// guard that skips it when the split keeps no byte of it, and the
// classification of every chunk as FULL, EDGE or DEAD (see TailPath) with the
// loads, the zeroing and the byte masks that each needs.

#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Registers.hpp>

#include "SkinnyGEMMEmitter.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::skinny;

// The largest byte shift of a dword mask: a shift of 4 bytes (32 bits) would
// not clear the mask, so the shift is clamped to 3 bytes and a dword that needs
// a shift of 4 bytes is cleared by the second mask of maskEdgeChunk.
static constexpr int32_t maxMaskBytes = 3;

// dst = 8 * clamp(src, 0, 3), the shift in bits of a mask of whole bytes.
static void clampedBitShift(IRBuilder &builder, RegisterRef dst,
                            RegisterRef src) {
  builder.maxsi32(dst, 0, src);
  builder.minsi32(dst, maxMaskBytes, dst);
  builder.shli32(dst, dst, 3);
}

void skinny::emitTailGuard(IRBuilder &builder, RegisterRef tailBytes,
                           const std::function<void()> &body) {
  std::shared_ptr<rocisa::Label> end = builder.createLabel("tail_skip");
  builder.create<rocisa::SCmpEQU32>(tailBytes.container(), 0);
  builder.create<rocisa::SCBranchSCC1>(end->getLabelName());
  builder.insert(builder.createLabel("tail_begin"));
  body();
  builder.insert(end);
}

// The SGPR with the bytes of the chunk that the split keeps, end less the bytes
// of the chunks before it. Chunk 0 uses end itself. It is computed before the
// compares that read SCC.
static std::optional<RegisterRange>
chunkEnd(IRBuilder &builder, const TailLanes &lanes, int32_t chunk) {
  if (chunk == 0)
    return std::nullopt;
  RegisterRange chunkHi = builder.allocateSgpr(1);
  builder.subi32(chunkHi.ref(), lanes.end, chunk * lanes.chunkStride);
  return chunkHi;
}

// Loads the lanes that are inside or straddle the end and clears the registers
// of the others: the steering is -1 for the lanes with no byte below the end,
// rem <= 0.
static void loadEdgeChunk(IRBuilder &builder, TailPath &path,
                          const TailLanes &lanes, RegisterRef chunkHi,
                          int32_t chunk) {
  if (!path.loadsChunks())
    return;
  RegisterRange steer = builder.allocateVgpr(1);
  builder.subi32(steer.ref(), chunkHi, lanes.laneBytes);
  builder.addi32(steer.ref(), steer.ref(), -1);
  builder.shrsi32(steer.ref(), steer.ref(), 31);
  path.loadChunk(chunk, steer.ref());
}

static void zeroChunk(IRBuilder &builder, TailPath &path, int32_t chunk) {
  path.beforeZero(chunk);
  for (RegisterRef data : path.chunkData(chunk))
    for (int32_t dword = 0; dword < 4; ++dword)
      builder.movb32(data.slice(dword), 0);
}

// The first pass over chunk u: loads it (FULL, EDGE) or zeroes it (DEAD), and
// for an EDGE chunk sets its bit in edgeChunks, an SGPR, for the second pass.
static void emitChunkArm(IRBuilder &builder, TailPath &path,
                         RegisterRef edgeChunks, int32_t chunk) {
  TailLanes lanes = path.lanes();
  // All SALU first: s_sub writes SCC, which the branches read.
  std::optional<RegisterRange> chunkHiStorage = chunkEnd(builder, lanes, chunk);
  RegisterRef chunkHi = chunkHiStorage ? chunkHiStorage->ref() : lanes.end;
  std::shared_ptr<rocisa::Label> edge = builder.createLabel("tail_edge");
  std::shared_ptr<rocisa::Label> zero = builder.createLabel("tail_zero");
  std::shared_ptr<rocisa::Label> done = builder.createLabel("tail_chunk_end");
  builder.create<rocisa::SCmpGeI32>(chunkHi.container(), lanes.laneSpan);
  builder.create<rocisa::SCBranchSCC0>(edge->getLabelName());
  if (path.loadsChunks())
    path.loadChunk(chunk, std::nullopt);
  builder.create<rocisa::SBranch>(done->getLabelName());
  builder.insert(edge);
  builder.create<rocisa::SCmpGtI32>(chunkHi.container(), 0);
  builder.create<rocisa::SCBranchSCC0>(zero->getLabelName());
  loadEdgeChunk(builder, path, lanes, chunkHi, chunk);
  builder.ori32(edgeChunks, edgeChunks, 1 << chunk);
  builder.create<rocisa::SBranch>(done->getLabelName());
  builder.insert(zero);
  zeroChunk(builder, path, chunk);
  builder.insert(done);
}

// Clears the bytes of an EDGE chunk at or after the end, in the data of every
// operand: the straddling lane keeps the bytes of its 16 below the end, and the
// lanes with none are cleared whole. A dword keeps its first bytes below the
// end; the shift alone keeps one byte of a dword that starts at or past the
// end, so a second mask clears it. rem, keepAny and mask are the only
// registers.
static void maskEdgeChunk(IRBuilder &builder, TailPath &path,
                          RegisterRef chunkHi, int32_t chunk) {
  TailLanes lanes = path.lanes();
  RegisterRange rem = builder.allocateVgpr(1);
  RegisterRange keepAny = builder.allocateVgpr(1);
  RegisterRange mask = builder.allocateVgpr(1);
  builder.subi32(rem.ref(), chunkHi, lanes.laneBytes);
  for (int32_t dword = 0; dword < 4; ++dword) {
    builder.subi32(keepAny.ref(), 4 * dword, rem.ref());
    builder.shrsi32(keepAny.ref(), keepAny.ref(), 31);
    builder.subi32(mask.ref(), 4 * dword + 4, rem.ref());
    clampedBitShift(builder, mask.ref(), mask.ref());
    builder.shrui32(mask.ref(), -1, mask.ref());
    builder.andi32(mask.ref(), mask.ref(), keepAny.ref());
    for (RegisterRef data : path.chunkData(chunk))
      builder.andi32(data.slice(dword), data.slice(dword), mask.ref());
  }
}

// The second pass over chunk u: masks it when the first pass found it EDGE.
static void emitChunkMask(IRBuilder &builder, TailPath &path,
                          RegisterRef edgeChunks, int32_t chunk) {
  TailLanes lanes = path.lanes();
  std::optional<RegisterRange> chunkHiStorage = chunkEnd(builder, lanes, chunk);
  RegisterRef chunkHi = chunkHiStorage ? chunkHiStorage->ref() : lanes.end;
  std::shared_ptr<rocisa::Label> skip = builder.createLabel("tail_mask_skip");
  builder.create<rocisa::SBitcmp1B32>(edgeChunks.container(), chunk);
  builder.create<rocisa::SCBranchSCC0>(skip->getLabelName());
  maskEdgeChunk(builder, path, chunkHi, chunk);
  builder.insert(skip);
}

void skinny::emitTailChunks(IRBuilder &builder, TailPath &path) {
  RegisterRange edgeChunks = builder.allocateSgpr(1);
  builder.movb32(edgeChunks.ref(), 0);
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    emitChunkArm(builder, path, edgeChunks.ref(), chunk);
  // The loads of every chunk are in flight, and the masks need them.
  waitForVectorLoads(builder, 0);
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk)
    emitChunkMask(builder, path, edgeChunks.ref(), chunk);
}

void skinny::emitChunkLiveBytes(IRBuilder &builder, RegisterRef dst,
                                const TailLanes &lanes) {
  RegisterRange below = builder.allocateVgpr(1);
  RegisterRange live = builder.allocateVgpr(1);
  // The chunk u starts below the end when laneBytes + u * chunkStride - end is
  // negative.
  builder.subi32(below.ref(), lanes.laneBytes, lanes.end);
  builder.movb32(dst, 0);
  for (int32_t chunk = 0; chunk < SkinnyDims::chunks; ++chunk) {
    builder.addi32(live.ref(), below.ref(), chunk * lanes.chunkStride);
    builder.shrsi32(live.ref(), live.ref(), 31);
    builder.andi32(live.ref(), live.ref(), 0xFF);
    builder.shli32(live.ref(), live.ref(), 8 * chunk);
    builder.ori32(dst, dst, live.ref());
  }
}
