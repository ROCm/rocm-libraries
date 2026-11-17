#pragma once

#include <hip/hip_runtime.h>

// Primitives for a persistent-grid launch on CDNA4 (gfx950 / MI355X).
//
// A persistent grid launches one workgroup per CU (PERSISTENT_GRID_SIZE total)
// and keeps each alive for the whole kernel. Each needs a partition of the work
// and a unique id within it, both derived from blockIdx.x rather than from the
// hardware XCC_ID register:
//
//   xcc_id       = blockIdx.x % NUM_XCD
//   workgroup_id = blockIdx.x / NUM_XCD
//
// The runtime runs each blockIdx.x exactly once, so this is a bijection onto the
// NUM_XCD * NUM_CU_PER_XCD space the traversal partitions over. Work coverage
// rests on that alone and holds however the blocks land.
//
// The locality is an assumption, not a guarantee. Calling the partition xcc_id
// assumes the runtime hands consecutive blocks to XCDs round-robin, putting a
// partition's workgroups on one XCD and behind one L2. Nothing here checks that,
// and no hardware or runtime documentation promises it. Where the assumption
// fails, a partition spreads across XCDs and the cost is L2 hit rate, never
// correctness.

namespace persistent
{

// MI355X topology, hardcoded until a device-specific table replaces it.
constexpr int NUM_XCD              = 8;
constexpr int NUM_CU_PER_XCD       = 32;
constexpr int PERSISTENT_GRID_SIZE = NUM_XCD * NUM_CU_PER_XCD; // 256

// This workgroup's partition (0 .. NUM_XCD-1) and its id within it.
//
// Block-uniform, so callers may readfirstlane both into SGPRs. xcc_id is only
// expected to match the physical XCD; see the file header.
struct WorkgroupIndex
{
    int xcc_id;
    int workgroup_id;
};

// Decode this workgroup's partition and id from blockIdx.x.
__device__ inline WorkgroupIndex workgroup_index()
{
    const int bid = static_cast<int>(blockIdx.x);
    return {bid % NUM_XCD, bid / NUM_XCD};
}

} // namespace persistent
