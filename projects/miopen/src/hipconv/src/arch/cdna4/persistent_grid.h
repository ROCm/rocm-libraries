#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

// Primitives for a persistent-grid launch on CDNA4 (gfx950 / MI355X).
//
// A persistent grid launches one workgroup per CU (PERSISTENT_GRID_SIZE total)
// and keeps each workgroup alive for the whole kernel, traversing the work
// assigned to its XCD. To do that a workgroup needs two facts: which XCD it owns
// (so it can pull XCD-local work) and a unique id within that XCD
// (0 .. NUM_CU_PER_XCD-1).
//
// Both are derived from blockIdx.x, NOT from the hardware XCC_ID register:
//   xcc_id       = blockIdx.x % NUM_XCD
//   workgroup_id = blockIdx.x / NUM_XCD
//
// The launch dispatches exactly PERSISTENT_GRID_SIZE = NUM_XCD * NUM_CU_PER_XCD
// blocks, and the runtime runs each blockIdx.x exactly once regardless of how
// the blocks are physically placed or serialized. So this mapping is an
// unconditional bijection onto the logical (xcc_id, workgroup_id) space the grid
// traversal partitions over: every (xcc_id, workgroup_id) pair is produced
// exactly once, so the work cover is exactly-once-correct even when a concurrent
// kernel on another stream occupies CUs and the blocks land unevenly across the
// physical XCDs.
//
// The earlier scheme read the physical XCC_ID register and claimed an XCD-local
// id with a per-XCD atomic counter. That is correct ONLY if every physical XCD
// receives exactly NUM_CU_PER_XCD of our blocks; under uneven placement an
// under-filled XCD's counter never reaches the ids whose X-slots hold real work,
// silently dropping it (the kernel's round loop trusts the id and has no in-loop
// validity break). The blockIdx mapping has no such dependence: physical
// placement now affects only locality-in-practice (whether two logically
// adjacent workgroups happen to share an L2), never coverage.

namespace persistent
{

// MI355X topology. Hardcoded for now; a device-specific table replaces these
// once we support other parts.
constexpr int NUM_XCD              = 8;
constexpr int NUM_CU_PER_XCD       = 32;
constexpr int PERSISTENT_GRID_SIZE = NUM_XCD * NUM_CU_PER_XCD; // 256

// This workgroup's owned XCD (0 .. NUM_XCD-1) and its unique id within that XCD
// (0 .. NUM_CU_PER_XCD-1), derived from blockIdx.x. Block-uniform (blockIdx.x is
// constant across the block), so callers may readfirstlane the results into
// SGPRs. See the file header for why this logical mapping is exactly-once-correct
// independent of physical XCD residency.
struct WorkgroupIndex
{
    int xcc_id;
    int workgroup_id;
};

__device__ inline WorkgroupIndex workgroup_index()
{
    const int bid = static_cast<int>(blockIdx.x);
    return {bid % NUM_XCD, bid / NUM_XCD};
}

} // namespace persistent
