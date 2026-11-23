#pragma once

#include <hip/hip_runtime.h>

namespace hipconv::cdna4::direct_l1
{

// First image and chunk of an unfold_chunks slot group.
struct ChunkGroup
{
    int image0;
    int chunk0;
};

__host__ __device__ inline ChunkGroup chunk_group(int group, int slots, int chunks_per_row)
{
    const int s0     = group * slots;
    const int image0 = s0 / chunks_per_row;
    return {image0, s0 - image0 * chunks_per_row};
}

// A slot's image, relative to its group's image0, and chunk within that image's row.
struct SlotChunk
{
    int image;
    int chunk;
};

// The image and chunk of slot `slot` of a group.
//
// At two or more chunks per row a group spans at most two images, so a conditional
// subtraction replaces the division.
__host__ __device__ inline SlotChunk slot_chunk(int chunk0, int slot, int chunks_per_row)
{
    const int t    = chunk0 + slot;
    const int next = t >= chunks_per_row ? 1 : 0;
    return {next, t - next * chunks_per_row};
}

} // namespace hipconv::cdna4::direct_l1
