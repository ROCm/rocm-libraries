#pragma once

namespace hipconv::cdna5
{

// The engine-fed reduce for the [partition][dW] layout. A block takes REDUCE_COLS adjacent
// dW elements across a run of partitions as one 2D tile, so its input arrives on TDM issues
// rather than on a strided load a lane. See the definition for why the columns are 32, and
// for what a pass costs.
constexpr int REDUCE_COLS  = 32;
constexpr int REDUCE_WAVES = 32;

// The partitions a pass may stage, given what LDS a block can have: the tile takes
// REDUCE_COLS floats a partition and the cross-wave sums take a row of their own. One pass
// is the fast case, so the caller hands over as deep a tile as fits and the kernel loops
// only when the partitions outrun it.
inline int reduce_partitions_per_pass(int num_partitions, size_t lds_budget_bytes)
{
    const size_t row  = (size_t)REDUCE_COLS * sizeof(float);
    const size_t keep = (size_t)REDUCE_WAVES * row; // the part[] rows, outside the tile
    const size_t fits = lds_budget_bytes > keep ? (lds_budget_bytes - keep) / row : 0;
    if(fits == 0)
        return 0;
    return (int)(fits < (size_t)num_partitions ? fits : (size_t)num_partitions);
}

inline size_t reduce_tdm_lds_bytes(int partitions_per_pass)
{
    return ((size_t)partitions_per_pass * REDUCE_COLS + (size_t)REDUCE_WAVES * REDUCE_COLS) *
           sizeof(float);
}

__global__ void conv2d_wgrad_reduce_tdm_cdna5(float* __restrict__ dW,
                                              const float* __restrict__ partials,
                                              int num_partitions,
                                              unsigned long long dW_total,
                                              int partitions_per_pass);

} // namespace hipconv::cdna5
