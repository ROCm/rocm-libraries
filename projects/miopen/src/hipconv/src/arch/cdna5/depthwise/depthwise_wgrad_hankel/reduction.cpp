#include "reduction.hpp"
#include "arch/cdna5/tdm_desc.h"
#include <hip/hip_runtime.h>

namespace hipconv::cdna5
{

// TDM reduction of [partition][dW] partials in LDS-sized passes.
__global__ __launch_bounds__(REDUCE_COLS* REDUCE_WAVES) void conv2d_wgrad_reduce_tdm_cdna5(
    float* __restrict__ dW,
    const float* __restrict__ partials,
    int num_partitions,
    unsigned long long dW_total,
    int partitions_per_pass)
{
    // Discard the body on a GPU this architecture does not serve; see docs/architecture-guards.md.
    if(!(__builtin_amdgcn_is_invocable(__builtin_amdgcn_tensor_load_to_lds) &&
         __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_wait_tensorcnt)))
        return;

    // TDM offsets require the tile at LDS offset zero; use no static shared arrays.
    extern __shared__ float smem[];
    float* tile = smem;                                                         // [pass][COLS]
    float* part = smem + (unsigned long long)partitions_per_pass * REDUCE_COLS; // [WAVES][COLS]

    const int lane              = threadIdx.x & (REDUCE_COLS - 1);
    const int wave              = threadIdx.x / REDUCE_COLS;
    const unsigned long long e0 = (unsigned long long)blockIdx.x * REDUCE_COLS;
    const unsigned long long e  = e0 + lane;

    // The descriptor extent zero-fills columns past dW_total.
    const unsigned have = (unsigned)(dW_total - e0 < (unsigned long long)REDUCE_COLS
                                         ? dW_total - e0
                                         : (unsigned long long)REDUCE_COLS);
    bunnies::TdmDesc d{};
    d.init((unsigned)sizeof(float), have, (unsigned)REDUCE_COLS, dW_total);

    float s = 0.0f;
    for(int q0 = 0; q0 < num_partitions; q0 += partitions_per_pass)
    {
        const int rows = min(partitions_per_pass, num_partitions - q0);

        // Waves 0 and 1 use both TDM engines; each waits before the shared barrier.
        const int half = rows / 2;
        const int mine = wave == 0 ? half : rows - half;
        if(wave < 2 && mine > 0)
        {
            const int r0 = wave == 0 ? 0 : half;
            d.set_dim1((unsigned)mine, (unsigned)mine);
            d.load((unsigned long long)reinterpret_cast<uintptr_t>(
                       partials + (unsigned long long)(q0 + r0) * dW_total + e0),
                   (unsigned)(r0 * REDUCE_COLS * sizeof(float)));
            __builtin_amdgcn_s_wait_tensorcnt(0);
        }
        __syncthreads();

        for(int q = wave; q < rows; q += REDUCE_WAVES)
            s += tile[(unsigned long long)q * REDUCE_COLS + lane];

        // Finish all reads before the next pass overwrites the tile.
        if(q0 + partitions_per_pass < num_partitions)
            __syncthreads();
    }
    part[wave * REDUCE_COLS + lane] = s;
    __syncthreads();
    if(wave == 0 && e < dW_total)
    {
        float t = 0.0f;
#pragma unroll
        for(int w = 0; w < REDUCE_WAVES; ++w)
            t += part[w * REDUCE_COLS + lane];
        dW[e] = t;
    }
}

} // namespace hipconv::cdna5
