#include <iomanip>
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "utils.hpp"
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

// Types
using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = col_major;
using DataLayoutB   = row_major;
using DataLayoutC   = col_major;
using DataLayoutLds = row_major;

/*
WAVE_SIZE for GFX11 is 32 -> 16x16x16 fragSize
WAVE_SIZE for CDNA  is 64 -> 32x32x16 fragSize
*/

constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
constexpr uint32_t WMMA_M    = WAVE_SIZE / 2u;
constexpr uint32_t WMMA_N    = WAVE_SIZE / 2u;
constexpr uint32_t WMMA_K    = 16u;
// WAVE tile: computed by each WAVE
constexpr uint32_t BLOCKS_X    = 2u;
constexpr uint32_t BLOCKS_Y    = 2u;
constexpr uint32_t WAVE_TILE_X = BLOCKS_X * WMMA_M;
constexpr uint32_t WAVE_TILE_Y = BLOCKS_Y * WMMA_N;

// Macro Tile: computed by each thread block (workgroup)
// Note: TBLOCK_X must be multiple of WAVE_SIZE.
constexpr uint32_t TBLOCK_X     = 128u;
constexpr uint32_t TBLOCK_Y     = 2u;
constexpr uint32_t WAVES_X      = TBLOCK_X / WAVE_SIZE;
constexpr uint32_t WAVES_Y      = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_X = WAVES_X * WAVE_TILE_X;
constexpr uint32_t MACRO_TILE_Y = WAVES_Y * WAVE_TILE_Y;
// For GFX11
// MACRO_TILE_X = (128 / 32) * 2 * 16 = 128
// MACRO_TILE_Y = 2 * 2 * 16 = 64
// MACRO_TILE_X = (128 / 64) * 2 * 32 = 128
// MACRO_TILE_Y = 2 * 2 * 32 = 128

/*
    Fragment Types
*/

//MFMA
using MfmaFragA   = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, InputT, DataLayoutA>;
using MfmaFragB   = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, InputT, DataLayoutB>;
using MfmaFragC   = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, OutputT, DataLayoutC>;
using MfmaFragD   = MfmaFragC;
using MfmaFragAcc = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, ComputeT>;

//Global read (MacroTile)
using GRBuffA = fragment<matrix_a, MACRO_TILE_X, WMMA_N, WMMA_K, InputT, DataLayoutA>;
using GRBuffB = fragment<matrix_b, WMMA_M, MACRO_TILE_Y, WMMA_K, InputT, DataLayoutB>;

/*
// Local write of global buffers (macro tile)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LWBuffA = ApplyDataLayout_t<GRBuffA, DataLayoutLds>;
using LWBuffB = ApplyDataLayout_t<ApplyTranspose_t<GRBuffB>, DataLayoutLds>;

// Local read (mfma frags)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LRFragA = ApplyDataLayout_t<MfmaFragA, DataLayoutLds>;
using LRFragB = ApplyDataLayout_t<ApplyTranspose_t<MfmaFragB>, DataLayoutLds>;
*/

__device__ inline void
    storePartials(float32_t* partials, int cta, const MfmaFragAcc fragAcc[BLOCKS_X][BLOCKS_Y])
{
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto xOffset = i * WAVE_TILE_X;
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            auto offset = (xOffset * MACRO_TILE_Y + j * WAVE_TILE_Y);
            for(int k = 0; k < fragAcc[i][j].num_elements; k++)
            {
                partials[cta * MACRO_TILE_X * MACRO_TILE_Y + (offset + k)] = fragAcc[i][j].x[k];
            }
        }
    }
}

//Reduce partial tiles. Note that a full fracAcc exists in *partials*, but it only has some of the K.
//The rest of the K lives in fragAcc.
__device__ inline void
    fixup(float32_t* partials, int cta, MfmaFragAcc (&fragAcc)[BLOCKS_X][BLOCKS_Y])
{
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto xOffset = i * WAVE_TILE_X;
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            auto offset = (xOffset * MACRO_TILE_Y + j * WAVE_TILE_Y);
            for(int k = 0; k < fragAcc[i][j].num_elements; k++)
            {
                fragAcc[i][j].x[k] += partials[cta * MACRO_TILE_X * MACRO_TILE_Y + (offset + k)];
            }
        }
    }
}

__device__ inline void signal(bool& flag)
{
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        flag = true;
    }
    __syncthreads();
}

__device__ inline void wait(bool& flag)
{
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        while(!flag)
        {
        }
    }
    __syncthreads();
}

__device__ inline void mfmaLoop(const MfmaFragA fragA[BLOCKS_X],
                                const MfmaFragB fragB[BLOCKS_Y],
                                MfmaFragAcc (&fragAcc)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            mma_sync(fragAcc[i][j], fragA[i], fragB[j], fragAcc[i][j]);
        }
    }
}

// Broadcast value to fragments in warp tile
template <typename FragT>
__device__ static inline void fill(FragT (&frags)[BLOCKS_X][BLOCKS_Y], GetDataType_t<FragT> value)
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            fill_fragment(frags[i][j], value);
        }
    }
}

__global__ void streamK(uint32_t         m,
                        uint32_t         n,
                        uint32_t         k,
                        float16_t const* a,
                        float16_t const* b,
                        float16_t const* c,
                        float16_t*       d,
                        uint32_t         lda,
                        uint32_t         ldb,
                        uint32_t         ldc,
                        uint32_t         ldd,
                        float16_t        alpha,
                        float16_t        beta,
                        bool*            flags,
                        float32_t*       partials)
{
    int itersPerTile = static_cast<int>(std::ceil(k / WMMA_K));
    int totalIters   = static_cast<int>(std::ceil(m / MACRO_TILE_X))
                     * static_cast<int>(std::ceil(n / MACRO_TILE_Y)) * itersPerTile;
    int itersPerCTA = totalIters / (gridDim.x);
    int cta         = blockIdx.x;
    int iter        = cta * itersPerCTA;
    int iterEnd     = iter + itersPerCTA;

    constexpr auto waveTileSize    = make_coord2d(WAVE_TILE_X, WAVE_TILE_Y);
    constexpr auto macroTileSize   = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y);
    constexpr auto waveDims        = make_coord2d(WAVES_X, WAVES_Y);
    auto           localWaveCoord  = make_coord2d(threadIdx.x / WAVE_SIZE, threadIdx.y);
    auto           localWaveOffset = localWaveCoord * waveTileSize;

    using Mapper1dA = GetDataLayout_t<MfmaFragA>;
    using Mapper1dB = GetDataLayout_t<MfmaFragB>;
    using FragShape = GetIOShape_t<MfmaFragD>;
    using Mapper1dD = GetDataLayout_t<MfmaFragD>;

    auto stepA
        = Mapper1dA::fromMatrixCoord(make_coord2d(GetIOShape_t<MfmaFragA>::BlockHeight, 0u), lda);
    auto stepB
        = Mapper1dB::fromMatrixCoord(make_coord2d(0u, GetIOShape_t<MfmaFragB>::BlockWidth), ldb);
    auto blockStepX = Mapper1dD::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldd);
    auto blockStepY = Mapper1dD::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldd);

    MfmaFragA   fragsA[BLOCKS_X];
    MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
    MfmaFragB   fragsB[BLOCKS_Y];
    MfmaFragD   fragsD[BLOCKS_X][BLOCKS_Y];
    while(iter < iterEnd)
    {
        int tileId       = iter / itersPerTile;
        int tileIter     = tileId * itersPerTile;
        int tileIterEnd  = tileIter + itersPerTile;
        int localIter    = iter - tileIter;
        int localIterEnd = min(iterEnd, tileIterEnd) - tileIter;

        auto xDim = (tileId / static_cast<int>(std::ceil(n / MACRO_TILE_Y))) * MACRO_TILE_X;
        auto yDim = (tileId % static_cast<int>(std::ceil(n / MACRO_TILE_Y))) * MACRO_TILE_Y;
        auto macroTileCoord = make_coord2d(xDim, yDim);
        auto waveTileCoord  = macroTileCoord + localWaveOffset;

        auto aAddr = Mapper1dA::fromMatrixCoord(make_coord2d(get<0>(waveTileCoord), 0u), lda);
        auto bAddr = Mapper1dB::fromMatrixCoord(make_coord2d(0u, get<1>(waveTileCoord)), ldb);

        //MFMA loop
        fill(fragsAcc, 0.0f);
        for(auto currentK = localIter; currentK < localIterEnd; currentK++)
        {
            auto kk      = currentK * WMMA_K;
            auto aOffset = aAddr;
            for(int i = 0; i < BLOCKS_X; i++)
            {
                load_matrix_sync(fragsA[i], a + aOffset + kk * lda, lda);
                aOffset += stepA;
            }

            auto bOffset = bAddr;
            for(int i = 0; i < BLOCKS_Y; i++)
            {
                load_matrix_sync(fragsB[i], b + bOffset + kk * ldb, ldb);
                bOffset += stepB;
            }
            mfmaLoop(fragsA, fragsB, fragsAcc);
        }
        bool tileStarted = iter == tileIter;
        bool tileEnded   = iterEnd >= tileIterEnd;

        if(!tileStarted)
        {
            storePartials(partials, cta, fragsAcc);
            signal(flags[cta]);
        }
        else
        {
            if(!tileEnded)
            {
                int ctaEnd = tileIterEnd / itersPerTile;
                for(int ctaNext = cta + 1; ctaNext <= ctaEnd / gridDim.x; ctaNext++)
                {
                    wait(flags[ctaNext]);
                    fixup(partials, ctaNext, fragsAcc);
                }
            }

            //Store Tile.
            auto dAddr = Mapper1dD::fromMatrixCoord(waveTileCoord, ldd);
            //#pragma unroll
            for(int i = 0; i < BLOCKS_X; i++)
            {
                auto offsetY = 0u;
                //#pragma unroll
                for(int j = 0; j < BLOCKS_Y; j++)
                {
                    for(int l = 0; l < fragsAcc[i][j].num_elements; l++)
                    {
                        fragsD[i][j].x[l] = static_cast<float16_t>(fragsAcc[i][j].x[l]);
                    }
                    store_matrix_sync(d + dAddr + offsetY, fragsD[i][j], ldd);
                    offsetY += blockStepY;
                }
                dAddr += blockStepX;
            }
        }
        iter = tileIterEnd;
    }
}

int main(int argc, char* argv[])
{
    bool test = false;
    if(argc > 1)
    {
        test = true;
    }

    const int              m = 3072;
    const int              n = 4096;
    const int              k = 4096;
    std::vector<float16_t> a(m * k); //, static_cast<float16_t>(2.0f));
    std::vector<float16_t> b(k * n); //, static_cast<float16_t>(1.0f));

    for(int i = 0; i < k; i++)
        for(int j = 0; j < m; j++)
        {
            a[i * m + j] = static_cast<float16_t>(static_cast<float>(i) / static_cast<float>(k));
        }

    for(int i = 0; i < k; i++)
        for(int j = 0; j < n; j++)
        {
            b[i * n + j] = static_cast<float16_t>(static_cast<float>(i) / static_cast<float>(n));
        }

    std::vector<float16_t> d(m * n, static_cast<float16_t>(0.f));

    std::vector<float16_t> dH(m * n, static_cast<float16_t>(0.f));

    if(test)
    {
        std::cout << "CPU MatMult" << std::endl;
        gemm_cpu_h<col_major, row_major, col_major>(
            m, n, k, a.data(), b.data(), dH.data(), dH.data(), 1.0f, 0.0f);
    }

    float16_t *dA, *dB, *dD;
    bool*      flags;
    float32_t* partials;

    dim3 dimBlock(TBLOCK_X, TBLOCK_Y);
    dim3 dimGrid(96);
    auto grid = dimGrid.x;

    CHECK_HIP_ERROR(hipMalloc(&dA, sizeof(float16_t) * a.size()));
    CHECK_HIP_ERROR(hipMalloc(&dB, sizeof(float16_t) * b.size()));
    CHECK_HIP_ERROR(hipMalloc(&dD, sizeof(float16_t) * d.size()));

    CHECK_HIP_ERROR(hipMemcpy(dA, a.data(), sizeof(float16_t) * a.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, b.data(), sizeof(float16_t) * b.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dD, d.data(), sizeof(float16_t) * d.size(), hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMalloc(&flags, sizeof(bool) * grid));
    CHECK_HIP_ERROR(hipMemset(flags, false, sizeof(bool) * grid));
    CHECK_HIP_ERROR(
        hipMalloc(&partials,
                  sizeof(float32_t) * grid * MACRO_TILE_X * MACRO_TILE_Y)); // use accumulator type.
    CHECK_HIP_ERROR(
        hipMemset(partials, 0.f, sizeof(float32_t) * grid * MACRO_TILE_X * MACRO_TILE_Y));

    std::cout << "Launching Kernel" << std::endl;
    streamK<<<dimGrid, dimBlock>>>(m,
                                   n,
                                   k,
                                   dA,
                                   dB,
                                   dD,
                                   dD,
                                   m,
                                   n,
                                   m,
                                   m,
                                   static_cast<float16_t>(1.0f),
                                   static_cast<float16_t>(0.0f),
                                   flags,
                                   partials);
    // Actual recorded runs
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    //warmup
    if(!test)
    {
        std::cout << "Benchmarking using 100 runs" << std::endl;
        for(int i = 0; i < 100; i++)
        {
            streamK<<<dimGrid, dimBlock>>>(m,
                                           n,
                                           k,
                                           dA,
                                           dB,
                                           dD,
                                           dD,
                                           m,
                                           n,
                                           m,
                                           m,
                                           static_cast<float16_t>(1.0f),
                                           static_cast<float16_t>(0.0f),
                                           flags,
                                           partials);
        }
        CHECK_HIP_ERROR(hipEventRecord(stopEvent));
        CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

        float elapsedTimeMs = 0.0f;
        CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
        std::cout << "Elapsed Time for Stream-K = " << elapsedTimeMs / 100 << " ms" << std::endl;
    }

    CHECK_HIP_ERROR(hipMemcpy(d.data(), dD, sizeof(float16_t) * d.size(), hipMemcpyDeviceToHost));
    if(test)
    {
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                float16_t diff = d[i * m + j] - static_cast<float16_t>(dH[i * m + j]);
                diff           = (diff > 0) ? diff : -diff;
                if(diff > 1.e-2)
                {
                    std::cout << "Wrong! Value = " << d[i * m + j]
                              << "\t Real Value = " << dH[i * m + j]
                              << "\tdiff =  " << static_cast<float>(diff) << std::endl;
                }
            }
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dD));
    CHECK_HIP_ERROR(hipFree(partials));
    CHECK_HIP_ERROR(hipFree(flags));
}
