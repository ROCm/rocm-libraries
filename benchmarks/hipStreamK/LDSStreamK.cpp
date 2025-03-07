/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2023-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cassert>
#include <iostream>
#include <vector>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "utils.hpp"

/**
 * This is a HIP Prototype for the Basic Stream-K Algorithm outlined in
 * Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU
 * by Osama et Al.
 *
 * This prototype assumes that the K dimension of the problem is a multiple of WMMA_K, defined below.
 * The executable takes in 6 optional arguments for Matrix Multiplication D = A * B where A is MxK and B is KxN
 * M: the first dimension of A //Default 3072
 * N: the second dimension of B //Defualt 4096
 * K: the accumulating dimension //Default 4096
 * numCUs: the grid size for the Stream-K kernel
 * test: to test the output or benchmark takes 0 (false) or 1 (true) //Default 0
 * numBench: number of benchmark runs to choose, note does not use if test == 1 //Default 10
 *
 * E.g. To run streamk 100 times for benchmark using the problem 3072 x 4096 x 4096 utilizing 120 CUs (workgroups)
 * ./streamk.exe 3072 4096 4096 120 0 100
 */

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
//*/

__device__ static inline void
    globalReadC(MfmaFragC (&fragC)[BLOCKS_X][BLOCKS_Y], OutputT const* gAddrC, uint32_t ldc)
{
    using FragShape = GetIOShape_t<MfmaFragC>;
    using Mapper1d  = GetDataLayout_t<MfmaFragC>;

    // Iterative offsets for each C block in the wave tile
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldc);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldc);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            load_matrix_sync(fragC[i][j], gAddrC + offsetY, ldc);
            offsetY += blockStepY;
        }
        gAddrC += blockStepX;
    }
}

// Global D reads for warp tile gemm, non-cooperative
__device__ static inline void
    globalWriteD(OutputT* gAddrD, MfmaFragD const (&fragsD)[BLOCKS_X][BLOCKS_Y], uint32_t ldd)
{
    using FragShape = GetIOShape_t<MfmaFragD>;
    using Mapper1d  = GetDataLayout_t<MfmaFragD>;

    // Iterative offsets for each D block in the warp tile
    auto blockStepX = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), ldd);
    auto blockStepY = Mapper1d::fromMatrixCoord(make_coord2d(0u, FragShape::BlockWidth), ldd);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto offsetY = 0u;
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            store_matrix_sync(gAddrD + offsetY, fragsD[i][j], ldd);
            offsetY += blockStepY;
        }
        gAddrD += blockStepX;
    }
}

__device__ inline void
    storePartials(ComputeT* partials, int cta, const MfmaFragAcc fragAcc[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto xOffset = i * WAVE_TILE_X;
#pragma unroll
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
__device__ static inline void
    fixup(ComputeT* partials, int cta, MfmaFragAcc (&fragAcc)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        auto xOffset = i * WAVE_TILE_X;
#pragma unroll
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
    synchronize_workgroup();
}

__device__ inline void wait(bool& flag)
{
    if(threadIdx.x == 0 && threadIdx.y == 0)
    {
        while(!flag)
        {
        }
    }
    synchronize_workgroup();
}

__device__ static inline void mfmaLoop(const MfmaFragA fragA[BLOCKS_X],
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

__device__ static inline void fma(MfmaFragD (&fragsD)[BLOCKS_X][BLOCKS_Y],
                                  ComputeT alpha,
                                  MfmaFragAcc const (&fragsAcc)[BLOCKS_X][BLOCKS_Y],
                                  ComputeT beta,
                                  MfmaFragC const (&fragsC)[BLOCKS_X][BLOCKS_Y])
{
#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
#pragma unroll
        for(int j = 0; j < BLOCKS_Y; j++)
        {
            for(int k = 0; k < fragsD[i][j].num_elements; k++)
            {
                // Perform computation in ComputeT and cast back to OutputT
                fragsD[i][j].x[k] = static_cast<OutputT>(
                    alpha * fragsAcc[i][j].x[k] + beta * static_cast<ComputeT>(fragsC[i][j].x[k]));
            }
        }
    }
}

template <uint32_t WaveCountA, uint32_t SplitCountA>
__device__ static inline void
    globalReadCoopA(GRBuffA& grBuffA, InputT const* gAddrA, uint32_t lda, uint32_t waveIndexA)
{
    load_matrix_coop_sync<WaveCountA, SplitCountA>(grBuffA, gAddrA, lda, waveIndexA);
}

// Global B reads in cooperative mode (macro tile)
template <uint32_t WaveCountB, uint32_t SplitCountB>
__device__ static inline void
    globalReadCoopB(GRBuffB& grBuffB, InputT const* gAddrB, uint32_t ldb, uint32_t waveIndexB)
{
    load_matrix_coop_sync<WaveCountB, SplitCountB>(grBuffB, gAddrB, ldb, waveIndexB);
}

// Local A writes in cooperative mode (macro tile)
template <uint32_t WaveCountA, uint32_t SplitCountA>
__device__ static inline void
    localWriteCoopA(InputT* ldsAddr, GRBuffA const& grBuffA, uint32_t WMMA_K, uint32_t waveIndexA)
{
    // No transpose, but apply the lds data layout
    store_matrix_coop_sync<WaveCountA, SplitCountA>(
        ldsAddr, applyDataLayout<DataLayoutLds>(grBuffA), WMMA_K, waveIndexA);
}

// Local B writes in cooperative mode (macro tile)
template <uint32_t WaveCountB, uint32_t SplitCountB>
__device__ static inline void
    localWriteCoopB(InputT* ldsAddr, GRBuffB const& grBuffB, uint32_t WMMA_K, uint32_t waveIndexB)
{
    // Transpose B and then apply lds data layout
    store_matrix_coop_sync<WaveCountB, SplitCountB>(
        ldsAddr, applyDataLayout<DataLayoutLds>(applyTranspose(grBuffB)), WMMA_K, waveIndexB);
}

// Local A reads for warp tile gemm, non-cooperative
__device__ static inline void
    localReadA(MfmaFragA (&fragsA)[BLOCKS_X], InputT const* ldsAddrA, uint32_t WMMA_K)
{
    using FragShape = GetIOShape_t<LRFragA>;
    using Mapper1d  = typename FragShape::DataLayout;

    // Each A block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), WMMA_K);

#pragma unroll
    for(int i = 0; i < BLOCKS_X; i++)
    {
        LRFragA tmp;
        load_matrix_sync(tmp, ldsAddrA, WMMA_K);
        fragsA[i] = applyDataLayout<DataLayoutA>(tmp);

        ldsAddrA += blockStep;
    }
}

// Local B reads for warp tile gemm, non-cooperative
__device__ static inline void
    localReadB(MfmaFragB (&fragsB)[BLOCKS_Y], InputT const* ldsAddrB, uint32_t WMMA_K)
{
    using FragShape = GetIOShape_t<LRFragB>;
    using Mapper1d  = GetDataLayout_t<LRFragB>;

    // Each B block is stacked vertically in LDS
    auto blockStep = Mapper1d::fromMatrixCoord(make_coord2d(FragShape::BlockHeight, 0u), WMMA_K);

#pragma unroll
    for(int i = 0; i < BLOCKS_Y; i++)
    {
        LRFragB tmp;
        load_matrix_sync(tmp, ldsAddrB, WMMA_K);

        // Transform back to MFMA tile
        fragsB[i] = applyDataLayout<DataLayoutB>(applyTranspose(tmp));

        ldsAddrB += blockStep;
    }
}

__device__ static inline auto macroTileCoord(const int tileId, const uint32_t n)
{
    return make_coord2d(
        (tileId
         / (static_cast<int>(std::ceil(static_cast<float>(n) / static_cast<float>(MACRO_TILE_Y)))))
            * MACRO_TILE_X,
        (tileId
         % (static_cast<int>(std::ceil(static_cast<float>(n) / static_cast<float>(MACRO_TILE_Y)))))
            * MACRO_TILE_Y);
}

__device__ static inline auto localWaveOffset()
{
    return make_coord2d((threadIdx.x / WAVE_SIZE) * WAVE_TILE_X, threadIdx.y * WAVE_TILE_Y);
}

__device__ static inline int waveIndex()
{
    return (threadIdx.x / WAVE_SIZE) * WAVES_Y + threadIdx.y;
}

__device__ static inline int totalIters(const uint32_t m, const uint32_t n, const uint32_t k)
{
    return ((k + WMMA_K - 1) / WMMA_K) * ((m + MACRO_TILE_X - 1) / MACRO_TILE_X)
           * ((n + MACRO_TILE_Y - 1) / MACRO_TILE_Y);
}

__device__ static inline int itersPerTile(const uint32_t k)
{
    return ((k + WMMA_K - 1) / WMMA_K);
}

__global__ void __launch_bounds__(256) streamK(uint32_t       m,
                                               uint32_t       n,
                                               uint32_t       k,
                                               InputT const*  a,
                                               InputT const*  b,
                                               OutputT const* c,
                                               OutputT*       d,
                                               uint32_t       lda,
                                               uint32_t       ldb,
                                               uint32_t       ldc,
                                               uint32_t       ldd,
                                               ComputeT       alpha,
                                               ComputeT       beta,
                                               bool*          flags,
                                               ComputeT*      partials)
{
    extern __shared__ InputT localMemPtr[];

    int itersPerCTA = totalIters(m, n, k) / (gridDim.x);
    int iter        = blockIdx.x * itersPerCTA;
    int iterEnd     = iter + itersPerCTA;

    int extraIters = totalIters(m, n, k) % (gridDim.x);
    if(extraIters != 0)
    {
        if(blockIdx.x < extraIters)
        {
            iter    = blockIdx.x * (itersPerCTA + 1);
            iterEnd = iter + itersPerCTA + 1;
        }
        else
        {
            iter    = blockIdx.x * itersPerCTA + extraIters;
            iterEnd = iter + itersPerCTA;
        }
    }

    using GRBuffAMap1d = GetDataLayout_t<GRBuffA>;
    using GRBuffBMap1d = GetDataLayout_t<GRBuffB>;
    using LWBuffAShape = GetIOShape_t<LWBuffA>;
    using LWBuffBShape = GetIOShape_t<LWBuffB>;
    using LWBuffAMap1d = GetDataLayout_t<LWBuffA>;
    using LWBuffBMap1d = GetDataLayout_t<LWBuffB>;
    using Mapper1dC    = GetDataLayout_t<MfmaFragC>;
    using Mapper1dD    = GetDataLayout_t<MfmaFragD>;

    GRBuffA grBuffA;
    GRBuffB grBuffB;

    constexpr auto splitCountA = std::min(static_cast<uint32_t>(GetIOTraits_t<GRBuffA>::IOCount),
                                          static_cast<uint32_t>(GetIOTraits_t<LWBuffA>::IOCount));
    constexpr auto splitCountB = std::min(static_cast<uint32_t>(GetIOTraits_t<GRBuffB>::IOCount),
                                          static_cast<uint32_t>(GetIOTraits_t<LWBuffB>::IOCount));

    MfmaFragA   fragsA[BLOCKS_X];
    MfmaFragAcc fragsAcc[BLOCKS_X][BLOCKS_Y];
    MfmaFragB   fragsB[BLOCKS_Y];

    //Setup LDS.
    auto* ldsPtrLo = localMemPtr;
    auto* ldsPtrHi = localMemPtr + (LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight) * WMMA_K;

    auto ldsWriteOffsetB
        = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), WMMA_K);
    auto ldsReadOffsetA = LWBuffAMap1d::fromMatrixCoord(
        make_coord2d(threadIdx.x / WAVE_SIZE * WAVE_TILE_X, 0u), WMMA_K); //LDSWRITE OFFSET is 0
    auto ldsReadOffsetB
        = ldsWriteOffsetB
          + LWBuffBMap1d::fromMatrixCoord(make_coord2d(threadIdx.y * WAVE_TILE_Y, 0u), WMMA_K);

    while(true)
    {
        int tileId       = iter / itersPerTile(k);
        int tileIter     = tileId * itersPerTile(k);
        int tileIterEnd  = tileIter + itersPerTile(k);
        int localIter    = iter - tileIter;
        int localIterEnd = min(iterEnd, tileIterEnd) - tileIter;

        auto globalReadOffsetA = GRBuffAMap1d::fromMatrixCoord(
            make_coord2d(get<0>(macroTileCoord(tileId, n)), 0u), lda);
        auto globalReadOffsetB = GRBuffBMap1d::fromMatrixCoord(
            make_coord2d(0u, get<1>(macroTileCoord(tileId, n))), ldb);

        //Initial read of global memory
        globalReadCoopA<WAVES_X * WAVES_Y, splitCountA>(
            grBuffA, a + globalReadOffsetA + localIter * WMMA_K * lda, lda, waveIndex());
        globalReadCoopB<WAVES_X * WAVES_Y, splitCountB>(
            grBuffB, b + globalReadOffsetB + localIter * WMMA_K * ldb, ldb, waveIndex());

        ///Write global prefetch to LDS
        localWriteCoopA<WAVES_X * WAVES_Y, splitCountA>(ldsPtrLo, grBuffA, WMMA_K, waveIndex());
        localWriteCoopB<WAVES_X * WAVES_Y, splitCountB>(
            ldsPtrLo + ldsWriteOffsetB, grBuffB, WMMA_K, waveIndex());
        //MFMA loop
        fill(fragsAcc, 0.0f);
        synchronize_workgroup();

        for(auto currentK = localIter; currentK < localIterEnd - 1; currentK++)
        {
            //local Read frags from first LDS Buffer
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, WMMA_K);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, WMMA_K);

            // Prefetch next round of global
            globalReadCoopA<WAVES_X * WAVES_Y, splitCountA>(
                grBuffA, a + globalReadOffsetA + (currentK + 1) * WMMA_K * lda, lda, waveIndex());
            globalReadCoopB<WAVES_X * WAVES_Y, splitCountB>(
                grBuffB, b + globalReadOffsetB + (currentK + 1) * WMMA_K * ldb, ldb, waveIndex());

            mfmaLoop(fragsA, fragsB, fragsAcc);

            localWriteCoopA<WAVES_X * WAVES_Y, splitCountA>(ldsPtrHi, grBuffA, WMMA_K, waveIndex());
            localWriteCoopB<WAVES_X * WAVES_Y, splitCountB>(
                ldsPtrHi + ldsWriteOffsetB, grBuffB, WMMA_K, waveIndex());

            synchronize_workgroup();

            //Swap Buffers!
            auto* tmp = ldsPtrLo;
            ldsPtrLo  = ldsPtrHi;
            ldsPtrHi  = tmp;
        }

        //Last MFMA
        // Local read mfma frags
        localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, WMMA_K);
        localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, WMMA_K);
        mfmaLoop(fragsA, fragsB, fragsAcc);

        if(iter != tileIter)
        {
            storePartials(partials, blockIdx.x, fragsAcc);
            signal(flags[blockIdx.x]);
        }
        else
        {
            if(iterEnd < tileIterEnd)
            {
                int ctaNext = blockIdx.x + 1;
                for(int end = tileIter; end <= tileIterEnd; end += itersPerCTA)
                {
                    wait(flags[ctaNext]);
                    fixup(partials, ctaNext, fragsAcc);
                    ctaNext++;
                }
            }

            auto      waveTileCoord = macroTileCoord(tileId, n) + localWaveOffset();
            MfmaFragC fragsC[BLOCKS_X][BLOCKS_Y];
            MfmaFragD fragsD[BLOCKS_X][BLOCKS_Y];
            globalReadC(fragsC, c + Mapper1dC::fromMatrixCoord(waveTileCoord, ldc), ldc);
            fma(fragsD, alpha, fragsAcc, beta, fragsC);
            globalWriteD(d + Mapper1dD::fromMatrixCoord(waveTileCoord, ldd), fragsD, ldd);
        }
        iter = tileIterEnd;
        if(iter >= iterEnd)
            break;
    }
}

int main(int argc, char* argv[])
{
    bool     test     = false;
    int      m        = 7680;
    int      n        = 8448;
    int      k        = 8192;
    int      numBench = 5;
    ComputeT alpha    = 1.f;
    ComputeT beta     = 1.f;

    hipDeviceProp_t devProp;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&devProp, 0));
    int  grid     = devProp.multiProcessorCount;
    auto waveSize = devProp.warpSize;
    auto macroTileSize
        = rocwmma::make_coord2d(TBLOCK_X / waveSize * WAVE_TILE_X, TBLOCK_Y * WAVE_TILE_Y);
    if(argc > 1)
    {
        if(strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        {
            std::cout << "usage: streamK.exe [m=" << m << "] [n=" << n << "] [k=" << k
                      << "] [grid=" << grid << "] [test=0] [numBench=" << numBench << "]"
                      << std::endl;
            return 0;
        }
        m = std::stoi(argv[1]);
        if(argc > 2)
            n = std::stoi(argv[2]);
        if(argc > 3)
            k = std::stoi(argv[3]);
        if(argc > 4)
            grid = std::stoi(argv[4]);
        if(argc > 5)
        {
            auto temp = std::stoi(argv[5]);
            test      = (temp == 1) ? true : false;
        }
        if(argc > 6 && !test)
            numBench = std::stoi(argv[6]);
    }

    auto testMode = (test) ? "Test" : "Benchmark with " + std::to_string(numBench) + " runs";

    std::cout << "Running StreamK with the following configuration:\n"
              << "\t M = " << m << "\n\t N = " << n << "\n\t K = " << k << "\n\t with " << grid
              << " CUs"
              << "\n\t Test Mode: " << testMode << std::endl;

    assert(("k must be aligned to WMMA_K" && k % WMMA_K == 0));
    std::vector<InputT> a(m * k);
    std::vector<InputT> b(k * n);

    for(int i = 0; i < k; i++)
        for(int j = 0; j < m; j++)
        {
            a[i * m + j] = static_cast<InputT>(static_cast<float>(i) / static_cast<float>(k));
        }

    for(int i = 0; i < k; i++)
        for(int j = 0; j < n; j++)
        {
            b[i * n + j] = static_cast<InputT>(static_cast<float>(i) / static_cast<float>(n));
        }

    std::vector<OutputT> c(m * n, static_cast<OutputT>(1.f));
    std::vector<OutputT> d(m * n, static_cast<OutputT>(0.f));

    std::vector<OutputT> dH(m * n, static_cast<OutputT>(0.f));

    if(test)
    {
        std::cout << "Starting CPU MatMult" << std::endl;
        gemm_cpu_h<col_major, row_major, col_major>(
            m, n, k, a.data(), b.data(), c.data(), dH.data(), alpha, beta);
    }

    InputT *  dA, *dB;
    OutputT * dC, *dD;
    bool*     flags;
    ComputeT* partials;

    dim3 dimBlock(TBLOCK_X, TBLOCK_Y);
    dim3 dimGrid(grid);

    CHECK_HIP_ERROR(hipMalloc(&dA, sizeof(InputT) * a.size()));
    CHECK_HIP_ERROR(hipMalloc(&dB, sizeof(InputT) * b.size()));
    CHECK_HIP_ERROR(hipMalloc(&dC, sizeof(OutputT) * c.size()));
    CHECK_HIP_ERROR(hipMalloc(&dD, sizeof(OutputT) * d.size()));

    CHECK_HIP_ERROR(hipMemcpy(dA, a.data(), sizeof(InputT) * a.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, b.data(), sizeof(InputT) * b.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, c.data(), sizeof(OutputT) * c.size(), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dD, d.data(), sizeof(OutputT) * d.size(), hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(hipMalloc(&flags, sizeof(bool) * grid));
    CHECK_HIP_ERROR(hipMemset(flags, false, sizeof(bool) * grid));
    CHECK_HIP_ERROR(
        hipMalloc(&partials,
                  sizeof(ComputeT) * grid * MACRO_TILE_X * MACRO_TILE_Y)); // use accumulator type.
    CHECK_HIP_ERROR(
        hipMemset(partials, 0.f, sizeof(ComputeT) * grid * MACRO_TILE_X * MACRO_TILE_Y));

    // Works as a warmup if benchmarking, or just the test results if test is true.
    std::cout << "Launching Kernel" << std::endl;
    uint32_t ldsSize
        = 2u * sizeof(InputT) * (get<0>(macroTileSize) + get<1>(macroTileSize)) * WMMA_K;

    auto streamKKernel = [&]() {
        hipExtLaunchKernelGGL(streamK,
                              dimGrid,
                              dimBlock,
                              ldsSize,
                              0,
                              nullptr,
                              nullptr,
                              0,
                              m,
                              n,
                              k,
                              dA,
                              dB,
                              dC,
                              dD,
                              m,
                              n,
                              m,
                              m,
                              alpha,
                              beta,
                              flags,
                              partials);
    };

    auto numWarmups = (!test) ? 2u : 1u;
    for(uint32_t i = 0; i < numWarmups; i++)
        streamKKernel();
    if(!test)
    {
        std::cout << "Benchmarking Kernel" << std::endl;
        hipEvent_t startEvent, stopEvent;
        CHECK_HIP_ERROR(hipEventCreate(&startEvent));
        CHECK_HIP_ERROR(hipEventCreate(&stopEvent));
        CHECK_HIP_ERROR(hipEventRecord(startEvent));
        float elapsedTimeMs = 0.0f;
        for(int i = 0; i < numBench; i++)
        {
            streamKKernel();
        }
        CHECK_HIP_ERROR(hipEventRecord(stopEvent));
        CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));
        CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
        std::cout << "Elapsed Time for Stream-K = " << elapsedTimeMs / numBench << " ms"
                  << std::endl;
    }

    CHECK_HIP_ERROR(hipMemcpy(d.data(), dD, sizeof(InputT) * d.size(), hipMemcpyDeviceToHost));
    if(test)
    {
        std::cout << "Comparing StreamK and CPU results." << std::endl;
        int starti = 0;
        int startj = 0;
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                InputT diff = d[i * m + j] - static_cast<InputT>(dH[i * m + j]);
                diff        = (diff > 0) ? diff : -diff;
                if(diff > 1.e-2)
                {
                    starti = i;
                    startj = j;
                    std::cout << "Diff found at " << i << "," << j << std::endl;
                    break;
                }
            }
            if(starti != 0 || startj != 0)
                break;
        }
        if(starti != 0 || startj != 0)
        {
            for(int i = starti; i < starti + 20; i++)
            {
                for(int j = startj; j < startj + 20; j++)
                {
                    std::cout << d[i * m + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "===========" << std::endl;
            for(int i = starti; i < starti + 20; i++)
            {
                for(int j = startj; j < startj + 20; j++)
                {
                    std::cout << dH[i * m + j] << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            std::cout << "No diffs found in results." << std::endl;
        }
    }

    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    CHECK_HIP_ERROR(hipFree(dD));
    CHECK_HIP_ERROR(hipFree(partials));
    CHECK_HIP_ERROR(hipFree(flags));
}
