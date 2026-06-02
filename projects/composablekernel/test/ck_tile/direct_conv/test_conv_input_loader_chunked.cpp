// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit test for the v3 ConvInputLoader's chunked prefetch path
// (c_slices_per_wave > 1).
//
// Verifies that prefetch_tile_to_lds<CS>(buf) loads chunk CS of the current
// input row into the LDS buffer at slot `buf`, with each chunk holding the
// correct waves_per_wg * channels_per_group channels from DRAM at the C
// offset CS * (waves_per_wg * cpg).
//

#include "gtest/gtest.h"

#include "ck_tile/host/hip_check_error.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/kernel/impl/conv_32c_tile_impl_v3.hpp"
#pragma clang diagnostic pop

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace v3 = ck_tile::direct_conv::conv_32c_tile::v3;

// Config 44: M16N16K32 fp16 Fprop, waves_per_wg=2, c_slices_per_wave=2,
// SwizzleType::None. total_block_c = 128, per-chunk block_c = 64.
static constexpr int CFG_CSPW2_FPROP = 44;

// Test kernel: construct ConvInputLoader, then for input row `target_y`
// prefetch chunk 0 into LDS buffer 0 and chunk 1 into LDS buffer 1, then
// copy both LDS buffers out to global memory.
//
// To exercise the loader's row-advance state (which is critical: the bug
// only manifests after multiple row advances), we step the loader forward
// using `fetch_tile_to_lds<0>` once per row from 0..target_y, discarding
// the intermediate data into buffer 0, then issue the two final prefetches.
template <int CfgIdx>
__global__ void test_chunked_prefetch_kernel(const _Float16* __restrict__ in,
                                              _Float16* __restrict__ lds_out,
                                              int N,
                                              int C,
                                              int hi,
                                              int wi,
                                              int px,
                                              int target_y)
{
#ifdef __HIP_DEVICE_COMPILE__
    constexpr auto cfg = v3::KernelConfigurations<>::configs_map.get(CfgIdx);
    using TC          = v3::TileConstants<cfg>;
    using BlockCoords = v3::ConvBlockCoordsT<cfg>;
    using InputLoader = v3::ConvInputLoader<cfg>;

    static_assert(cfg.c_slices_per_wave == 2,
                  "test_chunked_prefetch_kernel is written for N=2");

    constexpr int BLOCK_SIZE = cfg.block_size();
    constexpr int LDS_BUF_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
    constexpr int LDS_TOTAL_FP16 = TC::NUM_INPUT_LDS_BUFFERS * LDS_BUF_FP16;
    constexpr int UNIFIED_UINT4 =
        TC::NUM_INPUT_LDS_BUFFERS * TC::INPUT_LDS_BUFFER_SIZE_C8;

    __shared__ uint4 lds_buf[UNIFIED_UINT4];

    // Sentinel-fill so any unwritten slot is visibly wrong if the prefetch
    // skips it.
    for(int i = threadIdx.x; i < UNIFIED_UINT4; i += BLOCK_SIZE)
    {
        lds_buf[i] = uint4{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu};
    }
    __syncthreads();

    // C_in is the kernel's "C" — full input channel count.
    BlockCoords bc(C, /*K_total=*/cfg.block_k_size());
    if(bc.block_n >= N)
        return;

    InputLoader il(bc, lds_buf, in, hi, wi, px, /*py=*/0,
                   /*dx=*/1, /*dy=*/1, /*sx=*/1, /*sy=*/1);

    // Initial prefetch: chunk 0 of row 0 into buffer 0.
    il.template prefetch_tile_to_lds<0>(0);

    // Advance through rows 1..target_y using fetch_tile_to_lds<0>.  Each call
    // advances the per-thread voffset and prefetches chunk 0 of the new row.
    // The data lands back in buffer 0; we don't care about it (we just need
    // the voffset to be at row `target_y`).
    for(int adv = 1; adv <= target_y; ++adv)
    {
        ck_tile::direct_conv::wait_vmcnt<0>();
        __syncthreads();
        il.template fetch_tile_to_lds<0>(0);
    }

    // At this point the loader is positioned at row `target_y`.  Now do the
    // two prefetches we actually want to inspect: chunk 0 of target_y into
    // buffer 0, chunk 1 of target_y into buffer 1.
    ck_tile::direct_conv::wait_vmcnt<0>();
    __syncthreads();
    il.template prefetch_tile_to_lds<0>(0);
    il.template prefetch_tile_to_lds<1>(1);
    ck_tile::direct_conv::wait_vmcnt<0>();
    __syncthreads();

    // Copy both LDS buffers to global memory for host verification.
    const _Float16* lds_fp16 = reinterpret_cast<const _Float16*>(lds_buf);
    for(int i = threadIdx.x; i < LDS_TOTAL_FP16; i += BLOCK_SIZE)
    {
        lds_out[i] = lds_fp16[i];
    }
#else
    (void)in; (void)lds_out; (void)N; (void)C; (void)hi; (void)wi; (void)px;
    (void)target_y;
#endif
}

class ConvInputLoaderChunkedTest : public ::testing::Test
{
protected:
    // Fill an NHWC input tensor with deterministic non-zero values so we can
    // catch zeros at positions that should be real data.
    static std::vector<_Float16> make_input(int N, int hi, int wi, int C)
    {
        const int total = N * hi * wi * C;
        std::vector<_Float16> v(total);
        for(int i = 0; i < total; ++i)
        {
            // Non-zero, in a value range that fits fp16 exactly.
            v[i] = static_cast<_Float16>(static_cast<float>((i % 251) + 1));
        }
        return v;
    }

    static float read_input(const std::vector<_Float16>& in,
                            int wi, int C,
                            int h, int w, int c)
    {
        return static_cast<float>(in[(h * wi + w) * C + c]);
    }

    // Run the kernel for cfg index, advance the loader to target_y, and
    // verify each LDS cell (chunk 0 in buf 0, chunk 1 in buf 1).
    template <int CfgIdx>
    void run_and_verify(int hi, int wi, int px, int target_y = 0)
    {
        constexpr auto cfg = v3::KernelConfigurations<>::configs_map.get(CfgIdx);
        using TC = v3::TileConstants<cfg>;

        constexpr int N_CSPW   = cfg.c_slices_per_wave;
        constexpr int BLOCK_W  = TC::BLOCK_W;
        constexpr int BLOCK_C8 = TC::BLOCK_C8;
        constexpr int LDS_BUF_FP16 = TC::INPUT_LDS_BUFFER_SIZE_FP16;
        constexpr int LDS_TOTAL_FP16 = TC::NUM_INPUT_LDS_BUFFERS * LDS_BUF_FP16;
        constexpr int BLOCK_C_PER_CHUNK = BLOCK_C8 * 8;   // = waves * cpg
        constexpr int TOTAL_BLOCK_C     = BLOCK_C_PER_CHUNK * N_CSPW;
        constexpr int BLOCK_SIZE        = cfg.block_size();

        // total_block_c is exactly the input C (single workgroup covers all C).
        const int C = TOTAL_BLOCK_C;
        const int N = 1;

        auto inp = make_input(N, hi, wi, C);

        _Float16 *d_in = nullptr, *d_lds = nullptr;
        ck_tile::hip_check_error(hipMalloc(&d_in,  inp.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_lds, LDS_TOTAL_FP16 * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_in, inp.data(), inp.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_chunked_prefetch_kernel<CfgIdx>
            <<<dim3(1, 1, 1), BLOCK_SIZE>>>(
                d_in, d_lds, N, C, hi, wi, px, target_y);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> lds_host(LDS_TOTAL_FP16);
        ck_tile::hip_check_error(hipMemcpy(
            lds_host.data(), d_lds, LDS_TOTAL_FP16 * sizeof(_Float16),
            hipMemcpyDeviceToHost));

        // Verify each (chunk, spatial, c8, c) cell.
        for(int cs = 0; cs < N_CSPW; ++cs)
        {
            for(int w = 0; w < BLOCK_W; ++w)
            {
                for(int c8 = 0; c8 < BLOCK_C8; ++c8)
                {
                    for(int c = 0; c < 8; ++c)
                    {
                        const int lds_idx = cs * LDS_BUF_FP16
                                          + (w * BLOCK_C8 + c8) * 8 + c;
                        const float actual = static_cast<float>(lds_host[lds_idx]);

                        // Chunk cs occupies absolute C-range
                        // [cs*BLOCK_C_PER_CHUNK, (cs+1)*BLOCK_C_PER_CHUNK).
                        const int abs_c   = cs * BLOCK_C_PER_CHUNK + c8 * 8 + c;
                        const int w_actual = w - px;  // block_q == 0 here

                        const float expected =
                            (w_actual >= 0 && w_actual < wi)
                                ? read_input(inp, wi, C, target_y, w_actual, abs_c)
                                : 0.0f;

                        EXPECT_EQ(actual, expected)
                            << "cfg=" << CfgIdx
                            << " target_y=" << target_y
                            << " cs=" << cs << " w=" << w
                            << " c8=" << c8 << " c=" << c
                            << " abs_c=" << abs_c
                            << " w_actual=" << w_actual;
                    }
                }
            }
        }

        ck_tile::hip_check_error(hipFree(d_in));
        ck_tile::hip_check_error(hipFree(d_lds));
    }
};

// hi/wi/px chosen to match the failing end-to-end shape:
// Pad1, hi=wi=8, C=128 (= 2 waves * 2 chunks * 32 cpg).
TEST_F(ConvInputLoaderChunkedTest, Cfg49_Pad1_Hi8_Wi8_Row0)
{
    run_and_verify<CFG_CSPW2_FPROP>(/*hi=*/8, /*wi=*/8, /*px=*/1, /*target_y=*/0);
}

// Row 7 (last input row) — this is where the end-to-end suite first sees
// the chunk-1 boundary bug, after the loader has advanced through 7 row
// strides.
TEST_F(ConvInputLoaderChunkedTest, Cfg49_Pad1_Hi8_Wi8_Row7)
{
    run_and_verify<CFG_CSPW2_FPROP>(/*hi=*/8, /*wi=*/8, /*px=*/1, /*target_y=*/7);
}

// Same shape without padding — exercises a different w_actual mapping.
TEST_F(ConvInputLoaderChunkedTest, Cfg49_NoPad_Hi8_Wi8_Row0)
{
    run_and_verify<CFG_CSPW2_FPROP>(/*hi=*/8, /*wi=*/8, /*px=*/0, /*target_y=*/0);
}

TEST_F(ConvInputLoaderChunkedTest, Cfg49_NoPad_Hi8_Wi8_Row7)
{
    run_and_verify<CFG_CSPW2_FPROP>(/*hi=*/8, /*wi=*/8, /*px=*/0, /*target_y=*/7);
}

// Slightly larger spatial — confirms the boundary issue isn't tied to the
// exact wi=8 case.
TEST_F(ConvInputLoaderChunkedTest, Cfg49_Pad1_Hi16_Wi16_Row0)
{
    run_and_verify<CFG_CSPW2_FPROP>(/*hi=*/16, /*wi=*/16, /*px=*/1, /*target_y=*/0);
}

TEST_F(ConvInputLoaderChunkedTest, Cfg49_Pad1_Hi16_Wi16_RowLast)
{
    run_and_verify<CFG_CSPW2_FPROP>(/*hi=*/16, /*wi=*/16, /*px=*/1, /*target_y=*/15);
}
