// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit test for the v3 WeightLoader's chunked register cache
// (c_slices_per_wave > 1).
//
// Verifies that read_from_lds_chunk<CS> populates the per-thread
// weights[(R * KW + S) * N + CS] slot with the correct fp16x8 from DRAM
// for the wave's CS-th C-section.

#include "gtest/gtest.h"

#include "ck_tile/host/hip_check_error.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wshadow"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/direct_convolution/kernel/impl/conv_32c_tile_impl_v3.hpp"
#include "ck_tile/ops/direct_convolution/configs/direct_conv_32c_dense_configs.hpp"
#pragma clang diagnostic pop

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace v3 = ck_tile::direct_conv::conv_32c_tile::v3;

// Config 44: M16N16K32 fp16 Fprop, waves_per_wg=2, c_slices_per_wave=2.
static constexpr int CFG_CSPW2_FPROP = 44;

// Test kernel: run the v3 weight prologue for both chunks, then dump every
// thread's full weights[KH*KW*N] array (one VecType per slot) into a
// global buffer for host inspection.
//
// Layout of `weights_out` (in fp16 elements):
//   weights_out[thread_id * (KH * KW * N) * 8 + slot * 8 + j]
// where slot = F * N + CS, F = R*KW + S.
template <int CfgIdx>
__global__ void test_chunked_weight_kernel(const _Float16* __restrict__ wei,
                                            _Float16* __restrict__ weights_out,
                                            int C_total)
{
#ifdef __HIP_DEVICE_COMPILE__
    constexpr auto cfg = v3::KernelConfigurations<>::configs_map.get(CfgIdx);
    using TC = v3::TileConstants<cfg>;
    using WeightLoader = v3::WeightLoader<cfg>;

    constexpr int N_CSPW     = cfg.c_slices_per_wave;
    constexpr int KH_KW      = cfg.kh * cfg.kw;
    constexpr int SLOTS      = KH_KW * N_CSPW;

    // Per-wave private LDS region — same shape as the production prologue.
    constexpr int WEIGHT_SLICE_UINT4 = TC::WEIGHT_LDS_SIZE_UINT4;
    __shared__ uint4 lds_buf[WEIGHT_SLICE_UINT4 * cfg.waves_per_wg];

    const int wave_id = static_cast<int>(threadIdx.x) / 64;
    uint4* wave_weight_lds = lds_buf + wave_id * WEIGHT_SLICE_UINT4;

    // weight_block_k = 0 (we're testing the K-tile starting at 0)
    constexpr int weight_block_k = 0;

    WeightLoader wl;

    // Same prologue pattern as conv_compute_loop_v3.hpp.
    for(int CS = 0; CS < N_CSPW; ++CS)
    {
        // Fprop only: wave w loads C-section (CS * waves + w).
        const int wave_section = CS * cfg.waves_per_wg + wave_id;
        WeightLoader::load_kyxc_to_lds_wave(
            wave_weight_lds, wei, weight_block_k, wave_section, C_total);
        ck_tile::direct_conv::wait_vmcnt<0>();
        __syncthreads();

        // Dispatch to the appropriate compile-time CS.
        if(CS == 0)
            wl.template read_from_lds_chunk<0>(wave_weight_lds);
        if constexpr(N_CSPW > 1)
            if(CS == 1)
                wl.template read_from_lds_chunk<1>(wave_weight_lds);
        if constexpr(N_CSPW > 2)
            if(CS == 2)
                wl.template read_from_lds_chunk<2>(wave_weight_lds);
        if constexpr(N_CSPW > 3)
            if(CS == 3)
                wl.template read_from_lds_chunk<3>(wave_weight_lds);

        __syncthreads();
    }

    // Dump each thread's weights[] array.
    const int tid = static_cast<int>(threadIdx.x);
    const _Float16* w_bytes = reinterpret_cast<const _Float16*>(&wl.weights[0]);
    for(int slot = 0; slot < SLOTS; ++slot)
    {
        for(int j = 0; j < 8; ++j)
        {
            weights_out[(tid * SLOTS + slot) * 8 + j] = w_bytes[slot * 8 + j];
        }
    }
#else
    (void)wei; (void)weights_out; (void)C_total;
#endif
}

class ConvWeightLoaderChunkedTest : public ::testing::Test
{
protected:
    // KYXC layout: weight[K, R, S, C].
    // Stride: K → kh*kw*C, R → kw*C, S → C, C → 1.
    static std::vector<_Float16> make_weights(int K, int kh, int kw, int C)
    {
        const int total = K * kh * kw * C;
        std::vector<_Float16> v(total);
        for(int i = 0; i < total; ++i)
            v[i] = static_cast<_Float16>(static_cast<float>((i % 251) + 1));
        return v;
    }

    static float read_weight(const std::vector<_Float16>& w,
                             int kw, int C,
                             int k, int r, int s, int c)
    {
        return static_cast<float>(w[((k * 3 + r) * kw + s) * C + c]);
    }

    template <int CfgIdx>
    void run_and_verify()
    {
        constexpr auto cfg = v3::KernelConfigurations<>::configs_map.get(CfgIdx);

        constexpr int N_CSPW = cfg.c_slices_per_wave;
        constexpr int KH     = cfg.kh;
        constexpr int KW     = cfg.kw;
        constexpr int KH_KW  = KH * KW;
        constexpr int SLOTS  = KH_KW * N_CSPW;
        constexpr int BLOCK_SIZE_T = cfg.block_size();

        const int K = cfg.block_k_size();
        const int C = cfg.waves_per_wg * N_CSPW * cfg.channels_per_group();

        auto wei = make_weights(K, KH, KW, C);

        _Float16 *d_wei = nullptr, *d_out = nullptr;
        const size_t out_elems = static_cast<size_t>(BLOCK_SIZE_T) * SLOTS * 8;
        ck_tile::hip_check_error(hipMalloc(&d_wei, wei.size() * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMalloc(&d_out, out_elems * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemset(d_out, 0, out_elems * sizeof(_Float16)));
        ck_tile::hip_check_error(hipMemcpy(
            d_wei, wei.data(), wei.size() * sizeof(_Float16), hipMemcpyHostToDevice));

        test_chunked_weight_kernel<CfgIdx>
            <<<dim3(1, 1, 1), BLOCK_SIZE_T>>>(d_wei, d_out, C);
        ck_tile::hip_check_error(hipDeviceSynchronize());

        std::vector<_Float16> out_host(out_elems);
        ck_tile::hip_check_error(hipMemcpy(
            out_host.data(), d_out, out_elems * sizeof(_Float16), hipMemcpyDeviceToHost));

        // Verify each (thread, R, S, CS, j) slot.
        for(int tid = 0; tid < BLOCK_SIZE_T; ++tid)
        {
            const int wave_id     = tid / 64;
            const int lane        = tid % 64;
            const int k_out       = lane % 16;   // MFMA M index (= K row)
            const int c_grp_local = lane / 16;   // 0..3
            for(int R = 0; R < KH; ++R)
            {
                for(int S = 0; S < KW; ++S)
                {
                    const int F = R * KW + S;
                    for(int CS = 0; CS < N_CSPW; ++CS)
                    {
                        const int wave_section = CS * cfg.waves_per_wg + wave_id;
                        const int section_C_base = wave_section * cfg.channels_per_group();
                        const int slot = F * N_CSPW + CS;
                        for(int j = 0; j < 8; ++j)
                        {
                            const int c_abs = section_C_base + c_grp_local * 8 + j;
                            const float expected =
                                read_weight(wei, KW, C, k_out, R, S, c_abs);
                            const float actual =
                                static_cast<float>(out_host[(tid * SLOTS + slot) * 8 + j]);
                            EXPECT_EQ(actual, expected)
                                << "cfg=" << CfgIdx
                                << " tid=" << tid << " wave=" << wave_id
                                << " R=" << R << " S=" << S << " CS=" << CS
                                << " k_out=" << k_out << " c_grp=" << c_grp_local
                                << " c_abs=" << c_abs;
                        }
                    }
                }
            }
        }

        ck_tile::hip_check_error(hipFree(d_wei));
        ck_tile::hip_check_error(hipFree(d_out));
    }
};

TEST_F(ConvWeightLoaderChunkedTest, Cfg49_CspwIs2)
{
    run_and_verify<CFG_CSPW2_FPROP>();
}
