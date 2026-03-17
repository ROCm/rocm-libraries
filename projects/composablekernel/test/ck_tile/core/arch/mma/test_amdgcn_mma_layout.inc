// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/core/arch/arch.hpp"

#include <cstdio>
#include <vector>

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace mma;

using F16       = fp16_t;
using F32       = fp32_t;
using Target90a = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX90A>());
using Target950 = decltype(make_amdgcn_gfx9_target<amdgcn_target_id::GFX950>());
using Target11  = decltype(make_amdgcn_gfx11_target<amdgcn_target_id::GFX1100>());
using Target12  = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1201>());

// MMA register layout validation test for amdgcn_mma structs.
//
// Strategy: for every (m, k, n) triple in the tile, the test constructs a pair of input tensors A
// and B that contain exactly one non-zero element each, placed so that their product contributes to
// a single output element C(m, n):
//
//         A  (M x K)                B  (K x N)              C = A * B  (M x N)
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . 1 . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . 1 . . . .           . . . . . 1 . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//         A(m,k) = 1                B(k,n) = 1                C(m,n) = 1
//
// The kernel uses TileDistrEncRegMap to scatter A and B into the correct (lane, vecIdx) positions
// of the MMA fragment registers, executes the intrinsic, then uses TileDistrEncRegMap again to
// gather back into C matrix. The position of "1" in C is checked against the expected (m, n)
// location.

/**
 * @class MmaLayoutTestKernel
 * @brief Device kernel that performs C = AB using a given Mma op
 * @tparam MmaOp Intrinsic (amdgcn_mma) to be tested
 */
template <typename MmaOp> // TODO: C++20 concept for MmaOp
struct MmaLayoutTestKernel
{
    static constexpr int kBlockSize = MmaOp::WaveSize;

    __device__ void operator()(uint32_t* error_flags) const
    {
        using ARegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding>;
        using BRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::BWarpDstrEncoding>;
        using CRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::CWarpDstrEncoding>;

        if constexpr(MmaOpTraits<MmaOp>::IsSupported)
        {
            using AVecType               = typename MmaOp::AVecType;
            using BVecType               = typename MmaOp::BVecType;
            using CVecType               = typename MmaOp::CVecType;
            constexpr index_t a_vec_size = vector_traits<AVecType>::vector_size;
            constexpr index_t b_vec_size = vector_traits<BVecType>::vector_size;
            constexpr index_t c_vec_size = vector_traits<CVecType>::vector_size;

            const index_t lane = threadIdx.x;

            AVecType a_frag{};
            BVecType b_frag{};
            CVecType c_frag{};

            // get (m, k, n), where "1" should be placed for this block
            const index_t case_idx = blockIdx.x;
            const index_t m        = case_idx / (MmaOp::kK * MmaOp::kN);
            const index_t k        = (case_idx / MmaOp::kN) % MmaOp::kK;
            const index_t n        = case_idx % MmaOp::kN;

            // place a single "1" in A/B fragments using (lane, vecIdx) -> (row, col) mapping
            for(index_t v = 0; v < a_vec_size; ++v)
            {
                auto a_coords = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
                if(a_coords[0] == m && a_coords[1] == k)
                {
                    a_frag[v] = 1;
                }
            }

            for(index_t v = 0; v < b_vec_size; ++v)
            {
                auto b_coords = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
                if(b_coords[0] == n && b_coords[1] == k)
                {
                    b_frag[v] = 1;
                }
            }

            c_frag = MmaOp::exec(a_frag, b_frag, c_frag);

            // TODO: this tolerance might not be suitable for all data types and
            // should be revisited if we add more configurations
            const float tol = 1.0e-1f;
            index_t err     = 0;

            for(index_t v = 0; v < c_vec_size; ++v)
            {
                auto c_coords = CRegMap::calc_matrix_indices_from_lane_vector(lane, v);

                const float expected = (c_coords[0] == m && c_coords[1] == n) ? 1 : 0;
                const float value    = static_cast<float>(c_frag[v]);
                if(fabsf(value - expected) > tol)
                {
                    err = 1;
                }
            }

            const uint32_t any_err = __any(err);
            if(threadIdx.x == 0)
            {
                error_flags[case_idx] = any_err;
            }
        }
    }
};

/**
 * @brief Test driver: runs the test for a given MMA configuration.
 *
 * The testlaunches (mkn) test cases (one per block) to check all possible positions of the "1" in
 * the A/B tensors.
 *   1. Constructs A and B tensors with a single 1 at A(m,k) and B(k,n).
 *   2. Executes MMA intrinsic to compute C tensor.
 *   3. Checks if C has the 1 in the expected position.
 *
 * @tparam MmaOp Intrinsic (amdgcn_mma) to be tested
 */
template <typename MmaOp> // TODO: C++20 concept for MmaOp
void run_mma_layout_test()
{
    EXPECT_TRUE(MmaOpTraits<MmaOp>::IsSupported) << "Unsupported MmaOp! Bad MmaOp in list!\n";

    int device_count = 0;
    hipDevice_t device{};
    HIP_CHECK_ERROR(hipGetDevice(&device));
    HIP_CHECK_ERROR(hipGetDeviceCount(&device_count));
    hipDeviceProp_t props{};
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));
    EXPECT_EQ(props.warpSize, static_cast<int>(MmaOp::WaveSize))
        << "Device wavesize " << props.warpSize << " != Mma wavesize " << MmaOp::WaveSize;

    EXPECT_TRUE(device_count > 0) << "No device found!";

    constexpr uint32_t total_cases = MmaOp::kM * MmaOp::kN * MmaOp::kK;
    ck_tile::DeviceMem d_errors(total_cases * sizeof(uint32_t));
    std::vector<uint32_t> h_errors(total_cases, 0u);

    auto* d_error_ptr = static_cast<uint32_t*>(d_errors.GetDeviceBuffer());

    std::ignore = hipGetLastError();

    using Kernel = MmaLayoutTestKernel<MmaOp>;

    std::ignore = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, false, 0, 0, 1},
        ck_tile::make_kernel(Kernel{}, dim3(total_cases), dim3(MmaOp::WaveSize), 0, d_error_ptr));

    HIP_CHECK_ERROR(hipMemcpyAsync(
        h_errors.data(), d_error_ptr, d_errors.GetBufferSize(), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));

    for(uint32_t case_idx = 0; case_idx < total_cases; ++case_idx)
    {
        const uint32_t m = case_idx / (MmaOp::kK * MmaOp::kN);
        const uint32_t k = (case_idx / MmaOp::kN) % MmaOp::kK;
        const uint32_t n = case_idx % MmaOp::kN;

        EXPECT_EQ(h_errors[case_idx], 0u) << "Mismatch for m=" << m << " k=" << k << " n=" << n;
    }
}

// List of intrinsics to test.
// clang-format off
using Gfx9Intrinsics = ::testing::Types<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultMfmaCtrlFlags,                Target90a, MmaOpFamily::DENSE> // mfma_f32_16x16x16f16
>;
using Gfx950Intrinsics = ::testing::Types<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 32u, DefaultMfmaCtrlFlags,                Target950, MmaOpFamily::DENSE> // mfma_f32_16x16x32_f16
>;
using Gfx11Intrinsics = ::testing::Types<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags<F16, F16, F32>, Target11,  MmaOpFamily::DENSE> // wmma_f32_16x16x16_f16_w32
>;
using Gfx12Intrinsics = ::testing::Types<
    amdgcn_mma<F16, F16, F32, 16u, 16u, 16u, DefaultWmmaCtrlFlags<F16, F16, F32>, Target12,  MmaOpFamily::DENSE>  // wmma_f32_16x16x16_f16_w32_gfx12
>;
// clang-format on

template <typename MmaOp>
class TestMmaLayout : public ::testing::Test
{
};
