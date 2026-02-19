// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/utility/env.hpp"

#include "test/ck_tile/core/arch/mma/test_amdgcn_mma_layout_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cstddef>
#include <string>

namespace ck  = ck_tile;
namespace mma = ck_tile::core::arch::mma;

// MMA register layout validation test for amdgcn_mma structs.
//
// Strategy: for every (m, k, n) triple in the tile, the test constructs a pair of input tensors
// A and B that contain exactly one non-zero element each, placed so that their product
// contributes to a single output element C(m, n):
//
//         A  (M x K)                B  (K x N)              C = A * B  (M x N)
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//      . . . 1 . . . .           . . . . . . . .           . . . . . . . .
//      . . . . . . . .           . . . 1 . . . .           . . . . . 1 . .
//      . . . . . . . .           . . . . . . . .           . . . . . . . .
//         A(m,k) = 1                B(k,n) = 1                C(m,n) = 1
//
// The kernel uses RegisterMap to scatter A and B into the correct (lane, vecIdx) positions
// of the MMA fragment registers, executes the intrinsic, then uses RegisterMap again to
// gather back into C matrix. The position of "1" in C is checked against the expected (m, n) location.

namespace {

/**
 * @class MmaLayoutTestKernel
 * @brief Device kernel that performs C = AB using a given Mma op
 *
 * @tparam ADataType     Data type of tensor A elements
 * @tparam BDataType     Data type of tensor B elements
 * @tparam CDataType     Data type of accumulator / output C elements
 * @tparam BlockM        M-dimension of the MMA tile
 * @tparam BlockN        N-dimension of the MMA tile
 * @tparam BlockK        K-dimension of the MMA tile
 * @tparam LaneGroupSize WaveSize / LaneGroupsPerWave
 * @tparam BlockSize     HIP block size
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          uint32_t LaneGroupSize,
          uint32_t BlockSize>
struct MmaLayoutTestKernel
{
    static constexpr int kBlockSize = BlockSize;

    __device__ void operator()(uint32_t* error_flags) const
    {
        using Selector =
            mma::MmaDefaultSelector<ADataType,
                                    BDataType,
                                    CDataType,
                                    BlockM,
                                    BlockN,
                                    BlockK,
                                    decltype(ck_tile::core::arch::get_compiler_target())>;
        using MmaOp                   = typename Selector::SelectedOp;
        using MmaTraits               = mma::MmaOpTraits<MmaOp>;
        using AVecType                = typename MmaTraits::AVecType;
        using BVecType                = typename MmaTraits::BVecType;
        using CVecType                = typename MmaTraits::CVecType;
        constexpr uint32_t a_vec_size = sizeof(AVecType) / sizeof(ADataType);
        constexpr uint32_t b_vec_size = sizeof(BVecType) / sizeof(BDataType);
        constexpr uint32_t c_vec_size = sizeof(CVecType) / sizeof(CDataType);

        // LaneGroupSize doesnt equal WaveSize on RDNA3 due to matrix replication
        const uint32_t lane = threadIdx.x % LaneGroupSize;

        AVecType a_frag{};
        BVecType b_frag{};
        CVecType c_frag{};

        constexpr uint32_t TileM = MmaTraits::BlockM;
        constexpr uint32_t TileN = MmaTraits::BlockN;
        constexpr uint32_t TileK = MmaTraits::BlockK;

        // get (m, k, n), where "1" should be placed for this block 
        const uint32_t case_idx = static_cast<uint32_t>(blockIdx.x);
        const uint32_t m = case_idx / (TileK * TileN);
        const uint32_t k = (case_idx / TileN) % TileK;
        const uint32_t n = case_idx % TileN;

        // place a single "1" in A/B fragments
        auto a_pos = RegisterMap<MmaOp>::A2RegisterMap(m, k);
        if(a_pos.lane == lane && a_pos.vecIdx < a_vec_size)
        {
            a_frag[a_pos.vecIdx] = static_cast<ADataType>(1);
        }

        auto b_pos = RegisterMap<MmaOp>::B2RegisterMap(k, n);
        if(b_pos.lane == lane && b_pos.vecIdx < b_vec_size)
        {
            b_frag[b_pos.vecIdx] = static_cast<BDataType>(1);
        }

        c_frag = MmaOp::exec(a_frag, b_frag, c_frag);
        __syncthreads();

        __shared__ uint32_t err;
        if(threadIdx.x == 0)
        {
            err = 0;
        }
        __syncthreads();

        const CDataType tol = static_cast<CDataType>(1.0e-1f); // TODO: this tolerance might not be suitable for all data types and should be revisited if we add more configurations
        for(uint32_t i = 0; i < TileM; ++i)
        {
            for(uint32_t j = 0; j < TileN; ++j)
            {
                auto pos = RegisterMap<MmaOp>::C2RegisterMap(i, j);
                if(pos.lane == threadIdx.x && pos.vecIdx < c_vec_size)
                {
                    const CDataType expected = (i == m && j == n)
                                                   ? static_cast<CDataType>(1)
                                                   : static_cast<CDataType>(0);
                    const CDataType value = static_cast<CDataType>(c_frag[pos.vecIdx]);
                    if(fabsf(static_cast<float>(value - expected)) > static_cast<float>(tol))
                    {
                        atomicExch(&err, 1);
                    }
                }
            }
        }

        __syncthreads();
        if(threadIdx.x == 0)
        {
            error_flags[case_idx] = err;
        }
    }
};

struct CaseDim
{
    uint32_t m;
    uint32_t k;
    uint32_t n;
};

// debug helper
template <typename Array>
void print_tensor(const char* label, const Array& tensor, uint32_t rows, uint32_t columns)
{
    std::printf("%s\n", label);
    for(uint32_t row = 0; row < rows; ++row)
    {
        std::printf("    row %2u:", row);
        for(uint32_t column = 0; column < columns; ++column)
        {
            float value = static_cast<float>(tensor[row * columns + column]);
            std::printf(" %7.3f", value);
        }
        std::printf("\n");
    }
}

/**
 * @class MmaLayoutTestConfig
 * @brief Gathers compile-time parameters needed for one MMA layout test variant.
 *
 * @tparam Selector_          MmaDefaultSelector instantiation for the target
 * @tparam CompilerTarget_    amdgcn_target describing the arch target
 * @tparam LaneGroupsPerWave_ Number of lane groups per wave
 */
template <typename Selector_, typename CompilerTarget_, uint32_t LaneGroupsPerWave_>
struct MmaLayoutTestConfig
{
    using Selector                     = Selector_;
    using CompilerTarget               = CompilerTarget_;
    static constexpr uint32_t WaveSize = static_cast<uint32_t>(CompilerTarget::WAVE_SIZE_ID);
    static constexpr uint32_t LaneGroupsPerWave = LaneGroupsPerWave_;
    static constexpr uint32_t LaneGroupSize     = WaveSize / LaneGroupsPerWave;
};

/**
 * @brief Test driver: runs the test for a given MMA configuration.
 *
 * The testlaunches (mkn) test cases (one per block) to check all possible positions of the "1" in the A/B tensors.
 *   1. Constructs A and B tensors with a single 1 at A(m,k) and B(k,n).
 *   2. Executes MMA intrinsic to compute C tensor.
 *   3. Checks if C has the 1 in the expected position.
 *
 * @tparam Config An MmaLayoutTestConfig instantiation
 * @return true if the test ran on hardware; false if skipped (no matching device)
 */
template <typename Config>
bool run_mma_layout_test_case()
{
    int device_count = 0;
    hipDevice_t device{};
    HIP_CHECK_ERROR(hipGetDevice(&device));
    HIP_CHECK_ERROR(hipGetDeviceCount(&device_count));

    hipDeviceProp_t props{};
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));

    const auto runtime_target =
        ck_tile::core::arch::hip_device_prop_gcn_arch_name_to_amdgcn_target_id(props.gcnArchName);
    const bool has_device = device_count > 0;

    if(!has_device || runtime_target == ck_tile::core::arch::amdgcn_target_id::HOST ||
       runtime_target != Config::CompilerTarget::TARGET_ID)
        return false;

    using Selector   = typename Config::Selector;
    using MmaOp      = typename Selector::SelectedOp;
    using MmaTraits  = mma::MmaOpTraits<MmaOp>;

    static_assert(MmaTraits::IsSupported, "Mma layout test requires supported register mappings");

    constexpr uint32_t total_cases =
        MmaTraits::BlockM * MmaTraits::BlockK * MmaTraits::BlockN;
    ck_tile::DeviceMem d_errors(total_cases * sizeof(uint32_t));
    std::vector<uint32_t> h_errors(total_cases, 0u);

    auto* d_error_ptr = static_cast<uint32_t*>(d_errors.GetDeviceBuffer());

    using Kernel = MmaLayoutTestKernel<typename MmaTraits::ADataType,
                                       typename MmaTraits::BDataType,
                                       typename MmaTraits::CDataType,
                                       MmaTraits::BlockM,
                                       MmaTraits::BlockN,
                                       MmaTraits::BlockK,
                                       Config::LaneGroupSize,
                                       Config::WaveSize>;

    (void)hipGetLastError();

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, false, 0, 0, 1},
        ck_tile::make_kernel(
            Kernel{}, dim3(total_cases), dim3(Config::WaveSize), 0, d_error_ptr));

    HIP_CHECK_ERROR(
        hipMemcpyAsync(h_errors.data(), d_error_ptr, d_errors.GetBufferSize(), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));

    for(uint32_t case_idx = 0; case_idx < total_cases; ++case_idx)
    {
        const uint32_t m = case_idx / (MmaTraits::BlockK * MmaTraits::BlockN);
        const uint32_t k = (case_idx / MmaTraits::BlockN) % MmaTraits::BlockK;
        const uint32_t n = case_idx % MmaTraits::BlockN;

        EXPECT_EQ(h_errors[case_idx], 0u)
            << "Mismatch for m=" << m << " k=" << k << " n=" << n;
    }

    return true;
}

} // namespace

// ==================== Test configurations per target ====================
// TODO: currently we have only 1 specific target per test. This should be revisited to enable all
// the targets within the family (gfx12, gfx11, gfx9)
using MmaGfx1201CompilerTarget = decltype(ck_tile::core::arch::make_amdgcn_gfx12_target<
                                          ck_tile::core::arch::amdgcn_target_id::GFX1201>());
using MmaGfx90aCompilerTarget  = decltype(ck_tile::core::arch::make_amdgcn_gfx9_target<
                                          ck_tile::core::arch::amdgcn_target_id::GFX90A>());
using MmaGfx1100CompilerTarget = decltype(ck_tile::core::arch::make_amdgcn_gfx11_target<
                                          ck_tile::core::arch::amdgcn_target_id::GFX1100>());

using MmaGfx1201Selector = mma::
    MmaDefaultSelector<ck::fp16_t, ck::fp16_t, ck::fp32_t, 16u, 16u, 16u, MmaGfx1201CompilerTarget>;
using MmaGfx90aSelector = mma::
    MmaDefaultSelector<ck::fp16_t, ck::fp16_t, ck::fp32_t, 16u, 16u, 16u, MmaGfx90aCompilerTarget>;
using MmaGfx1100Selector = mma::
    MmaDefaultSelector<ck::fp16_t, ck::fp16_t, ck::fp32_t, 16u, 16u, 16u, MmaGfx1100CompilerTarget>;

struct MmaGfx12Config : MmaLayoutTestConfig<MmaGfx1201Selector,
                                            MmaGfx1201CompilerTarget,
                                            1u> // LaneGroupsPerWave
{
};

struct MmaGfx9Config : MmaLayoutTestConfig<MmaGfx90aSelector,
                                           MmaGfx90aCompilerTarget,
                                           1u> // LaneGroupsPerWave
{
};

struct MmaGfx11Config : MmaLayoutTestConfig<MmaGfx1100Selector,
                                            MmaGfx1100CompilerTarget,
                                            2u> // LaneGroupsPerWave
{
};
// ========================================================================

TEST(TestMmaLayout, Mma_16x16x16_F16_F16_F32_GFX9)
{
    bool executed = run_mma_layout_test_case<MmaGfx9Config>();

    if(!executed)
    {
        GTEST_SKIP() << "No gfx90a HIP device found. Skipping test.";
    }
}

TEST(TestMmaLayout, Mma_16x16x16_F16_F16_F32_GFX11)
{
    bool executed = run_mma_layout_test_case<MmaGfx11Config>();

    if(!executed)
    {
        GTEST_SKIP() << "No gfx1100 HIP device found. Skipping test.";
    }
}

TEST(TestMmaLayout, Mma_16x16x16_F16_F16_F32_GFX12)
{
    bool executed = run_mma_layout_test_case<MmaGfx12Config>();

    if(!executed)
    {
        GTEST_SKIP() << "No gfx1201 HIP device found. Skipping test.";
    }
}

