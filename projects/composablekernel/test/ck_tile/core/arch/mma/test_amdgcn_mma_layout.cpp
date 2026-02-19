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
// gather back into C matrix. The result is compared to a host-side reference GEMM.

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

    __device__ void operator()(ADataType* a, BDataType* b, CDataType* c) const
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

        for(uint32_t m = 0; m < TileM; ++m)
        {
            for(uint32_t k = 0; k < TileK; ++k)
            {
                auto pos = RegisterMap<MmaOp>::A2RegisterMap(m, k);
                if(pos.lane == lane && pos.vecIdx < a_vec_size)
                {
                    a_frag[pos.vecIdx] = static_cast<ADataType>(a[m * TileK + k]);
                }
            }
        }

        for(uint32_t k = 0; k < TileK; ++k)
        {
            for(uint32_t n = 0; n < TileN; ++n)
            {
                auto pos = RegisterMap<MmaOp>::B2RegisterMap(k, n);
                if(pos.lane == lane && pos.vecIdx < b_vec_size)
                {
                    b_frag[pos.vecIdx] = static_cast<BDataType>(b[k * TileN + n]);
                }
            }
        }

        c_frag = MmaOp::exec(a_frag, b_frag, c_frag);

        for(uint32_t m = 0; m < TileM; ++m)
        {
            for(uint32_t n = 0; n < TileN; ++n)
            {
                auto pos = RegisterMap<MmaOp>::C2RegisterMap(m, n);
                if(pos.lane == threadIdx.x && pos.vecIdx < c_vec_size)
                {
                    c[m * TileN + n] = static_cast<CDataType>(c_frag[pos.vecIdx]);
                }
            }
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
 * For every (m, k, n) in the tensor it:
 *   1. Constructs A and B tensors with a single 1 at A(m,k) and B(k,n).
 *   2. Launches MmaLayoutTestKernel which uses RegisterMap to load fragments, executes the
 *      MMA intrinsic, and writes C back.
 *   3. Computes a host-side reference C = AB.
 *   4. Compares the reference and device C element-by-element.
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
    using AInputType = typename MmaTraits::ADataType;
    using BInputType = typename MmaTraits::BDataType;
    using AccType    = typename MmaTraits::CDataType;

    static_assert(MmaTraits::IsSupported, "Mma layout test requires supported register mappings");

    ck_tile::DeviceMem d_a(MmaTraits::BlockM * MmaTraits::BlockK * sizeof(AInputType));
    ck_tile::DeviceMem d_b(MmaTraits::BlockK * MmaTraits::BlockN * sizeof(BInputType));
    ck_tile::DeviceMem d_c(MmaTraits::BlockM * MmaTraits::BlockN * sizeof(AccType));
    std::array<AInputType, MmaTraits::BlockM * MmaTraits::BlockK> h_a;
    std::array<BInputType, MmaTraits::BlockK * MmaTraits::BlockN> h_b;
    std::array<AccType, MmaTraits::BlockM * MmaTraits::BlockN> h_c;
    std::array<AccType, MmaTraits::BlockM * MmaTraits::BlockN> h_c_result;

    auto* d_a_ptr = static_cast<AInputType*>(d_a.GetDeviceBuffer());
    auto* d_b_ptr = static_cast<BInputType*>(d_b.GetDeviceBuffer());
    auto* d_c_ptr = static_cast<AccType*>(d_c.GetDeviceBuffer());

    // test all possible (m, k, n) positions within the tile
    // e.g. for a 16x16x16 tile this is 4096 test cases
    std::vector<CaseDim> cases;
    cases.reserve(MmaTraits::BlockM * MmaTraits::BlockK * MmaTraits::BlockN);
    for(uint32_t m = 0; m < MmaTraits::BlockM; ++m)
    {
        for(uint32_t k = 0; k < MmaTraits::BlockK; ++k)
        {
            for(uint32_t n = 0; n < MmaTraits::BlockN; ++n)
            {
                cases.push_back({m, k, n});
            }
        }
    }

    for(auto const& test_case : cases)
    {
        std::fill(h_a.begin(), h_a.end(), static_cast<AInputType>(0));
        std::fill(h_b.begin(), h_b.end(), static_cast<BInputType>(0));
        std::fill(h_c.begin(), h_c.end(), static_cast<AccType>(0));

        // place a single 1 in A(m,k) and B(k,n)
        h_a[test_case.m * MmaTraits::BlockK + test_case.k] = static_cast<AInputType>(1);
        h_b[test_case.k * MmaTraits::BlockN + test_case.n] = static_cast<BInputType>(1);

        HIP_CHECK_ERROR(hipMemcpyAsync(
            d_a_ptr, h_a.data(), h_a.size() * sizeof(AInputType), hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(hipMemcpyAsync(
            d_b_ptr, h_b.data(), h_b.size() * sizeof(BInputType), hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(hipMemcpyAsync(
            d_c_ptr, h_c.data(), h_c.size() * sizeof(AccType), hipMemcpyHostToDevice));

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
                Kernel{}, dim3(1), dim3(Config::WaveSize), 0, d_a_ptr, d_b_ptr, d_c_ptr));

        HIP_CHECK_ERROR(
            hipMemcpyAsync(h_c_result.data(), d_c_ptr, d_c.GetBufferSize(), hipMemcpyDeviceToHost));
        HIP_CHECK_ERROR(hipStreamSynchronize(nullptr));

        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            std::printf("MMA tensors for m=%u k=%u n=%u\n", test_case.m, test_case.k, test_case.n);
            print_tensor("  Device tensor C", h_c_result, MmaTraits::BlockM, MmaTraits::BlockN);
        }

        // check if the 1 that was placed A(m,k) and B(k,n) found its way to C(m,n) in the device
        // tensor
        const AccType tol = 1.0e-1f; // TODO: this tolerance might not be suitable for all data
                                     // types and should be revisited if we add more configurations
        for(uint32_t m = 0; m < MmaTraits::BlockM; ++m)
        {
            for(uint32_t n = 0; n < MmaTraits::BlockN; ++n)
            {
                const AccType expected = (m == test_case.m && n == test_case.n)
                                             ? static_cast<AccType>(1)
                                             : static_cast<AccType>(0);
                EXPECT_NEAR(h_c_result[m * MmaTraits::BlockN + n], expected, tol)
                    << "Mismatch at C(" << m << "," << n << ")";
            }
        }
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

