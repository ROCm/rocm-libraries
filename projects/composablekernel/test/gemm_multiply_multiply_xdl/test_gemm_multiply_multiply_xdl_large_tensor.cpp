// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Large-tensor coverage for DeviceGemmMultiD_Xdl_CShuffle_V3 with the a8w8 row-col quant
// layout used by aiter: E[M,N] = (A[M,K] * B[N,K]^T) * d0[N] * d1[M], RCR, bf16 output.
//
// When any of A, Ds or E is larger than 2GB the device op splits the problem along M on the
// host (Partitioner in device_gemm_multiple_d_xdl_cshuffle_v3.hpp). These tests use the
// customer shape [seq_len, 12288, 4096]. A bf16 E is larger than 2GB from M = 87382, but the
// kernel writes whole MPerBlock (128) row tiles, so the padded E tile grid already reaches past
// 2GB from M = 87297. Masked padded stores past 2GB must not land anywhere in E.
//
// A CPU reference is too slow at these sizes, so everything stays on the GPU:
//  1. Exact check: the full (split) run is compared bit for bit against the same instance run
//     on M slices that are small enough not to split. Integer-valued inputs make the K
//     reduction exact, so any difference is a partitioning bug (wrong A/D/E offsets).
//  2. Reference check: a naive GPU GEMM computes selected row windows (first rows, last rows
//     and the rows around every split point) and they are compared exactly.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/gpu_verification.hpp"

namespace {

using F8   = ck::f8_t;
using I8   = int8_t;
using I32  = int32_t;
using F32  = float;
using BF16 = ck::bhalf_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough      = ck::tensor_operation::element_wise::PassThrough;
using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

// Same tile configuration as the shipped instances
// device_gemm_multiply_multiply_xdl_{i8_i8,f8_f8}_bf16_mk_nk_mn_comp (256x128x128x128, v3).
template <typename ADataType, typename AccDataType>
using DeviceOp = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
    // clang-format off
    Row, Col, ck::Tuple<Row, Col>, Row,
    ADataType, ADataType, ck::Tuple<F32, F32>, BF16, AccDataType, AccDataType,
    PassThrough, PassThrough, MultiplyMultiply,
    ck::tensor_operation::device::GemmSpecialization::Default,
    256, 128, 128, 128, 16, 16, 32, 32, 2, 2,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
    1, 1, S<1, 32, 1, 8>, S<8, 8, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, ADataType>;
// clang-format on

// Rows per slice in the sliced run. Below the 87381-row split threshold (bf16 E, N = 12288),
// and the slice edges do not line up with the split points, so both runs cut M differently.
constexpr ck::index_t kSliceRows = 40960;
// Rows checked against the naive reference around each point of interest.
constexpr ck::index_t kWindowRows = 16;

// Naive reference for selected rows. Each thread computes one output; the epilogue uses the
// same expression as MultiplyMultiply so the result is bit exact for integer-valued inputs.
template <typename ADataType, typename AccDataType>
__global__ void reference_rows_kernel(const ADataType* __restrict__ a,
                                      const ADataType* __restrict__ b,
                                      const F32* __restrict__ d0,
                                      const F32* __restrict__ d1,
                                      const ck::index_t* __restrict__ rows,
                                      BF16* __restrict__ out,
                                      ck::index_t num_rows,
                                      ck::index_t N,
                                      ck::index_t K)
{
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t r = blockIdx.y;
    if(n >= N || r >= num_rows)
        return;

    const int64_t m        = rows[r];
    const ADataType* a_row = a + m * K;
    const ADataType* b_row = b + n * K;
    AccDataType acc        = 0;
    for(ck::index_t k = 0; k < K; ++k)
    {
        acc += ck::type_convert<AccDataType>(a_row[k]) * ck::type_convert<AccDataType>(b_row[k]);
    }
    MultiplyMultiply{}(out[r * N + n], acc, d0[n], d1[m]);
}

// Compares the selected rows of E against the reference. Counts mismatches and records the
// first mismatching flat index.
__global__ void compare_rows_kernel(const BF16* __restrict__ e,
                                    const BF16* __restrict__ ref,
                                    const ck::index_t* __restrict__ rows,
                                    ck::index_t num_rows,
                                    ck::index_t N,
                                    unsigned long long* __restrict__ num_errors,
                                    unsigned long long* __restrict__ first_error)
{
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t r = blockIdx.y;
    if(n >= N || r >= num_rows)
        return;

    // Bit-exact comparison of the bf16 values.
    static_assert(sizeof(BF16) == sizeof(uint16_t));
    const int64_t m = rows[r];
    const auto x    = reinterpret_cast<const uint16_t*>(e)[m * N + n];
    const auto y    = reinterpret_cast<const uint16_t*>(ref)[r * N + n];
    if(x != y)
    {
        atomicAdd(num_errors, 1ull);
        atomicMin(first_error, static_cast<unsigned long long>(m * N + n));
    }
}

template <typename ADataType>
class TestGemmMultiplyMultiplyXdlLargeTensor : public ::testing::Test
{
    protected:
    using AccDataType = std::conditional_t<std::is_same_v<ADataType, I8>, I32, F32>;
    using Op          = DeviceOp<ADataType, AccDataType>;

    static constexpr ck::index_t kN = 12288;
    static constexpr ck::index_t kK = 4096;

    // First row of every leaf after the first, as split by the device op itself. Used only to
    // place the reference windows; the exact check below covers every row regardless.
    static std::vector<ck::index_t> split_points(const typename Op::Argument& arg)
    {
        typename Op::Partitioner partitioner(
            arg.M, arg.N, arg.StrideA, arg.StrideB, arg.StrideDs, arg.StrideC);
        std::vector<ck::index_t> points;
        for(const auto& leaf : partitioner.partitionGemmProblem(arg))
        {
            const auto first_row = (leaf.p_c_grid - arg.p_c_grid) / arg.StrideC;
            if(first_row > 0)
                points.push_back(static_cast<ck::index_t>(first_row));
        }
        std::sort(points.begin(), points.end());
        return points;
    }

    auto make_argument(const void* a,
                       const void* b,
                       const void* d0,
                       const void* d1,
                       void* e,
                       ck::index_t M,
                       ck::index_t N,
                       ck::index_t K)
    {
        // d0 is a per-column scale (RowMajor, stride 0); d1 is a per-row scale (ColMajor,
        // stride 0). This is the aiter row-col quant layout.
        return Op::MakeArgument(a,
                                b,
                                {d0, d1},
                                e,
                                M,
                                N,
                                K,
                                K,
                                K,
                                {0, 0},
                                N,
                                1,
                                PassThrough{},
                                PassThrough{},
                                MultiplyMultiply{});
    }

    void Run(ck::index_t M, ck::index_t N = kN, ck::index_t K = kK)
    {
        const std::size_t a_bytes  = static_cast<std::size_t>(M) * K * sizeof(ADataType);
        const std::size_t b_bytes  = static_cast<std::size_t>(N) * K * sizeof(ADataType);
        const std::size_t e_bytes  = static_cast<std::size_t>(M) * N * sizeof(BF16);
        const std::size_t ds_bytes = (static_cast<std::size_t>(M) + N) * sizeof(F32);

        std::cout << "M=" << M << " N=" << N << " K=" << K << " E=" << e_bytes / (1 << 20) << " MiB"
                  << std::endl;

        // The reference rows below need a few MiB more; the 1 GiB headroom covers them.
        const std::size_t need_bytes =
            a_bytes + b_bytes + 2 * e_bytes + ds_bytes + (std::size_t{1} << 30);

        std::size_t free_bytes = 0, total_bytes = 0;
        ck::hip_check_error(hipMemGetInfo(&free_bytes, &total_bytes));
        if(free_bytes < need_bytes)
        {
            GTEST_SKIP() << "SKIPPED large tensor case M=" << M << ": needs " << (need_bytes >> 20)
                         << " MiB of GPU memory, only " << (free_bytes >> 20) << " MiB free";
        }

        ck::DeviceMem a(a_bytes), b(b_bytes), d0(N * sizeof(F32)), d1(M * sizeof(F32));
        ck::DeviceMem e_split(e_bytes), e_slices(e_bytes);

        // Integer values keep the K reduction exact for both int8 and fp8.
        a.FillUniformRandInteger<ADataType>(-5, 5);
        b.FillUniformRandInteger<ADataType>(-5, 5);
        d0.FillUniformRandFp<F32>(0.5f, 1.5f);
        d1.FillUniformRandFp<F32>(0.5f, 1.5f);
        e_split.SetZero();
        e_slices.SetZero();

        // 1. Full problem in one call; the device op splits it if needed.
        auto arg = make_argument(a.GetDeviceBuffer(),
                                 b.GetDeviceBuffer(),
                                 d0.GetDeviceBuffer(),
                                 d1.GetDeviceBuffer(),
                                 e_split.GetDeviceBuffer(),
                                 M,
                                 N,
                                 K);
        ASSERT_TRUE(Op::IsSupportedArgument(arg)) << "full problem rejected, M=" << M;
        Op::MakeInvoker().Run(arg, StreamConfig{nullptr, false});

        // 2. The same problem as slices that each fit in 2GB, with offsets computed here.
        for(ck::index_t m0 = 0; m0 < M; m0 += kSliceRows)
        {
            const ck::index_t ms = std::min(kSliceRows, M - m0);
            auto slice_arg       = make_argument(
                static_cast<const ADataType*>(a.GetDeviceBuffer()) + static_cast<int64_t>(m0) * K,
                b.GetDeviceBuffer(),
                d0.GetDeviceBuffer(),
                static_cast<const F32*>(d1.GetDeviceBuffer()) + m0,
                static_cast<BF16*>(e_slices.GetDeviceBuffer()) + static_cast<int64_t>(m0) * N,
                ms,
                N,
                K);
            ASSERT_TRUE(Op::IsSupportedArgument(slice_arg)) << "slice rejected, m0=" << m0;
            Op::MakeInvoker().Run(slice_arg, StreamConfig{nullptr, false});
        }
        ck::hip_check_error(hipDeviceSynchronize());

        const auto exact = ck::profiler::gpu_verify<BF16>(e_split.GetDeviceBuffer(),
                                                          e_slices.GetDeviceBuffer(),
                                                          0.0f,
                                                          0.0f,
                                                          static_cast<std::size_t>(M) * N);
        EXPECT_TRUE(exact) << "split run differs from sliced run in " << exact.error_count << " of "
                           << exact.total << " elements, max error " << exact.max_error;

        // 3. Naive reference on the rows at the start, the end and both sides of every split.
        const auto splits = split_points(arg);
        std::cout << "split points: " << splits.size() << std::endl;
        std::set<ck::index_t> row_set;
        auto add_window = [&](ck::index_t center) {
            for(ck::index_t m = center - kWindowRows / 2; m < center + kWindowRows / 2; ++m)
                if(m >= 0 && m < M)
                    row_set.insert(m);
        };
        add_window(kWindowRows / 2);
        add_window(M - kWindowRows / 2);
        for(auto p : splits)
            add_window(p);
        const std::vector<ck::index_t> rows(row_set.begin(), row_set.end());
        const ck::index_t num_rows = static_cast<ck::index_t>(rows.size());
        ck::DeviceMem e_ref(static_cast<std::size_t>(num_rows) * N * sizeof(BF16));
        ck::DeviceMem rows_dev(rows.size() * sizeof(ck::index_t));
        rows_dev.ToDevice(rows.data());
        const dim3 block(256);
        const dim3 grid((N + block.x - 1) / block.x, num_rows);
        reference_rows_kernel<ADataType, AccDataType>
            <<<grid, block>>>(static_cast<const ADataType*>(a.GetDeviceBuffer()),
                              static_cast<const ADataType*>(b.GetDeviceBuffer()),
                              static_cast<const F32*>(d0.GetDeviceBuffer()),
                              static_cast<const F32*>(d1.GetDeviceBuffer()),
                              static_cast<const ck::index_t*>(rows_dev.GetDeviceBuffer()),
                              static_cast<BF16*>(e_ref.GetDeviceBuffer()),
                              num_rows,
                              N,
                              K);
        ck::hip_check_error(hipGetLastError());

        unsigned long long counters[2] = {0, ~0ull};
        ck::DeviceMem counters_dev(sizeof(counters));
        counters_dev.ToDevice(counters);
        auto* counters_ptr = static_cast<unsigned long long*>(counters_dev.GetDeviceBuffer());
        compare_rows_kernel<<<grid, block>>>(
            static_cast<const BF16*>(e_split.GetDeviceBuffer()),
            static_cast<const BF16*>(e_ref.GetDeviceBuffer()),
            static_cast<const ck::index_t*>(rows_dev.GetDeviceBuffer()),
            num_rows,
            N,
            counters_ptr,
            counters_ptr + 1);
        ck::hip_check_error(hipGetLastError());
        counters_dev.FromDevice(counters);
        EXPECT_EQ(counters[0], 0ull)
            << "reference mismatch in " << counters[0] << " of " << rows.size() * N
            << " checked elements, first at m=" << counters[1] / N << " n=" << counters[1] % N;
    }
};

using ADataTypes = ::testing::Types<I8, F8>;
TYPED_TEST_SUITE(TestGemmMultiplyMultiplyXdlLargeTensor, ADataTypes);

// Customer minimum seq_len; no split.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, NoSplit_M6144) { this->Run(6144); }

// Largest M whose padded tile grid (a multiple of 128 rows) still fits in 2GB; no split.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, LargestNoSplit_M87296) { this->Run(87296); }

// The bf16 E fits in 2GB (2147475456 bytes), but the last row tile, padded to 87424 rows,
// reaches past 2GB. Its masked stores must not wrap around onto the first rows of E.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, PaddedTilePast2GB_M87381) { this->Run(87381); }

// Same shape with a short K (the v3 pipeline needs more than 2 K loops), cheap enough for
// functional emulators.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, PaddedTilePast2GB_M87381_K512)
{
    this->Run(87381, 12288, 512);
}

// First M that splits: two leaves, 43776 + 43606 (uneven tail).
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, FirstSplit_M87382) { this->Run(87382); }

// Same split with a short K (the v3 pipeline needs more than 2 K loops), cheap enough for
// functional emulators.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, FirstSplit_M87382_K512)
{
    this->Run(87382, 12288, 512);
}

// Two even leaves of 65536.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, EvenSplit_M131072) { this->Run(131072); }

// Customer seq_len: 15 uneven leaves; A is larger than 2GB as well.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, Customer_M700000) { this->Run(700000); }

// Customer maximum seq_len: 32 leaves.
TYPED_TEST(TestGemmMultiplyMultiplyXdlLargeTensor, CustomerMax_M1600000) { this->Run(1600000); }

} // namespace
