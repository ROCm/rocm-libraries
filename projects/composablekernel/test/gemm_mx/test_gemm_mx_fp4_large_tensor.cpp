// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Large-tensor coverage for DeviceGemmMX_Xdl_CShuffleV3 with fp4 (a4w4) inputs, RCR, fp16
// output: C[M,N] = sum_k A[m,k] * B[n,k] * scale_a[m,k/32] * scale_b[n,k/32].
//
// For fp4 A the device op splits the problem along M on the host when A or C is larger than
// 2GB (Partitioner in device_gemm_xdl_cshuffle_v3_mx.hpp). These tests use the customer shape
// [seq_len, 12288, 4096]. An fp16 C is larger than 2GB from M = 87382, but the kernel writes
// whole MPerBlock (128) row tiles, so the padded C tile grid already reaches past 2GB from
// M = 87297. Masked padded stores past 2GB must not land anywhere in C.
//
// A CPU reference is too slow at these sizes, so everything stays on the GPU:
//  1. Exact check: the full (split) run is compared bit for bit against the same instance run
//     on M slices that are small enough not to split. Scales are 1 or 2, so the fp32
//     accumulation of fp4 products is exact and any difference is a partitioning bug.
//  2. Reference check: a naive GPU GEMM computes selected row windows (first rows, last rows
//     and the rows around every split point) and they are compared exactly.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/gpu_verification.hpp"

namespace {

using F4     = ck::f4x2_pk_t;
using F16    = ck::half_t;
using F32    = float;
using E8M0PK = int32_t; // four e8m0 scales

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

constexpr ck::index_t kScaleBlockSize = 32;

// Same tile configuration as the shipped instance device_gemm_mx_xdl_f4_f4_f16_mk_nk_mn
// (256x128x256x128, v3).
using DeviceOp = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3<
    // clang-format off
    Row, Col, Row, F4, E8M0PK, F4, E8M0PK, F16, F32, F16,
    PassThrough, PassThrough, PassThrough,
    ck::tensor_operation::device::GemmSpecialization::Default, kScaleBlockSize,
    256, 128, 256, 128, 16, 16, 16, 16, 4, 8,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, true,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, true,
    2, 2, S<1, 32, 1, 8>, 8,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3>;
// clang-format on

// Rows per slice in the sliced run. Below the 87381-row split threshold (fp16 C, N = 12288),
// a multiple of 32 (the scale shuffle group), and the slice edges do not line up with the
// split points, so both runs cut M differently.
constexpr ck::index_t kSliceRows = 40960;
// Rows checked against the naive reference around each point of interest.
constexpr ck::index_t kWindowRows = 16;

// Position of scale (mn, k) in the gfx950 pre-shuffled scale buffer (see
// preShuffleScaleBuffer_gfx950 in profile_gemm_mx_impl.hpp). Each group of 32 rows is
// contiguous, so a row offset that is a multiple of 32 is a plain pointer offset.
__device__ int64_t shuffled_scale_index(int64_t mn, int64_t k, int64_t k_scales)
{
    const int64_t n0 = mn / 32, n1 = mn % 16, n2 = (mn % 32) / 16;
    const int64_t k0 = k / 8, k1 = k % 4, k2 = (k % 8) / 4;
    return n0 * 32 * k_scales + k0 * 256 + k1 * 64 + n1 * 4 + k2 * 2 + n2;
}

__device__ float fp4_to_float(uint8_t v)
{
    constexpr float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float x          = lut[v & 0x7];
    return (v & 0x8) ? -x : x;
}

__device__ float e8m0_to_float(uint8_t e) { return ldexpf(1.0f, static_cast<int>(e) - 127); }

// splitmix64 of (element index, seed). DeviceMem::FillUniformRand* seeds every GPU thread the
// same way on each call, so two tensors of the same size get the same bytes; a seed per tensor
// keeps the inputs independent.
__device__ uint64_t hash_index(uint64_t i, uint64_t seed)
{
    uint64_t z = i + seed * 0x9E3779B97F4A7C15ull;
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Fills every byte with lo + hash % range.
__global__ void
fill_bytes_kernel(uint8_t* p, uint64_t n, uint64_t seed, uint32_t lo, uint32_t range)
{
    for(uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; i < n;
        i += static_cast<uint64_t>(blockDim.x) * gridDim.x)
        p[i] = static_cast<uint8_t>(lo + hash_index(i, seed) % range);
}

void fill_bytes(ck::DeviceMem& buf, uint64_t seed, uint32_t lo, uint32_t range)
{
    fill_bytes_kernel<<<1024, 256>>>(
        static_cast<uint8_t*>(buf.GetDeviceBuffer()), buf.GetBufferSize(), seed, lo, range);
    ck::hip_check_error(hipGetLastError());
}

// Naive reference for selected rows. Each thread computes one output in fp32.
__global__ void reference_rows_kernel(const uint8_t* __restrict__ a,
                                      const uint8_t* __restrict__ a_scale,
                                      const uint8_t* __restrict__ b,
                                      const uint8_t* __restrict__ b_scale,
                                      const ck::index_t* __restrict__ rows,
                                      F16* __restrict__ out,
                                      ck::index_t num_rows,
                                      ck::index_t N,
                                      ck::index_t K)
{
    const int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t r = blockIdx.y;
    if(n >= N || r >= num_rows)
        return;

    const int64_t m        = rows[r];
    const int64_t k_scales = K / kScaleBlockSize;
    float acc              = 0.0f;
    for(int64_t kb = 0; kb < k_scales; ++kb)
    {
        float block_acc = 0.0f;
        for(int64_t k = kb * kScaleBlockSize; k < (kb + 1) * kScaleBlockSize; k += 2)
        {
            const uint8_t av = a[(m * K + k) / 2];
            const uint8_t bv = b[(n * K + k) / 2];
            block_acc += fp4_to_float(av & 0xf) * fp4_to_float(bv & 0xf);
            block_acc += fp4_to_float(av >> 4) * fp4_to_float(bv >> 4);
        }
        acc += block_acc * e8m0_to_float(a_scale[shuffled_scale_index(m, kb, k_scales)]) *
               e8m0_to_float(b_scale[shuffled_scale_index(n, kb, k_scales)]);
    }
    out[r * N + n] = ck::type_convert<F16>(acc);
}

// Compares the selected rows of C against the reference, bit for bit. Counts mismatches and
// records the first mismatching flat index.
__global__ void compare_rows_kernel(const F16* __restrict__ c,
                                    const F16* __restrict__ ref,
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

    static_assert(sizeof(F16) == sizeof(uint16_t));
    const int64_t m = rows[r];
    const auto x    = reinterpret_cast<const uint16_t*>(c)[m * N + n];
    const auto y    = reinterpret_cast<const uint16_t*>(ref)[r * N + n];
    if(x != y)
    {
        atomicAdd(num_errors, 1ull);
        atomicMin(first_error, static_cast<unsigned long long>(m * N + n));
    }
}

class TestGemmMXFp4LargeTensor : public ::testing::Test
{
    protected:
    static constexpr ck::index_t kN = 12288;
    static constexpr ck::index_t kK = 4096;

    void SetUp() override
    {
        if(ck::get_device_name() != "gfx950")
            GTEST_SKIP() << "MX fp4 large tensor tests need gfx950, found "
                         << ck::get_device_name();
    }

    // First row of every leaf after the first, as split by the device op itself. Used only to
    // place the reference windows; the exact check below covers every row regardless.
    static std::vector<ck::index_t> split_points(const DeviceOp::Argument& arg)
    {
        DeviceOp::Partitioner partitioner(
            arg.M, arg.N, arg.StrideA, arg.StrideScaleA, arg.StrideB, arg.StrideC);
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

    static auto make_argument(const void* a,
                              const void* a_scale,
                              const void* b,
                              const void* b_scale,
                              void* c,
                              ck::index_t M,
                              ck::index_t N,
                              ck::index_t K)
    {
        const ck::index_t k_scales = K / kScaleBlockSize;
        return DeviceOp::MakeArgument(static_cast<const F4*>(a),
                                      static_cast<const E8M0PK*>(a_scale),
                                      static_cast<const F4*>(b),
                                      static_cast<const E8M0PK*>(b_scale),
                                      static_cast<F16*>(c),
                                      M,
                                      N,
                                      K,
                                      K,        // StrideA
                                      k_scales, // StrideScaleA (row-major M x K/32)
                                      K,        // StrideB
                                      k_scales, // StrideScaleB (column-major K/32 x N)
                                      N,        // StrideC
                                      1,
                                      PassThrough{},
                                      PassThrough{},
                                      PassThrough{});
    }

    void Run(ck::index_t M, ck::index_t N = kN, ck::index_t K = kK)
    {
        const ck::index_t m_padded      = (M + 31) / 32 * 32; // scale rows come in groups of 32
        const ck::index_t k_scales      = K / kScaleBlockSize;
        const std::size_t a_bytes       = static_cast<std::size_t>(M) * K / 2;
        const std::size_t b_bytes       = static_cast<std::size_t>(N) * K / 2;
        const std::size_t a_scale_bytes = static_cast<std::size_t>(m_padded) * k_scales;
        const std::size_t b_scale_bytes = static_cast<std::size_t>(N) * k_scales;
        const std::size_t c_bytes       = static_cast<std::size_t>(M) * N * sizeof(F16);

        std::cout << "M=" << M << " N=" << N << " K=" << K << " C=" << c_bytes / (1 << 20) << " MiB"
                  << std::endl;

        // The reference rows below need a few MiB more; the 1 GiB headroom covers them.
        const std::size_t need_bytes = a_bytes + b_bytes + a_scale_bytes + b_scale_bytes +
                                       2 * c_bytes + (std::size_t{1} << 30);

        std::size_t free_bytes = 0, total_bytes = 0;
        ck::hip_check_error(hipMemGetInfo(&free_bytes, &total_bytes));
        if(free_bytes < need_bytes)
        {
            GTEST_SKIP() << "SKIPPED large tensor case M=" << M << ": needs " << (need_bytes >> 20)
                         << " MiB of GPU memory, only " << (free_bytes >> 20) << " MiB free";
        }

        ck::DeviceMem a(a_bytes), b(b_bytes), a_scale(a_scale_bytes), b_scale(b_scale_bytes);
        ck::DeviceMem c_split(c_bytes), c_slices(c_bytes);

        // Every 4-bit pattern is a finite fp4 (e2m1) value. Scales are 2^0 or 2^1, which keeps
        // the fp32 sums exact.
        fill_bytes(a, 1, 0, 256);
        fill_bytes(b, 2, 0, 256);
        fill_bytes(a_scale, 3, 127, 2);
        fill_bytes(b_scale, 4, 127, 2);
        c_split.SetZero();
        c_slices.SetZero();

        // 1. Full problem in one call; the device op splits it if needed.
        auto arg = make_argument(a.GetDeviceBuffer(),
                                 a_scale.GetDeviceBuffer(),
                                 b.GetDeviceBuffer(),
                                 b_scale.GetDeviceBuffer(),
                                 c_split.GetDeviceBuffer(),
                                 M,
                                 N,
                                 K);
        ASSERT_TRUE(DeviceOp::IsSupportedArgument(arg)) << "full problem rejected, M=" << M;
        DeviceOp::MakeInvoker().Run(arg, StreamConfig{nullptr, false});

        // 2. The same problem as slices that each fit in 2GB, with offsets computed here.
        for(ck::index_t m0 = 0; m0 < M; m0 += kSliceRows)
        {
            const ck::index_t ms = std::min(kSliceRows, M - m0);
            auto slice_arg       = make_argument(
                static_cast<const uint8_t*>(a.GetDeviceBuffer()) + static_cast<int64_t>(m0) * K / 2,
                static_cast<const uint8_t*>(a_scale.GetDeviceBuffer()) +
                    static_cast<int64_t>(m0) * k_scales,
                b.GetDeviceBuffer(),
                b_scale.GetDeviceBuffer(),
                static_cast<F16*>(c_slices.GetDeviceBuffer()) + static_cast<int64_t>(m0) * N,
                ms,
                N,
                K);
            ASSERT_TRUE(DeviceOp::IsSupportedArgument(slice_arg)) << "slice rejected, m0=" << m0;
            DeviceOp::MakeInvoker().Run(slice_arg, StreamConfig{nullptr, false});
        }
        ck::hip_check_error(hipDeviceSynchronize());

        const auto exact = ck::profiler::gpu_verify<F16>(c_split.GetDeviceBuffer(),
                                                         c_slices.GetDeviceBuffer(),
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
        ck::DeviceMem c_ref(static_cast<std::size_t>(num_rows) * N * sizeof(F16));
        ck::DeviceMem rows_dev(rows.size() * sizeof(ck::index_t));
        rows_dev.ToDevice(rows.data());
        const dim3 block(256);
        const dim3 grid((N + block.x - 1) / block.x, num_rows);
        reference_rows_kernel<<<grid, block>>>(
            static_cast<const uint8_t*>(a.GetDeviceBuffer()),
            static_cast<const uint8_t*>(a_scale.GetDeviceBuffer()),
            static_cast<const uint8_t*>(b.GetDeviceBuffer()),
            static_cast<const uint8_t*>(b_scale.GetDeviceBuffer()),
            static_cast<const ck::index_t*>(rows_dev.GetDeviceBuffer()),
            static_cast<F16*>(c_ref.GetDeviceBuffer()),
            num_rows,
            N,
            K);
        ck::hip_check_error(hipGetLastError());

        unsigned long long counters[2] = {0, ~0ull};
        ck::DeviceMem counters_dev(sizeof(counters));
        counters_dev.ToDevice(counters);
        auto* counters_ptr = static_cast<unsigned long long*>(counters_dev.GetDeviceBuffer());
        compare_rows_kernel<<<grid, block>>>(
            static_cast<const F16*>(c_split.GetDeviceBuffer()),
            static_cast<const F16*>(c_ref.GetDeviceBuffer()),
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

// Customer minimum seq_len; no split.
TEST_F(TestGemmMXFp4LargeTensor, NoSplit_M6144) { Run(6144); }

// The fp16 C fits in 2GB (2147475456 bytes), but the last row tile, padded to 87424 rows,
// reaches past 2GB. Its masked stores must not wrap around onto the first rows of C.
TEST_F(TestGemmMXFp4LargeTensor, PaddedTilePast2GB_M87381) { Run(87381); }

// Same shape with a short K (the v3 pipeline needs more than 2 K loops), cheap enough for
// functional emulators.
TEST_F(TestGemmMXFp4LargeTensor, PaddedTilePast2GB_M87381_K1024) { Run(87381, 12288, 1024); }

// First M that splits on the fp16 C: two leaves, 43776 + 43606 (uneven tail).
TEST_F(TestGemmMXFp4LargeTensor, FirstSplit_M87382) { Run(87382); }

// Same split with a short K (the v3 pipeline needs more than 2 K loops), cheap enough for
// functional emulators.
TEST_F(TestGemmMXFp4LargeTensor, FirstSplit_M87382_K1024) { Run(87382, 12288, 1024); }

// Two even leaves of 65536.
TEST_F(TestGemmMXFp4LargeTensor, EvenSplit_M131072) { Run(131072); }

// Customer seq_len: 15 uneven leaves.
TEST_F(TestGemmMXFp4LargeTensor, Customer_M700000) { Run(700000); }

// Customer maximum seq_len: 32 leaves; the fp4 A is larger than 2GB as well.
TEST_F(TestGemmMXFp4LargeTensor, CustomerMax_M1600000) { Run(1600000); }

} // namespace
