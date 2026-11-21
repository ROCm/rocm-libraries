// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

// cast_tile of a float tile whose per-thread buffer is a multiple of 4 converts with the packed
// fp8/bf8 instructions; for bf8_t it used the fp8 (E4M3) one, so bf8 tiles held fp8 codes
// (https://github.com/ROCm/rocm-libraries/issues/15786). The conversions are compared on the
// device, where the fp8 interpretation (OCP or FNUZ) is the target's.

namespace {

constexpr int kThreads   = 64;
constexpr int kPerThread = 8; // two packed groups of four
constexpr int kElements  = kThreads * kPerThread;

// kPerThread floats per thread.
using Encoding =
    ck_tile::tile_distribution_encoding<ck_tile::sequence<>,
                                        ck_tile::tuple<ck_tile::sequence<kThreads, kPerThread>>,
                                        ck_tile::tuple<ck_tile::sequence<1>>,
                                        ck_tile::tuple<ck_tile::sequence<0>>,
                                        ck_tile::sequence<1>,
                                        ck_tile::sequence<1>>;

// For each element: the code cast_tile gives, the code type_convert gives, and cast_tile's code
// converted back to float.
template <typename F8>
__global__ void
cast_tile_kernel(const float* x, uint8_t* tile_code, uint8_t* scalar_code, float* back)
{
    const int base = static_cast<int>(threadIdx.x) * kPerThread;

    auto tile = ck_tile::make_static_distributed_tensor<float>(
        ck_tile::make_static_tile_distribution(Encoding{}));
    static_assert(decltype(tile)::get_thread_buffer_size() == kPerThread);
    ck_tile::static_for<0, kPerThread, 1>{}(
        [&](auto j) { tile.get_thread_buffer()(j) = x[base + j]; });

    const auto out = ck_tile::cast_tile<F8>(tile);

    ck_tile::static_for<0, kPerThread, 1>{}([&](auto j) {
        const F8 code         = out.get_thread_buffer()[j];
        tile_code[base + j]   = ck_tile::bit_cast<uint8_t>(code);
        scalar_code[base + j] = ck_tile::bit_cast<uint8_t>(ck_tile::type_convert<F8>(x[base + j]));
        back[base + j]        = ck_tile::type_convert<float>(code);
    });
}

// Normal values inside the range of every fp8 and bf8 format (OCP and FNUZ), and zero.
std::vector<float> Inputs()
{
    const float fractions[] = {0.0f, 0.1f, 0.25f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f};
    std::vector<float> x;
    for(int i = 0; static_cast<int>(x.size()) < kElements; ++i)
    {
        const int exponent   = -6 + (i / 8) % 13; // 2^-6 .. 2^6
        const float fraction = fractions[i % 8];
        const float sign     = (i / 104) % 2 == 0 ? 1.0f : -1.0f;
        x.push_back(i == 0 ? 0.0f : sign * std::ldexp(1.0f + fraction, exponent));
    }
    return x;
}

// relative_error: half an ulp relative to the value, 2^-4 for fp8 (3 mantissa bits) and 2^-3
// for bf8 (2 mantissa bits).
template <typename F8>
void CheckCastTile(float relative_error)
{
    const std::vector<float> x = Inputs();
    std::vector<uint8_t> tile_code(kElements), scalar_code(kElements);
    std::vector<float> back(kElements);

    ck_tile::DeviceMem x_buf(kElements * sizeof(float));
    ck_tile::DeviceMem tile_code_buf(kElements);
    ck_tile::DeviceMem scalar_code_buf(kElements);
    ck_tile::DeviceMem back_buf(kElements * sizeof(float));
    x_buf.ToDevice(x.data());

    cast_tile_kernel<F8><<<1, kThreads>>>(static_cast<const float*>(x_buf.GetDeviceBuffer()),
                                          static_cast<uint8_t*>(tile_code_buf.GetDeviceBuffer()),
                                          static_cast<uint8_t*>(scalar_code_buf.GetDeviceBuffer()),
                                          static_cast<float*>(back_buf.GetDeviceBuffer()));
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    tile_code_buf.FromDevice(tile_code.data());
    scalar_code_buf.FromDevice(scalar_code.data());
    back_buf.FromDevice(back.data());

    for(int i = 0; i < kElements; ++i)
    {
        EXPECT_EQ(tile_code[i], scalar_code[i]) << "x = " << x[i];
        EXPECT_NEAR(back[i], x[i], relative_error * std::abs(x[i])) << "x = " << x[i];
    }
}

} // namespace

TEST(CastTileF8, Fp8) { CheckCastTile<ck_tile::fp8_t>(1.0f / 16); }

TEST(CastTileF8, Bf8) { CheckCastTile<ck_tile::bf8_t>(1.0f / 8); }
