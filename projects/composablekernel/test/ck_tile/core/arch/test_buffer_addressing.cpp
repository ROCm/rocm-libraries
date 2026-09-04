// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

// Covers cast_to_amdgpu_buffer_rsrc_t(), which reinterprets the 128-bit buffer
// descriptor ck_tile builds by hand as the opaque __amdgpu_buffer_rsrc_t that the
// raw buffer builtins take.
//
// The conversion has to be an exact bit copy: base address in dwords 0-1,
// num_records in dword 2, config in dword 3. Reading one element the descriptor
// covers and one it does not exercises both halves of that -- an out-of-range
// read only returns zero if num_records survived the copy intact, and an
// in-range read only returns the right value if the base address did.
//
// The function is defined twice, in two mutually exclusive headers:
// amd_buffer_addressing_builtins.hpp is compiled when
// CK_TILE_USE_BUFFER_ADDRESSING_BUILTIN is 1 and amd_buffer_addressing.hpp when
// it is 0, so a single translation unit can only ever reach one of them. CMake
// therefore builds this file twice, once per setting of that macro.

namespace {

constexpr int kNumElements = 64;

// num_records is set to cover only the first 48 floats, so the last 16 lie
// outside the descriptor while still being backed by real memory
constexpr int kVisibleElements = 48;

__global__ void kernel_buffer_rsrc_roundtrip(const float* src, float* dst)
{
    const auto i      = static_cast<ck_tile::index_t>(threadIdx.x);
    const auto offset = i * static_cast<ck_tile::index_t>(sizeof(float));

    const auto src_rsrc = ck_tile::make_wave_buffer_resource(
        src, static_cast<uint32_t>(kVisibleElements * sizeof(float)));
    const auto dst_rsrc = ck_tile::make_wave_buffer_resource(
        dst, static_cast<uint32_t>(kNumElements * sizeof(float)));

    // buffer_load/buffer_store are the raw path; where the buffer builtins are
    // available they are the callers of cast_to_amdgpu_buffer_rsrc_t
    float value = -1.0f;
    ck_tile::buffer_load<4>{}(value, src_rsrc, offset, 0, 0);
    ck_tile::buffer_store<4>{}(value, dst_rsrc, offset, 0, 0);
}

} // namespace

TEST(BufferAddressing, BufferResourceRoundTrip)
{
    std::vector<float> host_src(kNumElements);
    for(int i = 0; i < kNumElements; ++i)
    {
        host_src[i] = static_cast<float>(i) + 0.5f;
    }

    ck_tile::DeviceMem src_buf(kNumElements * sizeof(float));
    ck_tile::DeviceMem dst_buf(kNumElements * sizeof(float));
    src_buf.ToDevice(host_src.data());
    dst_buf.SetZero();

    kernel_buffer_rsrc_roundtrip<<<1, kNumElements>>>(
        static_cast<const float*>(src_buf.GetDeviceBuffer()),
        static_cast<float*>(dst_buf.GetDeviceBuffer()));
    ck_tile::hip_check_error(hipDeviceSynchronize());

    std::vector<float> host_dst(kNumElements);
    dst_buf.FromDevice(host_dst.data());

    // base address and config dwords survived: in-range elements round-trip
    for(int i = 0; i < kVisibleElements; ++i)
    {
        EXPECT_FLOAT_EQ(host_dst[i], host_src[i]) << "element " << i;
    }

    // num_records dword survived: accesses past it are dropped by the hardware,
    // so the loads return zero instead of what is actually in memory there
    for(int i = kVisibleElements; i < kNumElements; ++i)
    {
        EXPECT_FLOAT_EQ(host_dst[i], 0.0f) << "out-of-range element " << i;
    }
}
