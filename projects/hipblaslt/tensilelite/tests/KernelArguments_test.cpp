// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// External-pointer mode (grouped GEMM) hands KernelArguments a caller-owned
// buffer sized to exactly the argument bytes. An append ending exactly at the end
// of that buffer must be accepted, and one past it must be refused.

#include <gtest/gtest.h>

#include <Tensile/KernelArguments.hpp>

#include <cstdint>
#include <cstring>
#include <vector>

using TensileLite::KernelArguments;

TEST(KernelArguments, ExternalBufferExactFit)
{
    std::vector<uint8_t> buf(16, 0xFF);
    KernelArguments      args(false);
    args.useExternalPointer(buf.data(), buf.size());

    args.append<uint64_t>("a", 0x1111111111111111ull);
    args.append<uint64_t>("b", 0x2222222222222222ull);

    EXPECT_EQ(args.size(), 16u);
    uint64_t b;
    std::memcpy(&b, buf.data() + 8, sizeof(b));
    EXPECT_EQ(b, 0x2222222222222222ull);
}

TEST(KernelArguments, ExternalBufferOverflowThrows)
{
    std::vector<uint8_t> buf(12);
    KernelArguments      args(false);
    args.useExternalPointer(buf.data(), buf.size());

    args.append<uint64_t>("a", 1);
    EXPECT_THROW(args.append<uint64_t>("b", 2), std::runtime_error);
    EXPECT_THROW(args.appendPadding(5), std::runtime_error);
}

TEST(KernelArguments, ExternalBufferTrailingPaddingExactFit)
{
    std::vector<uint8_t> buf(12);
    KernelArguments      args(false);
    args.useExternalPointer(buf.data(), buf.size());

    args.append<uint64_t>("a", 1);
    args.appendPadding(4);
    EXPECT_EQ(args.size(), 12u);
}
