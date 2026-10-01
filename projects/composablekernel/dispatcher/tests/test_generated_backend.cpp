// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for the supports() gate of the generated GEMM backends.
/// Note: Uses a stub kernel struct; no kernel is compiled or launched.

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/dispatcher/backends/generated_kernel_backend.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend.hpp"
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

namespace {

// Fully padded fp16 kernel, as the codegen emits for every fixed-width (_vec) kernel.
struct PaddedStubKernel
{
    using ADataType             = ck_tile::half_t;
    using BDataType             = ck_tile::half_t;
    using CDataType             = ck_tile::half_t;
    using AccDataType           = float;
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
    static constexpr int TileM  = 128;
    static constexpr int TileN  = 128;
    static constexpr int TileK  = 32;
    static float launch(const ck_tile::GemmHostArgs&, const ck_tile::stream_config&) { return 0.f; }
};

using LegacyInstance = backends::GeneratedKernelInstance<PaddedStubKernel>;
using TileInstance   = backends::GeneratedTileKernelInstance<PaddedStubKernel,
                                                             ck_tile::half_t,
                                                             ck_tile::half_t,
                                                             ck_tile::half_t,
                                                             float>;

KernelKey make_vec_key()
{
    KernelKey key               = make_test_key(128, 128, 32, "gfx950");
    key.algorithm.vector_size_a = 1;
    key.algorithm.vector_size_b = 1;
    key.algorithm.vector_size_c = 8;
    return key;
}

// Padding must not hide the global vector widths: native fp16 rcr needs K % 8,
// the _vec1_1_8 kernel only needs N % 8.
template <typename Instance>
void expect_vector_gate()
{
    const Instance native(make_test_key(128, 128, 32, "gfx950"), "native");
    const Instance vec(make_vec_key(), "vec");

    EXPECT_TRUE(native.supports(Problem(512, 512, 512)));
    EXPECT_FALSE(native.supports(Problem(512, 512, 257)));
    EXPECT_TRUE(vec.supports(Problem(512, 512, 257)));
    EXPECT_FALSE(vec.supports(Problem(512, 129, 257)));
}

} // anonymous namespace

TEST(GeneratedBackendTest, GeneratedKernelInstanceGatesVectorWidths)
{
    expect_vector_gate<LegacyInstance>();
}

TEST(GeneratedBackendTest, GeneratedTileKernelInstanceGatesVectorWidths)
{
    expect_vector_gate<TileInstance>();
}

// The register_all_kernels.hpp wrappers use GeneratedKernelInstance; first-fit
// must skip the native kernel and pick the narrower one, or return none.
TEST(GeneratedBackendTest, FirstFitFallsThroughToNarrowerKernel)
{
    Registry registry;
    auto native = std::make_shared<LegacyInstance>(make_test_key(128, 128, 32, "gfx950"), "native");
    auto vec    = std::make_shared<LegacyInstance>(make_vec_key(), "vec");
    ASSERT_TRUE(registry.register_kernel(native));
    ASSERT_TRUE(registry.register_kernel(vec));

    Dispatcher dispatcher(&registry);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 512, 257)), vec);
    EXPECT_EQ(dispatcher.select_kernel(Problem(512, 129, 257)), nullptr);
}
