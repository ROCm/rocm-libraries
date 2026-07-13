// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// (RFC 0012 §7.1 Phase-1 exit criterion): a hand-written
// v9-only translation unit that includes ONLY the cuDNN-compatibility shim
// umbrella header compiles and links, using the FE-namespace enums, the error
// types, and the §4.7 C-API types it needs. This is the source-compatibility
// proof for the enum/error/macro surface delivered by this story; the graph
// and attribute wrappers arrive in Phase 2. Gated behind
// HIPDNN_ENABLE_CUDNN_COMPATIBILITY in the frontend tests CMakeLists.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

// Mirror the consumer hipification step (RFC 0012 §4.2 workflow 1): alias the
// namespace so in-code `cudnn_frontend::` symbols resolve to the shim.
namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

// FE-namespace enums resolve through the shim aliases and are the *same* enum
// as the hipDNN type (aliased, not re-declared).
static_assert(std::is_same_v<cudnn_frontend::DataType_t, hipdnn_frontend::DataType_t>,
              "cudnn_frontend::DataType_t must alias the hipDNN enum");

namespace
{
constexpr cudnn_frontend::DataType_t K_IO_TYPE = cudnn_frontend::DataType_t::HALF;
constexpr cudnn_frontend::DataType_t K_COMPUTE_TYPE = cudnn_frontend::DataType_t::FLOAT;
constexpr cudnn_frontend::HeurMode_t K_HEUR_MODE = cudnn_frontend::HeurMode_t::A;

// error_t / error_code_t resolve through the shim aliases.
cudnn_frontend::error_t makeOk()
{
    return {};
}

// §4.7 C-API types come from the stub <cudnn.h> pulled in by the umbrella.
cudnnStatus_t statusFor(cudnnHandle_t /*handle*/)
{
    return CUDNN_STATUS_SUCCESS;
}

TEST(TestCudnnShimV9Tu, V9OnlyTranslationUnitCompilesAndLinks)
{
    // The graph sub-namespace resolves even though it is empty at this phase
    // (this using-directive would fail to compile if the namespace did not
    // exist; Phase 2 will populate it with the Graph/attribute wrappers).
    using namespace cudnn_frontend::graph;

    EXPECT_TRUE(makeOk().is_good());
    EXPECT_EQ(K_IO_TYPE, cudnn_frontend::DataType_t::HALF);
    EXPECT_EQ(K_COMPUTE_TYPE, cudnn_frontend::DataType_t::FLOAT);
    EXPECT_EQ(K_HEUR_MODE, cudnn_frontend::HeurMode_t::A);
    EXPECT_EQ(statusFor(nullptr), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(CUDNN_FRONTEND_VERSION, 12400);

    SUCCEED() << "v9-only TU compiled and linked against the shim umbrella.";
}

} // namespace
