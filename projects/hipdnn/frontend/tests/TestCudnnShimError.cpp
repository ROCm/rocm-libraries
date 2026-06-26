// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// error_object construction, code mapping, and message
// preservation through the aliased hipdnn_frontend::Error, plus the re-exported
// error macros. The shim aliases the hipDNN error types 1:1 (no wrapper), so
// these assertions also pin that the alias is exact. Gated behind
// HIPDNN_ENABLE_CUDNN_COMPATIBILITY in the frontend tests CMakeLists.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace cfe = hipdnn_frontend::compatibility::cudnn_frontend;

// The error types must be the *same* types as hipDNN's, not parallel
// re-declarations (RFC 0012 §4.6 — alias, do not wrap).
static_assert(std::is_same_v<cfe::error_t, hipdnn_frontend::error_t>,
              "cudnn_frontend::error_t must alias hipdnn_frontend::error_t");
static_assert(std::is_same_v<cfe::error_object, hipdnn_frontend::error_object>,
              "cudnn_frontend::error_object must alias hipdnn_frontend::error_object");
static_assert(std::is_same_v<cfe::error_code_t, hipdnn_frontend::error_code_t>,
              "cudnn_frontend::error_code_t must alias hipdnn_frontend::error_code_t");

namespace
{
// Mirror a hipified consumer: bring the shim names into scope so the bare
// macro identifiers (error_code_t, error_t) resolve as they do upstream.
using namespace hipdnn_frontend::compatibility::cudnn_frontend;

TEST(TestCudnnShimError, DefaultObjectIsOkAndGood)
{
    const error_t e;
    EXPECT_TRUE(e.is_good());
    EXPECT_FALSE(e.is_bad());
    EXPECT_EQ(e.get_code(), error_code_t::OK);
    EXPECT_TRUE(e == error_code_t::OK);
}

TEST(TestCudnnShimError, PreservesCodeAndMessage)
{
    const error_object e{error_code_t::GRAPH_NOT_SUPPORTED, "no engine accepted graph"};
    EXPECT_TRUE(e.is_bad());
    EXPECT_FALSE(e.is_good());
    EXPECT_EQ(e.get_code(), error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_EQ(e.get_message(), "no engine accepted graph");
    EXPECT_TRUE(e == error_code_t::GRAPH_NOT_SUPPORTED);
    EXPECT_TRUE(e != error_code_t::OK);
}

// Because the shim error type *is* hipDNN's Error, a value produced on one side
// is read on the other with no conversion.
TEST(TestCudnnShimError, AliasInteroperatesWithHipdnnError)
{
    const hipdnn_frontend::error_t native{error_code_t::INVALID_VALUE, "bad value"};
    // Same type on both sides — a cudnn_frontend::error_t reference binds to a
    // hipdnn_frontend::error_t with no conversion.
    const error_t& aliased = native;
    EXPECT_EQ(aliased.get_code(), error_code_t::INVALID_VALUE);
    EXPECT_EQ(aliased.get_message(), "bad value");
}

// Helpers exercising the re-exported error macros.
error_t failsIfNegative(int x)
{
    RETURN_CUDNN_FRONTEND_ERROR_IF(x < 0, error_code_t::INVALID_VALUE, "x must be >= 0");
    return {};
}

error_t checkThenSucceed(int x)
{
    CHECK_CUDNN_FRONTEND_ERROR(failsIfNegative(x));
    return {};
}

TEST(TestCudnnShimError, ReturnErrorIfMacro)
{
    auto bad = failsIfNegative(-1);
    EXPECT_TRUE(bad.is_bad());
    EXPECT_EQ(bad.get_code(), error_code_t::INVALID_VALUE);
    EXPECT_EQ(bad.get_message(), "x must be >= 0");

    EXPECT_TRUE(failsIfNegative(5).is_good());
}

TEST(TestCudnnShimError, CheckErrorMacroPropagates)
{
    auto bad = checkThenSucceed(-1);
    EXPECT_TRUE(bad.is_bad());
    EXPECT_EQ(bad.get_code(), error_code_t::INVALID_VALUE);

    EXPECT_TRUE(checkThenSucceed(2).is_good());
}

} // namespace
