// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Exercises HipdnnDynamicBackendWrapper by instantiating it directly. The test
// executable links the real backend, so the wrapper resolves entry points from
// the already-loaded libhipdnn_backend at runtime via dlopen/dlsym. This keeps
// the dynamic resolution path covered even in the default (direct-link) build.

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>

#include <array>
#include <string_view>

using namespace hipdnn_frontend::detail;
using namespace hipdnn_data_sdk::utilities;
using namespace ::testing;

namespace
{
HipdnnDynamicBackendWrapper makeWrapper()
{
    return HipdnnDynamicBackendWrapper(Version{std::string_view(hipdnnVersionString_ext())});
}
} // namespace

TEST(TestHipdnnDynamicBackendWrapper, VersionReturnsConstructedVersion)
{
    const Version expected{std::string_view(hipdnnVersionString_ext())};
    HipdnnDynamicBackendWrapper wrapper(expected);
    EXPECT_EQ(wrapper.version(), expected);
}

TEST(TestHipdnnDynamicBackendWrapper, VersionStringResolvesFromBackend)
{
    auto wrapper = makeWrapper();
    EXPECT_STREQ(wrapper.versionString(), hipdnnVersionString_ext());
}

TEST(TestHipdnnDynamicBackendWrapper, VersionEqualsVersionString)
{
    auto wrapper = makeWrapper();
    EXPECT_EQ(wrapper.version(), Version{std::string_view(wrapper.versionString())});
}

TEST(TestHipdnnDynamicBackendWrapper, BackendGetSerializedExecutionPlanExtForwardsToBackend)
{
    auto wrapper = makeWrapper();
    size_t planByteSize = 0;

    EXPECT_EQ(wrapper.backendGetSerializedExecutionPlanExt(nullptr, 0, &planByteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestHipdnnDynamicBackendWrapper, BackendCreateAndDeserializeExecutionPlanExtForwardsToBackend)
{
    auto wrapper = makeWrapper();
    hipdnnBackendDescriptor_t descriptor = nullptr;
    const std::array<uint8_t, 1> serializedPlan{0};

    EXPECT_EQ(wrapper.backendCreateAndDeserializeExecutionPlanExt(
                  nullptr, &descriptor, serializedPlan.data(), serializedPlan.size()),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// Repeated calls must reuse the cached function pointer and stay correct.
TEST(TestHipdnnDynamicBackendWrapper, RepeatedCallsUseCachedSymbol)
{
    auto wrapper = makeWrapper();
    size_t planByteSize = 0;

    for(int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(wrapper.backendGetSerializedExecutionPlanExt(nullptr, 0, &planByteSize, nullptr),
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    }
}

TEST(TestHipdnnDynamicBackendWrapper, SetHeuristicPluginPathsExtForwardsToBackend)
{
    auto wrapper = makeWrapper();
    EXPECT_EQ(wrapper.setHeuristicPluginPathsExt(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_SUCCESS);
}
