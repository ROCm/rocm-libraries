// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Exercises HipdnnDynamicBackendWrapper through the runtime-load frontend backend
// factory. The test executable links hipdnn_frontend_dynamic, not
// libhipdnn_backend, so all backend entry points must be resolved through
// hipdnnBackend() / dlopen / dlsym.

#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>

#include <array>
#include <string_view>

using namespace hipdnn_frontend::detail;
using namespace hipdnn_data_sdk::utilities;
using namespace ::testing;

namespace
{
class TestHipdnnDynamicBackendWrapper : public Test
{
protected:
    void SetUp() override
    {
        _backend = hipdnnBackend();
        if(_backend->versionString()[0] == '\0')
        {
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
        }
    }

    HipdnnDynamicBackendWrapper makeWrapper() const
    {
        return HipdnnDynamicBackendWrapper(_backend->version());
    }

    std::shared_ptr<IHipdnnBackend> _backend;
};
} // namespace

TEST_F(TestHipdnnDynamicBackendWrapper, VersionReturnsConstructedVersion)
{
    const Version expected = _backend->version();
    HipdnnDynamicBackendWrapper wrapper(expected);
    EXPECT_EQ(wrapper.version(), expected);
}

TEST_F(TestHipdnnDynamicBackendWrapper, VersionStringResolvesFromBackend)
{
    auto wrapper = makeWrapper();
    EXPECT_STREQ(wrapper.versionString(), _backend->versionString());
}

TEST_F(TestHipdnnDynamicBackendWrapper, VersionEqualsVersionString)
{
    auto wrapper = makeWrapper();
    EXPECT_EQ(wrapper.version(), Version{std::string_view(wrapper.versionString())});
}

TEST_F(TestHipdnnDynamicBackendWrapper, BackendGetSerializedExecutionPlanExtForwardsToBackend)
{
    auto wrapper = makeWrapper();
    size_t planByteSize = 0;

    EXPECT_EQ(wrapper.backendGetSerializedExecutionPlanExt(nullptr, 0, &planByteSize, nullptr),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestHipdnnDynamicBackendWrapper,
       BackendCreateAndDeserializeExecutionPlanExtForwardsToBackend)
{
    auto wrapper = makeWrapper();
    hipdnnBackendDescriptor_t descriptor = nullptr;
    const std::array<uint8_t, 1> serializedPlan{0};

    EXPECT_EQ(wrapper.backendCreateAndDeserializeExecutionPlanExt(
                  nullptr, &descriptor, serializedPlan.data(), serializedPlan.size()),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// Repeated calls must reuse the cached function pointer and stay correct.
TEST_F(TestHipdnnDynamicBackendWrapper, RepeatedCallsUseCachedSymbol)
{
    auto wrapper = makeWrapper();
    size_t planByteSize = 0;

    for(int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(wrapper.backendGetSerializedExecutionPlanExt(nullptr, 0, &planByteSize, nullptr),
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
    }
}

TEST_F(TestHipdnnDynamicBackendWrapper, SetHeuristicPluginPathsExtForwardsToBackend)
{
    auto wrapper = makeWrapper();
    EXPECT_EQ(wrapper.setHeuristicPluginPathsExt(0, nullptr, HIPDNN_PLUGIN_LOADING_ADDITIVE),
              HIPDNN_STATUS_SUCCESS);
}
