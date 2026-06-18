// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "descriptors/ProfilingControlDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>

#include <string>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;

class TestProfilingControlDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<ProfilingControlDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<ProfilingControlDescriptor>();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<ProfilingControlDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

TEST_F(TestProfilingControlDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_PROFILING_CONTROL_EXT);
}

// Guards against a silent 1/0 render of the boolean state fields in toString():
// a freshly created descriptor has all state flags false, so each must render as
// the literal token "false" (never "0" / "1").
TEST_F(TestProfilingControlDescriptor, ToStringRendersBooleanTokens)
{
    const std::string str = getDescriptor()->toString();

    EXPECT_NE(str.find("eventsCreated=false"), std::string::npos);
    EXPECT_NE(str.find("startRecorded=false"), std::string::npos);
    EXPECT_NE(str.find("stopRecorded=false"), std::string::npos);
    EXPECT_NE(str.find("finalized=false"), std::string::npos);

    EXPECT_EQ(str.find("eventsCreated=0"), std::string::npos);
    EXPECT_EQ(str.find("eventsCreated=1"), std::string::npos);
}
