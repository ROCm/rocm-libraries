// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>

#include <memory>
#include <string_view>

using namespace hipdnn_frontend::detail;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class TestRuntimeLoadBackendFactory : public testing::Test
{
protected:
    void SetUp() override
    {
#ifdef HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
        ASSERT_NE(backendLibraryHandle(), nullptr);
#else
        if(backendLibraryHandle() == nullptr)
        {
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
        }
#endif

        IHipdnnBackend::resetInstance();
        _backend = hipdnnBackend();
        if(_backend->versionString()[0] == '\0')
        {
#ifdef HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
            FAIL() << "hipDNN backend library was found, but runtime symbol loading failed";
#else
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
#endif
        }
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
    }

    std::shared_ptr<IHipdnnBackend> _backend;
};

} // namespace

TEST_F(TestRuntimeLoadBackendFactory, TryToUseDynamicBackendInterfaceCreatesDynamicWrapper)
{
    EXPECT_TRUE(
        std::dynamic_pointer_cast<HipdnnDynamicBackendWrapper>(tryToUseDynamicBackendInterface()));
}

TEST_F(TestRuntimeLoadBackendFactory, HipdnnBackendCreatesDynamicWrapper)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<HipdnnDynamicBackendWrapper>(_backend));
}

TEST_F(TestRuntimeLoadBackendFactory, HipdnnBackendUsesBackendVersion)
{
    EXPECT_EQ(_backend->version(), Version{std::string_view(_backend->versionString())});
}
