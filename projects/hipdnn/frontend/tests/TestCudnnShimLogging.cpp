// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <memory>
#include <string>

namespace
{
namespace sdk_logging = hipdnn_data_sdk::logging;
namespace sdk_utilities = hipdnn_data_sdk::utilities;
using hipdnn_frontend::detail::IHipdnnBackend;
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;
using testing::_;
using testing::Return;
using testing::StrictMock;

constexpr const char* CUDNN_LOG_INFO_ENV = "CUDNN_FRONTEND_LOG_INFO";
constexpr const char* CUDNN_LOG_FILE_ENV = "CUDNN_FRONTEND_LOG_FILE";
constexpr const char* CUDNN_DISABLE_LOGGING_ENV = "NV_CUDNN_FRONTEND_DISABLE_LOGGING";

class TestCudnnShimLogging : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _logInfoGuard = std::make_unique<ScopedEnvironmentVariableSetter>(CUDNN_LOG_INFO_ENV);
        _logFileGuard = std::make_unique<ScopedEnvironmentVariableSetter>(CUDNN_LOG_FILE_ENV);
        _disableLoggingGuard
            = std::make_unique<ScopedEnvironmentVariableSetter>(CUDNN_DISABLE_LOGGING_ENV);
        _hipdnnLogLevelGuard
            = std::make_unique<ScopedEnvironmentVariableSetter>("HIPDNN_LOG_LEVEL");
        _hipdnnLogFileGuard = std::make_unique<ScopedEnvironmentVariableSetter>("HIPDNN_LOG_FILE");

        clearLoggingEnv();
        sdk_logging::setLogLevel(HIPDNN_SEV_OFF);
        _mockBackend = std::make_shared<StrictMock<Mock_hipdnn_backend>>();
        IHipdnnBackend::setInstance(_mockBackend);
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
        clearLoggingEnv();
        sdk_logging::setLogLevel(HIPDNN_SEV_OFF);
        sdk_logging::resetLogLevelCache();
    }

    static void clearLoggingEnv()
    {
        sdk_utilities::unsetEnv(CUDNN_LOG_INFO_ENV);
        sdk_utilities::unsetEnv(CUDNN_LOG_FILE_ENV);
        sdk_utilities::unsetEnv(CUDNN_DISABLE_LOGGING_ENV);
        sdk_utilities::unsetEnv("HIPDNN_LOG_LEVEL");
        sdk_utilities::unsetEnv("HIPDNN_LOG_FILE");
    }

    void setCudnnLogInfo(const std::string& value)
    {
        _logInfoGuard->setValue(value);
    }

    void setCudnnLogFile(const std::string& value)
    {
        _logFileGuard->setValue(value);
    }

    void setCudnnDisableLogging(const std::string& value)
    {
        _disableLoggingGuard->setValue(value);
    }

    void expectCreateSucceeds()
    {
        auto fakeHandle = reinterpret_cast<cudnnHandle_t>(0x1234);
        EXPECT_CALL(*_mockBackend, create(_)).WillOnce([&fakeHandle](hipdnnHandle_t* out) {
            *out = fakeHandle;
            return HIPDNN_STATUS_SUCCESS;
        });

        cudnnHandle_t handle = nullptr;
        EXPECT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(handle, fakeHandle);
    }

    std::shared_ptr<StrictMock<Mock_hipdnn_backend>> _mockBackend;

private:
    std::unique_ptr<ScopedEnvironmentVariableSetter> _logInfoGuard;
    std::unique_ptr<ScopedEnvironmentVariableSetter> _logFileGuard;
    std::unique_ptr<ScopedEnvironmentVariableSetter> _disableLoggingGuard;
    std::unique_ptr<ScopedEnvironmentVariableSetter> _hipdnnLogLevelGuard;
    std::unique_ptr<ScopedEnvironmentVariableSetter> _hipdnnLogFileGuard;
};

TEST_F(TestCudnnShimLogging, CudnnCreateLeavesHipdnnLoggingUntouchedWithoutCudnnEnv)
{
    EXPECT_CALL(*_mockBackend, backendSetGlobalLogLevelExt(_)).Times(0);

    expectCreateSucceeds();

    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_LEVEL"), "");
    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_FILE"), "");
    EXPECT_EQ(sdk_logging::getLogLevel(), HIPDNN_SEV_OFF);
}

TEST_F(TestCudnnShimLogging, CudnnCreateDisablesHipdnnLoggingWithoutTarget)
{
    setCudnnLogInfo("1");
    EXPECT_CALL(*_mockBackend, backendSetGlobalLogLevelExt(HIPDNN_SEV_OFF))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    expectCreateSucceeds();

    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_LEVEL"), "off");
    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_FILE"), "");
    EXPECT_EQ(sdk_logging::getLogLevel(), HIPDNN_SEV_OFF);
}

TEST_F(TestCudnnShimLogging, CudnnCreateMapsFileTargetToHipdnnLoggingEnv)
{
    setCudnnLogInfo("1");
    setCudnnLogFile("shim.log");
    EXPECT_CALL(*_mockBackend, backendSetGlobalLogLevelExt(HIPDNN_SEV_INFO))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    expectCreateSucceeds();

    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_LEVEL"), "info");
    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_FILE"), "shim.log");
    EXPECT_EQ(sdk_logging::getLogLevel(), HIPDNN_SEV_INFO);
}

TEST_F(TestCudnnShimLogging, CudnnCreateMapsStdTargetsToHipdnnConsoleLogging)
{
    setCudnnLogInfo("10");
    setCudnnLogFile("stderr");
    EXPECT_CALL(*_mockBackend, backendSetGlobalLogLevelExt(HIPDNN_SEV_INFO))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    expectCreateSucceeds();

    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_LEVEL"), "info");
    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_FILE"), "");
    EXPECT_EQ(sdk_logging::getLogLevel(), HIPDNN_SEV_INFO);
}

TEST_F(TestCudnnShimLogging, CudnnCreateDisableEnvOverridesPositiveLevelAndTarget)
{
    setCudnnLogInfo("1");
    setCudnnLogFile("shim.log");
    setCudnnDisableLogging("1");
    EXPECT_CALL(*_mockBackend, backendSetGlobalLogLevelExt(HIPDNN_SEV_OFF))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    expectCreateSucceeds();

    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_LEVEL"), "off");
    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_FILE"), "");
    EXPECT_EQ(sdk_logging::getLogLevel(), HIPDNN_SEV_OFF);
}

TEST_F(TestCudnnShimLogging, CudnnCreateDisableEnvZeroDoesNotOverride)
{
    setCudnnLogInfo("1");
    setCudnnLogFile("shim.log");
    setCudnnDisableLogging("0");
    EXPECT_CALL(*_mockBackend, backendSetGlobalLogLevelExt(HIPDNN_SEV_INFO))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    expectCreateSucceeds();

    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_LEVEL"), "info");
    EXPECT_EQ(sdk_utilities::getEnv("HIPDNN_LOG_FILE"), "shim.log");
    EXPECT_EQ(sdk_logging::getLogLevel(), HIPDNN_SEV_INFO);
}

} // namespace
