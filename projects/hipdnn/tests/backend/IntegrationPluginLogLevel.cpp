// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <test_plugins/TestPluginConstants.hpp>

class IntegrationPluginLogLevel : public ::testing::Test
{
private:
    hipdnnStatus_t setIsolatedRecorderAsUserCallback(hipdnnSeverity_t minLevel)
    {
        return hipdnnSetUserLogCallback_ext(
            hipdnn_test_sdk::utilities::IsolatedLogRecorder::getIsolatedUserRecordingCallback(),
            minLevel,
            HIPDNN_LOG_CALLBACK_SYNC,
            this);
    }

protected:
    hipdnnHandle_t _handle = nullptr;
    hipdnnSeverity_t _originalLevel{HIPDNN_SEV_OFF};

    void SetUp() override
    {
        ASSERT_EQ(hipdnnBackendGetGlobalLogLevel_ext(&_originalLevel), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        // Clear any user log callback registered by tests
        setIsolatedRecorderAsUserCallback(HIPDNN_SEV_OFF);

        if(_handle != nullptr)
        {
            EXPECT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
            _handle = nullptr;
        }

        ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(_originalLevel), HIPDNN_STATUS_SUCCESS);
    }

    void registerIsolatedRecorderAsUserCallback(hipdnnSeverity_t minLevel)
    {
        ASSERT_EQ(setIsolatedRecorderAsUserCallback(minLevel), HIPDNN_STATUS_SUCCESS);
    }

    void loadTestGoodPlugin()
    {
        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }
};

TEST_F(IntegrationPluginLogLevel, PluginCreatesInfoLogs)
{
    registerIsolatedRecorderAsUserCallback(HIPDNN_SEV_INFO);

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    auto recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    loadTestGoodPlugin();

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "pluginSetLogLevel level=0"))
        << "Expected plugin to create INFO-level logs. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
}

TEST_F(IntegrationPluginLogLevel, PluginReceivesWarnLevelNotification)
{
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    auto recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    loadTestGoodPlugin();

    registerIsolatedRecorderAsUserCallback(HIPDNN_SEV_INFO);

    // Change level to WARN — plugin should log "pluginSetLogLevel" at WARN severity
    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_WARN), HIPDNN_STATUS_SUCCESS);

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "pluginSetLogLevel level=1"))
        << "Expected plugin to create WARN-level logs. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "pluginSetLogLevel"))
        << "Expected plugin to NOT create INFO-level logs. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
}

TEST_F(IntegrationPluginLogLevel, PluginDoesNotCreateLogsWhenLogLevelOff)
{
    registerIsolatedRecorderAsUserCallback(HIPDNN_SEV_INFO);

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_OFF), HIPDNN_STATUS_SUCCESS);

    auto recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    loadTestGoodPlugin();

    EXPECT_EQ(recorder.getRecordedLogCount(), 0)
        << "Expected plugin to NOT create any logs when level is OFF. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
}

TEST_F(IntegrationPluginLogLevel, WarnThenInfoProducesLogs)
{
    registerIsolatedRecorderAsUserCallback(HIPDNN_SEV_INFO);

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_WARN), HIPDNN_STATUS_SUCCESS);

    auto recorder
        = hipdnn_test_sdk::utilities::IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);

    loadTestGoodPlugin();

    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, ""))
        << "Expected plugin to NOT create INFO-level logs. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

    ASSERT_EQ(hipdnnBackendSetGlobalLogLevel_ext(HIPDNN_SEV_INFO), HIPDNN_STATUS_SUCCESS);

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "pluginSetLogLevel level=0"))
        << "Expected plugin to create INFO-level logs after changing to INFO. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
}
