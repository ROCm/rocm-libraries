// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <array>
#include <sstream>

using namespace hipdnn_test_sdk::utilities;

class TestLogRecorder : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear any previous logs and stop recording (use SHARED for infrastructure tests)
        LogRecording::instance(LogRecording::Id::SHARED).stopRecording();
        LogRecording::instance(LogRecording::Id::SHARED).clearLogs();
    }

    void TearDown() override
    {
        // Clean up after each test
        LogRecording::instance(LogRecording::Id::SHARED).stopRecording();
        LogRecording::instance(LogRecording::Id::SHARED).clearLogs();
    }
};

// === Basic Recording Control ===

TEST_F(TestLogRecorder, RecordingStartsOff)
{
    EXPECT_FALSE(LogRecording::instance(LogRecording::Id::SHARED).isRecording());
}

TEST_F(TestLogRecorder, StartRecordingEnablesCapture)
{
    LogRecording::instance(LogRecording::Id::SHARED).startRecording();
    EXPECT_TRUE(LogRecording::instance(LogRecording::Id::SHARED).isRecording());
}

TEST_F(TestLogRecorder, StopRecordingDisablesCapture)
{
    LogRecording::instance(LogRecording::Id::SHARED).startRecording();
    LogRecording::instance(LogRecording::Id::SHARED).stopRecording();
    EXPECT_FALSE(LogRecording::instance(LogRecording::Id::SHARED).isRecording());
}

TEST_F(TestLogRecorder, RecordLogOnlyWhenRecording)
{
    // Recording is off - should not capture
    LogRecording::instance(LogRecording::Id::SHARED)
        .recordLog(HIPDNN_SEV_INFO, "should not capture");
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 0);

    // Enable recording - should capture
    LogRecording::instance(LogRecording::Id::SHARED).startRecording();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "should capture");
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 1);
}

TEST_F(TestLogRecorder, ClearLogsEmptiesBuffer)
{
    LogRecording::instance(LogRecording::Id::SHARED).startRecording();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "test log 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "test log 2");
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 2);

    LogRecording::instance(LogRecording::Id::SHARED).clearLogs();
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 0);
}

// === RAII Log Level Management ===

TEST_F(TestLogRecorder, WithOverrideLevelChangesLevel)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    {
        auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
        EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);
    }
}

TEST_F(TestLogRecorder, WithOverrideLevelRestoresOriginal)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);

    {
        auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
        EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);
    }

    // Should restore to WARN
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
}

TEST_F(TestLogRecorder, WithCurrentLevelPreservesLevel)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);

    {
        auto recorder = SharedLogRecorder::withCurrentLevel();
        EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
    }

    // Should still be WARN
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
}

TEST_F(TestLogRecorder, WithCurrentLevelRestoresOnDestroy)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    {
        auto recorder = SharedLogRecorder::withCurrentLevel();
        // Manually change level during recording
        hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_ERROR);
    }

    // Should restore to original level (INFO)
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_INFO);
}

TEST_F(TestLogRecorder, GetSavedLogLevelReturnsOriginal)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);

    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
    EXPECT_EQ(recorder.getSavedLogLevel(), HIPDNN_SEV_WARN);
}

// === Query Methods ===

TEST_F(TestLogRecorder, HasLogContainingFindsSubstring)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED)
        .recordLog(HIPDNN_SEV_INFO, "This is a test message");

    EXPECT_TRUE(recorder.hasLogContaining("test message"));
    EXPECT_TRUE(recorder.hasLogContaining("This is"));
    EXPECT_FALSE(recorder.hasLogContaining("message not present"));
}

TEST_F(TestLogRecorder, HasLogContainingWithSeverityFilters)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "info message");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "warn message");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_ERROR, "error message");

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "info message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "warn message"));
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "error message"));

    // Wrong severity should not match
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_ERROR, "info message"));
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_INFO, "warn message"));
}

TEST_F(TestLogRecorder, CountLogsAtLevelReturnsAccurateCount)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "info 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "info 2");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "warn 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_ERROR, "error 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "info 3");

    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_INFO), 3);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_WARN), 1);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 1);
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_FATAL), 0);
}

TEST_F(TestLogRecorder, GetRecordedLogCountReturnsTotal)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    EXPECT_EQ(recorder.getRecordedLogCount(), 0);

    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 1");
    EXPECT_EQ(recorder.getRecordedLogCount(), 1);

    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "log 2");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_ERROR, "log 3");
    EXPECT_EQ(recorder.getRecordedLogCount(), 3);
}

TEST_F(TestLogRecorder, GetRecordedLogsReturnsVector)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "message 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "message 2");

    auto logs = recorder.getRecordedLogs();
    ASSERT_EQ(logs.size(), 2);
    EXPECT_EQ(logs[0].severity, HIPDNN_SEV_INFO);
    EXPECT_EQ(logs[0].message, "message 1");
    EXPECT_EQ(logs[1].severity, HIPDNN_SEV_WARN);
    EXPECT_EQ(logs[1].message, "message 2");
}

TEST_F(TestLogRecorder, SharedRecorderClearLogsEmptiesBuffer)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "log 2");
    EXPECT_EQ(recorder.getRecordedLogCount(), 2);

    recorder.clearLogs();

    EXPECT_EQ(recorder.getRecordedLogCount(), 0);
    EXPECT_FALSE(recorder.hasLogContaining("log 1"));
    EXPECT_FALSE(recorder.hasLogContaining("log 2"));
}

TEST_F(TestLogRecorder, SharedRecorderClearLogsAllowsNewRecording)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "before clear");
    recorder.clearLogs();

    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "after clear");

    EXPECT_EQ(recorder.getRecordedLogCount(), 1);
    EXPECT_FALSE(recorder.hasLogContaining("before clear"));
    EXPECT_TRUE(recorder.hasLogContaining("after clear"));
}

// === getRecordedLogsAsString() ===

TEST_F(TestLogRecorder, GetRecordedLogsAsStringShowsEmptyMessage)
{
    // No logs recorded
    auto recorder = SharedLogRecorder::withCurrentLevel();
    EXPECT_EQ(recorder.getRecordedLogsAsString(), "(No logs captured.)\n");
}

TEST_F(TestLogRecorder, GetRecordedLogsAsStringFormatsLogs)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "info message");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "warn message");

    const std::string output = recorder.getRecordedLogsAsString();
    EXPECT_NE(output.find("[info] info message"), std::string::npos);
    EXPECT_NE(output.find("[warn] warn message"), std::string::npos);
}

TEST_F(TestLogRecorder, GetRecordedLogsAsStringWithMaxLogsLimitsOutput)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 2");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 3");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 4");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 5");

    const std::string output = recorder.getRecordedLogsAsString(3);

    // Should contain first 3 logs
    EXPECT_NE(output.find("log 1"), std::string::npos);
    EXPECT_NE(output.find("log 2"), std::string::npos);
    EXPECT_NE(output.find("log 3"), std::string::npos);

    // Should NOT contain logs 4 and 5
    EXPECT_EQ(output.find("log 4"), std::string::npos);
    EXPECT_EQ(output.find("log 5"), std::string::npos);

    // Should contain skip message
    EXPECT_NE(output.find("(Skipped 2 additional logs.)"), std::string::npos);
}

TEST_F(TestLogRecorder, GetRecordedLogsAsStringSkipMessageSingular)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 2");

    const std::string output = recorder.getRecordedLogsAsString(1);

    // Should say "log" (singular) not "logs"
    EXPECT_NE(output.find("(Skipped 1 additional log.)"), std::string::npos);
}

TEST_F(TestLogRecorder, GetRecordedLogsAsStringNoLimitShowsAll)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 2");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 3");

    const std::string output = recorder.getRecordedLogsAsString(0); // 0 = no limit

    // Should contain all logs
    EXPECT_NE(output.find("log 1"), std::string::npos);
    EXPECT_NE(output.find("log 2"), std::string::npos);
    EXPECT_NE(output.find("log 3"), std::string::npos);

    // Should NOT contain skip message
    EXPECT_EQ(output.find("Skipped"), std::string::npos);
}

TEST_F(TestLogRecorder, GetRecordedLogsAsStringMaxLogsGreaterThanCountShowsAll)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 2");

    const std::string output = recorder.getRecordedLogsAsString(10); // More than available

    // Should contain all logs
    EXPECT_NE(output.find("log 1"), std::string::npos);
    EXPECT_NE(output.find("log 2"), std::string::npos);

    // Should NOT contain skip message
    EXPECT_EQ(output.find("Skipped"), std::string::npos);
}

TEST_F(TestLogRecorder, GetRecordedLogsAsStringFormatsSeverities)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "info msg");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_ERROR, "error msg");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "warn msg");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_FATAL, "fatal msg");

    const std::string output = recorder.getRecordedLogsAsString();

    // Verify all severity levels formatted correctly
    EXPECT_NE(output.find("[info] info msg"), std::string::npos);
    EXPECT_NE(output.find("[error] error msg"), std::string::npos);
    EXPECT_NE(output.find("[warn] warn msg"), std::string::npos);
    EXPECT_NE(output.find("[fatal] fatal msg"), std::string::npos);

    // Verify order preserved
    EXPECT_LT(output.find("[info]"), output.find("[error]"));
    EXPECT_LT(output.find("[error]"), output.find("[warn]"));
    EXPECT_LT(output.find("[warn]"), output.find("[fatal]"));
}

// === LogRecordingOutput Filtering ===

TEST_F(TestLogRecorder, LogRecordingOutputCallsCallbackWhenLevelAllows)
{
    static bool s_callbackWasCalled = false; // NOLINT(readability-identifier-naming)
    s_callbackWasCalled = false;

    auto testCallback = [](hipdnnSeverity_t severity, const char* message) {
        (void)severity;
        (void)message;
        s_callbackWasCalled = true;
    };

    LogRecordingOutput::instance().initialize(HIPDNN_SEV_INFO, testCallback);
    LogRecordingOutput::instance().outputToChainedCallback(HIPDNN_SEV_WARN, "test", false);

    EXPECT_TRUE(s_callbackWasCalled);
}

TEST_F(TestLogRecorder, LogRecordingOutputDoesNotCallCallbackWhenLevelBlocks)
{
    static bool s_callbackWasCalled = false; // NOLINT(readability-identifier-naming)
    s_callbackWasCalled = false;

    auto testCallback = [](hipdnnSeverity_t severity, const char* message) {
        (void)severity;
        (void)message;
        s_callbackWasCalled = true;
    };

    // Set level to ERROR - INFO should be blocked
    LogRecordingOutput::instance().initialize(HIPDNN_SEV_ERROR, testCallback);
    LogRecordingOutput::instance().outputToChainedCallback(HIPDNN_SEV_INFO, "test", false);

    EXPECT_FALSE(s_callbackWasCalled);
}

// === Edge Cases ===

TEST_F(TestLogRecorder, EmptyLogMessagesAreRecorded)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "");

    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 1);
    auto logs = recorder.getRecordedLogs();
    EXPECT_EQ(logs[0].message, "");
}

TEST_F(TestLogRecorder, SpecialCharactersInMessages)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED)
        .recordLog(HIPDNN_SEV_INFO, "Message with \n newlines \t tabs \"quotes\"");

    EXPECT_TRUE(recorder.hasLogContaining("newlines"));
    EXPECT_TRUE(recorder.hasLogContaining("tabs"));
    EXPECT_TRUE(recorder.hasLogContaining("quotes"));
}

TEST_F(TestLogRecorder, MultipleRecordersCanExist)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    {
        auto recorder1 = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
        {
            auto recorder2 = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
            EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);
        }
        // Inner recorder destroyed - should restore to outer's level
        EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
    }
    // Outer recorder destroyed - should restore to original
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_INFO);
}

TEST_F(TestLogRecorder, SeverityToStringReturnsCorrectStrings)
{
    EXPECT_STREQ(severityToString(HIPDNN_SEV_INFO), "info");
    EXPECT_STREQ(severityToString(HIPDNN_SEV_WARN), "warn");
    EXPECT_STREQ(severityToString(HIPDNN_SEV_ERROR), "error");
    EXPECT_STREQ(severityToString(HIPDNN_SEV_FATAL), "fatal");
    EXPECT_STREQ(severityToString(HIPDNN_SEV_OFF), "off");
    // Cast invalid integer to severity type
    auto invalidSeverity = static_cast<hipdnnSeverity_t>(999);
    EXPECT_STREQ(severityToString(invalidSeverity), "UNKNOWN");
}

TEST_F(TestLogRecorder, LogRecordingCallbackIgnoresNullMessage)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const size_t countBefore = recorder.getRecordedLogCount();

    logRecordingCallback(HIPDNN_SEV_INFO, nullptr);

    // Should not record anything
    EXPECT_EQ(recorder.getRecordedLogCount(), countBefore);
}

TEST_F(TestLogRecorder, LogRecordingCallbackRecordsValidMessage)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_OFF);
    logRecordingCallback(HIPDNN_SEV_WARN, "test message");

    // Should record to SHARED instance
    EXPECT_EQ(recorder.getRecordedLogCount(), 1);
    auto logs = recorder.getRecordedLogs();
    EXPECT_EQ(logs[0].message, "test message");
    EXPECT_EQ(logs[0].severity, HIPDNN_SEV_WARN);
}

TEST_F(TestLogRecorder, LogChainedRecordingCallbackRecordsValidMessage)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_OFF);
    logChainedRecordingCallback(HIPDNN_SEV_ERROR, "chained test message");

    // Should record to SHARED instance
    EXPECT_EQ(recorder.getRecordedLogCount(), 1);
    auto logs = recorder.getRecordedLogs();
    EXPECT_EQ(logs[0].message, "chained test message");
    EXPECT_EQ(logs[0].severity, HIPDNN_SEV_ERROR);
}

TEST_F(TestLogRecorder, LogChainedRecordingCallbackIgnoresNullMessage)
{
    auto recorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    const size_t countBefore = recorder.getRecordedLogCount();

    logChainedRecordingCallback(HIPDNN_SEV_INFO, nullptr);

    // Should not record anything
    EXPECT_EQ(recorder.getRecordedLogCount(), countBefore);
}

TEST_F(TestLogRecorder, SimpleStderrOutputCallbackOutputsValidMessage)
{
    // Redirect stderr to capture output
    const std::stringstream capturedOutput;
    std::streambuf* oldCerr = std::cerr.rdbuf(capturedOutput.rdbuf());

    simpleStderrOutputCallback(HIPDNN_SEV_WARN, "test warning message");

    // Restore stderr
    std::cerr.rdbuf(oldCerr);

    // Verify output format: "[severity] message\n"
    const std::string output = capturedOutput.str();
    EXPECT_EQ(output, "[warn] test warning message\n");
}

TEST_F(TestLogRecorder, SimpleStderrOutputCallbackHandlesNullMessage)
{
    // Should not crash with nullptr message
    simpleStderrOutputCallback(HIPDNN_SEV_INFO, nullptr);
    SUCCEED();
}

TEST_F(TestLogRecorder, SimpleStderrOutputCallbackFormatsAllSeverities)
{
    struct TestCase
    {
        hipdnnSeverity_t severity;
        const char* expectedOutput;
    };
    const std::array<TestCase, 5> testCases = {{{HIPDNN_SEV_INFO, "[info] test\n"},
                                                {HIPDNN_SEV_WARN, "[warn] test\n"},
                                                {HIPDNN_SEV_ERROR, "[error] test\n"},
                                                {HIPDNN_SEV_FATAL, "[fatal] test\n"},
                                                {HIPDNN_SEV_OFF, "[off] test\n"}}};

    for(const auto& tc : testCases)
    {
        const std::stringstream capturedOutput;
        std::streambuf* oldCerr = std::cerr.rdbuf(capturedOutput.rdbuf());

        simpleStderrOutputCallback(tc.severity, "test");

        std::cerr.rdbuf(oldCerr);

        const std::string output = capturedOutput.str();
        EXPECT_EQ(output, tc.expectedOutput)
            << "Failed for severity: " << severityToString(tc.severity);
    }
}

// === IsolatedLogRecorder ===

TEST_F(TestLogRecorder, IsolatedWithOverrideLevelChangesLevel)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    {
        auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
        EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);
    }
}

TEST_F(TestLogRecorder, IsolatedWithOverrideLevelRestoresOriginal)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);

    {
        auto recorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);
        EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);
    }

    // Should restore to WARN
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
}

TEST_F(TestLogRecorder, IsolatedWithCurrentLevelPreservesLevel)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_WARN);

    {
        auto recorder = IsolatedLogRecorder::withCurrentLevel();
        EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
    }

    // Should still be WARN
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_WARN);
}

TEST_F(TestLogRecorder, IsolatedGetCallbackReturnsValidCallback)
{
    hipdnnCallback_t callback = IsolatedLogRecorder::getIsolatedRecordingCallback();
    EXPECT_NE(callback, nullptr);
}

TEST_F(TestLogRecorder, IsolatedCallbackRecordsToIsolatedInstance)
{
    // Clear ISOLATED instance
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();

    // Start recording on ISOLATED instance
    LogRecording::instance(LogRecording::Id::ISOLATED).startRecording();

    // Get the callback and invoke it directly
    hipdnnCallback_t callback = IsolatedLogRecorder::getIsolatedRecordingCallback();
    callback(HIPDNN_SEV_INFO, "test isolated message");

    // Verify log was recorded to ISOLATED instance
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogCount(), 1);
    auto logs = LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogs();
    EXPECT_EQ(logs[0].message, "test isolated message");
    EXPECT_EQ(logs[0].severity, HIPDNN_SEV_INFO);

    // Cleanup
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();
}

TEST_F(TestLogRecorder, IsolatedGetUserCallbackReturnsValidCallback)
{
    hipdnnUserLogCallback_t callback = IsolatedLogRecorder::getIsolatedUserRecordingCallback();
    EXPECT_NE(callback, nullptr);
}

TEST_F(TestLogRecorder, IsolatedUserCallbackRecordsToIsolatedInstance)
{
    // Clear ISOLATED instance
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();

    // Start recording on ISOLATED instance
    LogRecording::instance(LogRecording::Id::ISOLATED).startRecording();

    // Get the user callback and invoke it directly
    hipdnnUserLogCallback_t callback = IsolatedLogRecorder::getIsolatedUserRecordingCallback();
    int dummyHandle = 0;
    callback(&dummyHandle, HIPDNN_SEV_WARN, "test isolated user message");

    // Verify log was recorded to ISOLATED instance
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogCount(), 1);
    auto logs = LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogs();
    EXPECT_EQ(logs[0].message, "test isolated user message");
    EXPECT_EQ(logs[0].severity, HIPDNN_SEV_WARN);

    // Cleanup
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();
}

TEST_F(TestLogRecorder, IsolatedUserCallbackIgnoresNullMessage)
{
    // Clear ISOLATED instance
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();

    // Start recording on ISOLATED instance
    LogRecording::instance(LogRecording::Id::ISOLATED).startRecording();

    // Invoke user callback with null message
    hipdnnUserLogCallback_t callback = IsolatedLogRecorder::getIsolatedUserRecordingCallback();
    int dummyHandle = 0;
    callback(&dummyHandle, HIPDNN_SEV_INFO, nullptr);

    // Should not record anything
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogCount(), 0);

    // Cleanup
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();
}

TEST_F(TestLogRecorder, IsolatedRecorderClearLogsEmptiesBuffer)
{
    auto recorder = IsolatedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::ISOLATED).recordLog(HIPDNN_SEV_INFO, "log 1");
    LogRecording::instance(LogRecording::Id::ISOLATED).recordLog(HIPDNN_SEV_WARN, "log 2");
    EXPECT_EQ(recorder.getRecordedLogCount(), 2);

    recorder.clearLogs();

    EXPECT_EQ(recorder.getRecordedLogCount(), 0);
    EXPECT_FALSE(recorder.hasLogContaining("log 1"));
    EXPECT_FALSE(recorder.hasLogContaining("log 2"));
}

TEST_F(TestLogRecorder, IsolatedRecorderClearLogsAllowsNewRecording)
{
    auto recorder = IsolatedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::ISOLATED).recordLog(HIPDNN_SEV_INFO, "before clear");
    recorder.clearLogs();

    LogRecording::instance(LogRecording::Id::ISOLATED).recordLog(HIPDNN_SEV_WARN, "after clear");

    EXPECT_EQ(recorder.getRecordedLogCount(), 1);
    EXPECT_FALSE(recorder.hasLogContaining("before clear"));
    EXPECT_TRUE(recorder.hasLogContaining("after clear"));
}

TEST_F(TestLogRecorder, IsolatedAndSharedInstancesAreIndependent)
{
    // Clear both instances
    LogRecording::instance(LogRecording::Id::SHARED).stopRecording();
    LogRecording::instance(LogRecording::Id::SHARED).clearLogs();
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();

    // Start recording on both instances
    LogRecording::instance(LogRecording::Id::SHARED).startRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).startRecording();

    // Record to SHARED only
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "shared message");

    // Record to ISOLATED only
    LogRecording::instance(LogRecording::Id::ISOLATED)
        .recordLog(HIPDNN_SEV_WARN, "isolated message");

    // Verify SHARED has only its message
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 1);
    auto sharedLogs = LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogs();
    EXPECT_EQ(sharedLogs[0].message, "shared message");

    // Verify ISOLATED has only its message
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogCount(), 1);
    auto isolatedLogs = LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogs();
    EXPECT_EQ(isolatedLogs[0].message, "isolated message");

    // Cleanup
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();
}

// === LogRecording ISOLATED Instance ===

TEST_F(TestLogRecorder, SharedAndIsolatedInstancesAreSeparate)
{
    // Get references to both instances
    auto& sharedInstance = LogRecording::instance(LogRecording::Id::SHARED);
    auto& isolatedInstance = LogRecording::instance(LogRecording::Id::ISOLATED);

    // Verify they are different instances (different addresses)
    EXPECT_NE(&sharedInstance, &isolatedInstance);
}

TEST_F(TestLogRecorder, LogsDoNotLeakBetweenInstances)
{
    // Clear both instances
    LogRecording::instance(LogRecording::Id::SHARED).stopRecording();
    LogRecording::instance(LogRecording::Id::SHARED).clearLogs();
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();

    // Start recording on SHARED only
    LogRecording::instance(LogRecording::Id::SHARED).startRecording();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "shared only");

    // ISOLATED should have no logs (not recording)
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogCount(), 0);

    // Start recording on ISOLATED and add a log
    LogRecording::instance(LogRecording::Id::ISOLATED).startRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).recordLog(HIPDNN_SEV_WARN, "isolated only");

    // Verify counts are independent
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 1);
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogCount(), 1);

    // Clear SHARED - should not affect ISOLATED
    LogRecording::instance(LogRecording::Id::SHARED).clearLogs();
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::SHARED).getRecordedLogCount(), 0);
    EXPECT_EQ(LogRecording::instance(LogRecording::Id::ISOLATED).getRecordedLogCount(), 1);

    // Cleanup
    LogRecording::instance(LogRecording::Id::ISOLATED).stopRecording();
    LogRecording::instance(LogRecording::Id::ISOLATED).clearLogs();
}

TEST_F(TestLogRecorder, SharedAndIsolatedRecordersWorkTogether)
{
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_INFO);

    // Create both recorders simultaneously
    auto sharedRecorder = SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);
    auto isolatedRecorder = IsolatedLogRecorder::withOverrideLevel(HIPDNN_SEV_ERROR);

    // The last one to set the level wins (IsolatedLogRecorder)
    EXPECT_EQ(hipdnn_data_sdk::logging::getLogLevel(), HIPDNN_SEV_ERROR);

    // Record logs to each instance via their respective mechanisms
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "shared log");
    LogRecording::instance(LogRecording::Id::ISOLATED).recordLog(HIPDNN_SEV_ERROR, "isolated log");

    // Each recorder should see only its own logs
    EXPECT_EQ(sharedRecorder.getRecordedLogCount(), 1);
    EXPECT_TRUE(sharedRecorder.hasLogContaining("shared log"));
    EXPECT_FALSE(sharedRecorder.hasLogContaining("isolated log"));

    EXPECT_EQ(isolatedRecorder.getRecordedLogCount(), 1);
    EXPECT_TRUE(isolatedRecorder.hasLogContaining("isolated log"));
    EXPECT_FALSE(isolatedRecorder.hasLogContaining("shared log"));

    // Verify getSavedLogLevel returns the level that was active when each was created
    // sharedRecorder was created first when level was INFO
    EXPECT_EQ(sharedRecorder.getSavedLogLevel(), HIPDNN_SEV_INFO);
    // isolatedRecorder was created second when level was WARN (set by sharedRecorder)
    EXPECT_EQ(isolatedRecorder.getSavedLogLevel(), HIPDNN_SEV_WARN);
}

// === waitForLogCount ===

TEST_F(TestLogRecorder, WaitForLogCountReturnsTrueWhenAlreadyMet)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 1");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 2");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "log 3");

    EXPECT_TRUE(recorder.waitForLogCount(3, std::chrono::milliseconds(100)));
}

TEST_F(TestLogRecorder, WaitForLogCountReturnsFalseOnTimeout)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();

    const auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(recorder.waitForLogCount(1, std::chrono::milliseconds(50)));
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    EXPECT_GE(elapsedMs, 50) << "Should wait at least 50ms before timing out";
    EXPECT_LE(elapsedMs, 20000) << "Should not wait significantly longer than the timeout";
}

TEST_F(TestLogRecorder, WaitForLogCountZeroTargetReturnsTrueImmediately)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    EXPECT_TRUE(recorder.waitForLogCount(0, std::chrono::milliseconds(100)));
}

// === waitForLogsContaining ===

TEST_F(TestLogRecorder, WaitForLogsContainingReturnsTrueWhenAlreadyPresent)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "The gamma log");
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_WARN, "beta");

    EXPECT_TRUE(recorder.waitForLogsContaining({"gamma", "beta"}, std::chrono::milliseconds(100)));
}

TEST_F(TestLogRecorder, WaitForLogsContainingReturnsFalseOnTimeout)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    LogRecording::instance(LogRecording::Id::SHARED).recordLog(HIPDNN_SEV_INFO, "The gamma log");

    EXPECT_FALSE(recorder.waitForLogsContaining({"gamma", "beta"}, std::chrono::milliseconds(10)));
}

TEST_F(TestLogRecorder, WaitForLogsContainingEmptyListReturnsTrueImmediately)
{
    auto recorder = SharedLogRecorder::withCurrentLevel();
    EXPECT_TRUE(recorder.waitForLogsContaining({}, std::chrono::milliseconds(100)));
}
