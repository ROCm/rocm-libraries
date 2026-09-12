/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

namespace
{

// The inventory includes disabled, filtered and sharded-out cases. Only callbacks
// from a complete passing iteration can satisfy a declared census obligation.
class CensusExecutionListener : public testing::EmptyTestEventListener
{
public:
    explicit CensusExecutionListener(const testing::TestSuite& suite)
        : _suite(suite)
    {
        for(int i = 0; i < suite.total_test_count(); ++i)
        {
            _completed.emplace(suite.GetTestInfo(i), false);
        }
    }

    void OnTestIterationStart(const testing::UnitTest& /*unitTest*/, int /*iteration*/) override
    {
        for(auto& [test, completed] : _completed)
        {
            completed = false;
        }
    }

    void OnTestEnd(const testing::TestInfo& test) override
    {
        const auto entry = _completed.find(&test);
        if(entry == _completed.end())
        {
            return;
        }
        entry->second = test.result()->Passed() && !test.result()->Skipped();
        if(!entry->second)
        {
            _valid = false;
            std::cerr << "Census: " << _suite.name() << "." << test.name()
                      << " did not pass without skipping.\n";
        }
    }

    void OnTestIterationEnd(const testing::UnitTest& /*unitTest*/, int iteration) override
    {
        _completedIteration = true;
        for(const auto& [test, completed] : _completed)
        {
            if(!completed)
            {
                _valid = false;
                std::cerr << "Census: iteration " << iteration << " did not complete "
                          << _suite.name() << "." << test->name() << " successfully.\n";
            }
        }
    }

    bool passed() const
    {
        if(!_completedIteration)
        {
            std::cerr << "Census: no test iteration completed for " << _suite.name() << ".\n";
        }
        return _valid && _completedIteration;
    }

private:
    const testing::TestSuite& _suite;
    std::unordered_map<const testing::TestInfo*, bool> _completed;
    bool _valid = true;
    bool _completedIteration = false;
};

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::unique_ptr<CensusExecutionListener> census;
    const auto censusSuite = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_CENSUS_SUITE");
    if(!censusSuite.empty())
    {
        // Validate the caller's explicit shard before default-root setup can supply
        // another tree. The production loader's fallback remains unchanged.
        const auto arch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        const auto root = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
        std::error_code error;
        if(arch.empty() || root.empty() || !std::filesystem::is_directory(root, error))
        {
            std::cerr << "Census requires a nonempty HIPDNN_TEST_EXPECTED_ARCH and an existing "
                         "explicit HIPDNN_DESCRIPTOR_DIR; arch='"
                      << arch << "', root='" << root << "'.\n";
            return 1;
        }

        const auto* unitTest = testing::UnitTest::GetInstance();
        const testing::TestSuite* suite = nullptr;
        for(int i = 0; i < unitTest->total_test_suite_count(); ++i)
        {
            const auto* candidate = unitTest->GetTestSuite(i);
            if(censusSuite == candidate->name())
            {
                suite = candidate;
                break;
            }
        }
        if(suite == nullptr || suite->total_test_count() == 0)
        {
            std::cerr << "Census suite '" << censusSuite << "' is absent or empty.\n";
            return 1;
        }
        census = std::make_unique<CensusExecutionListener>(*suite);
    }

    // Keep the ingestor's winner cache out of the developer's ~/.cache/hipdnn: the
    // dispatch cases benchmark, and benchmarking writes a shard through to disk.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir("hip-kernel-provider-unit");

#ifdef HIPDNN_TEST_DESCRIPTOR_DIR
    // Point this binary at the descriptors staged beside the build's plugin. The engine
    // implementation is linked in statically here, so its module-relative lookup has no
    // module to measure from and falls through to the install prefix, which a build tree
    // has never written. Done here rather than in the CTest environment so the binary
    // runs standalone -- and so a path from this machine stays out of the install-time
    // CTest file, which is generated from that same environment.
    //
    // Never overrides a value the caller set, and never sets one naming nothing: on an
    // installed run the staged tree is absent and resolution should reach the installed
    // copy instead.
    if(std::error_code notFound;
       hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR").empty()
       && std::filesystem::is_directory(HIPDNN_TEST_DESCRIPTOR_DIR, notFound))
    {
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_DESCRIPTOR_DIR", HIPDNN_TEST_DESCRIPTOR_DIR);
    }
#endif

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend by the recordingCallback returned here
    // which is the desired behaviour because this is a plugin unit test harness.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    // Initialize plugin logger with test recording callback so that plugin logs
    // logs are first routed to the log recorder for capture and use by the unit tests.
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("hip_kernel-provider_tests",
                                                          recordingCallback);

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    auto hipErrorHandler = std::make_unique<hipdnn_test_sdk::utilities::HipErrorHandler>();
    listeners.Append(hipErrorHandler.release());
    const auto* censusResult = census.get();
    if(census)
    {
        listeners.Append(census.release());
    }

    const int result = RUN_ALL_TESTS();
    const bool censusPassed = censusResult == nullptr || censusResult->passed();
    if(result != 0)
    {
        return result;
    }
    return censusPassed ? 0 : 1;
}
