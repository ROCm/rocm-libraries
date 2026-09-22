/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "TestDescriptorRoot.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

// Every census diagnostic in this file is matched as text by the CTest entries in
// descriptor-packaging/cmake/HkpPackaging.cmake: each control keys on one refusal's
// wording, and the census entry keys on the "Census: " prefix. Rewording either breaks
// those gates silently.
namespace
{

// Comma-separated, because the value arrives through the CTest ENVIRONMENT property,
// itself a semicolon-separated list of VAR=VALUE. Empty entries are dropped so a trailing
// separator is not read as a case named "".
std::set<std::string> splitCaseNames(const std::string& packed)
{
    std::set<std::string> names;
    for(std::string::size_type begin = 0; begin <= packed.size();)
    {
        const auto end = packed.find(',', begin);
        const auto stop = end == std::string::npos ? packed.size() : end;
        if(stop > begin)
        {
            names.insert(packed.substr(begin, stop - begin));
        }
        if(end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return names;
}

// The inventory includes disabled, filtered and sharded-out cases; only callbacks from a
// complete passing iteration satisfy a census obligation. Obligations are built from the
// suite itself, so a suite that loses cases loses obligations with them. The pinned set
// closes that hole and is compared by name in both directions, since a lost case and a new
// one cancel in a count but call for opposite remedies.
//
// The pin is optional; absent, only the execution guard runs.
class CensusExecutionListener : public testing::EmptyTestEventListener
{
public:
    CensusExecutionListener(const testing::TestSuite& suite, const std::string& expectedCases)
        : _suite(suite)
    {
        std::set<std::string> registered;
        for(int i = 0; i < suite.total_test_count(); ++i)
        {
            const auto* test = suite.GetTestInfo(i);
            _completed.emplace(test, false);
            registered.insert(test->name());
        }

        if(expectedCases.empty())
        {
            return;
        }

        const auto expected = splitCaseNames(expectedCases);
        for(const auto& name : expected)
        {
            if(registered.count(name) == 0)
            {
                _valid = false;
                std::cerr << "Census: " << _suite.name() << "." << name
                          << " is expected but not registered, so the suite has lost a "
                             "case. Restore it, or drop it from EXPECTED_CASES.\n";
            }
        }
        for(const auto& name : registered)
        {
            if(expected.count(name) == 0)
            {
                _valid = false;
                std::cerr << "Census: " << _suite.name() << "." << name
                          << " is registered but not expected, so the suite has gained a "
                             "case the pin does not cover. Add it to EXPECTED_CASES.\n";
            }
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
    // Captured and validated ahead of the default-root block below, which can write
    // HIPDNN_DESCRIPTOR_DIR itself: the shard a census is verdicted against is never one
    // this binary chose.
    const auto censusArch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
    const auto censusRoot = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR");
    if(!censusSuite.empty())
    {
        std::error_code error;
        if(censusArch.empty() || censusRoot.empty()
           || !std::filesystem::is_directory(censusRoot, error))
        {
            std::cerr << "Census requires a nonempty HIPDNN_TEST_EXPECTED_ARCH and an existing "
                         "explicit HIPDNN_DESCRIPTOR_DIR; arch='"
                      << censusArch << "', root='" << censusRoot << "'.\n";
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
        // Before RUN_ALL_TESTS: this reads the static registration rather than results, and
        // a suite that shrank should say so even when the surviving cases pass.
        const auto expectedCases
            = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_CENSUS_EXPECTED_CASES");
        census = std::make_unique<CensusExecutionListener>(*suite, expectedCases);
    }

    // Keep the ingestor's winner cache out of the developer's ~/.cache/hipdnn: the
    // dispatch cases benchmark, and benchmarking writes a shard through to disk.
    const hipdnn_test_sdk::utilities::ScopedTestCacheDir cacheDir("hip-kernel-provider-unit");

#ifdef HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR
    // Point this binary at the descriptors staged beside it. The engine implementation is
    // linked in statically here, so its module-relative lookup measures from this
    // executable and would otherwise fall through to the install prefix, which a build
    // tree has never written. Done here rather than in the CTest environment so the binary
    // runs standalone, and so nothing machine-specific reaches the install-time CTest
    // file, which is generated from that same environment.
    //
    // Never overrides a value the caller set. Fail the process when the resolved root
    // holds no descriptor, which is otherwise indistinguishable from a run on a device
    // the descriptors do not cover.
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_DIR").empty())
    {
        const auto descriptors = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR);
        const auto unusable
            = hip_kernel_provider::testing::describeUnusableDescriptorRoot(descriptors);
        if(!unusable.empty())
        {
            std::cerr << unusable
                      << ". Build the descriptor staging targets, or set "
                         "HIPDNN_DESCRIPTOR_DIR to a root that holds them.\n";
            return 1;
        }

        hipdnn_data_sdk::utilities::setEnv("HIPDNN_DESCRIPTOR_DIR", descriptors.string().c_str());
    }
#endif

    // Initialize test logging infrastructure to forward logs to std::cerr based
    // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
    // NOTE: Logs are not routed to the backend by the recordingCallback returned here
    // which is the desired behaviour because this is a plugin unit test harness.
    auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

    // Initialize plugin logger with test recording callback so that plugin logs
    // are first routed to the log recorder for capture and use by the unit tests.
    hipdnn_plugin_sdk::logging::initializeCallbackLogging("hip_kernel-provider_tests",
                                                          recordingCallback);

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    if(!censusSuite.empty())
    {
        // The packer stamps every emitted copy of a pack with the architecture of the
        // shard it writes into, so those stamps say which shard arrived. Uncompared, an
        // entry declared for one architecture passes identically on another's shard.
        //
        // Below the logging setup because this call loads and memoizes for the process:
        // each loader diagnostic is dispatched once.
        //
        // A second stamp means the root spans shards, which a per-arch entry exists to
        // avoid. An arch-independent pack is refused too: it cannot show which shard
        // arrived.
        std::size_t loadedSets = 0;
        std::size_t loadedPacks = 0;
        std::set<std::string> stamped;
        for(const auto& descriptorSet :
            hip_kernel_provider::kernel_ingestor_engine::discoverDescriptorSets())
        {
            ++loadedSets;
            loadedPacks += descriptorSet.packs.size();
            for(const auto& pack : descriptorSet.packs)
            {
                stamped.insert(pack.arch.begin(), pack.arch.end());
            }
        }
        if(stamped.size() != 1 || *stamped.begin() != censusArch)
        {
            std::string detail;
            if(loadedSets == 0)
            {
                detail = "no descriptor set loaded from this root";
            }
            else if(loadedPacks == 0)
            {
                detail = std::to_string(loadedSets)
                         + " descriptor sets loaded from this root, holding no pack";
            }
            else if(stamped.empty())
            {
                detail = std::to_string(loadedPacks)
                         + " packs loaded from this root, none of them architecture-stamped";
            }
            else
            {
                std::string found;
                for(const auto& stamp : stamped)
                {
                    found += (found.empty() ? "'" : ", '") + stamp + "'";
                }
                detail = std::to_string(loadedPacks)
                         + " packs loaded from this root carry the stamps " + found;
            }
            std::cerr << "Census requires every loaded pack to be stamped for the expected "
                         "architecture; arch='"
                      << censusArch << "', root='" << censusRoot << "'; " << detail << ".\n";
            return 1;
        }
    }
#endif

    // Register HipErrorHandler to check and clear HIP errors after each test
    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    auto hipErrorHandler = std::make_unique<hipdnn_test_sdk::utilities::HipErrorHandler>();
    listeners.Append(hipErrorHandler.release());
    // Append takes ownership: gtest deletes its listeners only when the UnitTest singleton
    // is torn down at static destruction, so this pointer stays valid past the release.
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
