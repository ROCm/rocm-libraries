// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Validates checked-in golden data against our own reference executors.
//
// This is a separate binary from hipdnn_integration_tests, not a flag on it.
// Verifying an engine and validating our own golden data are different jobs with
// different failure meanings, and the separation is structural here: this main()
// loads no plugin, creates no hipdnnHandle_t, builds no LoadedEngineTable and
// enforces no support claims, so there is no configuration in which a golden-data
// run can silently become an engine run or vice versa.
//
// It is also why this is not driven by add_external_integration_test_target():
// that helper parameterizes a binary over a provider's plugin and engine name.
// Nothing here involves an engine, so running it once per provider would repeat
// identical work and give three chances to disagree about our own data.

#include <argparse.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleRegistration.hpp"

int main(int argc, char** argv) noexcept
{
    try
    {
        argparse::ArgumentParser parser(
            "hipdnn_golden_data_tests", "", argparse::default_arguments::help);
        parser.add_argument("--reference")
            .help("Which reference to validate against: 'cpu', 'gpu', or 'both' "
                  "(default). The GPU suite needs a device and skips without one; "
                  "the CPU suite reads and writes host memory and needs neither a "
                  "device nor a plugin.");
        parser.add_argument("--gd", "--golden-data-dir")
            .help("Path to the integration test bundle data directory. "
                  "Defaults to <exe>/../lib/integration-test-bundles/. "
                  "Can also be set via HIPDNN_TEST_GOLDEN_DATA_DIR env var.");
        parser.add_argument("--tc", "--test-config")
            .help("Path to a TOML configuration file for per-test tolerance overrides.");

        std::vector<std::string> remainingArgs;
        try
        {
            remainingArgs = parser.parse_known_args(argc, argv);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cerr << parser;
            return 1;
        }

        bool runCpu = true;
        bool runGpu = true;
        if(parser.is_used("--reference"))
        {
            const auto value = parser.get<std::string>("--reference");
            if(value == "cpu")
            {
                runGpu = false;
            }
            else if(value == "gpu")
            {
                runCpu = false;
            }
            else if(value != "both")
            {
                std::cerr << "Error: --reference must be 'cpu', 'gpu', or 'both'\n";
                return 1;
            }
        }

        std::optional<std::filesystem::path> goldenDataDir;
        if(parser.is_used("--golden-data-dir"))
        {
            goldenDataDir = parser.get<std::string>("--golden-data-dir");
            if(!std::filesystem::is_directory(*goldenDataDir))
            {
                std::cerr << "Error: --golden-data-dir is not a directory: " << *goldenDataDir
                          << "\n";
                return 1;
            }
        }

        std::optional<std::filesystem::path> configPath;
        if(parser.is_used("--test-config"))
        {
            const auto configPathArg = parser.get<std::string>("--test-config");
            try
            {
                configPath = std::filesystem::canonical(configPathArg);
            }
            catch(const std::filesystem::filesystem_error&)
            {
                std::cerr << "Error: Config path does not exist: " << configPathArg << '\n';
                return 1;
            }
        }

        std::vector<char*> gtestArgv;
        gtestArgv.reserve(remainingArgs.size() + 2);
        gtestArgv.push_back(argv[0]);
        for(auto& arg : remainingArgs)
        {
            gtestArgv.push_back(arg.data());
        }
        gtestArgv.push_back(nullptr);
        auto gtestArgc = static_cast<int>(remainingArgs.size()) + 1;
        ::testing::InitGoogleTest(&gtestArgc, gtestArgv.data());

        auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();
        hipdnn_plugin_sdk::logging::initializeCallbackLogging("hipdnn_golden_data_tests",
                                                              recordingCallback);

        hipdnn_integration_tests::TestConfigOptions opts;
        opts.goldenDataDir = std::move(goldenDataDir);
        opts.configPath = std::move(configPath);
        hipdnn_integration_tests::TestConfig::initialize(std::move(opts));

        if(runCpu)
        {
            hipdnn_integration_tests::bundle::registerGoldenDataValidationTests(
                hipdnn_integration_tests::ReferenceExecutorType::CPU);
        }
        if(runGpu)
        {
            hipdnn_integration_tests::bundle::registerGoldenDataValidationTests(
                hipdnn_integration_tests::ReferenceExecutorType::GPU);
        }

        const int result = RUN_ALL_TESTS();

        // An empty run here is not automatically an error: golden `.bin` blobs are
        // DVC-managed, so a tree that has not pulled them registers nothing and has
        // nothing to say. Only a run whose data directory is actually present and
        // still selected nothing is suspicious.
        const auto* unitTest = ::testing::UnitTest::GetInstance();
        if(unitTest->test_to_run_count() == 0)
        {
            const auto dataDir = hipdnn_integration_tests::bundle::resolveDataDir();
            std::cerr << "No golden-data validation tests ran. Bundle data directory: " << dataDir
                      << (std::filesystem::exists(dataDir) ? " (present)" : " (missing)")
                      << "\n       Golden .bin blobs are DVC-managed; run `dvc pull` in "
                         "integration-test-bundles/ if this is unexpected.\n";
        }

        return result;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
    catch(...)
    {
        std::cerr << "Fatal error: unknown exception\n";
        return 1;
    }
}
