// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <argparse.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common/Utilities.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/SupportMatrixCollector.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/UnverifiableBundleReport.hpp"

namespace
{

// Teardown for the shared handle and stream lives in main's scope, not at each
// return site. Deliberately *not* folded into getSharedHandle()'s static: that
// would defer hipdnnDestroy to static-destruction time, whose order against the
// HIP runtime's own statics is unspecified.
class HandleGuard
{
public:
    explicit HandleGuard(hipdnnHandle_t handle)
        : _handle(handle)
    {
    }

    HandleGuard(const HandleGuard&) = delete;
    HandleGuard& operator=(const HandleGuard&) = delete;
    HandleGuard(HandleGuard&&) = delete;
    HandleGuard& operator=(HandleGuard&&) = delete;

    ~HandleGuard()
    {
        static_cast<void>(hipdnnDestroy(_handle));
    }

private:
    hipdnnHandle_t _handle;
};

class StreamGuard
{
public:
    explicit StreamGuard(hipStream_t stream)
        : _stream(stream)
    {
    }

    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    StreamGuard(StreamGuard&&) = delete;
    StreamGuard& operator=(StreamGuard&&) = delete;

    ~StreamGuard()
    {
        static_cast<void>(hipStreamDestroy(_stream));
    }

private:
    hipStream_t _stream;
};

} // namespace

int main(int argc, char** argv) noexcept
{
    // Shared hipdnn handle + HIP stream are created below before any fixture
    // runs, so per-fixture SKIP_IF_NO_DEVICES is too late. Bail early on a
    // no-GPU runner so ctest reports PASS.
    int deviceCount = 0;
    auto deviceStatus = hipGetDeviceCount(&deviceCount);
    if(deviceStatus == hipErrorNoDevice || deviceCount == 0)
    {
        std::cout << "No HIP devices available; skipping " << argv[0] << "\n";
        return 0;
    }

    try
    {
        // Parse custom arguments before InitGoogleTest to avoid unknown flag warnings
        argparse::ArgumentParser parser(
            "hipdnn_integration_tests", "", argparse::default_arguments::help);
        parser.add_argument("--ta", "--test-article")
            .help("Full path to the hipdnn engine plugin .so to test. "
                  "Omit to use hipDNN's default plugin discovery.");
        parser.add_argument("--te", "--test-engine")
            .help("Engine name to test against (e.g., MIOPEN_ENGINE). "
                  "Omit to let hipDNN select the engine.");
        parser.add_argument("--fail-on-unsupported")
            .default_value(false)
            .implicit_value(true)
            .help("FAIL instead of SKIP when no engine supports a graph");
        parser.add_argument("--skip-graph-validation")
            .default_value(false)
            .implicit_value(true)
            .help("PASS immediately after confirming engine support, "
                  "without executing or validating the graph");
        parser.add_argument("--tc", "--test-config")
            .help("Path to a TOML configuration file for per-test tolerance overrides.");
        parser.add_argument("--reference-executor")
            .help("Reference executor for validation: 'cpu' (default) or 'gpu'. "
                  "Can also be set via HIPDNN_TEST_REFERENCE_EXECUTOR env var.");
        parser.add_argument("--generate-support-matrix")
            .default_value(std::string("support_matrix.md"))
            .implicit_value(std::string("support_matrix.md"))
            .help("Generate a markdown support matrix file (default: support_matrix.md).");
        parser.add_argument("--no-bundles")
            .default_value(false)
            .implicit_value(true)
            .help("Disable bundle test registration, leaving only the C++ tests built "
                  "into this binary. Equivalent to HIPDNN_TEST_ALLOW_BUNDLES=0.");
        parser.add_argument("--gd", "--golden-data-dir")
            .help("Path to the integration test bundle data directory. "
                  "Defaults to <exe>/../lib/integration-test-bundles/. "
                  "Can also be set via HIPDNN_TEST_GOLDEN_DATA_DIR env var.");
        // --verification-mode governs BUNDLE tests (how the engine's output is
        // verified). It is independent of --reference-executor, which governs the
        // parameterized tests (which ref executor is exercised as the SUT).
        parser.add_argument("--vm", "--verification-mode")
            .help("How bundle engine output is verified: 'auto' (default; golden -> "
                  "GPU ref -> CPU ref -> skip), 'golden', 'gpu', 'cpu', or "
                  "'golden-check' (validate golden data against CPU ref, no engine). "
                  "Can also be set via HIPDNN_TEST_VERIFICATION_MODE env var.");
        parser.add_argument("--capture-bundles")
            .help("Capture C++ graph tests as JSON bundles into the given directory. "
                  "Each test writes a {suite}/{case}/{case}.json + .meta.json pair.");
        parser.add_argument("--enforce-support-claims")
            .default_value(false)
            .implicit_value(true)
            .help("Enforce engine support claims from .support.json sidecars. "
                  "A broken claim (engine no longer supports a claimed graph) becomes "
                  "a test FAIL instead of a silent SKIP.");

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

        // Parse --test-engine, --fail-on-unsupported, and --test-config arguments
        std::optional<std::string> engineName;
        if(parser.is_used("--test-engine"))
        {
            engineName = parser.get<std::string>("--test-engine");
        }
        auto failOnUnsupported = parser.get<bool>("--fail-on-unsupported");
        auto skipGraphValidation = parser.get<bool>("--skip-graph-validation");

        std::optional<std::filesystem::path> configPath;
        if(parser.is_used("--test-config"))
        {
            auto configPathArg = parser.get<std::string>("--test-config");
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

        // Parse --reference-executor argument (case-insensitive)
        std::optional<hipdnn_integration_tests::ReferenceExecutorType> refExecType;
        if(parser.is_used("--reference-executor"))
        {
            auto val = parser.get<std::string>("--reference-executor");
            std::transform(val.begin(), val.end(), val.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if(val == "gpu")
            {
                refExecType = hipdnn_integration_tests::ReferenceExecutorType::GPU;
            }
            else if(val == "cpu")
            {
                refExecType = hipdnn_integration_tests::ReferenceExecutorType::CPU;
            }
            else
            {
                std::cerr << "Error: --reference-executor must be 'cpu' or 'gpu'\n";
                return 1;
            }
        }

        // Bundles are on by default; --no-bundles is the explicit opt-out.
        // HIPDNN_TEST_ALLOW_BUNDLES (handled in TestConfig) overrides.
        auto allowBundles = !parser.get<bool>("--no-bundles");

        std::optional<std::filesystem::path> goldenDataDir;
        if(parser.is_used("--golden-data-dir"))
        {
            goldenDataDir = parser.get<std::string>("--golden-data-dir");
            if(!std::filesystem::exists(*goldenDataDir))
            {
                std::cerr << "Error: --golden-data-dir path does not exist: " << *goldenDataDir
                          << "\n";
                return 1;
            }
            if(!std::filesystem::is_directory(*goldenDataDir))
            {
                std::cerr << "Error: --golden-data-dir is not a directory: " << *goldenDataDir
                          << "\n";
                return 1;
            }
        }

        // Parse --verification-mode (case-insensitive); invalid value -> exit 1.
        std::optional<hipdnn_integration_tests::VerificationMode> verificationMode;
        if(parser.is_used("--verification-mode"))
        {
            try
            {
                verificationMode = hipdnn_integration_tests::parseVerificationMode(
                    parser.get<std::string>("--verification-mode"));
            }
            catch(const std::exception& e)
            {
                std::cerr << "Error: " << e.what() << '\n';
                return 1;
            }
        }

        // Parse --capture-bundles argument
        std::optional<std::filesystem::path> captureDir;
        if(parser.is_used("--capture-bundles"))
        {
            captureDir = parser.get<std::string>("--capture-bundles");
        }

        // Parse --test-article argument and load explicit plugin if provided
        std::optional<std::filesystem::path> articlePath;
        if(parser.is_used("--test-article"))
        {
            // Validate and canonicalize article path (resolves relative paths)
            auto articlePathArg = parser.get<std::string>("--test-article");
            try
            {
                articlePath = std::filesystem::canonical(articlePathArg);
            }
            catch(const std::filesystem::filesystem_error&)
            {
                std::cerr << "Error: Article path does not exist: " << articlePathArg << '\n';
                return 1;
            }

            // Set engine plugin path to the plugin file (not the directory)
            const std::string articlePathStr = articlePath->string();
            const char* pluginPath = articlePathStr.c_str();
            if(hipdnnSetEnginePluginPaths_ext(1, &pluginPath, HIPDNN_PLUGIN_LOADING_ABSOLUTE)
               != HIPDNN_STATUS_SUCCESS)
            {
                std::cerr << "Error: Failed to set engine plugin path\n";
                return 1;
            }
        }

        // Enable support matrix generation if requested
        if(parser.is_used("--generate-support-matrix"))
        {
            auto outputFile = parser.get<std::string>("--generate-support-matrix");
            hipdnn_integration_tests::SupportMatrixCollector::get().setEnabled(true);
            hipdnn_integration_tests::SupportMatrixCollector::get().setOutputPath(outputFile);
        }

        hipdnn_integration_tests::TestConfigOptions opts;
        opts.articlePath = std::move(articlePath);
        opts.engineName = std::move(engineName);
        opts.failOnUnsupported = failOnUnsupported;
        opts.skipGraphValidation = skipGraphValidation;
        opts.configPath = std::move(configPath);
        opts.referenceExecutorType = refExecType;
        opts.allowBundles = allowBundles;
        opts.goldenDataDir = std::move(goldenDataDir);
        opts.verificationMode = verificationMode;
        opts.captureDir = std::move(captureDir);
        opts.enforceSupportClaims = parser.get<bool>("--enforce-support-claims");
        hipdnn_integration_tests::TestConfig::initialize(std::move(opts));

        // Reconstruct argc/argv for GTest from remaining (unknown) args.
        // argv[0] (program name) must be first — GTest requires it.
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

        // Initialize test logging infrastructure to forward logs to std::cerr based
        // on the current environment HIPDNN_LOG_LEVEL value when this function is called.
        auto recordingCallback = hipdnn_test_sdk::utilities::initializeTestLogRecordingShared();

        // Initialize plugin logger with test recording callback so that plugin logs
        // are routed to the log recorder for capture.
        hipdnn_plugin_sdk::logging::initializeCallbackLogging("hipdnn_integration_tests",
                                                              recordingCallback);

        // Register HipErrorHandler to check and clear HIP errors after each test
        testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
        listeners.Append(new hipdnn_test_sdk::utilities::HipErrorHandler);

        // Create shared handle (triggers engine loading). The guards below own
        // teardown for every exit path from here on, including the outer catch,
        // so no return site cleans up by hand. Declaration order matters: the
        // stream is destroyed first, then the handle it was set on.
        auto handle = hipdnn_integration_tests::getSharedHandle();
        const HandleGuard handleGuard(handle);

        // Set stream on shared handle
        hipStream_t stream;
        if(hipStreamCreate(&stream) != hipSuccess)
        {
            std::cerr << "Failed to create HIP stream\n";
            return 1;
        }
        const StreamGuard streamGuard(stream);

        if(hipdnnSetStream(handle, stream) != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "Failed to set stream on shared handle\n";
            return 1;
        }

        try
        {
            hipdnn_integration_tests::bundle::LoadedEngineTable::get().build(handle);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << "\n";
            return 1;
        }

        if(hipdnn_integration_tests::TestConfig::get().hasEngineName()
           && !hipdnn_integration_tests::bundle::LoadedEngineTable::get().isLoaded(
               hipdnn_integration_tests::TestConfig::get().getEngineName()))
        {
            std::cerr << "Error: Engine '"
                      << hipdnn_integration_tests::TestConfig::get().getEngineName()
                      << "' is not loaded. Check the plugin path.\n";
            return 1;
        }

        // Enumerated before any test records support data (see setEngineNames); the
        // vector keeps enumeration order for the table columns below.
        std::vector<std::string> loadedEngineNames;
        if(hipdnn_integration_tests::SupportMatrixCollector::get().isEnabled())
        {
            std::map<int64_t, std::string> engineNamesById;
            size_t numEngines = 0;
            if(hipdnnGetEngineCount_ext(handle, &numEngines) == HIPDNN_STATUS_SUCCESS)
            {
                for(size_t i = 0; i < numEngines; ++i)
                {
                    auto info = hipdnn_integration_tests::getEngineInfo(handle, i);
                    loadedEngineNames.push_back(info.engineName);
                    engineNamesById.emplace(info.engineId, std::move(info.engineName));
                }
            }
            hipdnn_integration_tests::SupportMatrixCollector::get().setEngineNames(
                std::move(engineNamesById));
        }

        hipdnn_integration_tests::bundle::registerBundleTests();

        const int result = RUN_ALL_TESTS();

        // Print bundles that ended without a verdict (no oracle / reference bug).
        // Informational only — these SKIP, so they do not affect `result`.
        hipdnn_integration_tests::bundle::UnverifiableBundleReport::get().print();
        hipdnn_integration_tests::bundle::printSupportClaimSummary(
            hipdnn_integration_tests::bundle::supportClaimCoverage(),
            hipdnn_integration_tests::bundle::SupportClaimVerdicts::get(),
            std::cerr);

        int exitCode = result;

        if(hipdnn_integration_tests::TestConfig::get().enforceSupportClaims()
           && hipdnn_integration_tests::bundle::verifiedNothing(
               hipdnn_integration_tests::bundle::supportClaimCoverage()))
        {
            std::cerr
                << "\nFATAL: --enforce-support-claims is active and "
                << hipdnn_integration_tests::bundle::supportClaimCoverage().graphsWithClaims
                << " graph(s) carrying support\n"
                   "       claims were discovered, but not one of them was ever queried. "
                   "Enforcement\n"
                   "       passed having verified nothing, so the run fails instead. Usual "
                   "causes:\n"
                   "         - no --test-engine was given, so there is no engine to check claims "
                   "against\n"
                   "         - the GPU or the engine plugin failed to load\n"
                   "         - a --gtest_filter selected only graphs without claims\n";
            exitCode = 1;
        }

        // Guard against a silently empty run: bundles are enabled, yet nothing
        // was selected. This must be checked *after* RUN_ALL_TESTS(). GTest only
        // applies --gtest_filter inside UnitTestImpl::RunAllTests(), via
        // FilterTests(), which is the sole place TestInfo::should_run_ is set;
        // it is default-constructed to false. So test_to_run_count() is
        // unconditionally 0 before RUN_ALL_TESTS(), no matter how many tests
        // were registered, and checking it earlier fails every engine-driven run.
        const auto* unitTest = ::testing::UnitTest::GetInstance();
        if(unitTest->test_to_run_count() == 0
           && hipdnn_integration_tests::TestConfig::get().allowBundles())
        {
            const auto dataDir = hipdnn_integration_tests::bundle::resolveDataDir();
            const bool dataDirFound = std::filesystem::exists(dataDir);

            // A run that named an engine, or one whose bundle data is actually
            // present, is expected to select something. A local build with
            // neither is allowed to run empty.
            if(hipdnn_integration_tests::TestConfig::get().hasEngineName() || dataDirFound)
            {
                // Print the counts, not a guess: "0 registered" is a build or
                // discovery problem, "N registered, 0 selected" is a filter
                // problem. They have different fixes and these numbers are the
                // only way to tell them apart from a CI log.
                const int suiteCount = unitTest->total_test_suite_count();
                std::cerr << "Error: zero tests ran.\n"
                          << "  registered:      " << unitTest->total_test_count() << " test(s) in "
                          << suiteCount << " suite(s)\n"
                          << "  selected:        0 (nothing matched --gtest_filter)\n"
                          << "  gtest_filter:    " << GTEST_FLAG_GET(filter) << "\n"
                          << "  bundle data dir: " << dataDir
                          << (dataDirFound ? " (exists)" : " (MISSING)") << "\n";

                constexpr int MAX_SUITES_TO_LIST = 10;
                for(int i = 0; i < suiteCount && i < MAX_SUITES_TO_LIST; ++i)
                {
                    std::cerr << "  registered suite: " << unitTest->GetTestSuite(i)->name()
                              << "\n";
                }
                if(suiteCount > MAX_SUITES_TO_LIST)
                {
                    std::cerr << "  ... and " << (suiteCount - MAX_SUITES_TO_LIST)
                              << " more suite(s)\n";
                }

                return 1;
            }
        }

        {
            const int total = unitTest->test_to_run_count();
            const int passed = unitTest->successful_test_count();
            const int skip = unitTest->skipped_test_count();
            const int failed = unitTest->failed_test_count();
            const double pct = total > 0 ? 100.0 * passed / total : 0.0;

            std::cerr << "\n==== TEST COVERAGE SUMMARY ====\n"
                      << "Passed:  " << passed << " / " << total << " (" << std::fixed
                      << std::setprecision(1) << pct << "%)\n"
                      << "Skipped: " << skip << "\n"
                      << "Failed:  " << failed << "\n";
        }

        // Generate support matrix if requested
        if(hipdnn_integration_tests::SupportMatrixCollector::get().isEnabled())
        {
            std::vector<std::string> allEngineNames;

            if(hipdnn_integration_tests::TestConfig::get().hasEngineName())
            {
                allEngineNames.emplace_back(
                    hipdnn_integration_tests::TestConfig::get().getEngineName());
            }
            else
            {
                allEngineNames = std::move(loadedEngineNames);
            }

            hipdnn_integration_tests::SupportMatrixCollector::get().writeMarkdown(allEngineNames);
        }

        // handleGuard / streamGuard clean up on the way out.
        return exitCode;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
