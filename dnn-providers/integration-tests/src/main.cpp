// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <argparse.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "common/Utilities.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/SupportMatrixCollector.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/SupportClaimWriter.hpp"
#include "harness/bundle/SupportEnforcementReport.hpp"
#include "harness/bundle/UnverifiableBundleReport.hpp"

namespace
{

using hipdnn_integration_tests::getEngineInfo;

bool engineIsLoaded(hipdnnHandle_t handle, std::string_view targetEngineName)
{
    size_t numEngines = 0;
    if(hipdnnGetEngineCount_ext(handle, &numEngines) != HIPDNN_STATUS_SUCCESS || numEngines == 0)
    {
        return false;
    }

    for(size_t i = 0; i < numEngines; ++i)
    {
        auto info = getEngineInfo(handle, i);
        if(info.engineName == targetEngineName)
        {
            return true;
        }
    }
    return false;
}

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
        parser.add_argument("--allow-bundles")
            .default_value(false)
            .implicit_value(true)
            .help("Enable bundle test registration (default: false). "
                  "Set --allow-bundles or HIPDNN_TEST_ALLOW_BUNDLES=1 env var to enable.");
        parser.add_argument("--gd", "--golden-data-dir")
            .help("Path to the integration test bundle data directory. "
                  "Defaults to <exe>/../lib/integration-test-bundles/. "
                  "Can also be set via HIPDNN_TEST_GOLDEN_DATA_DIR env var.");
        // --verification-mode governs BUNDLE tests (how the engine's output is
        // verified). It is independent of --reference-executor, which governs the
        // parameterized tests (which ref executor is exercised as the SUT).
        parser.add_argument("--vm", "--verification-mode")
            .help("How bundle engine output is verified: 'auto' (default; golden -> "
                  "GPU ref -> CPU ref -> skip), 'golden', 'gpu', or 'cpu'. "
                  "Can also be set via HIPDNN_TEST_VERIFICATION_MODE env var.");
        parser.add_argument("--capture-bundles")
            .help("Capture C++ graph tests as JSON bundles into the given directory. "
                  "Each test writes a {suite}/{case}/{case}.json + .meta.json pair.");
        parser.add_argument("--write-support-claims")
            .default_value(false)
            .implicit_value(true)
            .help("RFC 0015: observe live engine support (and, for buildable/full bundles, "
                  "plan-build/execute results) and write support-claim sidecars for every "
                  "exercised bundle. Requires --test-article (mode B or C); a mode-A run "
                  "(neither --test-article nor --test-engine) writes nothing.");
        parser.add_argument("--test-config-dir")
            .help("Directory of per-engine TOML configs (<dir>/<EngineName>.toml), used only "
                  "in mode B (--test-article without --test-engine): each pass over the "
                  "plugin's engines resolves its own tolerance/skip overrides from this "
                  "directory. Mutually exclusive with --test-config.");

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

        // Parse --write-support-claims / --test-config-dir (RFC 0015 §9.1, §9.4).
        // Mode detection here only needs whether --test-article/--test-engine were
        // *used* on the CLI -- the resolved articlePath is parsed further below.
        auto writeSupportClaims = parser.get<bool>("--write-support-claims");
        const bool hasArticleArg = parser.is_used("--test-article");
        const bool hasEngineArg = engineName.has_value();

        if(writeSupportClaims && !hasArticleArg)
        {
            std::cerr << "Error: --write-support-claims requires --test-article (mode B or C); "
                         "mode A (neither flag) cannot attribute an observation to any engine.\n";
            return 1;
        }

        std::optional<std::filesystem::path> testConfigDir;
        if(parser.is_used("--test-config-dir"))
        {
            if(configPath.has_value())
            {
                std::cerr << "Error: --test-config and --test-config-dir are mutually "
                             "exclusive\n";
                return 1;
            }
            auto testConfigDirArg = parser.get<std::string>("--test-config-dir");
            try
            {
                testConfigDir = std::filesystem::canonical(testConfigDirArg);
            }
            catch(const std::filesystem::filesystem_error&)
            {
                std::cerr << "Error: --test-config-dir path does not exist: " << testConfigDirArg
                          << '\n';
                return 1;
            }
            if(!std::filesystem::is_directory(*testConfigDir))
            {
                std::cerr << "Error: --test-config-dir is not a directory: " << testConfigDirArg
                          << '\n';
                return 1;
            }
            if(hasEngineArg || !hasArticleArg)
            {
                std::cerr << "Warning: --test-config-dir only applies in mode B (--test-article "
                             "without --test-engine); ignoring it for this run.\n";
                testConfigDir.reset();
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

        // Parse --allow-bundles, --golden-data-dir, --verification-mode
        auto allowBundles = parser.get<bool>("--allow-bundles");

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
        opts.writeSupportClaims = writeSupportClaims;
        opts.testConfigDir = std::move(testConfigDir);
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

        // Create shared handle (triggers engine loading)
        auto handle = hipdnn_integration_tests::getSharedHandle();

        // Set stream on shared handle
        hipStream_t stream;
        if(hipStreamCreate(&stream) != hipSuccess)
        {
            std::cerr << "Failed to create HIP stream\n";
            return 1;
        }
        if(hipdnnSetStream(handle, stream) != HIPDNN_STATUS_SUCCESS)
        {
            std::cerr << "Failed to set stream on shared handle\n";
            static_cast<void>(hipStreamDestroy(stream));
            return 1;
        }

        // Verify target engine is loaded (only when --test-engine was provided)
        if(hipdnn_integration_tests::TestConfig::get().hasEngineName()
           && !engineIsLoaded(handle, hipdnn_integration_tests::TestConfig::get().getEngineName()))
        {
            std::cerr << "Error: Engine '"
                      << hipdnn_integration_tests::TestConfig::get().getEngineName()
                      << "' is not loaded. Check the plugin path.\n";
            static_cast<void>(hipStreamDestroy(stream));
            return 1;
        }

        hipdnn_integration_tests::bundle::registerBundleTests();

        auto printCoverageSummary = [](const std::string& label) {
            const auto* unit = ::testing::UnitTest::GetInstance();
            const int total = unit->test_to_run_count();
            const int passed = unit->successful_test_count();
            const int skip = unit->skipped_test_count();
            const int failed = unit->failed_test_count();
            const double pct = total > 0 ? 100.0 * passed / total : 0.0;

            std::cerr << "\n==== TEST COVERAGE SUMMARY (" << label << ") ====\n"
                      << "Passed:  " << passed << " / " << total << " (" << std::fixed
                      << std::setprecision(1) << pct << "%)\n"
                      << "Skipped: " << skip << "\n"
                      << "Failed:  " << failed << "\n";
        };

        int result = 0;

        // RFC 0015 §7.3: mode B (--test-article without --test-engine) enumerates
        // every engine the plugin exposes and runs the whole suite once per
        // engine, pinning each in turn, so buildable/full checks (and
        // --write-support-claims observations) are attributable to a specific
        // engine instead of whichever one hipDNN's heuristic auto-selects. Mode
        // C pins its one named engine (already handled by TestConfig); mode A
        // never pins anything (unchanged auto-select behavior).
        if(hasArticleArg && !hasEngineArg)
        {
            auto engineNames = hipdnn_integration_tests::listLoadedEngineNames(handle);
            if(engineNames.empty())
            {
                std::cerr << "Error: --test-article was provided but the loaded plugin exposes "
                             "no engines.\n";
                static_cast<void>(hipStreamDestroy(stream));
                hipdnnDestroy(handle);
                return 1;
            }

            for(const auto& loopEngineName : engineNames)
            {
                std::cerr << "\n==== Pass for engine: " << loopEngineName << " ====\n";
                hipdnn_integration_tests::bundle::EnginePassContext::get().set(
                    loopEngineName, hipdnn_data_sdk::utilities::engineNameToId(loopEngineName));
                hipdnn_integration_tests::TestConfig::get().loadActiveTestSettingsForEngine(
                    loopEngineName);

                result |= RUN_ALL_TESTS();
                printCoverageSummary(loopEngineName);
            }
            hipdnn_integration_tests::bundle::EnginePassContext::get().clear();
        }
        else
        {
            if(hasEngineArg)
            {
                hipdnn_integration_tests::bundle::EnginePassContext::get().set(
                    std::string(hipdnn_integration_tests::TestConfig::get().getEngineName()),
                    hipdnn_integration_tests::TestConfig::get().getEngineId());
            }

            result = RUN_ALL_TESTS();
            printCoverageSummary(
                hasEngineArg
                    ? std::string(hipdnn_integration_tests::TestConfig::get().getEngineName())
                    : std::string("auto-select"));
            hipdnn_integration_tests::bundle::EnginePassContext::get().clear();
        }

        // Print bundles that ended without a verdict (no oracle / reference bug).
        // Informational only — these SKIP, so they do not affect `result`.
        hipdnn_integration_tests::bundle::UnverifiableBundleReport::get().print();

        // Print engines that turned out to support a claim-bearing bundle's
        // graph on an (engine, arch, platform) its support.json does not
        // list (RFC 0015 §7.1). Informational only -- never affects `result`.
        hipdnn_integration_tests::bundle::UnclaimedSupportReport::get().print();

        // RFC 0015 §7.2: the run-level empty-query guard. If enforcement was
        // expected (>=1 claim-bearing bundle registered) but zero support
        // queries were observed anywhere in the run, fail loudly rather than
        // report green -- a claim that is never queried is silently
        // unenforced, and this is the only floor that catches "nothing ran".
        if(hipdnn_integration_tests::bundle::SupportQueryGuard::get().tripped())
        {
            std::cerr << "\nERROR: RFC 0015 empty-query guard tripped: "
                      << hipdnn_integration_tests::bundle::SupportQueryGuard::get()
                             .claimBearingBundleCount()
                      << " claim-bearing bundle(s) were registered but zero engine-support "
                         "queries were observed in this run (no GPU, plugin failed to load, or "
                         "an over-narrow --gtest_filter). Failing rather than reporting green.\n";
            result = 1;
        }

        // RFC 0015 §9.2: the write tool's empty-write guard. A degenerate run
        // that recorded zero observations at all must not silently report
        // success -- it refuses to touch any file rather than risk masking a
        // real failure (no GPU, plugin load failure, or an over-narrow
        // --gtest_filter that excluded every claim-bearing bundle) as "no
        // change needed".
        if(hipdnn_integration_tests::TestConfig::get().writeSupportClaims())
        {
            if(hipdnn_integration_tests::bundle::ClaimObservationCollector::get().empty())
            {
                std::cerr << "\nERROR: --write-support-claims requested but zero support "
                             "observations were recorded this run. Refusing to write anything "
                             "(this would otherwise risk silently nulling existing claims).\n";
                result = 1;
            }
            else
            {
                try
                {
                    auto written
                        = hipdnn_integration_tests::bundle::ClaimObservationCollector::get()
                              .writeAll();
                    std::cerr << "\nWrote " << written.size() << " support-claim file(s):\n";
                    for(const auto& path : written)
                    {
                        std::cerr << "  " << path << "\n";
                    }
                }
                catch(const std::exception& e)
                {
                    std::cerr << "\nERROR: failed to write support-claim file(s): " << e.what()
                              << "\n";
                    result = 1;
                }
            }
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
                // Enumerate all loaded engines from the handle
                size_t numEngines = 0;
                if(hipdnnGetEngineCount_ext(handle, &numEngines) == HIPDNN_STATUS_SUCCESS)
                {
                    for(size_t i = 0; i < numEngines; ++i)
                    {
                        auto info = getEngineInfo(handle, i);
                        allEngineNames.push_back(std::move(info.engineName));
                    }
                }
            }

            hipdnn_integration_tests::SupportMatrixCollector::get().writeMarkdown(allEngineNames);
        }

        // Clean up shared handle and stream
        static_cast<void>(hipStreamDestroy(stream));
        hipdnnDestroy(handle);
        return result;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
