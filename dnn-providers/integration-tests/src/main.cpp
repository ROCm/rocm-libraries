// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <argparse.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/HipErrorHandler.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/Utilities.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/SupportClaims.hpp"
#include "harness/SupportClaimsAutoGen.hpp"
#include "harness/SupportClaimsVerifier.hpp"
#include "harness/SupportMatrixCollector.hpp"
#include "harness/TestConfig.hpp"

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
        parser.add_argument("--enforce-support-claims")
            .default_value(false)
            .implicit_value(true)
            .help("After RUN_ALL_TESTS, evaluate the engine's [[supported]] sidecar against "
                  "observed support. Process exits non-zero on Rule A/B/C failures. See "
                  "docs/support-claims-schema.md.");
        parser.add_argument("--write-support-claims")
            .default_value(false)
            .implicit_value(true)
            .help("Regenerate the engine's <ENGINE>.supported.toml [[supported]] block for "
                  "the current arch+platform from observed support. Engineer-driven; CI must "
                  "not invoke. See docs/support-claims-bringup.md.");

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

        const auto enforceClaims = parser.get<bool>("--enforce-support-claims");
        const auto writeClaims = parser.get<bool>("--write-support-claims");
        if(enforceClaims && writeClaims)
        {
            std::cerr << "Error: --enforce-support-claims and --write-support-claims are "
                         "mutually exclusive.\n";
            return 1;
        }
        // Both verifier and auto-gen need observations — they piggyback
        // on the support matrix collector's record stream.
        if(enforceClaims || writeClaims)
        {
            hipdnn_integration_tests::SupportMatrixCollector::get().setEnabled(true);
        }

        // NOTE: do not std::move(configPath) — the post-RUN_ALL_TESTS code
        // below (--write-support-claims) reads *configPath to derive the
        // sidecar path. Moving leaves the optional's contained path in a
        // valid-but-empty state and the sidecar would land at ".supported"
        // in CWD instead of next to the main config.
        hipdnn_integration_tests::TestConfig::initialize(std::move(articlePath),
                                                         std::move(engineName),
                                                         failOnUnsupported,
                                                         skipGraphValidation,
                                                         configPath,
                                                         refExecType);

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

        // RFC 0012 §6.4 / §8.3 preconditions. Both enforcement and
        // regeneration require an engine name + sidecar; both require
        // unsharded, single-plugin runs because the verifier/condenser
        // operate per-engine and a partial observation set produces
        // either false positives or unsafe matchers.
        if(enforceClaims || writeClaims)
        {
            if(!hipdnn_integration_tests::TestConfig::get().hasEngineName())
            {
                std::cerr << "Error: --enforce-support-claims/--write-support-claims require "
                             "--test-engine to be set so the verifier knows which engine's "
                             "support to evaluate.\n";
                static_cast<void>(hipStreamDestroy(stream));
                return 1;
            }
            // RFC 0012 §6.4: refuse multi-plugin runs. Multi-engine
            // attribution is deferred to v2 (§13) — the verifier scopes
            // per-engine via supportingEngines.find(name), but the writer
            // can't safely condense observations across engines and the
            // RFC text is the contract. If you're tripping this with a
            // reference engine loading alongside the test plugin, point
            // --test-article at *only* the engine you want measured.
            size_t numEngines = 0;
            if(hipdnnGetEngineCount_ext(handle, &numEngines) == HIPDNN_STATUS_SUCCESS
               && numEngines > 1)
            {
                std::cerr << "Error: more than one plugin loaded (" << numEngines
                          << " engines). --enforce-support-claims/--write-support-claims "
                             "require a single engine per RFC 0012 §6.4. Multi-engine "
                             "attribution is deferred to v2 (§13).\n";
                static_cast<void>(hipStreamDestroy(stream));
                return 1;
            }

            // MSVC mode treats std::getenv as deprecated under -Werror; use
            // the project's cross-platform getEnv (returns "" for unset).
            if(const auto shardTotal = hipdnn_data_sdk::utilities::getEnv("GTEST_TOTAL_SHARDS");
               !shardTotal.empty() && shardTotal != "1")
            {
                std::cerr << "Error: GTEST_TOTAL_SHARDS is set (" << shardTotal
                          << "). Sharding is deferred to v2; run unsharded "
                             "(RFC 0012 §9).\n";
                static_cast<void>(hipStreamDestroy(stream));
                return 1;
            }
            if(!hipdnn_data_sdk::utilities::getEnv("GTEST_SHARD_INDEX").empty())
            {
                std::cerr << "Error: GTEST_SHARD_INDEX is set. Sharding is deferred to v2 "
                             "(RFC 0012 §9).\n";
                static_cast<void>(hipStreamDestroy(stream));
                return 1;
            }
            if(::testing::GTEST_FLAG(break_on_failure))
            {
                std::cerr << "Error: --gtest_break_on_failure aborts before the verifier "
                             "runs; remove it when using --enforce-support-claims.\n";
                static_cast<void>(hipStreamDestroy(stream));
                return 1;
            }
            if(::testing::GTEST_FLAG(repeat) > 1)
            {
                std::cerr << "Error: --gtest_repeat with N>1 produces duplicate records that "
                             "confuse the verifier; remove it when enforcing claims.\n";
                static_cast<void>(hipStreamDestroy(stream));
                return 1;
            }
#ifndef NDEBUG
            std::cerr << "Error: --enforce-support-claims/--write-support-claims require a "
                         "release build; debug builds produce non-deterministic gtest param "
                         "strings (RFC 0012 §6.4/§8.3).\n";
            static_cast<void>(hipStreamDestroy(stream));
            return 1;
#endif
            if(writeClaims && ::testing::GTEST_FLAG(filter) != ""
               && ::testing::GTEST_FLAG(filter) != "*")
            {
                std::cerr << "Error: --write-support-claims refuses partial runs "
                             "(--gtest_filter='"
                          << ::testing::GTEST_FLAG(filter)
                          << "'); the sidecar must reflect the full observed set.\n";
                static_cast<void>(hipStreamDestroy(stream));
                return 1;
            }
        }

        const int result = RUN_ALL_TESTS();

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

        // Post-RUN_ALL_TESTS: enforcement or regeneration. We deliberately
        // run after the support matrix write so an engineer who passed
        // both flags still gets the markdown for review.
        int claimsExit = 0;
        if(writeClaims)
        {
            if(!configPath.has_value())
            {
                std::cerr << "Error: --write-support-claims requires --test-config so the "
                             "sidecar location is known.\n";
                claimsExit = 1;
            }
            else
            {
                const auto& cfg = hipdnn_integration_tests::TestConfig::get();
                const auto sidecar = hipdnn_integration_tests::sidecarPathFor(*configPath);
                const bool ok = hipdnn_integration_tests::generateSupportClaimsForCurrentArch(
                    sidecar,
                    std::string(cfg.getEngineName()),
                    cfg.getCurrentArchToken(),
                    std::optional<std::string>(cfg.getCurrentPlatform()));
                claimsExit = ok ? 0 : 1;
            }
        }
        else if(enforceClaims)
        {
            const auto& cfg = hipdnn_integration_tests::TestConfig::get();
            if(!cfg.hasSupportClaims())
            {
                std::cerr << "[--enforce-support-claims] no sidecar loaded for engine '"
                          << cfg.getEngineName() << "' on arch=" << cfg.getCurrentArchToken()
                          << "; treating as 'not enforced' (RFC 0012 §6).\n";
            }
            else
            {
                const bool fullCi = hipdnn_integration_tests::detectFullCiMode();
                hipdnn_integration_tests::SupportClaimsVerifier verifier(
                    cfg.getSupportClaims(),
                    std::string(cfg.getEngineName()),
                    cfg.getCurrentArchToken(),
                    cfg.getCurrentPlatform(),
                    fullCi);
                const bool clean = verifier.runAndReport();
                claimsExit = clean ? 0 : 1;
            }
        }

        // Clean up shared handle and stream
        static_cast<void>(hipStreamDestroy(stream));
        hipdnnDestroy(handle);
        return result != 0 ? result : claimsExit;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
}
