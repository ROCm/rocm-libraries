// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/catalog-sweep/CatalogSweepRegistration.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "harness/ReferenceExecutorPool.hpp"
#include "harness/TestConfig.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/ProductionPolicy.hpp"
#include "harness/catalog-sweep/FrontendCatalogGraphRunner.hpp"

namespace hipdnn_integration_tests::catalog_sweep
{

namespace
{

/// Sweeps one bundle graph. The body is thin on purpose: every decision that carries the
/// accounting lives in sweepGraph(), which is tested without a device.
class CatalogSweepTest : public ::testing::Test
{
public:
    CatalogSweepTest(std::shared_ptr<CatalogSweepRun> run,
                     bundle::detail::LoadedBundle loaded,
                     std::weak_ptr<DeviceInputFiller> filler)
        : _run(std::move(run))
        , _loaded(std::move(loaded))
        , _filler(std::move(filler))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void TestBody() override
    {
        const auto graphName = _loaded.fullName();
        const auto policy = bundle::productionPolicy(bundle::TensorPlacement::DEVICE);

        // The guards every verification test applies before it runs: a graph this device
        // cannot run, or this engine's TOML excludes, is a skip, not a sweep.
        std::optional<std::string> skipReason;
        if(bundle::noHipDevicesAvailable())
        {
            skipReason = "No devices available";
        }
        else if(auto reason = checkTomlSkip(graphName))
        {
            skipReason = "[arch " + policy.arch + "] " + *reason;
        }
        else if(auto vram = checkVramRequirement(_loaded.bundle->metadata, policy.deviceVramMb))
        {
            skipReason = *vram;
        }
        else if(auto arch = checkArchCompatibility(_loaded.bundle->metadata, policy.arch))
        {
            skipReason = *arch;
        }
        if(skipReason.has_value())
        {
            _run->noteSkipped(graphName, *skipReason, std::cout);
            GTEST_SKIP() << *skipReason;
        }

        FrontendCatalogGraphRunner::Config config;
        config.bundle = _loaded.bundle;
        config.testName = graphName;
        config.bundlePath = _loaded.jsonPath.string();
        config.policy = policy;
        config.deviceFiller = _filler.lock();
        config.references = sharedReferenceExecutors();
        FrontendCatalogGraphRunner runner(std::move(config));

        GraphSweepRequest request;
        request.graph = graphName;
        request.filter = &_run->options().filter;
        request.pinnedEngine = _run->pinnedEngine();
        request.failOnUnsupported = TestConfig::get().failOnUnsupported();
        request.context = _run->context();

        const auto result = sweepGraph(runner, request, _run->sink(), std::cout);
        _run->note(result);

        if(result.outcome == GraphOutcome::SKIPPED)
        {
            GTEST_SKIP() << result.message;
        }
        if(result.outcome == GraphOutcome::FAILED)
        {
            FAIL() << "catalog sweep could not be carried out: " << result.message;
        }
        if(!result.tally.reconciles())
        {
            ADD_FAILURE() << "catalog reconciliation does not hold: "
                          << formatReconcileLine(graphName, result.tally);
        }
        if(result.tally.failed() > 0)
        {
            ADD_FAILURE() << result.tally.failed()
                          << " catalog entr(ies) failed (fail_numeric=" << result.tally.failNumeric
                          << " fail_build=" << result.tally.failBuild
                          << " fail_execute=" << result.tally.failExecute
                          << " no_catalog=" << result.tally.noCatalog
                          << "); see the per-entry records in " << _run->options().reportPath;
        }
    }

private:
    std::shared_ptr<CatalogSweepRun> _run;
    bundle::detail::LoadedBundle _loaded;
    std::weak_ptr<DeviceInputFiller> _filler;
};

} // namespace

SweepRegistrationStats
    registerCatalogSweepTests(const std::shared_ptr<CatalogSweepRun>& run,
                              const std::shared_ptr<DeviceInputFiller>& deviceFiller)
{
    SweepRegistrationStats stats;

    auto discovered = bundle::detail::discoverDataDirBundles();
    if(!discovered.has_value())
    {
        return stats;
    }
    stats.discovered = discovered->bundles.size();

    // --gtest_filter keeps meaning "which graphs", spelled as the bundle's own name.
    auto split = bundle::detail::splitByGTestFilter(std::move(discovered->bundles),
                                                    GTEST_FLAG_GET(filter));
    stats.excludedByGTestFilter = split.excluded.size();

    // The shard decides on the graph's name alone, so the choice does not depend on what
    // else was selected.
    std::vector<bundle::DiscoveredBundle> owned;
    owned.reserve(split.selected.size());
    for(auto& graph : split.selected)
    {
        if(shardOwns(run->options().shard, graph.fullName()))
        {
            owned.push_back(std::move(graph));
        }
        else
        {
            ++stats.excludedByShard;
        }
    }

    // Only the sweep's own tests may run: the C++ graph tests built into this binary
    // would otherwise run beside it, and none of them is a sweep.
    GTEST_FLAG_SET(filter, std::string(K_SWEEP_SUITE_PREFIX) + "*");

    if(owned.empty())
    {
        return stats;
    }

    const bundle::detail::DiscoveredBundleSet set{std::move(discovered->dataDir), std::move(owned)};
    auto loaded
        = bundle::detail::loadDiscoveredBundles(set, /*countFound=*/false, /*countClaims=*/false);
    if(!loaded.has_value())
    {
        return stats;
    }

    for(auto& graph : *loaded)
    {
        const auto suite = std::string(K_SWEEP_SUITE_PREFIX) + graph.suiteName;
        ::testing::RegisterTest(
            suite.c_str(),
            graph.testName.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [run, graph, filler = std::weak_ptr<DeviceInputFiller>(deviceFiller)]()
                -> ::testing::Test* { return new CatalogSweepTest(run, graph, filler); });
        ++stats.registered;
    }

    HIPDNN_PLUGIN_LOG_INFO("Registered " << stats.registered << " catalog sweep test(s)");
    return stats;
}

} // namespace hipdnn_integration_tests::catalog_sweep
