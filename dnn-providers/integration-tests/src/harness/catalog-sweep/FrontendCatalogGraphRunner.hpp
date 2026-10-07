// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <hipdnn_frontend/knob/KnobSetting.hpp>

#include "harness/IReferenceExecutors.hpp"
#include "harness/ValidationSite.hpp"
#include "harness/bundle/BundleInputs.hpp"
#include "harness/bundle/HarnessPolicy.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/OutputComparison.hpp"
#include "harness/catalog-sweep/CatalogSweep.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/input-init/InputFillRecipes.hpp"

// Held by pointer so this header stays free of the frontend's Graph.hpp.
namespace hipdnn_frontend::graph
{
class Graph;
}

namespace hipdnn_integration_tests::catalog_sweep
{

/// The real thing behind ICatalogGraphRunner: one frontend Graph built from a bundle,
/// swept engine by engine, entry by entry, against a reference run once.
///
/// Loop isolation, in the order the sweep depends on it:
///  - the same Graph object serves the reference check, the engine listing, every catalog
///    fetch and every entry; create_execution_plan_ext() replaces the plan each time;
///  - the inputs are built once for the reference (with the bundle's seeded recipes) and
///    rebuilt from the same recipes before every entry, so no entry reads what a
///    previous kernel wrote;
///  - every entry gets freshly allocated outputs prefilled with the NaN sentinel, so a
///    kernel that writes nothing leaves NaN behind rather than the previous entry's
///    answer;
///  - the reference's outputs are copied to the host once and never handed to a plan:
///    plans only ever see the entry's own device buffers;
///  - only the graph's registered outputs are compared, and every comparison builds its
///    validators afresh, so no comparison state survives into the next entry.
class FrontendCatalogGraphRunner final : public ICatalogGraphRunner
{
public:
    struct Config
    {
        std::shared_ptr<bundle::IntegrationTestBundle> bundle;
        /// The bundle's "Suite.Test" name: what the TOML tolerance and validator
        /// overrides are keyed on, so the sweep grades with the tolerances the normal
        /// run of this bundle uses.
        std::string testName;
        std::string bundlePath; ///< for messages
        bundle::HarnessPolicy policy;
        std::shared_ptr<DeviceInputFiller> deviceFiller;
        std::shared_ptr<IReferenceExecutors> references;
    };

    explicit FrontendCatalogGraphRunner(Config config);
    ~FrontendCatalogGraphRunner() override;

    ReferenceResult prepareReference() override;
    EngineList listEngines() override;
    CatalogFetch fetchCatalog(const CatalogEngine& engine) override;
    EntryOutcome runEntry(const CatalogEngine& engine, std::size_t entryIndex) override;

private:
    enum class RefStatus
    {
        RAN,
        CAPABILITY_MISS,
        RUNTIME_ERROR,
        HARNESS_ERROR,
    };
    struct RefAttempt
    {
        RefStatus status = RefStatus::RAN;
        std::string message;
    };

    /// from_binary, once. Empty on success, otherwise the reason.
    std::string openGraph();

    RefAttempt runReference(ReferenceExecutorType type, bundle::BundleInputs& inputs);

    /// Compares `actual` against the host-resident reference and records the result in
    /// `outcome`.
    void compareOutputs(bundle::OutputTensors& actual,
                        const std::string& contextLine,
                        EntryOutcome& outcome);

    Config _config;
    std::unique_ptr<hipdnn_frontend::graph::Graph> _graph;
    bool _opened = false;
    std::string _openError;
    InputFillRecipes _recipes;
    bundle::OutputTensors _expected;
    std::map<std::int64_t, std::vector<std::vector<hipdnn_frontend::KnobSetting>>> _catalogs;
};

} // namespace hipdnn_integration_tests::catalog_sweep
