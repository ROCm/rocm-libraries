// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/catalog-sweep/FrontendCatalogGraphRunner.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <set>
#include <sstream>
#include <utility>
#include <variant>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>

#include "harness/EngineNotApplicableError.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/BundleInputs.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"
#include "harness/catalog-sweep/OutputDiffStats.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests::catalog_sweep
{

namespace
{

double millisecondsSince(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

std::vector<CatalogKnob> toCatalogKnobs(const std::vector<hipdnn_frontend::KnobSetting>& settings)
{
    std::vector<CatalogKnob> knobs;
    knobs.reserve(settings.size());
    for(const auto& setting : settings)
    {
        CatalogKnob knob;
        knob.id = setting.knobId();
        std::visit([&knob](const auto& value) { knob.value = value; }, setting.value());
        knobs.push_back(std::move(knob));
    }
    return knobs;
}

// A comparison report is a full per-element drift profile; the JSONL line wants the
// verdict and a hint, not a page.
std::string truncated(const std::string& text, std::size_t maxChars)
{
    return text.size() <= maxChars ? text : text.substr(0, maxChars) + "...";
}

} // namespace

FrontendCatalogGraphRunner::FrontendCatalogGraphRunner(Config config)
    : _config(std::move(config))
{
    // The same recipes the verification harness derives from a bundle's metadata, so a
    // graph is filled here exactly as it is in a normal run.
    const auto& metadata = _config.bundle->metadata;
    if(metadata.seed.has_value())
    {
        _recipes.setGlobalSeed(static_cast<unsigned int>(*metadata.seed));
    }
    if(metadata.inputs.has_value())
    {
        _recipes.loadFromJson(*metadata.inputs);
    }
}

FrontendCatalogGraphRunner::~FrontendCatalogGraphRunner() = default;

std::string FrontendCatalogGraphRunner::openGraph()
{
    if(_opened)
    {
        return _openError;
    }
    _opened = true;

    _graph = std::make_unique<hipdnn_frontend::graph::Graph>();
    const auto& buffer = _config.bundle->graphBuffer;
    const std::vector<uint8_t> graphBytes(buffer.data(), buffer.data() + buffer.size());
    if(auto err = _graph->from_binary(getSharedHandle(), graphBytes); !err.is_good())
    {
        _openError = "from_binary failed: " + err.get_message();
    }
    return _openError;
}

FrontendCatalogGraphRunner::RefAttempt
    FrontendCatalogGraphRunner::runReference(ReferenceExecutorType type,
                                             bundle::BundleInputs& inputs)
{
    auto& testBundle = *_config.bundle;
    const auto wrapper = testBundle.graphWrapper();
    bool useDevice = false;

    try
    {
        IReferenceGraphExecutor& executor = _config.references->get(type);
        if(!executor.isApplicable(testBundle.graphBuffer.data(), testBundle.graphBuffer.size()))
        {
            return {RefStatus::CAPABILITY_MISS, "reference is not applicable for this graph"};
        }

        useDevice = _config.policy.useDevice() && executor.requiresDeviceMemory();
        _expected = bundle::detail::allocateSentinelOutputs(
            wrapper.getTensorMap(), testBundle.outputTensorUids, useDevice);

        // The packed copy is what a device executor reads when an input is sub-byte.
        auto& source
            = useDevice && !inputs.packedInputs.empty() ? inputs.packedInputs : inputs.inputs;
        auto variantPack = bundle::detail::buildVariantPack(
            source, _expected, wrapper.getTensorMap(), testBundle.outputTensorUids, useDevice);

        executor.execute(testBundle.graphBuffer.data(), testBundle.graphBuffer.size(), variantPack);
    }
    catch(const ReferenceCapabilityError& e)
    {
        return {RefStatus::CAPABILITY_MISS, e.what()};
    }
    catch(const bundle::detail::DeviceOutputError& e)
    {
        return {RefStatus::HARNESS_ERROR, e.what()};
    }
    catch(const std::exception& e)
    {
        return {RefStatus::RUNTIME_ERROR, e.what()};
    }

    bundle::detail::markOutputsModified(_expected, useDevice);
    return {RefStatus::RAN, {}};
}

ReferenceResult FrontendCatalogGraphRunner::prepareReference()
{
    using Status = ReferenceResult::Status;
    const auto result = [](Status status, std::string message) {
        ReferenceResult r;
        r.status = status;
        r.message = std::move(message);
        return r;
    };

    try
    {
        if(auto error = openGraph(); !error.empty())
        {
            return result(Status::FAILED, error);
        }

        auto& testBundle = *_config.bundle;
        if(testBundle.metadata.enforcementLevel != EnforcementLevel::FULL)
        {
            return result(Status::UNAVAILABLE,
                          "bundle enforcement_level is below FULL: its outputs are not compared "
                          "against a reference");
        }
        if(testBundle.outputTensorUids.empty())
        {
            return result(Status::UNAVAILABLE, "bundle has no output tensors to compare");
        }

        // The one seeded fill. The reference runs on exactly these values, and every entry
        // is refilled from the same recipes.
        auto inputs = bundle::prepareBundleInputs(
            testBundle,
            _recipes,
            _config.policy.useDevice() ? _config.deviceFiller.get() : nullptr);
        switch(inputs.status)
        {
        case bundle::BundleInputs::Status::READY:
            break;
        case bundle::BundleInputs::Status::UNVERIFIABLE:
            return result(Status::UNAVAILABLE, inputs.message);
        case bundle::BundleInputs::Status::DEVICE_FILL_FAILED:
        case bundle::BundleInputs::Status::TENSOR_LOAD_FAILED:
        default:
            return result(Status::FAILED, inputs.message);
        }

        const auto mode = _config.policy.mode;

        if(mode == VerificationMode::GOLDEN
           || (mode == VerificationMode::AUTO && testBundle.hasGoldenOutputs))
        {
            if(!testBundle.hasGoldenOutputs)
            {
                return result(Status::FAILED,
                              "verification-mode=golden was requested but this bundle has no "
                              "golden data");
            }
            for(const int64_t uid : testBundle.outputTensorUids)
            {
                auto it = inputs.inputs.find(uid);
                if(it == inputs.inputs.end())
                {
                    return result(Status::FAILED,
                                  "golden data has no tensor for output " + std::to_string(uid));
                }
                _expected[uid] = std::move(it->second);
            }
            return result(Status::READY, {});
        }

        RefAttempt attempt;
        if(mode == VerificationMode::CPU)
        {
            attempt = runReference(ReferenceExecutorType::CPU, inputs);
        }
        else
        {
            attempt = runReference(ReferenceExecutorType::GPU, inputs);
            // In auto mode a GPU reference that cannot run, or broke, falls through to the
            // CPU one; a device or harness fault does not.
            if(mode == VerificationMode::AUTO
               && (attempt.status == RefStatus::CAPABILITY_MISS
                   || attempt.status == RefStatus::RUNTIME_ERROR))
            {
                attempt = runReference(ReferenceExecutorType::CPU, inputs);
            }
        }

        switch(attempt.status)
        {
        case RefStatus::RAN:
            break;
        case RefStatus::CAPABILITY_MISS:
            return result(Status::UNAVAILABLE,
                          "no reference can run this graph: " + attempt.message);
        case RefStatus::RUNTIME_ERROR:
        case RefStatus::HARNESS_ERROR:
        default:
            return result(Status::FAILED, "reference run failed: " + attempt.message);
        }

        // Copy the reference outputs to the host once. Every entry is compared against this
        // copy, so neither the device nor the reference executor is consulted again.
        for(auto& [uid, tensor] : _expected)
        {
            tensor->rawHostData();
        }
        return result(Status::READY, {});
    }
    catch(const std::exception& e)
    {
        return result(Status::FAILED, std::string("preparing the reference threw: ") + e.what());
    }
}

EngineList FrontendCatalogGraphRunner::listEngines()
{
    EngineList list;
    if(auto error = openGraph(); !error.empty())
    {
        list.ok = false;
        list.error = error;
        return list;
    }

    std::vector<hipdnn_frontend::EngineConfigInfo> configs;
    if(auto err = _graph->get_engine_configs(getSharedHandle(), configs); !err.is_good())
    {
        list.ok = false;
        list.error = err.get_message();
        // The frontend's answer for "no engine has a configuration for this graph": the
        // same code the verification harness's isResolved() accepts as a decline.
        list.notSupported = err.get_code() == hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED;
        return list;
    }

    list.engines.reserve(configs.size());
    for(const auto& config : configs)
    {
        list.engines.push_back({config.engineId, config.engineName});
    }
    return list;
}

CatalogFetch FrontendCatalogGraphRunner::fetchCatalog(const CatalogEngine& engine)
{
    CatalogFetch fetch;
    std::vector<std::vector<hipdnn_frontend::KnobSetting>> catalog;
    try
    {
        if(auto err = _graph->get_knob_catalog_for_engine(engine.id, catalog); !err.is_good())
        {
            fetch.ok = false;
            fetch.error = err.get_message();
            return fetch;
        }
    }
    catch(const std::exception& e)
    {
        fetch.ok = false;
        fetch.error = std::string("get_knob_catalog_for_engine threw: ") + e.what();
        return fetch;
    }

    fetch.entries.reserve(catalog.size());
    for(const auto& entry : catalog)
    {
        fetch.entries.push_back({toCatalogKnobs(entry)});
    }
    _catalogs[engine.id] = std::move(catalog);
    return fetch;
}

EntryOutcome FrontendCatalogGraphRunner::runEntry(const CatalogEngine& engine,
                                                  std::size_t entryIndex)
{
    EntryOutcome outcome;
    EntryStage stage = EntryStage::CREATE_PLAN;
    const auto fail = [&](EntryStage failedAt, std::string message) {
        outcome.failedStage = failedAt;
        outcome.message = std::move(message);
        return outcome;
    };

    try
    {
        auto& graph = *_graph;
        const auto& settings = _catalogs.at(engine.id).at(entryIndex);

        // Time spent building the plan: engines that compile at plan-build time spend
        // nearly all of it here.
        const auto compileStart = std::chrono::steady_clock::now();
        const auto buildStage = [&](EntryStage at, const hipdnn_frontend::Error& err) {
            stage = at;
            outcome.compileTimeMs = millisecondsSince(compileStart);
            return err.is_good() ? std::string{} : err.get_message();
        };

        if(auto message = buildStage(EntryStage::CREATE_PLAN,
                                     graph.create_execution_plan_ext(engine.id, settings));
           !message.empty())
        {
            return fail(EntryStage::CREATE_PLAN, std::move(message));
        }
        if(auto message = buildStage(EntryStage::CHECK_SUPPORT, graph.check_support());
           !message.empty())
        {
            return fail(EntryStage::CHECK_SUPPORT, std::move(message));
        }
        if(auto message = buildStage(EntryStage::BUILD_PLANS, graph.build_plans());
           !message.empty())
        {
            return fail(EntryStage::BUILD_PLANS, std::move(message));
        }

        stage = EntryStage::EXECUTE;
        int64_t workspaceSize = 0;
        if(auto err = graph.get_workspace_size(workspaceSize); !err.is_good())
        {
            return fail(EntryStage::EXECUTE, "get_workspace_size: " + err.get_message());
        }
        if(workspaceSize < 0)
        {
            return fail(EntryStage::EXECUTE, "the engine reported a negative workspace size");
        }
        const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        // Refill the inputs and reset the outputs to the NaN sentinel: nothing a previous
        // entry's kernel wrote may be visible to this one.
        auto inputs = bundle::prepareBundleInputs(
            *_config.bundle,
            _recipes,
            _config.policy.useDevice() ? _config.deviceFiller.get() : nullptr);
        if(inputs.status != bundle::BundleInputs::Status::READY)
        {
            return fail(EntryStage::EXECUTE, "could not prepare the inputs: " + inputs.message);
        }

        const auto wrapper = _config.bundle->graphWrapper();
        auto outputs = bundle::detail::allocateSentinelOutputs(
            wrapper.getTensorMap(), _config.bundle->outputTensorUids, /*onDevice=*/true);
        auto variantPack = bundle::detail::buildVariantPack(
            !inputs.packedInputs.empty() ? inputs.packedInputs : inputs.inputs,
            outputs,
            wrapper.getTensorMap(),
            _config.bundle->outputTensorUids,
            /*useDevice=*/true);

        if(auto err = graph.execute(getSharedHandle(), variantPack, workspace.get());
           !err.is_good())
        {
            return fail(EntryStage::EXECUTE, err.get_message());
        }
        bundle::detail::markOutputsModified(outputs, /*device=*/true);

        stage = EntryStage::COMPARE;
        compareOutputs(outputs,
                       "Bundle: " + _config.bundlePath + " engine=" + engine.name
                           + " entry=" + std::to_string(entryIndex),
                       outcome);
        return outcome;
    }
    // A provider that declines late is still an engine that did not run this entry.
    catch(const EngineNotApplicableError& e)
    {
        return fail(stage, std::string("engine declined: ") + e.what());
    }
    catch(const std::exception& e)
    {
        return fail(stage, std::string("threw: ") + e.what());
    }
}

void FrontendCatalogGraphRunner::compareOutputs(bundle::OutputTensors& actual,
                                                const std::string& contextLine,
                                                EntryOutcome& outcome)
{
    const auto& testBundle = *_config.bundle;
    const auto wrapper = testBundle.graphWrapper();
    const auto& tensorMap = wrapper.getTensorMap();
    // The reference now lives on the host, so that is where the comparison runs unless
    // --validator says otherwise.
    const auto site = resolveValidationSite(_config.policy.validator, ValidationSite::HOST);

    std::size_t measuredElements = 0;
    std::size_t mismatchedElements = 0;
    double maxAbsErr = 0.0;
    bool measuredAny = false;
    std::ostringstream message;

    for(const int64_t uid : testBundle.outputTensorUids)
    {
        const auto* attrs = tensorMap.at(uid);
        const auto label = bundle::tensorLabel(uid, *attrs);
        const auto dataType = attrs->data_type();

        // The same grading a normal run of this bundle applies, TOML overrides included.
        const float defaultValue = tolerance::defaultTolerance(wrapper, dataType);
        const auto grading = gradingForTensor(_config.testName, label, defaultValue, defaultValue);

        auto& expected = *_expected.at(uid);
        auto& output = *actual.at(uid);

        const auto mismatch
            = bundle::compareTensor(uid, *attrs, expected, output, grading, site, contextLine);

        // RMS grading has no per-element budget, so a per-element mismatch share would
        // describe a check that was never made.
        std::optional<OutputDiffStats> stats;
        if(grading.kind != bundle::ValidatorKind::RMS)
        {
            stats = computeOutputDiffStats(dataType, expected, output, grading.atol, grading.rtol);
        }

        if(stats.has_value())
        {
            measuredAny = true;
            measuredElements += stats->totalElements;
            mismatchedElements += stats->mismatchedElements;
            maxAbsErr = std::max(maxAbsErr, stats->maxAbsErr);
            outcome.nanElements += stats->nanElements;
        }

        if(mismatch.has_value())
        {
            outcome.outputMismatch = true;
            message << "output " << label << " mismatches the reference";
            if(stats.has_value())
            {
                message << " (max_abs_err=" << stats->maxAbsErr
                        << ", mismatched=" << stats->mismatchedElements << "/"
                        << stats->totalElements << ", nan=" << stats->nanElements << ")";
            }
            else
            {
                message << ": " << truncated(mismatch->report, 400);
            }
            message << "; ";
        }
        else if(stats.has_value() && stats->nanElements > 0)
        {
            message << "output " << label << " holds " << stats->nanElements << " NaN element(s); ";
        }
    }

    if(measuredAny)
    {
        outcome.maxAbsErr = maxAbsErr;
        outcome.mismatchPct = measuredElements == 0
                                  ? 0.0
                                  : 100.0 * static_cast<double>(mismatchedElements)
                                        / static_cast<double>(measuredElements);
    }
    outcome.message = message.str();
}

} // namespace hipdnn_integration_tests::catalog_sweep
