// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "harness/BundleMetadata.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/ComparisonReport.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "harness/ReferenceCapabilityError.hpp"
#include "harness/TomlGuards.hpp"
#include "harness/bundle/LoadedEngine.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportObservationLog.hpp"
#include "harness/bundle/SupportVerdict.hpp"
#include "harness/bundle/VariantPackBuilder.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace
{

// The --verification-mode value that selects `mode`, for messages.
const char* modeName(VerificationMode mode)
{
    switch(mode)
    {
    case VerificationMode::AUTO:
        return "auto";
    case VerificationMode::GOLDEN:
        return "golden";
    case VerificationMode::GPU:
        return "gpu";
    case VerificationMode::CPU:
        return "cpu";
    default:
        return "unknown";
    }
}

} // namespace

// ---- the one graph, the one query ------------------------------------------

GraphSession IntegrationBundleVerificationHarness::openGraph()
{
    if(_bundle == nullptr)
    {
        return GraphSession{};
    }
    return _deps.engineRunner->openGraph(*_bundle, _engineUnderTest);
}

void IntegrationBundleVerificationHarness::applyMetadataGuards() const
{
    if(auto reason = checkVramRequirement(_bundle->metadata, _deps.policy.deviceVramMb))
    {
        noteSkipBeforeObservation();
        GTEST_SKIP() << *reason;
    }

    if(auto reason = checkArchCompatibility(_bundle->metadata, _deps.policy.arch))
    {
        noteSkipBeforeObservation();
        GTEST_SKIP() << *reason;
    }
}

// ---- support claims --------------------------------------------------------

SupportObservation
    IntegrationBundleVerificationHarness::observeSupportClaims(const GraphSession& session)
{
    if(_bundle == nullptr || !shouldObserveClaims())
    {
        return {};
    }

    if(session.buildFailed)
    {
        // Silent on purpose: runComparison() reports this same build failure as the
        // test's outcome, and one fault deserves one message. NOT_QUERIED rather
        // than the default NONE so the coverage rules can tell "the query was
        // impossible" from "a sidecar sat there and nothing looked at it" — the
        // second is a harness bug, this is not.
        return SupportObservation{SidecarState::NOT_QUERIED, {}};
    }

    return _deps.claimObserver->observe(session.engines,
                                        _claimLocator,
                                        *_engineUnderTest,
                                        baseArchToken(_deps.policy.arch),
                                        _deps.policy.platform);
}

ClaimPhase IntegrationBundleVerificationHarness::observeClaims(const GraphSession& session)
{
    ClaimPhase phase;

    phase.observation = observeSupportClaims(session);

    // Everything from here down derives from the observation, so a throw above loses
    // only facts that were not yet true. graphsReachedBody is the exception and is
    // published from TestBody() instead, ahead of anything that can throw.
    const CoverageUpdate update = coverageFor(phase.observation, shouldObserveClaims());

    _deps.reporter->recordCoverage(update);

    phase.complaint = missedQueryComplaint(update, _bundlePath.string());
    return phase;
}

void IntegrationBundleVerificationHarness::raiseComplaints(
    std::initializer_list<std::optional<HarnessComplaint>> complaints)
{
    for(const auto& complaint : complaints)
    {
        if(!complaint.has_value())
        {
            continue;
        }

        // ADD_FAILURE() rather than FAIL(): FAIL() returns, and the caller still has
        // an outcome to report and possibly a second complaint to raise.
        ADD_FAILURE() << complaint->message;
    }
}

// The decision is finalizeClaims(); this only publishes what it returns. results is
// only ever non-empty when an engine was injected, so there is no engine-less case
// to handle here.
void IntegrationBundleVerificationHarness::commitClaims(const std::vector<SupportResult>& results,
                                                        const VerificationOutcome& outcome)
{
    if(!_engineUnderTest.has_value())
    {
        return;
    }

    for(const auto& record :
        finalizeClaims(results, _engineUnderTest->name, outcome, bundleRequiredDepth()))
    {
        _deps.reporter->recordVerdict(record);
    }
}

void IntegrationBundleVerificationHarness::reportOutcome(const VerificationOutcome& outcome)
{
    switch(outcome.status)
    {
    case OutcomeStatus::PASSED:
        return;
    case OutcomeStatus::SKIPPED:
        GTEST_SKIP() << outcome.message;
    case OutcomeStatus::FAILED:
        if(!outcome.message.empty())
        {
            FAIL() << outcome.message;
        }
        // A silent return is honest only for the one producer that promises the
        // detail is already on the record. Anything else reaching here is a failure
        // with nothing to say, and a bare failure beats a green run.
        if(!outcome.alreadyReported)
        {
            FAIL() << "verification failed at " << toString(outcome.depth) << " ("
                   << toString(outcome.origin) << ") with no message";
        }
        return;
    default:
        FAIL() << "Unknown outcome status";
        return;
    }
}

// ---- top-level dispatch ----------------------------------------------------

VerificationOutcome IntegrationBundleVerificationHarness::enforceAtLevel(EnforcementLevel level,
                                                                         GraphSession& session)
{
    if(level == EnforcementLevel::FULL)
    {
        return VerificationOutcome::failed(
            VerificationDepth::NOT_REACHED,
            FailureOrigin::HARNESS,
            "enforceAtLevel() handles APPLICABILITY/BUILDABLE only; FULL uses the normal path");
    }

    // The rung itself needs a named engine to check applicability against; claim
    // enforcement already ran in TestBody() and is not affected by this return.
    if(!_engineUnderTest.has_value())
    {
        return unverifiable("enforcement_level requires --test-engine");
    }

    const std::string rung
        = level == EnforcementLevel::APPLICABILITY ? "applicability" : "buildable";

    // Same applicability answer the claim verdict and the executor read.
    if(!session.engines.accepted)
    {
        return unverifiable("Engine " + _engineUnderTest->name
                            + " does not support this graph (enforcement_level=" + rung + ")");
    }

    if(level == EnforcementLevel::APPLICABILITY)
    {
        return VerificationOutcome::passed(VerificationDepth::APPLICABLE);
    }

    // BUILDABLE: additionally compile plans. A rung failure is the engine's doing,
    // and says so through the outcome rather than through a bare assertion.
    const EngineOpResult built = _deps.engineRunner->buildPlans(session, _engineUnderTest);

    // A provider is allowed to decline later than the ranked list suggested. That
    // is the same answer as `!accepted` above, just arrived at later, so it lands
    // in the same place rather than being blamed on the engine as a break.
    if(built.declined)
    {
        return unverifiable("Engine " + _engineUnderTest->name
                            + " declined this graph while compiling plans (enforcement_level="
                            + rung + "): " + built.message);
    }
    if(!built.ok)
    {
        return VerificationOutcome::failed(VerificationDepth::APPLICABLE,
                                           FailureOrigin::ENGINE,
                                           "[rung=buildable] " + built.message);
    }
    return VerificationOutcome::passed(VerificationDepth::BUILDABLE);
}

std::vector<ObservedGraphSupport> IntegrationBundleVerificationHarness::observeSupportOnly(
    const GraphSession& session, const std::vector<LoadedEngine>& engines)
{
    if(session.buildFailed)
    {
        HIPDNN_PLUGIN_LOG_WARN("observeSupportOnly: from_binary failed for " << _bundlePath << ": "
                                                                             << session.buildError);
        return {};
    }

    if(!isResolved(session.engines.status.get_code()))
    {
        HIPDNN_PLUGIN_LOG_WARN("observeSupportOnly: unresolved query for "
                               << _bundlePath << ": " << session.engines.status.get_message());
        return {};
    }

    const std::string arch = baseArchToken(_deps.policy.arch);
    const auto& rankedIds = session.engines.rankedIds;

    auto observe = [&](const LoadedEngine& engine) {
        const bool engineIsSupported
            = std::find(rankedIds.begin(), rankedIds.end(), engine.id) != rankedIds.end();
        return ObservedGraphSupport{
            _claimLocator, engine.name, arch, _deps.policy.platform, engineIsSupported};
    };

    std::vector<ObservedGraphSupport> observations;

    if(_engineUnderTest)
    {
        // --test-engine was given: observe only that engine.
        observations.push_back(observe(*_engineUnderTest));
    }
    else
    {
        // No --test-engine: observe every loaded engine plugin.
        observations.reserve(engines.size());
        for(const auto& engine : engines)
        {
            observations.push_back(observe(engine));
        }
    }

    return observations;
}

void IntegrationBundleVerificationHarness::observeAndRecordSupport(const GraphSession& session)
{
    // Handed over whole, empty result included -- the early returns above leave a
    // graph this run cannot refresh, and the log counts those for the summary.
    SupportObservationLog::get().recordGraph(
        observeSupportOnly(session, LoadedEngineTable::get().all()));
}

VerificationOutcome IntegrationBundleVerificationHarness::runComparison(GraphSession& session)
{
    // A graph that would not load is the engine's problem, at every level, and it is
    // the reason nothing below can run. Checked once, here, so the rungs and the
    // modes can all assume a usable session.
    if(session.buildFailed)
    {
        return VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                           FailureOrigin::ENGINE,
                                           "from_binary failed: " + session.buildError);
    }

    if(_bundle->metadata.enforcementLevel != EnforcementLevel::FULL)
    {
        return enforceAtLevel(_bundle->metadata.enforcementLevel, session);
    }

    if(_bundle->outputTensorUids.empty())
    {
        return unverifiable("bundle has no output tensors to compare");
    }

    // A graph the engine declined never reads its inputs: every mode below reaches
    // runEngine() -- which reports the decline -- before anything touches
    // _inputs. Filling first made a declined graph pay the full host-side
    // allocation and RNG fill for its tensors, which on a 57M-element sweep case is
    // seconds per skip, and reading a golden bundle's blobs first made it pay for
    // those too.
    if(session.engines.accepted)
    {
        if(auto unavailable = prepareInputs())
        {
            return *unavailable;
        }
    }

    switch(_deps.policy.mode)
    {
    case VerificationMode::GOLDEN:
        return runGoldenMode(session);
    case VerificationMode::GPU:
    case VerificationMode::CPU:
    case VerificationMode::AUTO:
        return runReferenceMode(session);
    default:
        return VerificationOutcome::failed(
            VerificationDepth::NOT_REACHED, FailureOrigin::HARNESS, "Unknown verification mode");
    }
}

IntegrationBundleVerificationHarness::OracleChain
    IntegrationBundleVerificationHarness::resolveOracles(VerificationMode mode)
{
    OracleChain chain;
    chain.mode = mode;
    switch(mode)
    {
    case VerificationMode::AUTO:
        chain.candidates = {ReferenceExecutorType::GPU, ReferenceExecutorType::CPU};
        break;
    case VerificationMode::GPU:
        chain.candidates = {ReferenceExecutorType::GPU};
        break;
    case VerificationMode::CPU:
        chain.candidates = {ReferenceExecutorType::CPU};
        break;
    case VerificationMode::GOLDEN:
    default:
        // runComparison() sends only the reference modes here: golden mode demands its
        // one oracle in runGoldenMode(), and an unknown mode fails there. An empty
        // chain would FAIL as a bundle no oracle can verify, blaming the bundle for a
        // harness bug.
        throw std::invalid_argument(
            std::string("resolveOracles: no reference chain for verification-mode=")
            + modeName(mode));
    }
    return chain;
}

std::optional<IntegrationBundleVerificationHarness::ResolvedReference>
    IntegrationBundleVerificationHarness::probeReference(OracleChain& chain,
                                                         ReferenceExecutorType type)
{
    const std::string label = refLabel(type);
    // Still null in the catch below if creating the executor is what threw.
    IReferenceGraphExecutor* executor = nullptr;
    try
    {
        executor = &_deps.referenceExecutors->get(type);
        if(executor->isApplicable(_bundle->graphBuffer.data(), _bundle->graphBuffer.size()))
        {
            return ResolvedReference{type, executor};
        }
        chain.declined(label + " (not applicable)");
    }
    catch(const ReferenceCapabilityError& e)
    {
        chain.declined(label + " (not applicable: " + e.what() + ")");
    }
    catch(const std::exception& e)
    {
        std::string detail
            = executor != nullptr ? "errored checking applicability" : "could not be created";
        detail.append(": ").append(e.what());
        std::string entry = label;
        entry.append(" (").append(detail).append(")");
        std::string error = label;
        error.append(" ").append(detail);
        referenceErrored(chain, std::move(entry), std::move(error));
    }
    return std::nullopt;
}

void IntegrationBundleVerificationHarness::referenceErrored(OracleChain& chain,
                                                            std::string entry,
                                                            std::string error)
{
    recordRefError(error);
    chain.tried.push_back({std::move(entry), std::move(error)});
}

VerificationOutcome
    IntegrationBundleVerificationHarness::engineDidNotRun(const EngineRunResult& run) const
{
    // The rung the engine actually cleared. A bundle that only has to compile still
    // gets credit for compiling, even though what came next did not work.
    const VerificationDepth reached
        = run.plansBuilt ? VerificationDepth::BUILDABLE : VerificationDepth::NOT_REACHED;

    if(run.status == EngineStatus::DECLINED)
    {
        std::ostringstream msg;
        msg << "Engine could not execute bundle " << _bundlePath;
        if(!run.message.empty())
        {
            msg << ": " << run.message;
        }
        return VerificationOutcome::skipped(reached, msg.str());
    }

    // ERRORED: the engine broke while compiling or executing, and the runner handed
    // back the frontend's own message. The engine is at fault.
    //
    // Named unconditionally: an empty message on a FAILED outcome means "the failure
    // is already on the gtest record", which only the comparison can promise. A
    // silent green test is the exact shape this harness exists to rule out.
    std::ostringstream msg;
    msg << "Engine failed on bundle " << _bundlePath;
    if(!run.message.empty())
    {
        msg << ": " << run.message;
    }
    return VerificationOutcome::failed(reached, FailureOrigin::ENGINE, msg.str());
}

VerificationOutcome IntegrationBundleVerificationHarness::runGoldenMode(GraphSession& session)
{
    // An explicit --verification-mode=golden is a demand for a specific oracle, not
    // a preference. Skipping when that oracle is absent means the run did not do
    // what it was asked and still went green — use `auto` if a fallback chain is
    // what you want.
    if(!_bundle->hasGoldenOutputs)
    {
        return VerificationOutcome::failed(
            VerificationDepth::NOT_REACHED,
            FailureOrigin::HARNESS,
            "verification-mode=golden was requested but this bundle has no golden "
            "data; run `dvc pull` for it, or use --verification-mode=auto");
    }

    auto engine = runEngine(session);
    if(engine.status != EngineStatus::RAN)
    {
        return engineDidNotRun(engine);
    }
    return compareAgainstGolden(engine.outputs);
}

VerificationOutcome IntegrationBundleVerificationHarness::runReferenceMode(GraphSession& session)
{
    // The engine runs before any oracle is looked at. A decline (a SKIP) or a break
    // (the engine's FAIL) needs no oracle to report, so only an engine that ran
    // creates or probes a reference.
    auto engine = runEngine(session);
    if(engine.status != EngineStatus::RAN)
    {
        return engineDidNotRun(engine);
    }

    auto oracles = resolveOracles(_deps.policy.mode);
    if(oracles.mode == VerificationMode::AUTO)
    {
        // Golden data is auto mode's first oracle; when the bundle has it, no
        // reference is consulted.
        if(_bundle->hasGoldenOutputs)
        {
            return compareAgainstGolden(engine.outputs);
        }
        oracles.declined("golden (absent)");
    }
    return runOracleChain(engine.outputs, oracles);
}

VerificationOutcome
    IntegrationBundleVerificationHarness::runOracleChain(OutputTensors& engineOutputs,
                                                         OracleChain& chain)
{
    // isApplicable() said yes, but execute() can still find a capability gap the
    // check could not see, or crash. Either way the next candidate gets its turn;
    // only once the chain is spent does the bundle go without a verdict.
    for(std::size_t i = 0; i < chain.candidates.size(); ++i)
    {
        const auto ref = probeReference(chain, chain.candidates[i]);
        if(!ref)
        {
            continue;
        }
        const bool lastResort = i + 1 == chain.candidates.size();

        OutputTensors refOutputs;
        const RefRunResult result = runReferenceCapturingOutputs(*ref, refOutputs);
        const std::string label = refLabel(ref->type);
        switch(result.status)
        {
        case RefStatus::RAN:
            return compareOutputs(engineOutputs, refOutputs, result.site, verifierFor(ref->type));
        case RefStatus::CAPABILITY_MISS:
            chain.declined(label + " (cannot run this op: " + result.message + ")");
            break;
        case RefStatus::RUNTIME_ERROR:
        {
            // "the next reference", not a name: the next candidate has not been probed
            // yet and may turn out not to be applicable.
            std::string context;
            if(chain.mode != VerificationMode::AUTO)
            {
                context = std::string("verification-mode=") + modeName(chain.mode);
            }
            else if(lastResort)
            {
                context = "auto mode, last resort";
            }
            else
            {
                context = "auto mode, falling through to the next reference";
            }
            std::string entry = label;
            entry.append(" (errored: ").append(result.message).append(")");
            std::string error = label;
            error.append(" errored (").append(context).append("): ").append(result.message);
            referenceErrored(chain, std::move(entry), std::move(error));
            break;
        }
        case RefStatus::HARNESS_ERROR:
            // Not the reference's fault, so it neither falls through to the next one nor
            // goes in the reference-error report.
            return VerificationOutcome::failed(VerificationDepth::EXECUTED,
                                               FailureOrigin::HARNESS,
                                               label + " was not run: " + result.message);
        default:
            return VerificationOutcome::failed(
                VerificationDepth::EXECUTED, FailureOrigin::HARNESS, "Unknown RefStatus");
        }
    }

    // The chain is spent. An oracle that broke and one that declined are different
    // verdicts: the first is a bug in the oracle, the second leaves the bundle
    // unverifiable.
    if(chain.lastErrored())
    {
        return lastOracleErrored(chain);
    }
    return noOracle(chain);
}

VerificationOutcome
    IntegrationBundleVerificationHarness::lastOracleErrored(const OracleChain& chain) const
{
    // Led by the reference error as recorded, so the FAIL and its entry in the
    // reference-error report read the same.
    std::string message = *chain.tried.back().error;
    message.append("; ").append(describeTried(chain));
    message.append(" (").append(_bundlePath.string()).append(")");
    return VerificationOutcome::failed(
        VerificationDepth::EXECUTED, FailureOrigin::ORACLE, std::move(message));
}

VerificationOutcome IntegrationBundleVerificationHarness::noOracle(const OracleChain& chain)
{
    const bool refErrored = chain.anyErrored();
    std::string reason;
    if(refErrored)
    {
        reason = "a reference executor errored (see the reference-error report) and ";
    }
    // Auto mode exhausted every oracle; an explicit mode only the one it demanded.
    reason.append(chain.mode == VerificationMode::AUTO
                      ? "no oracle can verify this bundle"
                      : "the requested oracle cannot verify this bundle");
    reason.append("; ").append(describeTried(chain));

    return VerificationOutcome::failed(VerificationDepth::EXECUTED,
                                       refErrored ? FailureOrigin::ORACLE : FailureOrigin::HARNESS,
                                       recordUnverifiable(reason));
}

std::string IntegrationBundleVerificationHarness::describeTried(const OracleChain& chain) const
{
    std::string tried = "tried: ";
    for(std::size_t i = 0; i < chain.tried.size(); ++i)
    {
        tried.append(i == 0 ? "" : ", ").append(chain.tried[i].entry);
    }
    // An explicit mode never consults golden data. Say so when it is there, so the
    // message does not send anyone looking for data that is sitting in the bundle.
    if(chain.mode != VerificationMode::AUTO && _bundle->hasGoldenOutputs)
    {
        tried.append("; golden data is present but not used under --verification-mode=")
            .append(modeName(chain.mode));
    }
    return tried;
}

// ---- inputs ----------------------------------------------------------------

std::optional<VerificationOutcome> IntegrationBundleVerificationHarness::prepareInputs()
{
    if(!_bundle->blobs.has_value())
    {
        return fillBundleInputs();
    }

    // Tensors read from blobs are unpacked, but the engine reads sub-byte operands
    // packed, and only fillBundleInputs() builds the packed set.
    const auto wrapper = _bundle->graphWrapper();
    const std::set<int64_t> outputUids(_bundle->outputTensorUids.begin(),
                                       _bundle->outputTensorUids.end());
    for(const auto& [uid, attrs] : wrapper.getTensorMap())
    {
        if(!attrs->virtual_() && outputUids.count(uid) == 0
           && hipdnn_test_sdk::detail::isSubByteDataType(attrs->data_type()))
        {
            return unverifiable("sub-byte input " + std::to_string(uid)
                                + " has no packed copy for the engine");
        }
    }

    // Read here, not at registration, so a golden bundle holds its tensors only while
    // its own test runs. A blob that is unreadable or the wrong size fails this test
    // instead of quietly dropping it from the run.
    try
    {
        _inputs = _bundle->loadTensors();
    }
    catch(const std::exception& e)
    {
        return VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                           FailureOrigin::HARNESS,
                                           std::string("tensor data failed to load: ") + e.what()
                                               + " (" + _bundlePath.string() + ")");
    }
    return std::nullopt;
}

std::optional<VerificationOutcome> IntegrationBundleVerificationHarness::fillBundleInputs()
{
    const auto wrapper = _bundle->graphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();

    // The generic fill recipes cannot produce a valid monotonic offset table, and
    // garbage offsets would send the engine reading out of bounds.
    if(!hipdnn_flatbuffers_sdk::utilities::hasNoRaggedTensorIds(tensorAttrMap))
    {
        return unverifiable("ragged inputs require golden blobs");
    }

    const std::set<int64_t> outputUids(_bundle->outputTensorUids.begin(),
                                       _bundle->outputTensorUids.end());

    InputTensorMap inputs;
    std::vector<int64_t> leafInputUids;
    bool anySubByte = false;
    for(const auto& [uid, attrs] : tensorAttrMap)
    {
        if(attrs->virtual_() || outputUids.count(uid) != 0)
        {
            continue;
        }
        inputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(*attrs);
        leafInputUids.push_back(uid);
        anySubByte = anySubByte || hipdnn_test_sdk::detail::isSubByteDataType(attrs->data_type());
    }

    // Sub-byte graphs fill twice, once unpacked and once packed, and the two sets must
    // hold the same values. rocRAND does not fill sub-byte types and its stream differs
    // from the host's, so a graph with any sub-byte tensor is filled on the host
    // throughout. That is a deliberately conservative choice, not a correctness
    // requirement: a large non-sub-byte operand would get the same device values in
    // both sets, so relaxing it would only buy back speed on the large MX shapes.
    DeviceInputFiller* const device
        = _deps.policy.useDevice() && !anySubByte ? _deps.deviceFiller.get() : nullptr;

    FillResult fillResult;
    try
    {
        fillResult = hipdnn_integration_tests::fillInputs(
            wrapper.getGraph(), inputs, leafInputUids, _inputFillRecipes, device);
    }
    catch(const DeviceInputError& e)
    {
        // The device failed, not the graph, so this is a failure and not an
        // "unverifiable" skip that would let the run go green.
        return VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                           FailureOrigin::HARNESS,
                                           std::string(e.what()) + " (" + _bundlePath.string()
                                               + "); set HIPDNN_TEST_HOST_INPUT_FILL=1 to fill on "
                                                 "the host");
    }
    if(!fillResult.filled)
    {
        return unverifiable(fillResult.reason);
    }
    _deviceFilledInputs = fillResult.deviceFilled;

    if(anySubByte)
    {
        InputTensorMap packed;
        for(const int64_t uid : leafInputUids)
        {
            packed[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(
                *tensorAttrMap.at(uid), /*packSubByteElements=*/true);
        }

        auto packedFill = hipdnn_integration_tests::fillInputs(
            wrapper.getGraph(), packed, leafInputUids, _inputFillRecipes, device);
        if(!packedFill.filled)
        {
            return unverifiable(packedFill.reason);
        }
        _packedInputs = std::move(packed);
    }

    _inputs = std::move(inputs);
    return std::nullopt;
}

// ---- engine + reference runs -----------------------------------------------

OutputTensors IntegrationBundleVerificationHarness::allocateSentinelOutputs(bool onDevice) const
{
    const auto wrapper = _bundle->graphWrapper();
    return detail::allocateSentinelOutputs(
        wrapper.getTensorMap(), _bundle->outputTensorUids, _inputs, onDevice);
}

std::unordered_map<int64_t, void*>
    IntegrationBundleVerificationHarness::buildVariantPack(OutputTensors& outputs, bool useDevice)
{
    const auto wrapper = _bundle->graphWrapper();
    TensorMap& inputs = (useDevice && !_packedInputs.empty()) ? _packedInputs : _inputs;
    return detail::buildVariantPack(
        inputs, outputs, wrapper.getTensorMap(), _bundle->outputTensorUids, useDevice);
}

IntegrationBundleVerificationHarness::EngineRunResult
    IntegrationBundleVerificationHarness::runEngine(GraphSession& session)
{
    EngineRunResult run;

    // Decided once, in openGraph(), so applicability is one fact rather than three
    // opinions. A test states what it is simulating by setting engines.accepted on
    // the session it hands back.
    if(!session.engines.accepted)
    {
        const auto summary
            = std::to_string(_bundle->outputTensorUids.size()) + " output tensor(s), "
              + std::to_string(session.engines.rankedIds.size()) + " ranked engine(s)";
        run.status = EngineStatus::DECLINED;
        run.message = _engineUnderTest.has_value()
                          ? "Engine " + _engineUnderTest->name + " does not support this graph ("
                                + summary + ")"
                          : "No engine supports this graph (" + summary + ")";
        return run;
    }

    run.outputs = allocateSentinelOutputs(_deps.policy.useDevice());
    auto variantPack = buildVariantPack(run.outputs, _deps.policy.useDevice());

    // The runner reports rather than asserts, so "the engine broke" is a value here
    // instead of a disposition read back off GTest, which cannot tell an engine
    // assertion apart from any other fatal failure in the same test.
    const EngineOpResult result
        = _deps.engineRunner->execute(session, _engineUnderTest, variantPack);

    run.plansBuilt = result.plansBuilt;

    if(result.declined)
    {
        run.status = EngineStatus::DECLINED;
        run.message = result.message;
        return run;
    }
    if(!result.ok)
    {
        run.status = EngineStatus::ERRORED;
        run.message = result.message;
        return run;
    }

    markOutputsModified(run.outputs);
    run.status = EngineStatus::RAN;
    return run;
}

IntegrationBundleVerificationHarness::RefRunResult
    IntegrationBundleVerificationHarness::runReferenceCapturingOutputs(const ResolvedReference& ref,
                                                                       OutputTensors& refOutputs)
{
    // Only an executor that asks for device pointers gets them. Handing host memory
    // to an executor that wants device memory — or the reverse — is a silent crash,
    // not an error, and the executor is the one that knows which it needs.
    bool useDevice = false;

    // isApplicable() already said yes in probeReference(); execute() can
    // still throw a ReferenceCapabilityError for what that check could not see.
    try
    {
        IReferenceGraphExecutor& executor = *ref.executor;
        useDevice = _deps.policy.useDevice() && executor.requiresDeviceMemory();
        refOutputs = allocateSentinelOutputs(useDevice);
        auto variantPack = buildVariantPack(refOutputs, useDevice);
        executor.execute(_bundle->graphBuffer.data(), _bundle->graphBuffer.size(), variantPack);
    }
    catch(const ReferenceCapabilityError& e)
    {
        return {RefStatus::CAPABILITY_MISS, e.what()};
    }
    catch(const detail::DeviceOutputError& e)
    {
        return {RefStatus::HARNESS_ERROR, e.what()};
    }
    catch(const std::exception& e)
    {
        return {RefStatus::RUNTIME_ERROR, e.what()};
    }

    detail::markOutputsModified(refOutputs, useDevice);
    return {RefStatus::RAN, {}, useDevice ? ValidationSite::DEVICE : ValidationSite::HOST};
}

void IntegrationBundleVerificationHarness::markOutputsModified(OutputTensors& outputs) const
{
    detail::markOutputsModified(outputs, _deps.policy.useDevice());
}

// ---- comparison ------------------------------------------------------------

VerificationOutcome
    IntegrationBundleVerificationHarness::compareAgainstGolden(OutputTensors& engineOutputs)
{
    return compareAgainst(
        engineOutputs,
        [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& { return *_inputs.at(uid); },
        ValidationSite::HOST,
        Verifier::GOLDEN);
}

VerificationOutcome IntegrationBundleVerificationHarness::compareOutputs(
    OutputTensors& engineOutputs, OutputTensors& expected, ValidationSite site, Verifier verifier)
{
    return compareAgainst(
        engineOutputs,
        [&](int64_t uid) -> hipdnn_data_sdk::utilities::ITensor& { return *expected.at(uid); },
        site,
        verifier);
}

VerificationOutcome
    IntegrationBundleVerificationHarness::compareAgainst(OutputTensors& engineOutputs,
                                                         const ExpectedTensorLookup& expectedFor,
                                                         ValidationSite site,
                                                         Verifier verifier)
{
    auto wrapper = _bundle->graphWrapper();
    // defaultTolerance() rather than resolveTolerance(): the TOML tolerance override is
    // applied by gradingForTensor(), which reads the validator override first and so is
    // the only place that can log the check that actually graded this tensor.
    const auto toleranceFor
        = [&](const std::string& label, hipdnn_flatbuffers_sdk::data_objects::DataType dataType) {
              const float value = tolerance::defaultTolerance(wrapper, dataType);
              return gradingForTensor(currentTestName(), label, value, value);
          };

    const auto mismatches
        = bundle::compareOutputs(wrapper,
                                 _bundle->outputTensorUids,
                                 engineOutputs,
                                 expectedFor,
                                 toleranceFor,
                                 resolveValidationSite(_deps.policy.validator, site),
                                 "Bundle: " + _bundlePath.string() + inputFillNote());

    // Reported one per tensor so each diff lands next to the tensor it describes;
    // the outcome carries no message because of it.
    for(const auto& mismatch : mismatches)
    {
        ADD_FAILURE() << mismatch.report;
    }

    return comparisonOutcome(mismatches.empty(), verifier);
}

// ---- reporting helpers -----------------------------------------------------

std::string IntegrationBundleVerificationHarness::recordUnverifiable(const std::string& reason)
{
    _deps.reporter->recordUnverifiable(_bundlePath.string(), reason);
    std::string message = "Unverifiable: ";
    message.append(reason).append(" (").append(_bundlePath.string()).append(")");
    return message;
}

VerificationOutcome IntegrationBundleVerificationHarness::unverifiable(const std::string& reason,
                                                                       VerificationDepth reached)
{
    return VerificationOutcome::skipped(reached, recordUnverifiable(reason));
}

void IntegrationBundleVerificationHarness::recordRefError(const std::string& reason)
{
    _deps.reporter->recordReferenceError(_bundlePath.string(), reason);
}

std::string IntegrationBundleVerificationHarness::inputFillNote() const
{
    if(_deviceFilledInputs == 0)
    {
        return {};
    }
    return " [" + std::to_string(_deviceFilledInputs)
           + " input tensor(s) filled on the device with rocRAND; set "
             "HIPDNN_TEST_HOST_INPUT_FILL=1 to reproduce with the host fill]";
}

std::string IntegrationBundleVerificationHarness::refLabel(ReferenceExecutorType type)
{
    return type == ReferenceExecutorType::GPU ? "GPU reference" : "CPU reference";
}

} // namespace hipdnn_integration_tests::bundle
