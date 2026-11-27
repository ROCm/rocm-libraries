// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/IngestorPreparedDispatch.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdLaunchValues.hpp"
#include "engines/kernel_ingestor_engine/serialization/IngestorPlanRefusal.hpp"

/**
 * @file TestFlydslSdpaBwdHandlerRestore.cpp
 * @brief The FlyDSL SDPA backward dispatch handlers save and restore their launches.
 *
 * Host cases: every refusal a handler's restoreLaunches owes, over saved launch values written
 * from the sample problem (launch_sample in FlydslSdpaBwdDeviceHarness.hpp) and saved kernels
 * that carry the right arguments and no code, since a refusal comes before the code loads.
 *
 * Device cases (TestGpuFlydslSdpaBwdHandlerRestore*): a plan is built through the engine, a
 * handler saves the launch inputs of one of its prepared dispatches and restores a second one
 * from them and from the code objects of every launch it names, and both launch over identical
 * inputs. Their outputs and workspace bytes must be equal, byte for byte. The split engine's
 * prep, dkdv, dq, dq_reduce, dkdv_u2 and dq_u2 handlers run on the gfx11-generic, gfx950, gfx942
 * and gfx90a content; the merged and merged_reduce handlers run on gfx950.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;
namespace sample = hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device::launch_sample;

constexpr std::string_view INCOMPATIBLE_PREFIX = serialization::INGESTOR_PLAN_INCOMPATIBLE_PREFIX;

// ---------------------------------------------------------------------------
// Host cases
// ---------------------------------------------------------------------------

const IKernelDispatchHandler<Handle>& handlerOf(std::string_view alias)
{
    registerNativeIngestorSymbols();
    const auto* handler = DispatchRegistry<Handle>::tryResolve(std::string(alias));
    if(handler == nullptr)
    {
        throw std::runtime_error("no handler is registered as " + std::string(alias));
    }
    return *handler;
}

/// A saved kernel with the arguments @p signature records and no code.
SavedKernelCode savedKernel(std::string launchName,
                            std::vector<KernelArgument> signature,
                            std::string symbol = "saved_symbol")
{
    SavedKernelCode code;
    code.sourceKind = KernelSourceKind::KPACK;
    code.symbol = std::move(symbol);
    code.target = "gfx1152";
    code.recordedSignature = std::move(signature);
    code.launchName = std::move(launchName);
    return code;
}

const std::vector<KernelArgument>& reduceSignature()
{
    return flydsl_sdpa_bwd_native::reduceSignature();
}

/// The stages whose launch set the host cases refuse on.
enum class Stage
{
    SPLIT_DQ,
    SPLIT_DQ_REDUCE,
    MERGED,
    MERGED_REDUCE,
    DKDV_U2,
    DQ_U2,
};

std::string_view stageName(Stage stage)
{
    switch(stage)
    {
    case Stage::SPLIT_DQ:
        return "SplitDq";
    case Stage::SPLIT_DQ_REDUCE:
        return "SplitDqReduce";
    case Stage::MERGED:
        return "Merged";
    case Stage::MERGED_REDUCE:
        return "MergedReduce";
    case Stage::DKDV_U2:
        return "SplitDkdvU2";
    case Stage::DQ_U2:
        return "SplitDqU2";
    default:
        break;
    }
    return {};
}

/// The versioned dispatch name of the handler that serves @p stage.
std::string_view dispatchSymbolOf(Stage stage)
{
    switch(stage)
    {
    case Stage::SPLIT_DQ:
        return FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1;
    case Stage::SPLIT_DQ_REDUCE:
        return FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1;
    case Stage::MERGED:
        return FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1;
    case Stage::MERGED_REDUCE:
        return FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1;
    case Stage::DKDV_U2:
        return FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1;
    case Stage::DQ_U2:
        return FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1;
    default:
        break;
    }
    return {};
}

/// What a stage launches for a problem: the dV-only kernel, the dBias reduce kernel, and the
/// tail-zero kernel.
struct StageLaunches
{
    bool reduces;
    bool zeroesTails;
    bool dvonly = false;
};

/// The sample problem as the dq_reduce and merged_reduce stages serve it: a broadcast dBias and a
/// padded graph with dense dK and dV.
constexpr StageLaunches FULL_STAGE{true, true};

/// The sample problem as the plain dq and merged stages serve it: the dBias is not broadcast, so
/// there is no reduce kernel, and the padded graph with dense dK and dV has a tail-zero kernel.
constexpr StageLaunches PLAIN_STAGE{false, true};

/// What the sample problem of the dkdv_u2 stage launches: the dV-only kernel beside dkdv.
constexpr StageLaunches DKDV_U2_STAGE{false, false, true};

/// What the sample problem of the dq_u2 stage launches: a padded graph with dense dQ, dK and
/// dV, and no bias, so a tail-zero kernel and no reduce.
constexpr StageLaunches DQ_U2_STAGE{false, true};

constexpr StageLaunches fullStageOf(Stage stage)
{
    switch(stage)
    {
    case Stage::DKDV_U2:
        return DKDV_U2_STAGE;
    case Stage::DQ_U2:
        return DQ_U2_STAGE;
    case Stage::SPLIT_DQ_REDUCE:
    case Stage::MERGED_REDUCE:
        return FULL_STAGE;
    case Stage::SPLIT_DQ:
    case Stage::MERGED:
    default:
        return PLAIN_STAGE;
    }
}

const IKernelDispatchHandler<Handle>& dqReduceHandler()
{
    return handlerOf(dispatchSymbolOf(Stage::SPLIT_DQ_REDUCE));
}

const IKernelDispatchHandler<Handle>& handlerOf(Stage stage)
{
    return handlerOf(dispatchSymbolOf(stage));
}

std::string primaryLaunchOf(Stage stage)
{
    switch(stage)
    {
    case Stage::DKDV_U2:
        return "dkdv";
    case Stage::MERGED:
    case Stage::MERGED_REDUCE:
        return std::string(FLYDSL_SDPA_BWD_MERGED_LAUNCH);
    case Stage::SPLIT_DQ:
    case Stage::SPLIT_DQ_REDUCE:
    case Stage::DQ_U2:
    default:
        break;
    }
    return std::string(FLYDSL_SDPA_BWD_DQ_LAUNCH);
}

std::vector<KernelArgument> primarySignatureOf(Stage stage)
{
    switch(stage)
    {
    case Stage::DKDV_U2:
        return flydslSdpaBwdSplitSignature(FlydslSdpaBwdSplitSlot::DKDV, true);
    case Stage::DQ_U2:
        return flydslSdpaBwdSplitSignature(FlydslSdpaBwdSplitSlot::DQ, true);
    case Stage::MERGED:
    case Stage::MERGED_REDUCE:
        return flydslSdpaBwdMergedSignature();
    case Stage::SPLIT_DQ:
    case Stage::SPLIT_DQ_REDUCE:
    default:
        break;
    }
    return flydslSdpaBwdSplitSignature(FlydslSdpaBwdSplitSlot::DQ);
}

/// The tail-zero kernel's arguments: the V-head variant's for the `dq_u2` handler.
std::vector<KernelArgument> tailzeroSignatureOf(Stage stage)
{
    return stage == Stage::DQ_U2 ? flydsl_sdpa_bwd_native::tailzeroHvSignature()
                                 : flydsl_sdpa_bwd_native::tailzeroSignature();
}

/// The launch names @p stage saves for a problem that launches what @p shape says: the single
/// unnamed launch when it launches nothing beside its primary kernel.
std::vector<std::string> expectedLaunchesOf(Stage stage, const StageLaunches& shape)
{
    if(!shape.reduces && !shape.zeroesTails && !shape.dvonly)
    {
        return {""};
    }
    std::vector<std::string> names{primaryLaunchOf(stage)};
    if(shape.dvonly)
    {
        names.emplace_back("dvonly");
    }
    if(shape.reduces)
    {
        names.emplace_back("dbias_reduce");
    }
    if(shape.zeroesTails)
    {
        names.emplace_back("tailzero");
    }
    return names;
}

/// Makes the launch's problem launch what @p shape says: a dBias that is not broadcast has no
/// reduce kernel, and a graph without a padding mask (so with nothing ragged) no tail-zero kernel.
template <typename TLaunch>
void shapeLaunch(TLaunch& launch, const StageLaunches& shape)
{
    auto& problem = launch.problem;
    if(problem.dbiasUid != flydsl_sdpa_bwd::NO_UID)
    {
        problem.biasBroadcastBatch = shape.reduces;
    }
    if(!shape.zeroesTails)
    {
        problem.paddingMask = false;
        problem.seqLenQUid = flydsl_sdpa_bwd::NO_UID;
        problem.seqLenKvUid = flydsl_sdpa_bwd::NO_UID;
        problem.ragged = decltype(problem.ragged){};
        // A tensor that is not ragged has a batch stride.
        problem.q.batch = 114;
        problem.k.batch = 124;
        problem.o.batch = 144;
        problem.dQ.batch = 164;
        problem.stats.batch = 320;
    }
}

/// The saved launch inputs of the sample problem of @p stage, shaped as @p shape.
SavedLaunchInputs inputsOf(Stage stage, const StageLaunches& shape)
{
    const std::string symbol(dispatchSymbolOf(stage));
    if(stage == Stage::DKDV_U2)
    {
        auto launch = sample::dkdvU2Inputs();
        shapeLaunch(launch, shape);
        return SavedLaunchInputs{symbol, flydslSdpaBwdDkdvLaunchValues(launch)};
    }
    if(stage == Stage::MERGED || stage == Stage::MERGED_REDUCE)
    {
        auto launch = sample::mergedInputs();
        shapeLaunch(launch, shape);
        return SavedLaunchInputs{symbol, flydslSdpaBwdMergedLaunchValues(launch)};
    }
    auto launch = stage == Stage::DQ_U2 ? sample::dqU2Inputs() : sample::dqInputs();
    shapeLaunch(launch, shape);
    return SavedLaunchInputs{symbol, flydslSdpaBwdDqLaunchValues(launch)};
}

/// The full valid saved launches of the sample problem of @p stage, shaped as @p shape: the
/// primary launch, named only when the stage launches more than one kernel, then the dBias
/// reduce and tail-zero launches the shape has. A test changes one thing in them.
std::vector<SavedKernelCode> savedLaunches(Stage stage, const StageLaunches& shape)
{
    const bool named = shape.reduces || shape.zeroesTails || shape.dvonly;
    std::vector<SavedKernelCode> codes;
    codes.push_back(
        savedKernel(named ? primaryLaunchOf(stage) : std::string(), primarySignatureOf(stage)));
    if(shape.dvonly)
    {
        codes.push_back(savedKernel("dvonly", primarySignatureOf(stage)));
    }
    if(shape.reduces)
    {
        codes.push_back(savedKernel("dbias_reduce", reduceSignature(), "reduce_sample"));
    }
    if(shape.zeroesTails)
    {
        codes.push_back(savedKernel("tailzero", tailzeroSignatureOf(stage)));
    }
    return codes;
}

SavedKernelCode& reduceOf(std::vector<SavedKernelCode>& codes)
{
    return codes[1];
}

std::vector<SavedKernelCode> dqAndReduce()
{
    return savedLaunches(Stage::SPLIT_DQ_REDUCE, FULL_STAGE);
}

SavedLaunchInputs dqReduceInputs()
{
    return inputsOf(Stage::SPLIT_DQ_REDUCE, FULL_STAGE);
}

/// Expects @p restore to refuse the plan as incompatible, with a message holding @p cause.
void expectRefusal(const std::function<void()>& restore, std::string_view cause)
{
    try
    {
        restore();
        ADD_FAILURE() << "the restore did not refuse; expected a refusal naming '" << cause << "'";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE) << error.getMessage();
        EXPECT_NE(error.getMessage().find(INCOMPATIBLE_PREFIX), std::string::npos)
            << error.getMessage();
        EXPECT_NE(error.getMessage().find(cause), std::string::npos) << error.getMessage();
    }
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAWrongDispatchVersion)
{
    auto inputs = dqReduceInputs();
    inputs.dispatchSymbol = "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch.v2";

    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, dqAndReduce(), 0); },
                  "names no launch-value contract");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAnUnknownLaunchValue)
{
    auto inputs = dqReduceInputs();
    inputs.values["not_a_value"] = int64_t{1};

    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, dqAndReduce(), 0); },
                  "'not_a_value' is not part of the contract");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAMissingLaunchValue)
{
    auto inputs = dqReduceInputs();
    inputs.values.erase("block_m");

    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, dqAndReduce(), 0); },
                  "needs launch value 'block_m'");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesALaunchValueOutOfRange)
{
    auto inputs = dqReduceInputs();
    inputs.values["batch"] = int64_t{0};

    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, dqAndReduce(), 0); },
                  "launch value 'batch'");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAProblemTheRowsTileCannotServe)
{
    auto inputs = dqReduceInputs();
    inputs.values["head_dim"] = int64_t{64};

    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, dqAndReduce(), 0); },
                  "exceeds the tile head_dim");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesASignatureTheHandlerDoesNotLaunch)
{
    auto codes = dqAndReduce();
    codes.front().recordedSignature = reduceSignature();

    expectRefusal([&] { dqReduceHandler().restoreLaunches(dqReduceInputs(), std::move(codes), 0); },
                  "do not match what this provider launches");
}

TEST(TestFlydslSdpaBwdHandlerRestore, ARecordedSignatureIsRefusedBeforeTheDispatchNameIsRead)
{
    auto inputs = dqReduceInputs();
    inputs.dispatchSymbol = "hipkernel.flydsl_sdpa_bwd_split.dq_reduce.dispatch.v2";
    auto codes = dqAndReduce();
    codes.back().recordedSignature = flydslSdpaBwdSplitSignature(FlydslSdpaBwdSplitSlot::DQ);

    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, std::move(codes), 0); },
                  "do not match what this provider launches");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAUserSuppliedScalarWithABakedValue)
{
    auto inputs = dqReduceInputs();
    // The sample's seed is user supplied: it saves no value.
    inputs.values["seed.baked"] = int64_t{5};

    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, dqAndReduce(), 0); },
                  "launch value 'seed.baked'");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesABindingOfAScalarTheProblemDoesNotUse)
{
    auto inputs = dqReduceInputs();
    inputs.values["scale_source"] = std::string("value");

    // The refusal names the first such value in the map's order: any of the scale's four.
    expectRefusal([&] { dqReduceHandler().restoreLaunches(inputs, dqAndReduce(), 0); },
                  "is not part of the contract");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAMergedProblemOfFullPrecision)
{
    auto launch = sample::mergedInputs();
    launch.problem.elementBytes = 4;
    launch.problem.biasElementBytes = 4;
    const SavedLaunchInputs inputs{std::string(FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1),
                                   flydslSdpaBwdMergedLaunchValues(launch)};
    std::vector<SavedKernelCode> codes;
    codes.push_back(savedKernel("merged", flydslSdpaBwdMergedSignature()));
    codes.push_back(savedKernel("dbias_reduce", reduceSignature(), "reduce_sample"));

    expectRefusal(
        [&] { handlerOf(Stage::MERGED_REDUCE).restoreLaunches(inputs, std::move(codes), 0); },
        "16-bit data only");
}

// ---------------------------------------------------------------------------
// The launch set a stage saves
// ---------------------------------------------------------------------------

/// The rule is `requireBwdLaunchSet`'s: the saved launch names are exactly the set the problem
/// implies. Each case starts from the full valid saved launches of the sample, changes one
/// thing, and expects the refusal to print the expected and the found set. Both sets are
/// spelled here, so a case that refused for any other reason would not match. The plain dq and
/// merged stages never launch the dBias reduce kernel, so a saved one is a set they refuse; the
/// dq_reduce and merged_reduce stages launch it exactly when the problem broadcasts its dBias;
/// dkdv_u2 saves {dkdv, dvonly} and dq_u2 {dq, tailzero}. The tests every stage shares run over
/// all of them; the standard stages alone have a reduce launch to refuse, and a single unnamed
/// launch to name.
class TestFlydslSdpaBwdLaunchSetRestore : public ::testing::TestWithParam<Stage>
{
};

class TestFlydslSdpaBwdStandardLaunchSetRestore : public ::testing::TestWithParam<Stage>
{
};

std::string setText(std::vector<std::string> names)
{
    std::sort(names.begin(), names.end());
    std::string text = "{";
    for(const auto& name : names)
    {
        text += (text.size() > 1 ? ", '" : "'") + name + "'";
    }
    return text + "}";
}

/// The refusal text of a saved launch set that is not the expected one.
std::string setRefusal(const std::vector<std::string>& expected,
                       const std::vector<SavedKernelCode>& saved)
{
    std::vector<std::string> found;
    found.reserve(saved.size());
    for(const auto& code : saved)
    {
        found.push_back(code.launchName);
    }
    return "launches " + setText(expected)
           + " for the saved problem, and the plan holds the launches " + setText(found);
}

void expectSetRefusal(Stage stage,
                      StageLaunches problem,
                      std::vector<SavedKernelCode> saved,
                      const std::vector<std::string>& expected)
{
    const std::string cause = setRefusal(expected, saved);
    expectRefusal(
        [&] { handlerOf(stage).restoreLaunches(inputsOf(stage, problem), std::move(saved), 0); },
        cause);
}

TEST_P(TestFlydslSdpaBwdLaunchSetRestore, RefusesAnExtraLaunch)
{
    const auto stage = GetParam();
    const auto full = fullStageOf(stage);
    auto saved = savedLaunches(stage, full);
    saved.push_back(savedKernel("extra", primarySignatureOf(stage)));

    expectSetRefusal(stage, full, std::move(saved), expectedLaunchesOf(stage, full));
}

TEST_P(TestFlydslSdpaBwdStandardLaunchSetRestore, RefusesAReduceLaunchWhenTheProblemHasNone)
{
    const auto stage = GetParam();

    expectSetRefusal(stage,
                     PLAIN_STAGE,
                     savedLaunches(stage, FULL_STAGE),
                     expectedLaunchesOf(stage, PLAIN_STAGE));
}

TEST_P(TestFlydslSdpaBwdLaunchSetRestore, RefusesAMissingLastLaunch)
{
    const auto stage = GetParam();
    const auto full = fullStageOf(stage);
    auto saved = savedLaunches(stage, full);
    saved.pop_back();

    expectSetRefusal(stage, full, std::move(saved), expectedLaunchesOf(stage, full));
}

TEST_P(TestFlydslSdpaBwdStandardLaunchSetRestore, RefusesATailzeroLaunchWhenTheProblemHasNone)
{
    const auto stage = GetParam();
    const auto full = fullStageOf(stage);
    const StageLaunches noTails{full.reduces, false};

    expectSetRefusal(
        stage, noTails, savedLaunches(stage, full), expectedLaunchesOf(stage, noTails));
}

TEST_P(TestFlydslSdpaBwdLaunchSetRestore, RefusesAWrongPrimaryLaunchName)
{
    const auto stage = GetParam();
    const auto full = fullStageOf(stage);
    auto saved = savedLaunches(stage, full);
    saved.front().launchName = "wrong";

    expectSetRefusal(stage, full, std::move(saved), expectedLaunchesOf(stage, full));
}

TEST_P(TestFlydslSdpaBwdLaunchSetRestore, RefusesAnUnnamedLaunchWhenTheStageLaunchesMore)
{
    const auto stage = GetParam();
    const auto full = fullStageOf(stage);
    const std::string cause = "launches " + setText(expectedLaunchesOf(stage, full))
                              + " for the saved problem, and the plan holds the launches {''}";

    // restoreLaunch hands the handler one code, with no launch name.
    expectRefusal(
        [&] {
            handlerOf(stage).restoreLaunch(
                inputsOf(stage, full), savedKernel("", primarySignatureOf(stage)), 0);
        },
        cause);
}

TEST_P(TestFlydslSdpaBwdStandardLaunchSetRestore, RefusesANamedLaunchWhenTheStageLaunchesOne)
{
    constexpr StageLaunches ONE_LAUNCH{false, false};
    const auto stage = GetParam();
    std::vector<SavedKernelCode> saved;
    saved.push_back(savedKernel(std::string(primaryLaunchOf(stage)), primarySignatureOf(stage)));

    // Without a reduce or tail-zero kernel the single launch is saved with no launch name.
    expectSetRefusal(stage, ONE_LAUNCH, std::move(saved), {""});
}

/// Expects the restore of @p inputs with the full valid launches of @p stage, shaped as @p shape,
/// to pass the rule named by @p refusalText: the saved kernels have no code, so the restore
/// fails when it loads the code, after that rule was applied, and not with its refusal.
void expectRestoreReachesTheCode(Stage stage,
                                 const StageLaunches& shape,
                                 const SavedLaunchInputs& inputs,
                                 std::string_view refusalText)
{
    try
    {
        handlerOf(stage).restoreLaunches(inputs, savedLaunches(stage, shape), 0);
        ADD_FAILURE() << "a kernel with no code loaded";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getMessage().find(refusalText), std::string::npos) << error.getMessage();
    }
    catch(const std::exception& error)
    {
        EXPECT_EQ(std::string(error.what()).find(refusalText), std::string::npos) << error.what();
    }
}

/// Expects the restore of the sample of @p stage, shaped as @p shape and saved with its full
/// valid launches, to pass the launch-set rule.
void expectSetAccepted(Stage stage, const StageLaunches& shape)
{
    expectRestoreReachesTheCode(stage, shape, inputsOf(stage, shape), "for the saved problem");
}

TEST_P(TestFlydslSdpaBwdStandardLaunchSetRestore, TakesTheSingleUnnamedLaunchPastTheSetRule)
{
    expectSetAccepted(GetParam(), StageLaunches{false, false});
}

TEST_P(TestFlydslSdpaBwdLaunchSetRestore, TakesTheFullLaunchSetPastTheSetRule)
{
    expectSetAccepted(GetParam(), fullStageOf(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaBwdLaunchSetRestore,
                         ::testing::Values(Stage::SPLIT_DQ,
                                           Stage::SPLIT_DQ_REDUCE,
                                           Stage::MERGED,
                                           Stage::MERGED_REDUCE,
                                           Stage::DKDV_U2,
                                           Stage::DQ_U2),
                         [](const ::testing::TestParamInfo<Stage>& info) {
                             return std::string(stageName(info.param));
                         });

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSdpaBwdStandardLaunchSetRestore,
    ::testing::Values(Stage::SPLIT_DQ, Stage::SPLIT_DQ_REDUCE, Stage::MERGED, Stage::MERGED_REDUCE),
    [](const ::testing::TestParamInfo<Stage>& info) { return std::string(stageName(info.param)); });

// ---------------------------------------------------------------------------
// The dBias reduce kernel belongs to the reduce dispatches
// ---------------------------------------------------------------------------

class TestFlydslSdpaBwdReduceLaunchRestore : public ::testing::TestWithParam<Stage>
{
};

class TestFlydslSdpaBwdPlainLaunchRestore : public ::testing::TestWithParam<Stage>
{
};

TEST_P(TestFlydslSdpaBwdReduceLaunchRestore, RefusesAMissingReduceLaunch)
{
    const auto stage = GetParam();
    auto saved = savedLaunches(stage, FULL_STAGE);
    saved.erase(saved.begin() + 1);

    expectSetRefusal(stage, FULL_STAGE, std::move(saved), expectedLaunchesOf(stage, FULL_STAGE));
}

TEST_P(TestFlydslSdpaBwdReduceLaunchRestore, RefusesAReduceKernelWithTheWrongSignature)
{
    const auto stage = GetParam();
    auto codes = savedLaunches(stage, FULL_STAGE);
    reduceOf(codes).recordedSignature = primarySignatureOf(stage);

    expectRefusal(
        [&] { handlerOf(stage).restoreLaunches(inputsOf(stage, FULL_STAGE), std::move(codes), 0); },
        "do not match what this provider launches");
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaBwdReduceLaunchRestore,
                         ::testing::Values(Stage::SPLIT_DQ_REDUCE, Stage::MERGED_REDUCE),
                         [](const ::testing::TestParamInfo<Stage>& info) {
                             return std::string(stageName(info.param));
                         });

TEST_P(TestFlydslSdpaBwdPlainLaunchRestore, RefusesAProblemWithABroadcastDbias)
{
    // The plain dispatch launches no reduce kernel: a problem that needs one is a plan of the
    // reduce dispatch, even when the saved launches include the reduce kernel.
    const auto stage = GetParam();

    expectRefusal(
        [&] {
            handlerOf(stage).restoreLaunches(
                inputsOf(stage, FULL_STAGE), savedLaunches(stage, FULL_STAGE), 0);
        },
        "dBias");
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaBwdPlainLaunchRestore,
                         ::testing::Values(Stage::SPLIT_DQ, Stage::MERGED),
                         [](const ::testing::TestParamInfo<Stage>& info) {
                             return std::string(stageName(info.param));
                         });

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesTheStandardTailzeroKernelForTheU2DqStage)
{
    // dq_u2 launches the V-head tail-zero kernel, which has one more argument.
    auto saved = savedLaunches(Stage::DQ_U2, DQ_U2_STAGE);
    saved.back().recordedSignature = flydsl_sdpa_bwd_native::tailzeroSignature();

    expectRefusal(
        [&] {
            handlerOf(Stage::DQ_U2)
                .restoreLaunches(inputsOf(Stage::DQ_U2, DQ_U2_STAGE), std::move(saved), 0);
        },
        "do not match what this provider launches");
}

TEST(TestFlydslSdpaBwdHandlerRestore, RefusesATailzeroLaunchWhenTheU2ProblemHasNone)
{
    constexpr StageLaunches NO_TAILS{false, false};

    expectSetRefusal(Stage::DQ_U2,
                     NO_TAILS,
                     savedLaunches(Stage::DQ_U2, DQ_U2_STAGE),
                     std::vector<std::string>{""});
}

TEST(TestFlydslSdpaBwdHandlerRestore, TakesTheSingleUnnamedU2LaunchPastTheSetRule)
{
    constexpr StageLaunches NO_TAILS{false, false};

    expectSetAccepted(Stage::DQ_U2, NO_TAILS);
}

// ---------------------------------------------------------------------------
// A saved problem whose launch grid is past a limit is refused
// ---------------------------------------------------------------------------

/// Changes the saved values of a stage's sample problem.
using EditInputs = std::function<void(SavedLaunchInputs&)>;

/// The sample problem of @p stage, shaped as @p shape, with @p edit applied.
SavedLaunchInputs inputsEdited(Stage stage, const StageLaunches& shape, const EditInputs& edit)
{
    auto inputs = inputsOf(stage, shape);
    edit(inputs);
    return inputs;
}

/// An edit that gives the problem @p batch batches.
EditInputs withBatch(int64_t batch)
{
    return [batch](SavedLaunchInputs& inputs) { inputs.values["batch"] = batch; };
}

/// An edit that gives the problem @p heads query heads and @p kvHeads key and value heads.
EditInputs withHeads(int64_t heads, int64_t kvHeads)
{
    return [heads, kvHeads](SavedLaunchInputs& inputs) {
        inputs.values["num_heads"] = heads;
        inputs.values["num_kv_heads"] = kvHeads;
        inputs.values["num_v_heads"] = kvHeads;
    };
}

void expectGridRefusal(Stage stage,
                       const StageLaunches& shape,
                       const EditInputs& edit,
                       std::string_view cause)
{
    expectRefusal(
        [&] {
            handlerOf(stage).restoreLaunches(
                inputsEdited(stage, shape, edit), savedLaunches(stage, shape), 0);
        },
        cause);
}

void expectPastTheGridRule(Stage stage, const StageLaunches& shape, const EditInputs& edit)
{
    expectRestoreReachesTheCode(stage, shape, inputsEdited(stage, shape, edit), "grid exceeds");
}

// A split launch's batch is its grid's z extent, limited to 65535.
TEST(TestFlydslSdpaBwdHandlerRestore, RefusesASplitProblemWhoseBatchPutsItsGridPastTheLimit)
{
    constexpr StageLaunches NO_TAILS{false, false};

    expectGridRefusal(Stage::SPLIT_DQ, NO_TAILS, withBatch(65536), "limits for kernel");
    expectPastTheGridRule(Stage::SPLIT_DQ, NO_TAILS, withBatch(65535));
}

// The dV-only launch of a dkdv_u2 stage runs the dK-only object's grid spelling over the V heads:
// with the key blocks first, the V heads are the y extent, limited to 65535. The key heads stay
// at 4, so the dK-only grid is inside the limit.
TEST(TestFlydslSdpaBwdHandlerRestore, RefusesADkdvU2ProblemWhoseDvOnlyGridIsPastTheLimit)
{
    const auto valueHeads = [](int64_t heads, int64_t vHeads) {
        return [heads, vHeads](SavedLaunchInputs& inputs) {
            inputs.values["grid"] = std::string("kvblocks_by_kheads_by_seqs");
            inputs.values["num_heads"] = heads;
            inputs.values["num_v_heads"] = vHeads;
        };
    };

    expectGridRefusal(
        Stage::DKDV_U2, DKDV_U2_STAGE, valueHeads(65536, 65536), "dV-only launch grid exceeds");
    expectPastTheGridRule(Stage::DKDV_U2, DKDV_U2_STAGE, valueHeads(262140, 65535));
}

// The dBias reduce launch has one plane for every batch and head the dBias is not broadcast over:
// the sample broadcasts it over the batch, so the planes are the heads, limited to 65535. The
// stage has no tail-zero launch, which has its own plane limit.
TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAReduceProblemWhoseDbiasPlanesArePastTheLimit)
{
    constexpr StageLaunches REDUCE_ONLY{true, false};

    for(const auto stage : {Stage::SPLIT_DQ_REDUCE, Stage::MERGED_REDUCE})
    {
        SCOPED_TRACE(stageName(stage));
        expectGridRefusal(
            stage, REDUCE_ONLY, withHeads(65536, 65536), "reduce launch grid exceeds");
        expectPastTheGridRule(stage, REDUCE_ONLY, withHeads(65535, 65535));
    }
}

// The tail-zero launch has one plane for every batch and head of Q, K and V: the sample's 8 query
// heads, 4 key heads and 4 value heads make 16 planes a batch, and a grid is limited to 65535
// planes.
TEST(TestFlydslSdpaBwdHandlerRestore, RefusesASplitProblemWhoseTailzeroGridIsPastTheLimit)
{
    expectGridRefusal(
        Stage::SPLIT_DQ, PLAIN_STAGE, withBatch(4096), "tail-zero launch grid exceeds");
    expectPastTheGridRule(Stage::SPLIT_DQ, PLAIN_STAGE, withBatch(4095));
}

// The merged stage's tail-zero launch has the same planes: the merged grid is far from its own
// limit at the same batch.
TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAMergedProblemWhoseTailzeroGridIsPastTheLimit)
{
    expectGridRefusal(Stage::MERGED, PLAIN_STAGE, withBatch(4096), "tail-zero launch grid exceeds");
    expectPastTheGridRule(Stage::MERGED, PLAIN_STAGE, withBatch(4095));
}

// The merged launch is one block for every (batch, key tile, key head) and (batch, query tile,
// query head), limited to 2^32 - 1 work-items: the sample's 4 key heads of 4 tiles of 32 keys and
// 8 query heads of 2 tiles of 64 rows make 32 blocks of 256 threads a batch.
TEST(TestFlydslSdpaBwdHandlerRestore, RefusesAMergedProblemWhoseGridIsPastTheLimit)
{
    constexpr StageLaunches NO_TAILS{false, false};

    expectGridRefusal(Stage::MERGED, NO_TAILS, withBatch(524288), "grid exceeds");
    expectPastTheGridRule(Stage::MERGED, NO_TAILS, withBatch(524287));
}

// ---------------------------------------------------------------------------
// Device cases
// ---------------------------------------------------------------------------

/// One saved kernel per launch the prepared dispatch names, with its code object read back.
std::vector<SavedKernelCode> savedCodesOf(const GenericPlan<Handle>& plan)
{
    std::vector<SavedKernelCode> codes;
    const auto* prepared = dynamic_cast<const IngestorPreparedDispatch*>(&plan.prepared());
    if(prepared == nullptr)
    {
        ADD_FAILURE() << "the prepared dispatch exposes no kernel code";
        return codes;
    }
    for(const auto& launched : prepared->launchedKernels())
    {
        auto object = launched.code->readCodeObject();
        SavedKernelCode code;
        code.kernelId = plan.kernel().kernelId;
        code.sourceKind = launched.sourceKind;
        code.symbol = std::move(object.symbol);
        code.target = std::move(object.target);
        code.sha256 = std::move(object.sha256);
        code.recordedSignature = launched.signature;
        code.codeObject = std::move(object.bytes);
        code.launchName = launched.launchName;
        codes.push_back(std::move(code));
    }
    return codes;
}

std::vector<std::string> launchNamesOf(const PreparedDispatch& prepared)
{
    std::vector<std::string> names;
    const auto* ingestor = dynamic_cast<const IngestorPreparedDispatch*>(&prepared);
    if(ingestor == nullptr)
    {
        ADD_FAILURE() << "the prepared dispatch exposes no launches";
        return names;
    }
    for(const auto& launched : ingestor->launchedKernels())
    {
        names.push_back(launched.launchName);
    }
    return names;
}

bool allBytes(const std::vector<uint8_t>& bytes, uint8_t value)
{
    return std::all_of(bytes.begin(), bytes.end(), [value](uint8_t byte) { return byte == value; });
}

/// What a restored handler is checked against.
struct RoundTrip
{
    /// The engine, the slot of the stage under test (empty for a single-kernel engine) and the
    /// case.
    std::string_view engine;
    std::string slot;
    BwdCase c;
    /// The versioned alias of the handler the stage must be served by.
    std::string_view handler;
    /// The launch names the stage's prepared dispatch reports.
    std::vector<std::string> launches;
    /// Restore through restoreLaunch, which forwards one code, rather than restoreLaunches.
    bool forwardOneLaunch = false;
};

/// Builds the plan, saves the stage's launch inputs, restores a second prepared dispatch from
/// them and the stage's code objects, launches both over identical inputs, and compares every
/// output and the whole workspace, byte for byte.
void expectRestoredLaunchesWhatThePreparedOneLaunches(const RoundTrip& trip)
{
    BwdPlanRig rig;
    rig.build(trip.engine, trip.c);
    if(::testing::Test::HasFatalFailure())
    {
        return;
    }
    const auto* stage = rig.kernelPlan(trip.slot);
    ASSERT_NE(stage, nullptr);
    const auto& handler = stage->handler();
    ASSERT_EQ(&handler, &handlerOf(trip.handler))
        << "the stage is served by another handler than " << trip.handler;
    ASSERT_EQ(launchNamesOf(stage->prepared()), trip.launches);

    const auto inputs = handler.saveLaunchInputs(stage->prepared());
    ASSERT_TRUE(inputs.has_value());
    EXPECT_EQ(inputs->dispatchSymbol, trip.handler);
    auto codes = savedCodesOf(*stage);
    ASSERT_EQ(codes.size(), trip.launches.size());
    const int ordinal = launchDeviceOrdinal(rig.handle().getStream());
    std::unique_ptr<PreparedDispatch> restored;
    if(trip.forwardOneLaunch)
    {
        ASSERT_EQ(codes.size(), 1U);
        ASSERT_NO_THROW(restored
                        = handler.restoreLaunch(*inputs, std::move(codes.front()), ordinal));
    }
    else
    {
        ASSERT_NO_THROW(restored = handler.restoreLaunches(*inputs, std::move(codes), ordinal));
    }
    ASSERT_NE(restored, nullptr);
    EXPECT_EQ(launchNamesOf(*restored), trip.launches);

    // The stage reads the workspace the stages before it wrote: run the engine's prep stage
    // first, then give both runs the same copy.
    const size_t bytes = rig.plan().getWorkspaceSize(rig.handle());
    const GuardedWorkspace original(bytes);
    const GuardedWorkspace again(bytes);
    if(bytes > 0)
    {
        ASSERT_NE(original.get(), nullptr);
        ASSERT_EQ(hipSuccess, hipMemset(original.get(), 0, bytes));
        ASSERT_EQ(hipSuccess, hipMemset(again.get(), 0, bytes));
    }
    if(trip.slot != "prep" && !trip.slot.empty())
    {
        const auto* prep = rig.kernelPlan("prep");
        ASSERT_NE(prep, nullptr);
        rig.launch(prep->handler(), prep->prepared(), original.get());
        ASSERT_EQ(hipSuccess,
                  hipMemcpy(again.get(), original.get(), bytes, hipMemcpyDeviceToDevice));
    }

    rig.poisonOutputs();
    rig.launch(handler, stage->prepared(), original.get());
    const auto expectedOutputs = rig.outputs();
    const auto expectedWorkspace = downloadBytes(original.get(), bytes);

    rig.poisonOutputs();
    rig.launch(handler, *restored, again.get());
    const auto restoredOutputs = rig.outputs();
    const auto restoredWorkspace = downloadBytes(again.get(), bytes);

    EXPECT_TRUE(restoredOutputs == expectedOutputs) << "an output differs from the prepared one's";
    EXPECT_TRUE(restoredWorkspace == expectedWorkspace) << "the workspace differs";
    EXPECT_EQ(original.overwrittenGuardBytes(), 0U);
    EXPECT_EQ(again.overwrittenGuardBytes(), 0U);

    // The comparison is of something: the stage wrote its results.
    if(trip.slot == "prep")
    {
        EXPECT_FALSE(allBytes(expectedWorkspace, 0)) << "prep wrote no row statistics";
    }
    else
    {
        bool wrote = false;
        for(const auto& output : expectedOutputs)
        {
            wrote = wrote || (!output.empty() && !allBytes(output, 0xFF));
        }
        EXPECT_TRUE(wrote) << "the stage wrote none of its outputs";
    }
}

class TestGpuFlydslSdpaBwdHandlerRestore : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdHandlerRestore()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

/// The merged handlers run on a gfx950 device: no other architecture ships them.
class TestGpuFlydslSdpaBwdHandlerRestoreMerged : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdHandlerRestoreMerged()
        : FlydslSdpaBwdDevice(MERGED_OPS)
    {
    }

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        const auto arch = packedArchDeviceProperties().gcnArchName;
        if(arch.rfind("gfx950", 0) != 0)
        {
            GTEST_SKIP() << "this round trip runs on a gfx950 device; " << arch
                         << " ships no such content";
        }
        FlydslSdpaBwdDevice::SetUp();
    }
};

TEST_F(TestGpuFlydslSdpaBwdHandlerRestore,
       RestoredDqReduceHandlerLaunchesWhatThePreparedOneLaunches)
{
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {SPLIT_ENGINE,
         "dq",
         broadcastBiasCase(),
         FLYDSL_SDPA_BWD_SPLIT_DQ_REDUCE_DISPATCH_SYMBOL_V1,
         {"dq", "dbias_reduce"}});
}

TEST_F(TestGpuFlydslSdpaBwdHandlerRestore, RestoredDqHandlerLaunchesWhatThePreparedOneLaunches)
{
    // Without a broadcast dBias one kernel runs, under the empty launch name, and it restores
    // through restoreLaunch.
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {SPLIT_ENGINE, "dq", plainCase(), FLYDSL_SDPA_BWD_SPLIT_DQ_DISPATCH_SYMBOL_V1, {""}, true});
}

TEST_F(TestGpuFlydslSdpaBwdHandlerRestore, RestoredPrepHandlerLaunchesWhatThePreparedOneLaunches)
{
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {SPLIT_ENGINE, "prep", plainCase(), FLYDSL_SDPA_BWD_SPLIT_PREP_DISPATCH_SYMBOL_V1, {""}});
}

TEST_F(TestGpuFlydslSdpaBwdHandlerRestore, RestoredDkdvHandlerLaunchesWhatThePreparedOneLaunches)
{
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {SPLIT_ENGINE, "dkdv", plainCase(), FLYDSL_SDPA_BWD_SPLIT_DKDV_DISPATCH_SYMBOL_V1, {""}});
}

// A V with other heads than K is served by the u2 handlers. The dV-only launch of dkdv_u2 runs
// over V's heads, and the V-head tail-zero launch of dq_u2 covers V's heads: a restored launch
// that took K's heads would write dV past its heads, or leave some unwritten.
TEST_F(TestGpuFlydslSdpaBwdHandlerRestore,
       RestoredDkdvU2HandlerLaunchesItsDvonlyKernelOverFewerValueHeads)
{
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {SPLIT_ENGINE,
         "dkdv",
         splitValueHeadsCase(DataType::HALF, 8, 4, 2, false),
         FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1,
         {"dkdv", "dvonly"}});
}

TEST_F(TestGpuFlydslSdpaBwdHandlerRestore,
       RestoredDkdvU2HandlerLaunchesItsDvonlyKernelOverMoreValueHeads)
{
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {SPLIT_ENGINE,
         "dkdv",
         splitValueHeadsCase(DataType::HALF, 8, 2, 4, false),
         FLYDSL_SDPA_BWD_SPLIT_DKDV_U2_DISPATCH_SYMBOL_V1,
         {"dkdv", "dvonly"}});
}

TEST_F(TestGpuFlydslSdpaBwdHandlerRestore, RestoredDqU2HandlerLaunchesWhatThePreparedOneLaunches)
{
    // Without a padded graph one kernel runs, under the empty launch name, and it restores
    // through restoreLaunch.
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {SPLIT_ENGINE,
         "dq",
         splitValueHeadsCase(DataType::HALF, 8, 4, 2, false),
         FLYDSL_SDPA_BWD_SPLIT_DQ_U2_DISPATCH_SYMBOL_V1,
         {""},
         true});
}

TEST_F(TestGpuFlydslSdpaBwdHandlerRestoreMerged,
       RestoredMergedHandlerLaunchesWhatThePreparedOneLaunches)
{
    // Without a broadcast dBias one kernel runs, under the empty launch name, and it restores
    // through restoreLaunch.
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {MERGED_ENGINE, "", plainCase(), FLYDSL_SDPA_BWD_MERGED_DISPATCH_SYMBOL_V1, {""}, true});
}

TEST_F(TestGpuFlydslSdpaBwdHandlerRestoreMerged,
       RestoredMergedReduceHandlerLaunchesWhatThePreparedOneLaunches)
{
    expectRestoredLaunchesWhatThePreparedOneLaunches(
        {MERGED_ENGINE,
         "",
         broadcastBiasCase(),
         FLYDSL_SDPA_BWD_MERGED_REDUCE_DISPATCH_SYMBOL_V1,
         {"merged", "dbias_reduce"}});
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
