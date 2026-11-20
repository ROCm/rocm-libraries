// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelIngestorStateManager.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdTestSupport.hpp"

/**
 * @file TestAttentionGeneralBwdReplay.cpp
 * @brief Replays the hipkernel:AttentionGeneralBwd corpus through the C++ matcher and plan.
 *
 * Every corpus entry is a hipDNN graph with the verdict rocKE's Python reaches on it: the
 * reason code, and for an accepted graph the whole plan -- the bound tokens, the workspace
 * layout and every launch with its kernel symbol, grid, block and each kernarg in ABI order.
 * The C++ lowering, checks and plan must reproduce it exactly, through both the evaluator
 * and the registered graph matcher. For an accepted graph the kernel matcher must then pick
 * exactly one prep kernel of the arch's catalog, the one the plan names, and the main
 * candidates (the class match, then the candidate filter) must hold the plan's main kernel
 * once, beside the alternates that can stand in for it.
 *
 * The committed subset runs here. `HKP_GENERAL_BWD_CORPUS` points the suite at a full
 * corpus (a file, or a directory of `corpus_<arch>.jsonl`) instead.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace general_bwd_test;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

constexpr std::size_t MAX_REPORTED_FAILURES = 40;

struct ReplayTally
{
    std::size_t entries = 0;
    std::size_t accepts = 0;
    std::size_t launches = 0;
    std::size_t failures = 0;
    std::size_t candidates = 0;
    std::size_t withAlternates = 0;
    std::size_t alternatesVerified = 0;
    std::size_t declinedVerified = 0;
    std::map<std::string, std::size_t> codes;
    std::map<std::string, std::size_t> mainKernelsHit;
};

class Replay
{
public:
    /// @p alternates are the Python reference plans of the alternate main kernels by entry id,
    /// or null when the corpus ships without them (the parity check is then skipped).
    Replay(std::string_view arch, const std::map<std::string, json>* alternates)
        : _arch(arch)
        , _kernels(archKernels(arch))
        , _alternates(alternates)
    {
        registerNativeIngestorSymbols();
        _graphMatcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
            std::string(GRAPH_MATCHER_SYMBOL));
        _kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
            std::string(KERNEL_MATCHER_SYMBOL));
        _candidateFilter = hipdnn_plugin_sdk::ingestor::CandidateFilterRegistry::resolve(
            std::string(KERNEL_MATCHER_SYMBOL));
        _mainScore = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(
            "hipkernel.attention_general_bwd.main_score");
    }

    void run(const json& entry)
    {
        ++_tally.entries;
        _id = entry.at("id").get<std::string>();
        const std::string code = entry.at("expect").at("code").get<std::string>();
        ++_tally.codes[code];

        std::unique_ptr<GraphFixture> fixture;
        try
        {
            fixture = std::make_unique<GraphFixture>(
                entry.at("graph"), _arch, entry.at("num_cus").get<int>());
        }
        catch(const std::exception& error)
        {
            fail(std::string("the graph JSON does not read: ") + error.what());
            return;
        }
        const auto context = fixture->context();

        const auto verdict = general_bwd::evaluate(context);
        const auto bound = _graphMatcher(context);
        if(const auto* decline = std::get_if<general_bwd::Decline>(&verdict))
        {
            if(code != general_bwd::reasonName(decline->code))
            {
                fail("verdict " + std::string(general_bwd::reasonName(decline->code)) + " ("
                     + decline->detail + "), expected " + code);
            }
            if(bound.has_value())
            {
                fail("the registered graph matcher accepted a graph the evaluator declined");
            }
            return;
        }

        ++_tally.accepts;
        if(code != "OK")
        {
            fail("verdict OK, expected " + code);
            return;
        }
        const auto& plan = std::get<general_bwd::Plan>(verdict);
        const auto& expect = entry.at("expect");
        checkTokens(general_bwd::tokensFor(plan), expect.at("tokens"), "evaluate()");
        if(!bound.has_value())
        {
            fail("the registered graph matcher declined an accepted graph");
            return;
        }
        checkTokens(*bound, expect.at("tokens"), "graph_match");
        checkSharedWorkspaceContract(*bound);
        checkWorkspace(plan.workspace, expect.at("workspace"));
        checkLaunches(plan, expect.at("launches"));
        checkSelection(context, *bound, plan);
    }

    const ReplayTally& tally() const
    {
        return _tally;
    }

private:
    void fail(const std::string& why)
    {
        if(_tally.failures < MAX_REPORTED_FAILURES)
        {
            ADD_FAILURE() << _arch << " corpus entry '" << _id << "': " << why;
        }
        ++_tally.failures;
    }

    void checkTokens(const BoundTokens& got, const json& want, std::string_view source)
    {
        if(got.size() != want.size())
        {
            fail(std::string(source) + " bound " + std::to_string(got.size()) + " tokens, expected "
                 + std::to_string(want.size()));
        }
        for(const auto& [name, value] : want.items())
        {
            const auto actual = tryGetBoundInt(got, name);
            if(!actual.has_value() || *actual != value.get<int64_t>())
            {
                fail(std::string(source) + " token " + name + " is "
                     + (actual ? std::to_string(*actual) : std::string("missing")) + ", expected "
                     + value.dump());
            }
        }
    }

    /// The slotted runtime's shared-workspace contract (RFC 0021): the `shared_workspace`
    /// token is a non-negative size, `ws.scratch_offset` is that size on the runtime's
    /// alignment, and every `ws.<region>.offset` is a non-negative offset inside it.
    void checkSharedWorkspaceContract(const BoundTokens& bound)
    {
        namespace ingestor = hipdnn_plugin_sdk::ingestor;
        const auto bytes = tryGetBoundInt(bound, "ws.bytes");
        const auto scratch = tryGetBoundInt(bound, ingestor::SCRATCH_OFFSET_TOKEN);
        if(!bytes || *bytes < 0 || !scratch
           || static_cast<std::size_t>(*scratch)
                  != ingestor::alignSharedWorkspace(static_cast<std::size_t>(*bytes)))
        {
            fail("the shared-workspace tokens break the slotted runtime's contract");
            return;
        }
        for(const auto& [name, value] : bound)
        {
            const bool offset = name.rfind("ws.", 0) == 0 && name.size() > 7
                                && name.compare(name.size() - 7, 7, ".offset") == 0;
            const auto* number = std::get_if<int64_t>(&value);
            if(offset && (number == nullptr || *number < 0 || *number >= *bytes))
            {
                fail("token " + name + " is not an offset inside the shared region");
            }
        }
    }

    void checkWorkspace(const general_bwd::WorkspaceLayout& got, const json& want)
    {
        if(got.bytes != want.at("bytes").get<int64_t>()
           || got.scratchOffset != want.at("scratch_offset").get<int64_t>())
        {
            fail("workspace bytes " + std::to_string(got.bytes) + " / scratch "
                 + std::to_string(got.scratchOffset) + ", expected " + want.dump());
        }
        const auto& regions = want.at("regions");
        if(got.regions.size() != regions.size())
        {
            fail("workspace has " + std::to_string(got.regions.size()) + " regions, expected "
                 + regions.dump());
        }
        for(const auto& [name, region] : regions.items())
        {
            const auto it = got.regions.find(name);
            if(it == got.regions.end() || it->second.offset != region.at(0).get<int64_t>()
               || it->second.bytes != region.at(1).get<int64_t>())
            {
                fail("workspace region " + name + " differs from " + region.dump());
            }
        }
    }

    void checkLaunches(const general_bwd::Plan& plan, const json& want)
    {
        if(plan.launches.size() != want.size())
        {
            fail(std::to_string(plan.launches.size()) + " launches, expected "
                 + std::to_string(want.size()));
            return;
        }
        for(std::size_t i = 0; i < want.size(); ++i)
        {
            const auto& launch = plan.launches[i];
            const auto& expected = want[i];
            const std::string name(general_bwd::launchName(launch.launch));
            const std::string where = "launch " + std::to_string(i) + " ('" + name + "')";
            const std::string slot = launch.launch == general_bwd::Launch::PREP ? "prep" : "main";
            if(expected.at("slot").get<std::string>() != slot
               || expected.at("launch").get<std::string>() != name)
            {
                fail(where + " is not " + expected.at("slot").dump() + "/"
                     + expected.at("launch").dump());
            }
            if(launch.kernel != expected.at("kernel").get<std::string>())
            {
                fail(where + " kernel " + std::string(launch.kernel) + ", expected "
                     + expected.at("kernel").dump());
            }
            const auto& grid = expected.at("grid");
            for(std::size_t axis = 0; axis < 3; ++axis)
            {
                if(launch.grid[axis] != grid.at(axis).get<uint32_t>())
                {
                    fail(where + " grid differs from " + grid.dump());
                    break;
                }
            }
            if(launch.block != expected.at("block").get<uint32_t>())
            {
                fail(where + " block " + std::to_string(launch.block) + ", expected "
                     + expected.at("block").dump());
            }
            const auto& kernargs = expected.at("kernargs");
            if(launch.kernargs.size() != kernargs.size())
            {
                fail(where + " has " + std::to_string(launch.kernargs.size())
                     + " kernargs, expected " + std::to_string(kernargs.size()));
                continue;
            }
            for(std::size_t a = 0; a < kernargs.size(); ++a)
            {
                const auto& arg = launch.kernargs[a];
                const auto& row = kernargs.at(a);
                const std::string kind(1, arg.kind);
                const auto value = kernargText(arg);
                if(arg.name != row.at(0).get<std::string>() || kind != row.at(1).get<std::string>()
                   || value != corpusKernargText(row.at(2)))
                {
                    std::ostringstream why;
                    why << where << " kernarg " << a << " is [" << arg.name << ", " << kind << ", "
                        << value << "], expected " << row.dump();
                    fail(why.str());
                }
            }
            ++_tally.launches;
        }
    }

    void checkSelection(const MatchContext& context,
                        const BoundTokens& bound,
                        const general_bwd::Plan& plan)
    {
        std::vector<const KernelDefinition*> admitted;
        for(const auto& kernel : _kernels.main)
        {
            if(_kernelMatcher(context, bound, kernel))
            {
                admitted.push_back(&kernel);
            }
        }
        const auto keep = _candidateFilter(context, bound, admitted);
        std::vector<const KernelDefinition*> mains;
        for(std::size_t i = 0; i < admitted.size() && i < keep.size(); ++i)
        {
            if(keep[i])
            {
                mains.push_back(admitted[i]);
            }
        }
        std::vector<const KernelDefinition*> preps;
        for(const auto& kernel : _kernels.prep)
        {
            if(_kernelMatcher(context, bound, kernel))
            {
                preps.push_back(&kernel);
            }
        }
        if(keep.size() != admitted.size() || preps.size() != 1)
        {
            fail("kernel_match selected " + std::to_string(preps.size())
                 + " prep kernels, expected "
                   "one, and the candidate filter answered "
                 + std::to_string(keep.size()) + " flags for " + std::to_string(admitted.size())
                 + " main kernels");
            return;
        }
        // Tuned keys can share one binary, so the pick is the candidate the native score ranks
        // first (its tuning key and head pack are the plan's), not the one with the plan's name.
        std::vector<const KernelDefinition*> picks;
        for(const auto* kernel : mains)
        {
            if(_mainScore(context, bound, *kernel) > 0.0)
            {
                picks.push_back(kernel);
            }
        }
        if(picks.size() != 1 || picks.front()->name != plan.row->mainKernel)
        {
            fail("the main candidates hold " + std::to_string(picks.size())
                 + " picks of the native score, expected one named "
                 + std::string(plan.row->mainKernel));
            return;
        }
        _tally.candidates += mains.size();
        _tally.withAlternates += mains.size() > 1 ? 1 : 0;
        checkAlternates(context, plan, mains);
        if(preps.front()->name != plan.row->prepKernel)
        {
            fail("kernel_match selected prep kernel " + preps.front()->name + ", the plan names "
                 + std::string(plan.row->prepKernel));
        }
        ++_tally.mainKernelsHit[std::string(plan.row->mainKernel) + "/"
                                + std::to_string(plan.row->tuningKeyId)];
    }

    using CandidateKey = std::pair<int64_t, int64_t>;

    static std::optional<CandidateKey> candidateKeyOf(const KernelDefinition& kernel)
    {
        const auto key = kernel.metadata.find("tuning_key_id");
        const auto pack = kernel.metadata.find("head_pack_on");
        if(key == kernel.metadata.end() || pack == kernel.metadata.end())
        {
            return std::nullopt;
        }
        const auto* keyValue = std::get_if<int64_t>(&key->second);
        const auto* packValue = std::get_if<int64_t>(&pack->second);
        if(keyValue == nullptr || packValue == nullptr)
        {
            return std::nullopt;
        }
        return CandidateKey{*keyValue, *packValue};
    }

    static std::string describe(const std::set<CandidateKey>& keys)
    {
        std::ostringstream text;
        for(const auto& [tuningKey, headPack] : keys)
        {
            text << " (tuning_key " << tuningKey << ", head_pack " << headPack << ')';
        }
        return keys.empty() ? std::string(" none") : text.str();
    }

    /// The Python reference of the alternate main kernels: the engine must admit the pick and
    /// exactly the alternates rocKE's planner builds for the graph, and each one's plan must
    /// reproduce rocKE's main launches (kernel, grid, block and every kernarg value).
    void checkAlternates(const MatchContext& context,
                         const general_bwd::Plan& pick,
                         const std::vector<const KernelDefinition*>& mains)
    {
        if(_alternates == nullptr)
        {
            return;
        }
        std::set<CandidateKey> admitted;
        for(const auto* kernel : mains)
        {
            const auto key = candidateKeyOf(*kernel);
            if(!key.has_value())
            {
                fail("main kernel " + kernel->name
                     + " carries no tuning_key_id / head_pack_on knob values");
                return;
            }
            if(!admitted.insert(*key).second)
            {
                fail("two admitted main kernels share one (tuning_key_id, head_pack_on)");
                return;
            }
        }

        const auto found = _alternates->find(_id);
        const json none = json::array();
        const json& alternates
            = found == _alternates->end() ? none : found->second.at("alternates");
        std::set<CandidateKey> expected{{pick.row->tuningKeyId, pick.row->headPack ? 1 : 0}};
        for(const auto& alternate : alternates)
        {
            expected.insert({alternate.at("tuning_key").get<int64_t>(),
                             alternate.at("head_pack").get<int64_t>()});
        }
        const json noneDeclined = json::array();
        const json& declined
            = found == _alternates->end() ? noneDeclined : found->second.at("declined");
        for(const auto& row : declined)
        {
            const CandidateKey key{row.at("tuning_key").get<int64_t>(),
                                   row.at("head_pack").get<int64_t>()};
            if(admitted.count(key) != 0)
            {
                fail("the engine admits (tuning_key " + std::to_string(key.first) + ", head_pack "
                     + std::to_string(key.second) + "), which its rules decline ("
                     + row.at("reason").get<std::string>() + ")");
            }
            else
            {
                ++_tally.declinedVerified;
            }
        }
        if(admitted != expected)
        {
            std::set<CandidateKey> extra;
            std::set<CandidateKey> missing;
            std::set_difference(admitted.begin(),
                                admitted.end(),
                                expected.begin(),
                                expected.end(),
                                std::inserter(extra, extra.begin()));
            std::set_difference(expected.begin(),
                                expected.end(),
                                admitted.begin(),
                                admitted.end(),
                                std::inserter(missing, missing.begin()));
            fail("the engine admits main kernels rocKE does not build:" + describe(extra)
                 + "; rocKE builds main kernels the engine does not admit:" + describe(missing));
        }

        if(alternates.empty())
        {
            return;
        }
        auto lowered = general_bwd::lowerGraph(context, general_bwd::computeUnitsOf(context));
        const auto* lowering = std::get_if<general_bwd::Lowered>(&lowered);
        const auto traits = general_bwd::shippedArchTraits(_arch);
        const auto in = lowering != nullptr ? general_bwd::planInputOf(lowering->request)
                                            : std::optional<general_bwd::PlanInput>();
        if(!traits.has_value() || !in.has_value())
        {
            fail("the graph has no plan input to build the alternates from");
            return;
        }
        for(const auto& alternate : alternates)
        {
            const auto tuningKey = alternate.at("tuning_key").get<int64_t>();
            general_bwd::Selection selection;
            selection.headPack = alternate.at("head_pack").get<int64_t>() != 0;
            if(tuningKey != 0)
            {
                selection.tuned = general_bwd::tunedEntryById(_arch, tuningKey);
                if(selection.tuned == nullptr)
                {
                    fail("alternate tuning_key " + std::to_string(tuningKey)
                         + " is not a tuned entry of " + _arch);
                    continue;
                }
            }
            auto result = general_bwd::resolveCandidate(*in, *traits, pick, selection);
            if(const auto* decline = std::get_if<general_bwd::Decline>(&result))
            {
                fail("alternate (tuning_key " + std::to_string(tuningKey)
                     + ") is not plannable: " + decline->detail);
                continue;
            }
            const auto failuresBefore = _tally.failures;
            checkAlternatePlan(std::get<general_bwd::Plan>(result), pick, alternate);
            if(_tally.failures == failuresBefore)
            {
                ++_tally.alternatesVerified;
            }
        }
    }

    void checkAlternatePlan(const general_bwd::Plan& plan,
                            const general_bwd::Plan& pick,
                            const json& alternate)
    {
        const std::string tag
            = "alternate (tuning_key " + std::to_string(alternate.at("tuning_key").get<int64_t>())
              + ", head_pack " + std::to_string(alternate.at("head_pack").get<int64_t>()) + ")";
        const auto mainsOf = [](const general_bwd::Plan& of) {
            std::vector<const general_bwd::LaunchPlan*> out;
            for(const auto& launch : of.launches)
            {
                if(launch.launch != general_bwd::Launch::PREP)
                {
                    out.push_back(&launch);
                }
            }
            return out;
        };
        const auto launches = mainsOf(plan);
        const auto pickLaunches = mainsOf(pick);
        const auto& want = alternate.at("launches");
        if(launches.size() != want.size() || launches.size() != pickLaunches.size())
        {
            fail(tag + " has " + std::to_string(launches.size()) + " main launches, rocKE "
                 + std::to_string(want.size()) + ", the pick "
                 + std::to_string(pickLaunches.size()));
            return;
        }
        for(std::size_t i = 0; i < want.size(); ++i)
        {
            const auto& launch = *launches[i];
            const auto& expected = want[i];
            const std::string where = tag + " launch " + std::to_string(i) + " ('"
                                      + std::string(general_bwd::launchName(launch.launch)) + "')";
            if(general_bwd::launchName(launch.launch) != expected.at("launch").get<std::string>())
            {
                fail(where + " is not " + expected.at("launch").dump());
            }
            if(launch.kernel != expected.at("kernel").get<std::string>())
            {
                fail(where + " kernel " + std::string(launch.kernel) + ", rocKE "
                     + expected.at("kernel").dump());
            }
            const auto& grid = expected.at("grid");
            for(std::size_t axis = 0; axis < 3; ++axis)
            {
                if(launch.grid[axis] != grid.at(axis).get<uint32_t>())
                {
                    fail(where + " grid differs from " + grid.dump());
                    break;
                }
            }
            if(launch.block != expected.at("block").get<uint32_t>())
            {
                fail(where + " block " + std::to_string(launch.block) + ", rocKE "
                     + expected.at("block").dump());
            }
            const auto& values = expected.at("values");
            const auto& pickArgs = pickLaunches[i]->kernargs;
            if(launch.kernargs.size() != values.size() || launch.kernargs.size() != pickArgs.size())
            {
                fail(where + " has " + std::to_string(launch.kernargs.size()) + " kernargs, rocKE "
                     + std::to_string(values.size()));
                continue;
            }
            for(std::size_t a = 0; a < values.size(); ++a)
            {
                const auto& arg = launch.kernargs[a];
                const auto value = kernargText(arg);
                if(arg.name != pickArgs[a].name || arg.kind != pickArgs[a].kind
                   || value != corpusKernargText(values.at(a)))
                {
                    std::ostringstream why;
                    why << where << " kernarg " << a << " is [" << arg.name << ", " << arg.kind
                        << ", " << value << "], rocKE " << values.at(a).dump() << " (the pick has ["
                        << pickArgs[a].name << ", " << pickArgs[a].kind << "])";
                    fail(why.str());
                }
            }
        }
    }

    std::string _arch;
    ArchKernels _kernels;
    const std::map<std::string, json>* _alternates;
    hipdnn_plugin_sdk::ingestor::GraphMatchFn _graphMatcher;
    hipdnn_plugin_sdk::ingestor::KernelMatcherFn _kernelMatcher;
    hipdnn_plugin_sdk::ingestor::CandidateFilterFn _candidateFilter;
    hipdnn_plugin_sdk::ingestor::ScoreFn _mainScore;
    ReplayTally _tally;
    std::string _id;
};

class TestAttentionGeneralBwdReplay : public ::testing::TestWithParam<std::string>
{
};

} // namespace

TEST_P(TestAttentionGeneralBwdReplay, EveryCorpusEntryReplaysItsVerdictAndPlan)
{
    const std::string& arch = GetParam();
    const auto path = corpusPath(arch);
    ASSERT_TRUE(std::filesystem::is_regular_file(path))
        << "no " << arch << " corpus at '" << path.string()
        << "': the fixture is staged beside this binary";

    const auto entries = loadCorpus(path, arch);
    ASSERT_FALSE(entries.empty()) << "the corpus '" << path.string() << "' holds no " << arch
                                  << " entries";

    std::map<std::string, json> alternates;
    const auto alternatesFile = alternatesPath(arch);
    // The committed corpus always has its alternates beside it (CMake stages both), so an absent
    // file there is a build slip that would silently turn the candidate parity off. Only a corpus
    // taken from the override may lack one.
    if(hipdnn_data_sdk::utilities::getEnv("HKP_GENERAL_BWD_CORPUS").empty())
    {
        ASSERT_TRUE(alternatesFile.has_value())
            << "no alternates_" << arch << ".jsonl beside the committed corpus '" << path.string()
            << "': the candidate parity against rocKE would be skipped";
    }
    if(alternatesFile.has_value())
    {
        alternates = loadAlternates(*alternatesFile);
    }
    Replay replay(arch, alternatesFile.has_value() ? &alternates : nullptr);
    for(const auto& entry : entries)
    {
        replay.run(entry);
    }

    const auto& tally = replay.tally();
    std::ostringstream summary;
    summary << arch << ": " << tally.entries << " entries, " << tally.accepts << " accepted, "
            << tally.launches << " launches, " << tally.mainKernelsHit.size()
            << " distinct main kernels, " << tally.candidates << " main candidates ("
            << tally.withAlternates << " graphs with alternates), ";
    if(alternatesFile.has_value())
    {
        summary << tally.alternatesVerified << " alternates verified against rocKE, "
                << tally.declinedVerified << " rocKE-built rows the engine declines, ";
    }
    else
    {
        summary << "no alternates file beside the corpus (alternate parity skipped), ";
    }
    summary << tally.failures << " failures; codes:";
    for(const auto& [code, count] : tally.codes)
    {
        summary << ' ' << code << '=' << count;
    }
    std::cout << summary.str() << '\n';

    EXPECT_EQ(tally.failures, 0u) << summary.str();
    EXPECT_GT(tally.accepts, 0u) << "a corpus that accepts nothing proves no plan";
    EXPECT_GT(tally.codes.size(), 1u) << "a corpus with one verdict proves no check order";
    if(alternatesFile.has_value())
    {
        EXPECT_GT(tally.alternatesVerified, 0u)
            << "the alternates file verified no candidate, so the candidate parity proves nothing";
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionGeneralBwdReplay,
                         ::testing::Values(std::string("gfx942"), std::string("gfx950")),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
