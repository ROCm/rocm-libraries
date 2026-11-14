// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
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
 * exactly one prep and one main kernel of the arch's catalog, the ones the plan names.
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
    std::map<std::string, std::size_t> codes;
    std::map<std::string, std::size_t> mainKernelsHit;
};

class Replay
{
public:
    explicit Replay(std::string_view arch)
        : _arch(arch)
        , _kernels(archKernels(arch))
    {
        registerNativeIngestorSymbols();
        _graphMatcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
            std::string(GRAPH_MATCHER_SYMBOL));
        _kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
            std::string(KERNEL_MATCHER_SYMBOL));
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
        std::vector<const KernelDefinition*> mains;
        for(const auto& kernel : _kernels.main)
        {
            if(_kernelMatcher(context, bound, kernel))
            {
                mains.push_back(&kernel);
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
        if(mains.size() != 1 || preps.size() != 1)
        {
            fail("kernel_match selected " + std::to_string(preps.size()) + " prep and "
                 + std::to_string(mains.size()) + " main kernels, expected one of each");
            return;
        }
        if(mains.front()->name != plan.row->mainKernel)
        {
            fail("kernel_match selected main kernel " + mains.front()->name + ", the plan names "
                 + std::string(plan.row->mainKernel));
        }
        if(preps.front()->name != plan.row->prepKernel)
        {
            fail("kernel_match selected prep kernel " + preps.front()->name + ", the plan names "
                 + std::string(plan.row->prepKernel));
        }
        ++_tally.mainKernelsHit[std::string(plan.row->mainKernel) + "/"
                                + std::to_string(plan.row->tuningKeyId)];
    }

    std::string _arch;
    ArchKernels _kernels;
    hipdnn_plugin_sdk::ingestor::GraphMatchFn _graphMatcher;
    hipdnn_plugin_sdk::ingestor::KernelMatcherFn _kernelMatcher;
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

    Replay replay(arch);
    for(const auto& entry : entries)
    {
        replay.run(entry);
    }

    const auto& tally = replay.tally();
    std::ostringstream summary;
    summary << arch << ": " << tally.entries << " entries, " << tally.accepts << " accepted, "
            << tally.launches << " launches, " << tally.mainKernelsHit.size()
            << " distinct main kernels, " << tally.failures << " failures; codes:";
    for(const auto& [code, count] : tally.codes)
    {
        summary << ' ' << code << '=' << count;
    }
    std::cout << summary.str() << '\n';

    EXPECT_EQ(tally.failures, 0u) << summary.str();
    EXPECT_GT(tally.accepts, 0u) << "a corpus that accepts nothing proves no plan";
    EXPECT_GT(tally.codes.size(), 1u) << "a corpus with one verdict proves no check order";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionGeneralBwdReplay,
                         ::testing::Values(std::string("gfx942"), std::string("gfx950")),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
