// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <miopen/config.h>

#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE && MIOPEN_BUILD_DRIVER
#include "miopendriver_common.hpp"

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db.hpp>
#include <miopen/perf_field.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>
#include <miopen/solver/implicitgemm_ck_util_common.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tmp_dir.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#endif

namespace {

#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE && MIOPEN_BUILD_DRIVER

using Loader       = miopen::solver::CkImplLibLoader;
using CKSolverType = miopen::solver::CKSolverType;

struct DriverCase
{
    int spatial_dims;
    int spatial;

    bool IsWrw() const { return spatial_dims == 3; }
    std::string Layout() const { return IsWrw() ? "NDHWC" : "NHWC"; }
    CKSolverType SolverType() const
    {
        return IsWrw() ? CKSolverType::GrpConv3dWrw : CKSolverType::GrpConvFwd;
    }
    std::string SolverName() const
    {
        return IsWrw() ? "ConvHipImplicitGemm3DGroupWrwXdlops"
                       : "ConvHipImplicitGemmGroupFwdXdlops";
    }

    std::string Arguments(bool search, bool unrestricted = false) const
    {
        // Match MakeProblem: N=2, G=2, C/G=K/G=64, packed channels-last,
        // filter=3, padding=1, stride=dilation=1 in every spatial dimension.
        std::ostringstream args;
        args << "convfp16 --spatial_dim " << spatial_dims << " -n 2 -c 128 -k 128 -g 2 -H "
             << spatial << " -W " << spatial
             << " -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -z default" << " --in_layout "
             << Layout() << " --fil_layout " << Layout() << " --out_layout " << Layout() << " -F "
             << (IsWrw() ? 4 : 1) << " -M 0 -i 1 -t 1 -V 1";
        if(IsWrw())
            args << " --in_d " << spatial
                 << " --fil_d 3 --pad_d 1 --conv_stride_d 1 --dilation_d 1";
        if(unrestricted)
        {
            // Default Find (-S -1, search=0, policy=0): only the environment's
            // NORMAL/SEARCH_DB_UPDATE settings request exhaustive tuning.
            if(!search)
                args << " -S 0";
        }
        else
            args << (search ? " -S -1 -s 1 --tuning_policy 4" : " -S 0 -s 0 --tuning_policy 1");
        return args.str();
    }
};

struct DriverProblem
{
    miopen::TensorDescriptor x;
    miopen::TensorDescriptor w;
    miopen::TensorDescriptor y;
    miopen::ConvolutionDescriptor conv;

    miopen::conv::ProblemDescription Describe(bool wrw) const
    {
        return {wrw ? y : x,
                w,
                wrw ? x : y,
                conv,
                wrw ? miopen::conv::Direction::BackwardWeights : miopen::conv::Direction::Forward};
    }
};

DriverProblem MakeProblem(const DriverCase& c)
{
    const auto layout = c.IsWrw() ? miopenTensorNDHWC : miopenTensorNHWC;
    std::vector<std::size_t> x_lengths{2, 128};
    std::vector<std::size_t> w_lengths{128, 64};
    for(int d = 0; d < c.spatial_dims; ++d)
    {
        x_lengths.push_back(c.spatial);
        w_lengths.push_back(3);
    }
    miopen::TensorDescriptor x(miopenHalf, layout, x_lengths);
    miopen::TensorDescriptor w(miopenHalf, layout, w_lengths);
    miopen::ConvolutionDescriptor conv(c.spatial_dims,
                                       miopenConvolution,
                                       miopenPaddingDefault,
                                       std::vector<int>(c.spatial_dims, 1),
                                       std::vector<int>(c.spatial_dims, 1),
                                       std::vector<int>(c.spatial_dims, 1),
                                       std::vector<int>(c.spatial_dims, 0),
                                       2);
    auto y = conv.GetForwardOutputTensor(x, w, miopenHalf);
    return {std::move(x), std::move(w), std::move(y), std::move(conv)};
}

struct IsolatedDriverState
{
    miopen::TmpDir root{"ck-tile-driver"};
    const miopen::fs::path user   = root / "user";
    const miopen::fs::path system = root / "system";
    const miopen::fs::path cache  = root / "cache";

    IsolatedDriverState()
    {
        miopen::fs::create_directories(user);
        miopen::fs::create_directories(system);
        miopen::fs::create_directories(cache);
    }

    ~IsolatedDriverState()
    {
        // Unlike compiler temporaries, these databases must be removed even
        // when MIOPEN_DEBUG_SAVE_TEMP_DIR is inherited from the test runner.
        std::error_code ec;
        miopen::fs::remove_all(root.path, ec);
    }

    miopen::ProcessEnvironmentMap
    Environment(const DriverCase& c, bool search, bool unrestricted = false) const
    {
        const auto unlimited = std::to_string(std::numeric_limits<std::size_t>::max());
        // Process inherits the device selection, including HIP_VISIBLE_DEVICES.
        // Do not change the parent's visible GPU or select a physical GPU here.
        miopen::ProcessEnvironmentMap environment{
            {"MIOPEN_USER_DB_PATH", user.string()},
            {"MIOPEN_SYSTEM_DB_PATH", system.string()},
            {"MIOPEN_CUSTOM_CACHE_DIR", cache.string()},
            {"MIOPEN_DEBUG_DISABLE_SYSTEM_DB", "1"},
            {"MIOPEN_DEBUG_DISABLE_USER_DB", "0"},
            {"MIOPEN_DISABLE_CACHE", "0"},
            {"MIOPEN_DEBUG_DISABLE_FIND_DB", "0"},
            {"MIOPEN_SEARCH_SKIP_PCT", "4294967295"},
            {"MIOPEN_COMPILE_PARALLEL_LEVEL", "1"},
            {"MIOPEN_ENABLE_LOGGING", "1"},
            {"MIOPEN_DEBUG_FIND_ONLY_SOLVER", c.SolverName()},
            {"MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", "1"},
            {"MIOPEN_DEBUG_CONV_DIRECT", "0"},
            {"MIOPEN_DEBUG_CONV_GEMM", "0"},
            {"MIOPEN_DEBUG_CONV_WINOGRAD", "0"},
            {"MIOPEN_DEBUG_CONV_FFT", "0"},
            {"MIOPEN_DEBUG_COMPILE_ONLY", "0"},
            {"MIOPEN_FIND_MODE", search ? "1" : "2"},
            {"MIOPEN_FIND_ENFORCE", search ? "4" : "1"},
            {"MIOPEN_DEBUG_TUNING_ITERATIONS_MAX", unlimited},
            {"MIOPEN_TUNING_PATIENCE", unlimited},
            {"MIOPEN_TUNING_TIME_MS_MAX", "7200000"},
            {"MIOPEN_TUNING_ITERATIONS", "10"},
            {"MIOPEN_SEARCH_CUTOFF", "0"},
            {"MIOPEN_LOG_LEVEL", "7"},
            {"MIOPEN_PERFORMANCE_LOGS", "1"},
            {"MIOPEN_DRIVER_USE_GPU_REFERENCE", search ? "0" : "1"}};
        if(unrestricted)
        {
            environment["MIOPEN_DEBUG_FIND_ONLY_SOLVER"] = "";
            for(const auto* algorithm : {"MIOPEN_DEBUG_CONV_IMPLICIT_GEMM",
                                         "MIOPEN_DEBUG_CONV_DIRECT",
                                         "MIOPEN_DEBUG_CONV_GEMM",
                                         "MIOPEN_DEBUG_CONV_WINOGRAD",
                                         "MIOPEN_DEBUG_CONV_FFT"})
                environment[algorithm] = "1";
            environment["MIOPEN_FIND_SKIP_PCT"] = "4294967295";
        }
        return environment;
    }
};

std::set<std::string> EligibleIds(const Loader& loader,
                                  const DriverCase& c,
                                  const miopen::conv::ProblemDescription& problem)
{
    std::set<std::string> ids;
    // Query the catalog independently of the performance-config iterator being
    // tested. In particular, do not call GetAllConfigs or SetNextValue here.
    for(const auto& base : loader.FillValidKernels(c.SolverType(), problem, miopenHalf, false))
    {
        if(!c.IsWrw())
        {
            if(loader.IsArgsSupported(c.SolverType(), problem, base, miopenHalf, false))
                ids.insert(base);
            continue;
        }
        // The WRW search domain is powers of two through 128 plus auto-deduce.
        // Check each actual ID, including every catalog entry's first +1 config.
        int split = 1;
        do
        {
            const auto id = base + "+" + std::to_string(split);
            if(loader.IsArgsSupported(c.SolverType(), problem, id, miopenHalf, false))
                ids.insert(id);
        } while(!miopen::solver::NextCKSplitkValue<1, 128>(split));
    }
    return ids;
}

std::string RunDriver(const DriverCase& c,
                      const IsolatedDriverState& state,
                      bool search,
                      bool unrestricted = false)
{
    std::ostringstream output;
    miopen::Process driver{MIOpenDriverExePath()};
    const auto args = c.Arguments(search, unrestricted);
    const int rc =
        driver(args, state.root.path, &output, state.Environment(c, search, unrestricted));
    EXPECT_EQ(rc, 0) << args << '\n' << output.str();
    return output.str(); // Process captures stdout and stderr together.
}

void CheckSuccessfulRun(const std::string& output, const DriverCase& c, bool search)
{
    for(const auto* failure : {"MIOpen: Error",
                               "Error:",
                               "FAILED",
                               "Failed rc=",
                               "Search Failed:",
                               "Invalid config loaded from Perf Db:",
                               "Skipping config",
                               "exhausted time budget",
                               "Ending Search by patience:",
                               "greater than cutoff:",
                               "Search skipped",
                               "Search cutoff or skipped for all kernels"})
        EXPECT_EQ(output.find(failure), std::string::npos) << failure << '\n' << output;
    const auto direction           = c.IsWrw() ? "backward_weights" : "forward";
    const auto reference           = search ? "CPU" : "GPU";
    const std::string verification = std::string{"\"verification\":{\"direction\":\""} + direction +
                                     "\",\"status\":\"OK\",\"reference\":\"" + reference + "\"";
    EXPECT_NE(output.find(verification), std::string::npos) << output;
}

struct Probe
{
    int calls   = 0;
    double time = 0;
};

struct SearchEvidence
{
    std::map<std::string, Probe> probes;
    std::string winner;
    double winner_mean = 0;
    int done_count     = 0;
};

SearchEvidence ParseSearch(const std::string& output)
{
    const std::regex candidate{R"(Info2 \[GenericSearch\] #([0-9]+)/([0-9]+)/([0-9]+) (.+)$)"};
    const std::regex timing{
        R"(\[GenericSearch\] ##\(n_current, n_failed, n_runs_total\):  ([0-9]+)/([0-9]+)/([0-9]+) elapsed_time: ([^,]+), best_time: [^,]+, (.+)$)"};
    const std::regex done{
        R"(\[GenericSearch\] Done: ([0-9]+)/([0-9]+)/([0-9]+), best #([0-9]+) ([^ ]+) (.+)$)"};
    SearchEvidence evidence;
    std::string current;
    std::istringstream lines{output};
    std::string line;
    std::smatch match;
    while(std::getline(lines, line))
    {
        if(std::regex_search(line, match, candidate))
        {
            EXPECT_EQ(match[2].str(), "0") << line;
            current = match[4].str();
            // Legacy factories can publish duplicate type strings. Coverage is
            // the independently eligible ID set, not incidental catalog multiplicity.
            evidence.probes.try_emplace(current, Probe{});
        }
        else if(std::regex_search(line, match, timing))
        {
            EXPECT_EQ(match[2].str(), "0") << line;
            EXPECT_EQ(match[5].str(), current) << line;
            if(!current.empty())
                evidence.probes.at(current).time = std::stod(match[4].str());
            current.clear();
        }
        else if(std::regex_search(line, match, done))
        {
            EXPECT_EQ(match[2].str(), "0") << line;
            evidence.winner      = match[6].str();
            evidence.winner_mean = std::stod(match[5].str());
            ++evidence.done_count;
        }
        else if(!current.empty())
        {
            const auto pos = line.find("kernel_name = ");
            if(pos != std::string::npos && line.substr(pos + 14) == current)
                ++evidence.probes.at(current).calls;
        }
    }
    EXPECT_TRUE(current.empty()) << "Candidate had no completed measurement: " << current;
    return evidence;
}

template <class Config>
void CheckPersistedWinner(const IsolatedDriverState& state,
                          const DriverCase& c,
                          const miopen::conv::ProblemDescription& problem,
                          const std::string& winner)
{
    int matching_records = 0;
    for(const auto& entry : miopen::fs::recursive_directory_iterator(state.user))
    {
        if(!entry.is_regular_file() || !entry.path().filename().string().ends_with(".udb.txt"))
            continue;
        miopen::PlainTextDb db{miopen::DbKinds::PerfDb, entry.path()};
        Config stored;
        if(db.Load(problem, c.SolverName(), stored))
        {
            ++matching_records;
            EXPECT_EQ(stored.kernel_id, winner) << entry.path();
        }
    }
    EXPECT_EQ(matching_records, 1) << "Winner missing from the exact problem/solver perf-db record";
}

void ExhaustiveSearchAndReload(const DriverCase& c)
{
    const auto device = miopen::solver::GetCurrentDeviceName();
    if(device.find("gfx1250") != 0)
        GTEST_SKIP() << "CK Tile driver tuning requires gfx1250";
    const auto& loader = Loader::Get(device);
    ASSERT_TRUE(loader.IsLoaded()) << "Enabled CK Tile tests require the gfx1250 plugin";
    const auto tensors  = MakeProblem(c);
    const auto problem  = tensors.Describe(c.IsWrw());
    const auto expected = EligibleIds(loader, c, problem);
    std::set<std::string> expected_tile;
    for(const auto& id : expected)
        if(id.find("CKTileGroupedConv") == 0)
            expected_tile.insert(id);
    ASSERT_FALSE(expected_tile.empty()) << "No eligible CK Tile driver candidates";
    if(c.IsWrw())
    {
        for(const auto& base : loader.FillValidKernels(c.SolverType(), problem, miopenHalf, false))
            if(base.find("CKTileGroupedConv") == 0)
                ASSERT_EQ(expected_tile.count(base + "+1"), 1u) << base;
        ASSERT_TRUE(std::any_of(expected_tile.begin(), expected_tile.end(), [](const auto& id) {
            return id.ends_with("+2");
        })) << "The WRW shape must exercise eligible nontrivial split-K candidates";
    }

    IsolatedDriverState state;
    const auto tuned = RunDriver(c, state, true);
    CheckSuccessfulRun(tuned, c, true);
    const miopen::DbRecord record{miopen::DbKinds::PerfDb, problem};
    ASSERT_NE(tuned.find("Search Started: " + record.GetKey() + " : " + c.SolverName()),
              std::string::npos)
        << tuned;
    const auto evidence = ParseSearch(tuned);
    ASSERT_EQ(evidence.done_count, 1) << tuned;
    ASSERT_TRUE(std::isfinite(evidence.winner_mean) && evidence.winner_mean > 0) << tuned;
    std::set<std::string> measured;
    for(const auto& [id, probe] : evidence.probes)
    {
        measured.insert(id);
        EXPECT_GE(probe.calls, 2) << "Candidate did not execute both probe invocations: " << id;
        EXPECT_TRUE(std::isfinite(probe.time) && probe.time > 0)
            << "Candidate has no positive execution timing: " << id;
    }
    // Count actual measured IDs, never the planned total printed by Done.
    EXPECT_EQ(measured, expected) << "Exhaustive driver search omitted or invented candidates";
    for(const auto& id : expected_tile)
        EXPECT_EQ(measured.count(id), 1u) << "Eligible Tile candidate was omitted: " << id;
    ASSERT_EQ(measured.count(evidence.winner), 1u) << tuned;

    // GenericSearch's Done mean is the persistence authority. JSON timings have
    // different sample trimming; their minimum is not an independent winner oracle.
    // Legacy kernels remain competitors, and a legacy winner is explicitly valid.
    if(c.IsWrw())
        CheckPersistedWinner<
            miopen::solver::conv::PerformanceConfigHipImplicitGemm3DGroupWrwXdlops>(
            state, c, problem, evidence.winner);
    else
        CheckPersistedWinner<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops>(
            state, c, problem, evidence.winner);

    // A separate driver process cannot reuse the tuner's registered invoker.
    const auto reloaded = RunDriver(c, state, false);
    CheckSuccessfulRun(reloaded, c, false);
    EXPECT_EQ(reloaded.find("Search Started:"), std::string::npos) << reloaded;
    EXPECT_NE(reloaded.find("Perf Db: record loaded: " + c.SolverName()), std::string::npos)
        << reloaded;
    // Level-7 logs also contain reference/zeroing kernels from the generic HIP
    // runner. Only the CK invoker's actual kernel_name identifies this winner.
    const std::regex invocation{R"(Info2 \[InitInvokerFactory[A-Za-z0-9_]+\] kernel_name = (.+)$)"};
    std::set<std::string> executed;
    std::istringstream lines{reloaded};
    std::string line;
    std::smatch match;
    while(std::getline(lines, line))
    {
        if(std::regex_search(line, match, invocation))
            executed.insert(match[1].str());
    }
    EXPECT_EQ(executed, std::set<std::string>{evidence.winner}) << reloaded;
}

std::set<std::string> ExecutionSolvers(const std::string& output)
{
    // Read the execution-phase solution identity, not JSON candidate timings.
    const std::regex execution{
        R"json(^\{"solution":"([^"]+)","solver_id":[0-9]+,"workspace_bytes":[0-9]+,"phase":"execution",)json"};
    std::set<std::string> solvers;
    std::istringstream lines{output};
    std::string line;
    std::smatch match;
    while(std::getline(lines, line))
        if(std::regex_search(line, match, execution))
            solvers.insert(match[1].str());
    return solvers;
}

void UnrestrictedFindAndReload()
{
    const DriverCase c{2, 8};
    const auto device = miopen::solver::GetCurrentDeviceName();
    if(device.find("gfx1250") != 0)
        GTEST_SKIP() << "CK Tile driver tuning requires gfx1250";
    const auto& loader = Loader::Get(device);
    ASSERT_TRUE(loader.IsLoaded()) << "Enabled CK Tile tests require the gfx1250 plugin";
    const auto tensors  = MakeProblem(c);
    const auto problem  = tensors.Describe(false);
    const auto expected = EligibleIds(loader, c, problem);
    ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [](const auto& id) {
        return id.find("CKTileGroupedConv") == 0;
    })) << "No eligible Tile candidates in the unrestricted driver problem";
    ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [](const auto& id) {
        return id.find("CKTileGroupedConv") != 0;
    })) << "The unrestricted Find must retain eligible legacy CK competition";

    IsolatedDriverState state;
    const auto tuned = RunDriver(c, state, true, true);
    CheckSuccessfulRun(tuned, c, true);
    const miopen::DbRecord perf_record{miopen::DbKinds::PerfDb, problem};
    const std::string search_key = perf_record.GetKey() + " : " + c.SolverName();
    const auto start             = tuned.find("Search Started: " + search_key);
    ASSERT_NE(start, std::string::npos) << tuned;
    const auto end = tuned.find("Search Ended: " + search_key, start);
    ASSERT_NE(end, std::string::npos) << tuned;
    // Other applicable tunable solvers produce their own GenericSearch Done
    // lines. Parse only the grouped-CK search's actual measured candidates.
    const auto evidence = ParseSearch(tuned.substr(start, end - start));
    ASSERT_EQ(evidence.done_count, 1) << tuned;
    ASSERT_TRUE(std::isfinite(evidence.winner_mean) && evidence.winner_mean > 0);
    std::set<std::string> measured;
    for(const auto& [id, probe] : evidence.probes)
    {
        measured.insert(id);
        EXPECT_GE(probe.calls, 2) << id;
        EXPECT_TRUE(std::isfinite(probe.time) && probe.time > 0) << id;
    }
    EXPECT_EQ(measured, expected)
        << "NORMAL/SEARCH_DB_UPDATE did not measure every eligible Tile and legacy CK ID";
    ASSERT_EQ(measured.count(evidence.winner), 1u);
    CheckPersistedWinner<miopen::solver::conv::PerformanceConfigHipImplicitGemmGroupFwdXdlops>(
        state, c, problem, evidence.winner);

    // Find-db holds the measured cross-solver competition. Its fastest entry,
    // not a GenericSearch winner or aggregated JSON minimum, selects -S 0.
    std::map<std::string, miopen::FindDbData> solutions;
    int matching_records = 0;
    for(const auto& entry : miopen::fs::recursive_directory_iterator(state.user))
    {
        if(!entry.is_regular_file() || !entry.path().filename().string().ends_with(".ufdb.txt"))
            continue;
        miopen::PlainTextDb db{miopen::DbKinds::FindDb, entry.path()};
        const auto record = db.FindRecord(problem);
        if(!record)
            continue;
        ++matching_records;
        for(const auto& solution : record->As<miopen::FindDbData>())
            solutions.emplace(solution);
    }
    ASSERT_EQ(matching_records, 1) << "Exact unrestricted Find-db problem record missing";
    ASSERT_EQ(solutions.count(c.SolverName()), 1u);
    const auto& ck_algorithm    = solutions.at(c.SolverName()).algorithm;
    bool other_backend_measured = false;
    float best_time             = std::numeric_limits<float>::max();
    for(const auto& [solver, data] : solutions)
    {
        ASSERT_TRUE(std::isfinite(data.time) && data.time > 0) << solver;
        best_time = std::min(best_time, data.time);
        if(data.algorithm != ck_algorithm)
        {
            EXPECT_NE(tuned.find("Evaluating Solver: " + data.algorithm + ":" + solver),
                      std::string::npos)
                << solver << '\n'
                << tuned;
            other_backend_measured = true;
        }
    }
    ASSERT_TRUE(other_backend_measured)
        << "Unrestricted Find did not measure any backend outside implicit GEMM";
    std::set<std::string> fastest_solvers;
    for(const auto& [solver, data] : solutions)
        if(data.time == best_time)
            fastest_solvers.insert(solver);
    const auto selected = ExecutionSolvers(tuned);
    ASSERT_EQ(selected.size(), 1u) << tuned;
    EXPECT_EQ(fastest_solvers.count(*selected.begin()), 1u)
        << "Driver did not execute a global Find-db winner";

    const auto reloaded = RunDriver(c, state, false, true);
    CheckSuccessfulRun(reloaded, c, false);
    EXPECT_EQ(reloaded.find("Search Started:"), std::string::npos) << reloaded;
    EXPECT_EQ(reloaded.find("Find-db regenerating."), std::string::npos) << reloaded;
    EXPECT_EQ(ExecutionSolvers(reloaded), selected)
        << "Fresh unrestricted immediate-mode process did not replay the global Find-db winner";
}

#endif

TEST(GPU_CKTileDriver_FP16, ExhaustiveFwd2DIncludesEveryEligibleCandidateAndReloadsWinner)
{
#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE && MIOPEN_BUILD_DRIVER
    ExhaustiveSearchAndReload({2, 8});
#else
    GTEST_SKIP() << "Requires HIP, CK Tile, and MIOpenDriver";
#endif
}

TEST(GPU_CKTileDriver_FP16, ExhaustiveWrw3DIncludesEveryEligibleSplitAndReloadsWinner)
{
#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE && MIOPEN_BUILD_DRIVER
    ExhaustiveSearchAndReload({3, 4});
#else
    GTEST_SKIP() << "Requires HIP, CK Tile, and MIOpenDriver";
#endif
}

TEST(GPU_CKTileDriver_FP16, NormalFindTunesAllBackendsAndReloadsGlobalWinner)
{
#if MIOPEN_BACKEND_HIP && defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE && MIOPEN_BUILD_DRIVER
    UnrestrictedFindAndReload();
#else
    GTEST_SKIP() << "Requires HIP, CK Tile, and MIOpenDriver";
#endif
}

} // namespace
