// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A deterministic Find must not reuse the find-db record or the invokers of a default-mode Find
// of the same problem. A default-mode record can hold the time of a kernel that deterministic
// mode does not allow (for example, split-K with atomics).

#include <gtest/gtest.h>

#include "../tensor_holder.hpp"
#include "find_db.hpp"

#include <miopen/handle.hpp>
#include <miopen/solver_id.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <string>

namespace {

using miopen::conv::Direction;
using miopen::conv::ProblemDescription;

struct FindDbPathOverride
{
    explicit FindDbPathOverride(const miopen::fs::path& path)
    {
        miopen::debug::testing_find_db_path_override() = path;
    }
    ~FindDbPathOverride() { miopen::debug::testing_find_db_path_override().reset(); }
};

// Returns the solver names in the find-db record of the problem. An empty set means no record.
// A find-db line has the form "<key>=<solver>:<values>;<solver>:<values>".
std::set<std::string> FindDbSolvers(const miopen::fs::path& find_db,
                                    const ProblemDescription& problem)
{
    std::ostringstream key;
    problem.Serialize(key);
    const auto prefix = key.str() + "=";

    std::set<std::string> solvers;
    std::ifstream file{find_db};
    for(std::string line; std::getline(file, line);)
    {
        if(line.compare(0, prefix.size(), prefix) != 0)
            continue;
        std::istringstream entries{line.substr(prefix.size())};
        for(std::string entry; std::getline(entries, entry, ';');)
            solvers.insert(entry.substr(0, entry.find(':')));
    }
    return solvers;
}

// Runs a wrw Find and returns the number of solutions it found.
int FindWrw(miopen::Handle& handle,
            const miopen::ConvolutionDescriptor& conv,
            const tensor<float>& x,
            const tensor<float>& w,
            const tensor<float>& y,
            const ProblemDescription& problem)
{
    auto x_dev = handle.Write(x.data);
    auto w_dev = handle.Write(w.data);
    auto y_dev = handle.Write(y.data);

    const auto ctx = miopen::ExecutionContext{&handle};
    Workspace wspace{conv.GetWorkSpaceSize(ctx, problem)};

    int ret_algo_count = 0;
    miopenConvAlgoPerf_t perf[1];
    conv.FindConvBwdWeightsAlgorithm(handle,
                                     y.desc,
                                     y_dev.get(),
                                     x.desc,
                                     x_dev.get(),
                                     w.desc,
                                     w_dev.get(),
                                     1,
                                     &ret_algo_count,
                                     perf,
                                     wspace.ptr(),
                                     wspace.size(),
                                     false);
    return ret_algo_count;
}

void RunTest()
{
    // The variable overrides the attribute of both descriptors. Then the test has one mode only.
    if(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC)
        GTEST_SKIP() << "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC is set";

    // One handle for both modes, so the invoker cache is shared as in an application.
    miopen::Handle handle{};

    auto conv_default = miopen::ConvolutionDescriptor{
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    conv_default.findMode.Set(miopen::FindMode::Values::Normal);
    auto conv_det = conv_default;
    conv_det.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_DETERMINISTIC, 1);

    const tensor<float> x{4, 8, 16, 16};
    const tensor<float> w{16, 8, 3, 3};
    const tensor<float> y{conv_default.GetForwardOutputTensor(x.desc, w.desc)};

    const auto problem_default =
        ProblemDescription{y.desc, w.desc, x.desc, conv_default, Direction::BackwardWeights};
    const auto problem_det =
        ProblemDescription{y.desc, w.desc, x.desc, conv_det, Direction::BackwardWeights};

    // One temporary file serves as the user and the system find-db.
    const miopen::TempFile temp_file{"miopen.test.conv_deterministic_find_db"};
    const miopen::fs::path find_db = temp_file;
    const FindDbPathOverride find_db_path_override{find_db};
    const miopen::TestRordbEmbedFsOverrideLock rordb_embed_fs_override;
    ASSERT_TRUE(miopen::debug::testing_find_db_enabled);

    ASSERT_GT(FindWrw(handle, conv_default, x, w, y, problem_default), 0)
        << "The default-mode Find returned no solution";
    const auto solvers_default = FindDbSolvers(find_db, problem_default);
    ASSERT_FALSE(solvers_default.empty());

    // The default-mode Find must not supply a record or an invoker to deterministic mode.
    EXPECT_TRUE(FindDbSolvers(find_db, problem_det).empty());
    const auto config_det = problem_det.MakeNetworkConfig();
    for(const auto& solver : solvers_default)
    {
        EXPECT_FALSE(handle.GetInvoker(config_det, miopen::solver::Id{solver}).has_value())
            << solver;
    }

    ASSERT_GT(FindWrw(handle, conv_det, x, w, y, problem_det), 0)
        << "The deterministic Find returned no solution";
    EXPECT_FALSE(FindDbSolvers(find_db, problem_det).empty());
    // The deterministic Find must not change the default-mode record.
    EXPECT_EQ(FindDbSolvers(find_db, problem_default), solvers_default);
}

} // namespace

// The parameter is not read. It exists only for the GPU_<Name>_FP32 naming convention and the
// "Smoke/" prefix that the CI test categories select.
class GPU_ConvDeterministicFindDb_FP32 : public testing::TestWithParam<miopenDataType_t>
{
};

TEST_P(GPU_ConvDeterministicFindDb_FP32, DefaultFindIsNotReused) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_ConvDeterministicFindDb_FP32, testing::Values(miopenFloat));
