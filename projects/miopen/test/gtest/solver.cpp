/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <miopen/convolution.hpp>
#include <miopen/db.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/temp_file.hpp>

#include <cstdlib>
#include <functional>
#include <sstream>
#include <string_view>
#include <typeinfo>
#include <utility>

#include "random.hpp"
#include "../get_handle.hpp"

namespace miopen {
namespace tests {

class TrivialTestSolver final : public solver::conv::ConvSolver
{
public:
    static const char* FileName() { return "TrivialTestSolver"; }

    const std::string& SolverDbId() const override { return GetSolverDbId<TrivialTestSolver>(); }

    bool IsApplicable(const ExecutionContext&,
                      const conv::ProblemDescription& problem) const override
    {
        return problem.GetInWidth() == 1;
    }

    solver::ConvSolution GetSolution(const ExecutionContext&,
                                     const conv::ProblemDescription&) const override
    {
        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = FileName();
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }
};

struct TestConfig : solver::PerfConfigBase<TestConfig>
{
    std::string str;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.str, "str");
    }
};

class SearchableTestSolver final : public solver::conv::ConvTunableSolver<TestConfig>
{
public:
    static int searches_done() { return _serches_done; }
    static const char* FileName() { return "SearchableTestSolver"; }
    static const char* NoSearchFileName() { return "SearchableTestSolver.NoSearch"; }

    const std::string& SolverDbId() const override { return GetSolverDbId<SearchableTestSolver>(); }

    bool IsApplicable(const ExecutionContext&, const conv::ProblemDescription&) const override
    {
        return true;
    }

    TestConfig GetDefaultPerformanceConfig(const ExecutionContext&,
                                           const conv::ProblemDescription&) const override
    {
        TestConfig config{};
        config.str = NoSearchFileName();
        return config;
    }

    bool IsValidPerformanceConfig(const ExecutionContext&,
                                  const conv::ProblemDescription&,
                                  const TestConfig&) const override
    {
        return true;
    }

    TestConfig Search(const ExecutionContext&,
                      const conv::ProblemDescription&,
                      const AnyInvokeParams&) const override
    {
        TestConfig config;
        config.str = FileName();
        _serches_done++;
        return config;
    }

    solver::ConvSolution GetSolution(const ExecutionContext&,
                                     const conv::ProblemDescription&,
                                     const TestConfig& config) const override
    {

        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = config.str;
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }

private:
    static int _serches_done; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)
};

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
int SearchableTestSolver::_serches_done = 0;

static solver::ConvSolution FindSolution(const ExecutionContext& ctx,
                                         const conv::ProblemDescription& problem,
                                         const fs::path& db_path)
{
    PlainTextDb db(DbKinds::PerfDb, db_path);

    const auto solvers = solver::SolverContainer<TrivialTestSolver, SearchableTestSolver>{};

    return solvers.SearchForAllSolutions(ctx, problem, db, {}, 1).front();
}

template <class TInstance>
class StaticContainer
{
public:
    inline static TInstance& Instance()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static TInstance data{};
        return data;
    }
};

struct TestParams
{
    std::string expected_kernel;
    std::vector<size_t> in;
    std::function<void(ExecutionContext&)> context_filter = [](ExecutionContext&) {};
};

using TestCase =
    std::tuple<std::vector<TestParams>,
               std::vector<TestParams>>; // pre-check parameters and post-check parameters

auto GenCases()
{
    return std::vector<TestCase>{std::make_tuple<std::vector<TestParams>, std::vector<TestParams>>(
        {TestParams{TrivialTestSolver::FileName(), {1, 1, 1, 1}, [](ExecutionContext&) {}},
         TestParams{TrivialTestSolver::FileName(),
                    {1, 1, 1, 1},
                    [](ExecutionContext& c) { c.do_search = true; }},
         TestParams{SearchableTestSolver::NoSearchFileName(),
                    {1, 1, 1, 2},
                    [](ExecutionContext& c) { c.do_search = false; }},
         TestParams{SearchableTestSolver::FileName(),
                    {1, 1, 1, 2},
                    [](ExecutionContext& c) { c.do_search = true; }}},
        {TestParams{SearchableTestSolver::FileName(), {1, 1, 1, 2}, [](ExecutionContext&) {}},
         TestParams{SearchableTestSolver::FileName(), {1, 1, 1, 2}, [](ExecutionContext& c) {
                        c.do_search = true;
                    }}})};
}

auto GetCases()
{
    static auto cases = GenCases();
    return cases;
}

} // namespace tests
} // namespace miopen

using namespace miopen;
using namespace miopen::tests;

class SolverTest : public ::testing::TestWithParam<TestCase>
{
public:
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        auto [pre_check_param, post_check_param] = GetParam();
        const TempFile db_path("miopen.tests.solver");

        for(auto const& param : pre_check_param)
        {
            ConstructTest(db_path, param.expected_kernel.c_str(), param.in, param.context_filter);
        }

        const auto& searchable_solver = StaticContainer<const SearchableTestSolver>::Instance();
        const auto searches           = SearchableTestSolver::searches_done();

        // Should read in both cases: result is already in DB, solver is searchable.

        for(auto const& param : post_check_param)
        {
            ConstructTest(db_path, param.expected_kernel.c_str(), param.in, param.context_filter);
        }

        // Checking no more searches were done.
        EXPECT_EQ(searches, searchable_solver.searches_done());
    }

protected:
    void ConstructTest(
        const fs::path& db_path,
        const char* expected_kernel,
        const std::vector<size_t>& in,
        const std::function<void(ExecutionContext&)>& context_filler = [](ExecutionContext&) {
        }) const
    {
        const auto problem = conv::ProblemDescription{TensorDescriptor{miopenFloat, in},
                                                      TensorDescriptor{miopenFloat, in},
                                                      TensorDescriptor{miopenFloat, in},
                                                      ConvolutionDescriptor{},
                                                      conv::Direction::Forward};
        auto ctx           = ExecutionContext{};
        ctx.SetStream(&get_handle());
        context_filler(ctx);

        const auto sol = FindSolution(ctx, problem, db_path);

        EXPECT_TRUE(sol.construction_params.size() > 0);
        EXPECT_EQ(sol.construction_params[0].kernel_file, expected_kernel);
    }
};

TEST_P(SolverTest, GPU_TestSolver_None) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke, SolverTest, ::testing::ValuesIn(GetCases()), [](auto const&) {
    return "SearchesDoneTest";
});
