#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/Operations/CommandArguments.hpp>

#include "BenchmarkSolution.hpp"
#include "GEMMParameters.hpp"

#include "../../../test/unit/Utilities.hpp"

using namespace rocRoller;

namespace rocRoller
{
    namespace Client
    {
        namespace GEMMClient
        {
            const int wavefrontSize = 64;

            template <typename A, typename B, typename C, typename D>
            class GEMMSolution : public BenchmarkSolution
            {
            protected:
                GEMMSolution() {}
                ProblemParameters m_problemParams;

            public:
                GEMMSolution(ProblemParameters const& problemParams)
                {
                    m_problemParams = problemParams;
                }

                std::pair<bool, double> validate(std::vector<A> h_A,
                                                 std::vector<B> h_B,
                                                 std::vector<C> h_C,
                                                 std::vector<D> h_D)
                {
                    // Host result
                    std::vector<D> h_result(m_problemParams.m * m_problemParams.n, 0.0);
                    CPUMM(h_result,
                          h_C,
                          h_A,
                          h_B,
                          m_problemParams.m,
                          m_problemParams.n,
                          m_problemParams.k,
                          m_problemParams.alpha,
                          m_problemParams.beta,
                          m_problemParams.transA == TransposeType::T,
                          m_problemParams.transB == TransposeType::T);

                    auto tol = gemmAcceptableError<A, B, D>(
                        m_problemParams.m,
                        m_problemParams.n,
                        m_problemParams.k,
                        getKernel()->getContext()->targetArchitecture().target());
                    auto res = compare(h_D, h_result, tol);

                    std::cout << "Result: " << (res.ok ? "Correct" : "Incorrect") << std::endl;
                    std::cout << "RNorm: " << res.relativeNormL2 << std::endl;
                    if(!res.ok)
                    {
                        std::cerr << "WARNING: Result incorrect.  " << res.message() << std::endl;
                    }
                    return {res.ok, res.relativeNormL2};
                }

                BenchmarkResults benchmark(RunParameters const& runParams,
                                           CommandArguments     runtimeArgs)
                {
                    BenchmarkResults result;

                    result = BenchmarkSolution::benchmark(runParams, runtimeArgs);

                    double totalTime = 0;
                    for(auto ke : result.kernelExecute)
                        totalTime += static_cast<double>(ke) / 1.e9;
                    double averageTime = totalTime / (runParams.numInner * runParams.numOuter);

                    std::cout << "Average GFLOPS: "
                              << (double)m_problemParams.m * m_problemParams.n * m_problemParams.k
                                     * 2.0 / averageTime * 1.e-9
                              << std::endl;

                    return result;
                }
            };
        }
    }
}
