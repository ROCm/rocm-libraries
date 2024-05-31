#pragma once
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/KernelOptions.hpp>

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

                    double rnorm = relativeNorm(h_D, h_result);

                    bool isCorrect = rnorm < 3e-5;
                    std::cout << "Result: " << (isCorrect ? "Correct" : "Incorrect") << std::endl;
                    std::cout << "RNorm: " << rnorm << std::endl;
                    if(!isCorrect)
                    {
                        std::cerr << "WARNING: Result incorrect. RNorm too large: " << rnorm
                                  << std::endl;
                    }
                    return {isCorrect, rnorm};
                }

                BenchmarkResults benchmark(RunParameters const&       runParams,
                                           rocRoller::KernelArguments runtimeArgs)
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
