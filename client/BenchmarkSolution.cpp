#include "include/BenchmarkSolution.hpp"

namespace rocRoller
{
    namespace Client
    {
        rocRoller::CommandPtr BenchmarkSolution::getCommand()
        {
            return m_command;
        }

        std::shared_ptr<rocRoller::CommandKernel> BenchmarkSolution::getKernel()
        {
            return m_kernel;
        }

        BenchmarkResults BenchmarkSolution::benchmark(RunParameters const& runParams,
                                                      CommandArguments     runtimeArgs)
        {
            BenchmarkResults result;
            result.runParams = runParams;

            // Benchmark runs
            for(int outer = 0; outer < runParams.numOuter; ++outer)
            {
                // Warmup runs
                for(int i = 0; i < runParams.numWarmUp; ++i)
                {
                    m_kernel->launchKernel(runtimeArgs.runtimeArguments());
                }
                HIP_TIMER(t_kernel, "GEMM", runParams.numInner);
                for(int inner = 0; inner < runParams.numInner; ++inner)
                {
                    m_kernel->launchKernel(runtimeArgs.runtimeArguments(), t_kernel, inner);
                }
                HIP_SYNC(t_kernel);
                t_kernel->sleep(50);
                auto nanoseconds = t_kernel->allNanoseconds();
                result.kernelExecute.insert(
                    result.kernelExecute.end(), nanoseconds.begin(), nanoseconds.end());
            }

            double totalTime = 0;
            for(auto ke : result.kernelExecute)
                totalTime += static_cast<double>(ke) / 1.e9;
            double averageTime = totalTime / (runParams.numInner * runParams.numOuter);

            std::cout << "Average runtime (s): " << averageTime << std::endl;

            result.kernelAssemble = TimerPool::nanoseconds("CommandKernel::assembleKernel");
            result.kernelGenerate = TimerPool::nanoseconds("CommandKernel::generateKernel");

            return result;
        }
    }
}
